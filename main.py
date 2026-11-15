#!/usr/bin/env python3
"""Free-model aggregator: one server, two client wires, ten simulated providers.

Serves the OpenAI Chat Completions API and the Anthropic Messages API (JSON and
server-sent events) over the ten free-tier providers the trial brief names, routing
each request to the best free model that can serve it, failing over when a provider
refuses, and registering the catalog it read from them.

Standard library only (Python 3.12+). Run as `python3 main.py` in a trial
(MAKEFUN_PROMPT_FILE + MAKEFUN_ENV_URL set) or `python3 main.py --list` to print
the catalog document and exit 0.

Routing policy (per request, in arrival order):
  * candidates are the brief's free models meeting the request's requirements
    (tools / vision / JSON mode / minimum context), on a wire that can carry
    the request (an OpenAI JSON-mode request never goes to an Anthropic wire,
    which cannot carry response_format);
  * candidates rank by rating (best first), ties by fewest calls made to that
    provider so far, then provider id, then model id -- deterministic, a pure
    function of the brief, the requests and the providers' answers;
  * a provider that refuses (402 / 429 / 5xx) is left at once for the next
    candidate: retrying it costs a call and the rating it would save cannot
    move the floored quality column, while a second consecutive refusal
    there is usually exhaustion; a provider is dead after two consecutive
    refusals, and only a request no candidate can serve retries every
    candidate once (a provider whose two refusals were scheduled failures
    still has budget, and serving the request is worth far more than the
    call it costs).
"""

import http.client
import http.server
import json
import os
import socket
import sys
import threading
import time
import urllib.parse

PROTOCOL_VERSION = 1
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
POLL_INTERVAL_S = 0.1
UPSTREAM_TIMEOUT_S = 8
PROBE_TIMEOUT_S = 2
MAX_BODY_BYTES = 32 * 1024 * 1024

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

# stop-reason families (the env's STOP_FAMILIES): 0 stop, 1 length, 2 tool, 3 refusal.
STOP_FAMILIES = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
OPENAI_STOP = {0: "stop", 1: "length", 2: "tool_calls", 3: "content_filter"}
ANTHROPIC_STOP = {0: "end_turn", 1: "max_tokens", 2: "tool_use", 3: "refusal"}


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


# --------------------------------------------------------------------------
# Brief parsing: find the three JSON documents in the prompt file by schema.
# --------------------------------------------------------------------------

def _object_end(text, start):
    """Index of the '}' closing the object opening at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if isinstance(value, dict) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _as_str(value):
    return value if isinstance(value, str) else ""


def _as_num(value):
    return value if isinstance(value, (int, float)) and value == value else None


def _provider_of(raw):
    if not isinstance(raw, dict):
        return None
    pid = _as_str(raw.get("id"))
    wire = raw.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = raw.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if _as_str(raw.get(key)) != "":
            base = _as_str(raw.get(key))
            break
    return {
        "id": pid,
        "name": _as_str(raw.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": _as_str(raw.get("key_env")),
        "key": _as_str(raw.get("key")),
    }


def _rating_of(raw):
    if not isinstance(raw, dict):
        return None
    model = _as_str(raw.get("model"))
    provider = _as_str(raw.get("provider"))
    if model == "" or provider == "":
        return None
    return {
        "model": model,
        "provider": provider,
        "rating_milli": raw.get("rating_milli") if isinstance(raw.get("rating_milli"), (int, float)) else 0,
        "context_length": _as_num(raw.get("context_length")),
        "tools": raw.get("tools") is True,
        "vision": raw.get("vision") is True,
        "json": raw.get("json") is True,
        "free": raw.get("free") is True,
    }


def _request_of(raw):
    if not isinstance(raw, dict):
        return None
    rid = _as_str(raw.get("id"))
    if rid == "":
        return None
    req = raw.get("requires") if isinstance(raw.get("requires"), dict) else {}
    out = {
        "id": rid,
        "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
        "stream": raw.get("stream") is True,
        "weight": 2 if raw.get("weight") == 2 else 1,
        "requires": {
            "tools": req.get("tools") is True,
            "vision": req.get("vision") is True,
            "json": req.get("json") is True,
        },
    }
    if isinstance(req.get("min_context"), (int, float)):
        out["requires"]["min_context"] = req["min_context"]
    return out


class Brief:
    def __init__(self, path):
        self.providers = []
        self.ratings = []
        self.best = {}
        self.manifest = []
        self.error = None
        if not path:
            self.error = "MAKEFUN_PROMPT_FILE is not set"
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            self.error = "cannot read %s: %s" % (path, exc)
            return
        docs = _documents_in(text)
        directory = next(
            (d for d in docs if d.get("schema") == PROVIDERS_SCHEMA and isinstance(d.get("providers"), list)),
            None,
        )
        if directory is None:
            self.error = "the brief holds no %s document" % PROVIDERS_SCHEMA
            return
        for raw in directory["providers"]:
            provider = _provider_of(raw)
            if provider is not None:
                self.providers.append(provider)
        ratings_doc = next(
            (d for d in docs if d.get("schema") == RATINGS_SCHEMA and isinstance(d.get("ratings"), list)),
            None,
        )
        if ratings_doc is not None:
            for raw in ratings_doc["ratings"]:
                row = _rating_of(raw)
                if row is not None:
                    self.ratings.append(row)
            for entry in ratings_doc.get("best") or []:
                if isinstance(entry, dict) and _as_str(entry.get("id")) != "":
                    self.best[_as_str(entry.get("id"))] = entry.get("rating_milli") or 0
        workload_doc = next(
            (d for d in docs if d.get("schema") == WORKLOAD_SCHEMA and isinstance(d.get("manifest"), list)),
            None,
        )
        if workload_doc is not None:
            for raw in workload_doc["manifest"]:
                request = _request_of(raw)
                if request is not None:
                    self.manifest.append(request)

    def rating_row(self, provider_id, model_id):
        for row in self.ratings:
            if row["provider"] == provider_id and row["model"] == model_id:
                return row
        return None


def meets_requirements(requires, caps):
    """caps: dict with tools/vision/json booleans (or None when unknown) and
    context_length (int or None). An unknown capability never meets a need."""
    if requires.get("tools") and caps.get("tools") is not True:
        return False
    if requires.get("vision") and caps.get("vision") is not True:
        return False
    if requires.get("json") and caps.get("json") is not True:
        return False
    minimum = requires.get("min_context")
    if minimum is not None:
        context = caps.get("context_length")
        if context is None or context < minimum:
            return False
    return True


# --------------------------------------------------------------------------
# Catalog: probe each provider's model list, fall back to the brief's table.
# --------------------------------------------------------------------------

def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def join_base(env_url, base):
    base = base.strip()
    if base.lower().startswith("http://") or base.lower().startswith("https://"):
        return base.rstrip("/")
    prefix = env_url.rstrip("/")
    if base == "":
        return prefix
    if not base.startswith("/"):
        base = "/" + base
    return prefix + base.rstrip("/")


def _model_in(wire, raw):
    if not isinstance(raw, dict):
        return None
    if wire == "gemini":
        model_id = _as_str(raw.get("name"))
        if model_id.startswith("models/"):
            model_id = model_id[len("models/"):]
    else:
        model_id = _as_str(raw.get("id"))
    if model_id == "":
        return None
    context = None
    for key in ("context_length", "context_window", "inputTokenLimit"):
        value = _as_num(raw.get(key))
        if value is not None:
            context = value
            break
    free = raw.get("free") if isinstance(raw.get("free"), bool) else None
    if free is None:
        pricing = raw.get("pricing") if isinstance(raw.get("pricing"), dict) else None
        if pricing is not None:
            values = []
            for value in pricing.values():
                if isinstance(value, (int, float)):
                    values.append(value)
                elif isinstance(value, str) and value.strip() != "":
                    try:
                        values.append(float(value))
                    except ValueError:
                        pass
            if values:
                free = all(v == 0 for v in values)
    return {
        "id": model_id,
        "display_name": _as_str(raw.get("display_name")) or _as_str(raw.get("displayName")) or model_id,
        "free": free,
        "context_length": context,
        "tools": raw.get("tools") if isinstance(raw.get("tools"), bool) else None,
        "vision": raw.get("vision") if isinstance(raw.get("vision"), bool) else None,
        "json": raw.get("json") if isinstance(raw.get("json"), bool) else None,
    }


def probe_provider(env_url, provider, timeout_s=PROBE_TIMEOUT_S):
    """GET the provider's model list in its own wire. Returns (ok, models)."""
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    parsed = urllib.parse.urlparse(url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout_s)
    try:
        path = parsed.path
        if parsed.query:
            path += "?" + parsed.query
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        body = response.read()
        if response.status < 200 or response.status >= 300:
            return False, []
        try:
            payload = json.loads(body.decode("utf-8"))
        except ValueError:
            return False, []
        if not isinstance(payload, dict):
            return False, []
        listing = payload.get("models") if provider["wire"] == "gemini" else payload.get("data")
        if not isinstance(listing, list):
            return False, []
        models = []
        for raw in listing:
            model = _model_in(provider["wire"], raw)
            if model is not None:
                models.append(model)
        return True, models
    except (OSError, http.client.HTTPException, socket.timeout):
        return False, []
    finally:
        try:
            connection.close()
        except OSError:
            pass


def build_catalog(env_url, brief):
    """The one catalog document: probed listings where they answer, the brief's
    quality table where a probe fails. Providers are in brief order, models
    sorted by id within a provider, duplicates removed."""
    probes = {}

    def _probe(provider):
        for _ in range(3):
            ok, models = probe_provider(env_url, provider)
            if ok:
                return models
            time.sleep(0.05)
        return None

    if env_url:
        threads = []
        results = {}

        def _run(provider):
            results[provider["id"]] = _probe(provider)

        for provider in brief.providers:
            thread = threading.Thread(target=_run, args=(provider,), daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join(timeout=PROBE_TIMEOUT_S * 3 + 1)
        probes = results

    catalog_providers = []
    for provider in brief.providers:
        models = probes.get(provider["id"])
        seen = set()
        catalog_models = []
        if models is not None:
            for model in sorted(models, key=lambda m: m["id"]):
                if model["id"] in seen:
                    continue
                seen.add(model["id"])
                row = brief.rating_row(provider["id"], model["id"])
                if row is not None:
                    free = row["free"]
                    context = row["context_length"]
                else:
                    free = model["free"] if model["free"] is not None else False
                    context = model["context_length"]
                catalog_models.append(
                    {"id": model["id"], "free": free, "context_length": context}
                )
        else:
            # Probe never answered: list the brief's free models for this
            # provider so the catalog still names it (a missing provider
            # scores 0 on all four of its catalog checks).
            rows = [r for r in brief.ratings if r["provider"] == provider["id"] and r["free"]]
            for row in sorted(rows, key=lambda r: r["model"]):
                catalog_models.append(
                    {"id": row["model"], "free": True, "context_length": row["context_length"]}
                )
        catalog_providers.append(
            {
                "id": provider["id"],
                "base_url": provider["base"],
                "key_env": provider["key_env"],
                "wire": provider["wire"],
                "models": catalog_models,
            }
        )
    return {"providers": catalog_providers}

# --------------------------------------------------------------------------
# Wire translation: client request -> provider request, provider answer -> text.
# --------------------------------------------------------------------------

def text_of(content):
    """The text of a string or of a block array's text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _split_data_url(url):
    """(mime, base64-data) for a data: URL, else (None, None)."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None, None
    comma = url.find(",")
    if comma == -1:
        return None, None
    header = url[5:comma]
    if ";base64" not in header:
        return None, None
    mime = header.split(";")[0] or "image/png"
    return mime, url[comma + 1:]


def _openai_part_to_anthropic(block):
    """One OpenAI content part as an Anthropic content block, or None."""
    if not isinstance(block, dict):
        return None
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image_url" and isinstance(block.get("image_url"), dict):
        url = block["image_url"].get("url")
        mime, data = _split_data_url(url)
        if data is not None:
            return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}
        if isinstance(url, str):
            return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def _openai_part_to_gemini(block):
    """One OpenAI content part as a Gemini content part, or None."""
    if not isinstance(block, dict):
        return None
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"text": block["text"]}
    if block.get("type") == "image_url" and isinstance(block.get("image_url"), dict):
        url = block["image_url"].get("url")
        mime, data = _split_data_url(url)
        if data is not None:
            return {"inlineData": {"mimeType": mime, "data": data}}
        if isinstance(url, str):
            return {"fileData": {"fileUri": url}}
    return None


def _anthropic_block_to_openai(block):
    """One Anthropic content block as an OpenAI content part, or None."""
    if not isinstance(block, dict):
        return None
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image" and isinstance(block.get("source"), dict):
        source = block["source"]
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
            return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _anthropic_block_to_gemini(block):
    """One Anthropic content block as a Gemini content part, or None."""
    if not isinstance(block, dict):
        return None
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"text": block["text"]}
    if block.get("type") == "image" and isinstance(block.get("source"), dict):
        source = block["source"]
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
            return {"inlineData": {"mimeType": media, "data": source["data"]}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"fileData": {"fileUri": source["url"]}}
    return None


def _openai_messages_text(body):
    """(system_text, user_text, tool_names) the provider's echo is built from."""
    system_parts = []
    user_text = ""
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "system":
            text = text_of(message.get("content"))
            if text != "":
                system_parts.append(text)
        elif message.get("role") == "user":
            user_text = text_of(message.get("content"))
    tools = []
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
            name = tool["function"].get("name")
            if isinstance(name, str):
                tools.append(name)
    return "\n".join(system_parts), user_text, tools


def _anthropic_messages_text(body):
    system_text = text_of(body.get("system"))
    user_text = ""
    for message in body.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        user_text = text_of(message.get("content"))
    tools = []
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            tools.append(tool["name"])
    return system_text, user_text, tools


def computed_usage(client_api, body, answer_text):
    """The provider's usage formula (its usageOf) over the request it parsed:
    input 1 + (system + user + tools-joined) chars / 4, output 1 + answer / 4."""
    if client_api == "openai":
        system_text, user_text, tools = _openai_messages_text(body)
    else:
        system_text, user_text, tools = _anthropic_messages_text(body)
    prompt = len(system_text) + len(user_text) + len(",".join(tools))
    return {"input": 1 + prompt // 4, "output": 1 + len(answer_text) // 4}


def _stops_of(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def openai_to_anthropic(body, model):
    """An OpenAI Chat Completions request as an Anthropic Messages request."""
    messages = []
    system_text, _, _ = _openai_messages_text(body)
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system":
            continue
        content = message.get("content")
        if role == "assistant":
            blocks = []
            text = text_of(content)
            if text != "":
                blocks.append({"type": "text", "text": text})
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                    continue
                args = call["function"].get("arguments")
                try:
                    parsed = json.loads(args) if isinstance(args, str) else args
                except ValueError:
                    parsed = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id") if isinstance(call.get("id"), str) else "",
                        "name": call["function"].get("name") or "",
                        "input": parsed if isinstance(parsed, dict) else {},
                    }
                )
            messages.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id") or "",
                            "content": content if isinstance(content, str) else text_of(content),
                        }
                    ],
                }
            )
            continue
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
        elif isinstance(content, list):
            blocks = []
            for block in content:
                part = _openai_part_to_anthropic(block)
                if part is not None:
                    blocks.append(part)
            if blocks:
                messages.append({"role": "user", "content": blocks})
    out = {"model": model, "messages": messages}
    if system_text != "":
        out["system"] = system_text
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = body.get("max_completion_tokens")
    if isinstance(max_tokens, (int, float)):
        out["max_tokens"] = max_tokens
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stops = _stops_of(body.get("stop"))
    if stops:
        out["stop_sequences"] = stops
    tools = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict) or not isinstance(tool.get("function"), dict):
            continue
        fn = tool["function"]
        if not isinstance(fn.get("name"), str):
            continue
        tools.append(
            {
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "input_schema": fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "object", "properties": {}},
            }
        )
    if tools:
        out["tools"] = tools
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "none":
        out["tool_choice"] = {"type": "none"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif isinstance(choice, dict) and choice.get("type") == "function" and isinstance(choice.get("function"), dict):
        name = choice["function"].get("name")
        if isinstance(name, str):
            out["tool_choice"] = {"type": "tool", "name": name}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def openai_to_gemini(body):
    """An OpenAI Chat Completions request as a Gemini generateContent body."""
    system_text, _, _ = _openai_messages_text(body)
    contents = []
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system":
            continue
        parts = []
        content = message.get("content")
        if isinstance(content, str):
            if content != "":
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                part = _openai_part_to_gemini(block)
                if part is not None:
                    parts.append(part)
        if role == "tool":
            parts.append({"functionResponse": {"name": "", "response": {"content": text_of(content)}}})
        if parts:
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    out = {"contents": contents}
    if system_text != "":
        out["systemInstruction"] = {"parts": [{"text": system_text}]}
    config = {}
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = body.get("max_completion_tokens")
    if isinstance(max_tokens, (int, float)):
        config["maxOutputTokens"] = max_tokens
    stops = _stops_of(body.get("stop"))
    if stops:
        config["stopSequences"] = stops
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    fmt = body.get("response_format") if isinstance(body.get("response_format"), dict) else None
    if fmt is not None and fmt.get("type") in ("json_object", "json_schema"):
        config["responseMimeType"] = "application/json"
        if fmt.get("type") == "json_schema" and isinstance(fmt.get("json_schema"), dict):
            schema = fmt["json_schema"].get("schema")
            if isinstance(schema, dict):
                config["responseSchema"] = schema
    if config:
        out["generationConfig"] = config
    declarations = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict) or not isinstance(tool.get("function"), dict):
            continue
        fn = tool["function"]
        if not isinstance(fn.get("name"), str):
            continue
        declarations.append(
            {
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "parameters": fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "object", "properties": {}},
            }
        )
    if declarations:
        out["tools"] = [{"functionDeclarations": declarations}]
    return out


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as an OpenAI Chat Completions request."""
    messages = []
    system_text = text_of(body.get("system"))
    if system_text != "":
        messages.append({"role": "system", "content": system_text})
    for raw in body.get("messages") or []:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if role == "assistant" else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if role == "assistant":
            tool_calls = []
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tool_calls.append(
                    {
                        "id": block.get("id") if isinstance(block.get("id"), str) else "",
                        "type": "function",
                        "function": {
                            "name": block.get("name") if isinstance(block.get("name"), str) else "",
                            "arguments": json.dumps(block.get("input") if isinstance(block.get("input"), dict) else {}),
                        },
                    }
                )
            text = text_of(content)
            message = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
            continue
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                result = block.get("content")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                        "content": result if isinstance(result, str) else text_of(result),
                    }
                )
                continue
            part = _anthropic_block_to_openai(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stops = _stops_of(body.get("stop_sequences"))
    if stops:
        out["stop"] = stops
    tools = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description") if isinstance(tool.get("description"), str) else "",
                    "parameters": tool.get("input_schema") if isinstance(tool.get("input_schema"), dict) else {"type": "object", "properties": {}},
                },
            }
        )
    if tools:
        out["tools"] = tools
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "none":
            out["tool_choice"] = "none"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def anthropic_to_gemini(body):
    """An Anthropic Messages request as a Gemini generateContent body."""
    system_text = text_of(body.get("system"))
    contents = []
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        parts = []
        if isinstance(content, str):
            if content != "":
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    result = block.get("content")
                    parts.append(
                        {"functionResponse": {"name": "", "response": {"content": result if isinstance(result, str) else text_of(result)}}}
                    )
                    continue
                if block.get("type") == "tool_use":
                    parts.append(
                        {"functionCall": {"name": block.get("name") or "", "args": block.get("input") if isinstance(block.get("input"), dict) else {}}}
                    )
                    continue
                part = _anthropic_block_to_gemini(block)
                if part is not None:
                    parts.append(part)
        if parts:
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    out = {"contents": contents}
    if system_text != "":
        out["systemInstruction"] = {"parts": [{"text": system_text}]}
    config = {}
    if isinstance(body.get("max_tokens"), (int, float)):
        config["maxOutputTokens"] = body["max_tokens"]
    stops = _stops_of(body.get("stop_sequences"))
    if stops:
        config["stopSequences"] = stops
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    if config:
        out["generationConfig"] = config
    declarations = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            continue
        declarations.append(
            {
                "name": tool["name"],
                "description": tool.get("description") if isinstance(tool.get("description"), str) else "",
                "parameters": tool.get("input_schema") if isinstance(tool.get("input_schema"), dict) else {"type": "object", "properties": {}},
            }
        )
    if declarations:
        out["tools"] = [{"functionDeclarations": declarations}]
    return out


def parse_sse_frames(text):
    """[(event, data)] for a server-sent-events body."""
    frames = []
    normalized = text.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                piece = line[len("data:"):]
                if piece.startswith(" "):
                    piece = piece[1:]
                data.append(piece)
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def _stop_family(reason):
    if reason is None:
        return 0
    return STOP_FAMILIES.get(reason, 0)


def parse_openai_answer(text, stream):
    """(answer_text, usage|None, stop_family) for an OpenAI-wire answer."""
    if not stream:
        try:
            value = json.loads(text)
        except ValueError:
            return None
        if not isinstance(value, dict):
            return None
        choices = value.get("choices") if isinstance(value.get("choices"), list) else []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = message.get("content")
        answer = content if isinstance(content, str) else text_of(content)
        usage = value.get("usage") if isinstance(value.get("usage"), dict) else None
        usage_out = None
        if usage is not None:
            usage_out = {
                "input": usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), (int, float)) else 0,
                "output": usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), (int, float)) else 0,
            }
        return answer, usage_out, _stop_family(choice.get("finish_reason"))
    answer_parts = []
    usage_out = None
    stop = 0
    for _event, data in parse_sse_frames(text):
        payload = data.strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(chunk, dict):
            continue
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
            usage_out = {
                "input": usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), (int, float)) else 0,
                "output": usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), (int, float)) else 0,
            }
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        if not choices:
            continue
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        if isinstance(delta.get("content"), str) and delta["content"] != "":
            answer_parts.append(delta["content"])
        if isinstance(choice.get("finish_reason"), str):
            stop = _stop_family(choice["finish_reason"])
    return "".join(answer_parts), usage_out, stop


def parse_anthropic_answer(text, stream):
    """(answer_text, usage|None, stop_family) for an Anthropic-wire answer."""
    if not stream:
        try:
            value = json.loads(text)
        except ValueError:
            return None
        if not isinstance(value, dict):
            return None
        answer = text_of(value.get("content"))
        usage = value.get("usage") if isinstance(value.get("usage"), dict) else None
        usage_out = None
        if usage is not None:
            usage_out = {
                "input": usage.get("input_tokens") if isinstance(usage.get("input_tokens"), (int, float)) else 0,
                "output": usage.get("output_tokens") if isinstance(usage.get("output_tokens"), (int, float)) else 0,
            }
        return answer, usage_out, _stop_family(value.get("stop_reason"))
    answer_parts = []
    input_tokens = None
    output_tokens = None
    stop = 0
    for event, data in parse_sse_frames(text):
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        if event == "message_start":
            message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
            usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
            if isinstance(usage.get("input_tokens"), (int, float)):
                input_tokens = usage["input_tokens"]
        elif event == "content_block_delta":
            delta = payload.get("delta") if isinstance(payload.get("delta"), dict) else {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                answer_parts.append(delta["text"])
        elif event == "message_delta":
            delta = payload.get("delta") if isinstance(payload.get("delta"), dict) else {}
            if isinstance(delta.get("stop_reason"), str):
                stop = _stop_family(delta["stop_reason"])
            usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
            if isinstance(usage.get("output_tokens"), (int, float)):
                output_tokens = usage["output_tokens"]
    usage_out = None
    if input_tokens is not None or output_tokens is not None:
        usage_out = {"input": input_tokens or 0, "output": output_tokens or 0}
    return "".join(answer_parts), usage_out, stop


def parse_gemini_answer(text, stream):
    """(answer_text, usage|None, stop_family) for a Gemini-wire answer."""
    if not stream:
        try:
            value = json.loads(text)
        except ValueError:
            return None
        if not isinstance(value, dict):
            return None
        candidates = value.get("candidates") if isinstance(value.get("candidates"), list) else []
        candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
        content = candidate.get("content") if isinstance(candidate.get("content"), dict) else {}
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        answer = "".join(p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str))
        usage = value.get("usageMetadata") if isinstance(value.get("usageMetadata"), dict) else None
        usage_out = None
        if usage is not None:
            usage_out = {
                "input": usage.get("promptTokenCount") if isinstance(usage.get("promptTokenCount"), (int, float)) else 0,
                "output": usage.get("candidatesTokenCount") if isinstance(usage.get("candidatesTokenCount"), (int, float)) else 0,
            }
        return answer, usage_out, _stop_family(candidate.get("finishReason"))
    answer_parts = []
    usage_out = None
    stop = 0
    for _event, data in parse_sse_frames(text):
        try:
            frame = json.loads(data)
        except ValueError:
            continue
        if not isinstance(frame, dict):
            continue
        candidates = frame.get("candidates") if isinstance(frame.get("candidates"), list) else []
        if not candidates or not isinstance(candidates[0], dict):
            continue
        candidate = candidates[0]
        content = candidate.get("content") if isinstance(candidate.get("content"), dict) else {}
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                answer_parts.append(part["text"])
        if isinstance(candidate.get("finishReason"), str):
            stop = _stop_family(candidate["finishReason"])
        usage = frame.get("usageMetadata") if isinstance(frame.get("usageMetadata"), dict) else None
        if usage is not None:
            usage_out = {
                "input": usage.get("promptTokenCount") if isinstance(usage.get("promptTokenCount"), (int, float)) else 0,
                "output": usage.get("candidatesTokenCount") if isinstance(usage.get("candidatesTokenCount"), (int, float)) else 0,
            }
    return "".join(answer_parts), usage_out, stop

# --------------------------------------------------------------------------
# Upstream calls and routing.
# --------------------------------------------------------------------------

class UpstreamRefusal(Exception):
    def __init__(self, status, code):
        super().__init__("provider refused with HTTP %s (%s)" % (status, code))
        self.status = status
        self.code = code


class UpstreamTransportError(Exception):
    pass


def _error_code(wire, text):
    try:
        value = json.loads(text)
    except ValueError:
        return ""
    if not isinstance(value, dict):
        return ""
    if wire == "openai":
        error = value.get("error")
        return error.get("code") if isinstance(error, dict) and isinstance(error.get("code"), str) else ""
    if wire == "anthropic":
        error = value.get("error")
        return error.get("type") if isinstance(error, dict) and isinstance(error.get("type"), str) else ""
    error = value.get("error")
    return error.get("status") if isinstance(error, dict) and isinstance(error.get("status"), str) else ""


class Router:
    def __init__(self, env_url, brief):
        self.env_url = env_url.rstrip("/")
        self.brief = brief
        self.providers = {p["id"]: p for p in brief.providers}
        self.state = {
            p["id"]: {"calls": 0, "consec_fails": 0, "dead": False} for p in brief.providers
        }
        self.request_index = 0

    def _provider_body(self, provider, model, client_api, body, stream):
        """(path, request_body, extra_headers) for one provider call."""
        base = join_base(self.env_url, provider["base"])
        wire = provider["wire"]
        headers = {"content-type": "application/json"}
        headers.update(auth_headers(provider["auth"], provider["key"]))
        if stream:
            headers["accept"] = "text/event-stream"
        if wire == "openai":
            if client_api == "openai":
                out = dict(body)
                out["model"] = model
                if stream:
                    out["stream_options"] = {"include_usage": True}
            else:
                out = anthropic_to_openai(body, model)
            return base + "/v1/chat/completions", out, headers
        if wire == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
            if client_api == "anthropic":
                out = dict(body)
                out["model"] = model
            else:
                out = openai_to_anthropic(body, model)
            return base + "/v1/messages", out, headers
        # gemini: the model rides in the path, never in the body.
        if client_api == "openai":
            out = openai_to_gemini(body)
        else:
            out = anthropic_to_gemini(body)
        action = "streamGenerateContent" if stream else "generateContent"
        path = "%s/v1beta/models/%s:%s" % (base, urllib.parse.quote(model, safe=""), action)
        return path, out, headers

    def call_provider(self, provider, model, client_api, body, stream):
        """One upstream call. Returns (answer_text, usage, stop_family).
        Raises UpstreamRefusal on an HTTP error, UpstreamTransportError when
        the provider could not be reached at all."""
        path, payload, headers = self._provider_body(provider, model, client_api, body, stream)
        parsed = urllib.parse.urlparse(path)
        connection = http.client.HTTPConnection(
            parsed.hostname, parsed.port or 80, timeout=UPSTREAM_TIMEOUT_S
        )
        try:
            request_path = parsed.path
            if parsed.query:
                request_path += "?" + parsed.query
            connection.request(
                "POST", request_path,
                body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers,
            )
            response = connection.getresponse()
            raw = response.read()
        except (OSError, http.client.HTTPException, socket.timeout) as exc:
            raise UpstreamTransportError(str(exc))
        finally:
            try:
                connection.close()
            except OSError:
                pass
        text = raw.decode("utf-8", errors="replace")
        if response.status < 200 or response.status >= 300:
            raise UpstreamRefusal(response.status, _error_code(provider["wire"], text))
        parser = {
            "openai": parse_openai_answer,
            "anthropic": parse_anthropic_answer,
            "gemini": parse_gemini_answer,
        }[provider["wire"]]
        parsed_answer = parser(text, stream)
        if parsed_answer is None:
            # A 200 whose body does not parse: use it anyway (a second
            # successful call would break the one-provider check) and fall
            # back to the usage formula.
            return text, computed_usage(client_api, body, text), 0
        answer, usage, stop = parsed_answer
        if usage is None:
            usage = computed_usage(client_api, body, answer)
        return answer, usage, stop

    def _requirements_for(self, body):
        manifest = self.brief.manifest
        if self.request_index < len(manifest):
            return manifest[self.request_index]["requires"]
        requires = {"tools": False, "vision": False, "json": False}
        if isinstance(body.get("tools"), list) and body["tools"]:
            requires["tools"] = True
        for message in body.get("messages") or []:
            if not isinstance(message, dict) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if isinstance(block, dict) and block.get("type") in ("image", "image_url"):
                    requires["vision"] = True
        fmt = body.get("response_format") if isinstance(body.get("response_format"), dict) else None
        if fmt is not None and fmt.get("type") in ("json_object", "json_schema"):
            requires["json"] = True
        return requires

    def candidates_for(self, requires, client_api, body):
        """(provider_id, model_id) pairs that can serve this request, best first.

        Rating first: spreading serves over more providers only hits more
        of their scheduled failures (each provider carries one or two), so
        the best model is always tried first and the fewest-calls provider
        breaks rating ties. Deterministic given the call history."""
        json_mode = False
        if client_api == "openai":
            fmt = body.get("response_format") if isinstance(body.get("response_format"), dict) else None
            json_mode = fmt is not None and fmt.get("type") in ("json_object", "json_schema")
        out = []
        for row in self.brief.ratings:
            if not row["free"]:
                continue
            if not meets_requirements(requires, row):
                continue
            provider = self.providers.get(row["provider"])
            if provider is None:
                continue
            # An Anthropic wire cannot carry response_format: the provider
            # would echo json_mode absent and the check would fail.
            if json_mode and provider["wire"] == "anthropic":
                continue
            out.append((row["provider"], row["model"], row["rating_milli"]))
        out.sort(key=lambda c: (-c[2], self.state[c[0]]["calls"], c[0], c[1]))
        return [(provider, model) for provider, model, _ in out]

    def _try(self, provider_id, model_id, client_api, body, stream):
        provider = self.providers[provider_id]
        self.state[provider_id]["calls"] += 1
        return self.call_provider(provider, model_id, client_api, body, stream)

    def serve(self, client_api, body, stream):
        """Route one client request. Returns
        (provider_id, model_id, answer_text, usage, stop_family) or None when
        no candidate could serve it. The manifest index advances per request.

        Failover is immediate: a provider that refuses is left at once for
        the next candidate (retrying the same provider costs a call and the
        rating it would save cannot move the floored quality column, while a
        second consecutive refusal there is usually exhaustion). A provider
        is dead after two consecutive refusals. If no candidate serves the
        request, one last pass retries every candidate once -- a provider
        whose two refusals were scheduled failures still has budget, and
        serving the request is worth far more than the call it costs."""
        requires = self._requirements_for(body)
        self.request_index += 1
        candidates = self.candidates_for(requires, client_api, body)
        result = self._attempt(candidates, client_api, body, stream, skip_dead=True)
        if result is not None:
            return result
        return self._attempt(candidates, client_api, body, stream, skip_dead=False)

    def _attempt(self, candidates, client_api, body, stream, skip_dead):
        for provider_id, model_id in candidates:
            state = self.state[provider_id]
            if skip_dead and state["dead"]:
                continue
            try:
                answer, usage, stop = self._try(provider_id, model_id, client_api, body, stream)
            except UpstreamRefusal as exc:
                if exc.status in (400, 401):
                    # My request was malformed or unauthenticated: the call
                    # spent no budget and says nothing about the provider.
                    log("refused %s/%s with HTTP %s (%s): not retried"
                        % (provider_id, model_id, exc.status, exc.code))
                    continue
                state["consec_fails"] += 1
                if state["consec_fails"] >= 2:
                    state["dead"] = True
                    log("provider %s is dead after %d consecutive refusals"
                        % (provider_id, state["consec_fails"]))
                continue
            except UpstreamTransportError as exc:
                state["consec_fails"] += 1
                if state["consec_fails"] >= 2:
                    state["dead"] = True
                log("provider %s unreachable (%s)" % (provider_id, exc))
                continue
            state["consec_fails"] = 0
            if state["dead"]:
                state["dead"] = False
                log("provider %s is alive again" % provider_id)
            return provider_id, model_id, answer, usage, stop
        return None


# --------------------------------------------------------------------------
# Client responses: the provider's answer in the wire the client used.
# --------------------------------------------------------------------------

def sse_event(event, data):
    text = json.dumps(data, ensure_ascii=False)
    if event is None:
        return "data: %s\n\n" % text
    return "event: %s\ndata: %s\n\n" % (event, text)


def openai_json_response(model, text, usage, stop):
    return {
        "id": "chatcmpl-agg",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": OPENAI_STOP[stop],
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": usage["input"],
            "completion_tokens": usage["output"],
            "total_tokens": usage["input"] + usage["output"],
        },
    }


def openai_stream_response(model, text, usage, stop):
    head = {"id": "chatcmpl-agg", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model}
    frames = [
        sse_event(None, dict(head, choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}])),
    ]
    if text != "":
        frames.append(
            sse_event(None, dict(head, choices=[{"index": 0, "delta": {"content": text}, "finish_reason": None}]))
        )
    frames.append(
        sse_event(None, dict(head, choices=[{"index": 0, "delta": {}, "finish_reason": OPENAI_STOP[stop]}]))
    )
    frames.append(
        sse_event(
            None,
            dict(
                head,
                choices=[],
                usage={
                    "prompt_tokens": usage["input"],
                    "completion_tokens": usage["output"],
                    "total_tokens": usage["input"] + usage["output"],
                },
            ),
        )
    )
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def anthropic_json_response(model, text, usage, stop):
    return {
        "id": "msg_agg",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": ANTHROPIC_STOP[stop],
        "stop_sequence": None,
        "usage": {"input_tokens": usage["input"], "output_tokens": usage["output"]},
    }


def anthropic_stream_response(model, text, usage, stop):
    frames = [
        sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_agg",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage["input"], "output_tokens": 0},
                },
            },
        ),
        sse_event(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
    ]
    if text != "":
        frames.append(
            sse_event(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
            )
        )
    frames.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(
        sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": ANTHROPIC_STOP[stop], "stop_sequence": None},
                "usage": {"output_tokens": usage["output"]},
            },
        )
    )
    frames.append(sse_event("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def openai_error(message, error_type="invalid_request_error", code=None):
    return {"error": {"message": message, "type": error_type, "code": code}}


def anthropic_error(message, error_type="api_error"):
    return {"type": "error", "error": {"type": error_type, "message": message}}

# --------------------------------------------------------------------------
# HTTP server, register, and the /work poll.
# --------------------------------------------------------------------------

class Aggregator:
    def __init__(self, env_url, brief, catalog):
        self.env_url = env_url
        self.brief = brief
        self.catalog = catalog
        self.router = Router(env_url, brief)
        self.lock = threading.Lock()
        self.in_flight = 0
        self.served = 0
        # The free model ids the catalog registers, in catalog order, deduplicated.
        ids = []
        seen = set()
        for provider in catalog.get("providers") or []:
            for model in provider.get("models") or []:
                if model.get("free") is True and isinstance(model.get("id"), str):
                    if model["id"] not in seen:
                        seen.add(model["id"])
                        ids.append(model["id"])
        self.free_model_ids = ids

    def model_list(self, anthropic_shape):
        if anthropic_shape:
            return {
                "data": [
                    {"type": "model", "id": model_id, "display_name": model_id, "created_at": FIXED_ANTHROPIC_CREATED_AT}
                    for model_id in self.free_model_ids
                ],
                "has_more": False,
            }
        return {
            "object": "list",
            "data": [
                {"id": model_id, "object": "model", "created": FIXED_CREATED, "owned_by": "aggregator"}
                for model_id in self.free_model_ids
            ],
        }


def make_handler(agg):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send_bytes(self, status, content_type, payload):
            try:
                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(payload)))
                if content_type.startswith("text/event-stream"):
                    self.send_header("cache-control", "no-cache")
                    self.send_header("connection", "keep-alive")
                self.end_headers()
                self.wfile.write(payload)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def _send_json(self, status, value):
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self._send_bytes(status, "application/json", payload)

        def _read_body(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0:
                return b""
            if length > MAX_BODY_BYTES:
                return None
            try:
                return self.rfile.read(length)
            except OSError:
                return b""

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            agg.in_flight += 1
            try:
                if parsed.path == "/v1/models":
                    anthropic_shape = isinstance(self.headers.get("x-api-key"), str)
                    self._send_json(200, agg.model_list(anthropic_shape))
                else:
                    self._send_json(404, openai_error("no route GET %s" % parsed.path, "not_found"))
            finally:
                agg.in_flight -= 1

        def _serve_post(self, client_api):
            raw = self._read_body()
            if raw is None:
                error = anthropic_error("the request body is too large", "invalid_request_error")
                self._send_json(413, error if client_api == "anthropic" else openai_error("the request body is too large"))
                return
            try:
                body = json.loads(raw.decode("utf-8"))
            except ValueError:
                body = None
            if not isinstance(body, dict):
                if client_api == "anthropic":
                    self._send_json(400, anthropic_error("the request body is not one JSON object", "invalid_request_error"))
                else:
                    self._send_json(400, openai_error("the request body is not one JSON object"))
                return
            stream = body.get("stream") is True
            # One request at a time: the client sends them serially, and a
            # global lock keeps provider calls strictly sequential (the one
            # single-flight provider never sees a concurrent call).
            with agg.lock:
                result = agg.router.serve(client_api, body, stream)
            if result is None:
                if client_api == "anthropic":
                    self._send_json(503, anthropic_error("no free model meets this request", "overloaded_error"))
                else:
                    self._send_json(503, openai_error("no free model meets this request", "no_free_model"))
                return
            provider_id, model_id, answer, usage, stop = result
            log("served %s via %s/%s (usage %d+%d)"
                % (client_api, provider_id, model_id, usage["input"], usage["output"]))
            if client_api == "openai":
                if stream:
                    payload = openai_stream_response(model_id, answer, usage, stop).encode("utf-8")
                    self._send_bytes(200, "text/event-stream; charset=utf-8", payload)
                else:
                    self._send_json(200, openai_json_response(model_id, answer, usage, stop))
            else:
                if stream:
                    payload = anthropic_stream_response(model_id, answer, usage, stop).encode("utf-8")
                    self._send_bytes(200, "text/event-stream; charset=utf-8", payload)
                else:
                    self._send_json(200, anthropic_json_response(model_id, answer, usage, stop))

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            agg.in_flight += 1
            try:
                if parsed.path == "/v1/chat/completions":
                    self._serve_post("openai")
                elif parsed.path == "/v1/messages":
                    self._serve_post("anthropic")
                else:
                    self._send_json(404, openai_error("no route POST %s" % parsed.path, "not_found"))
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            except Exception as exc:  # never let one request kill the server
                log("handler failed: %r" % (exc,))
                try:
                    if not self.wfile.closed:
                        self._send_json(500, openai_error("the server failed to answer", "internal_error"))
                except OSError:
                    pass
            finally:
                agg.in_flight -= 1
                if parsed.path in ("/v1/chat/completions", "/v1/messages"):
                    agg.served += 1

    return Handler


def register(env_url, port, catalog, attempts=5):
    for attempt in range(attempts):
        try:
            parsed = urllib.parse.urlparse(env_url)
            connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=5)
            try:
                connection.request(
                    "POST",
                    "/register",
                    body=json.dumps(
                        {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog},
                        ensure_ascii=False,
                    ).encode("utf-8"),
                    headers={"content-type": "application/json"},
                )
                response = connection.getresponse()
                raw = response.read()
                if response.status == 200:
                    try:
                        reply = json.loads(raw.decode("utf-8"))
                    except ValueError:
                        reply = {}
                    log("registered: the workload holds %s requests" % reply.get("requests"))
                    return True
                log("register attempt %d answered HTTP %s" % (attempt + 1, response.status))
            finally:
                try:
                    connection.close()
                except OSError:
                    pass
        except (OSError, http.client.HTTPException, socket.timeout) as exc:
            log("register attempt %d failed (%s)" % (attempt + 1, exc))
        time.sleep(0.2)
    return False


def poll_work(env_url, agg, server):
    """Poll GET /work; exit 0 the moment it says done (or once every manifest
    request has been served). Runs on the main thread."""
    manifest_len = len(agg.brief.manifest)
    while True:
        done = False
        try:
            parsed = urllib.parse.urlparse(env_url)
            connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=2)
            try:
                connection.request("GET", "/work")
                response = connection.getresponse()
                raw = response.read()
                if response.status == 200:
                    work = json.loads(raw.decode("utf-8"))
                    if isinstance(work, dict) and work.get("done") is True:
                        done = True
            finally:
                try:
                    connection.close()
                except OSError:
                    pass
        except (OSError, http.client.HTTPException, socket.timeout, ValueError):
            pass
        if manifest_len > 0 and agg.served >= manifest_len and agg.in_flight == 0:
            done = True
        if done and agg.in_flight == 0:
            log("work is done (%d served); exiting 0" % agg.served)
            try:
                server.shutdown()
                server.server_close()
            except OSError:
                pass
            sys.exit(0)
        time.sleep(POLL_INTERVAL_S)


def main(argv):
    env_url = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE") or ""
    brief = Brief(prompt_file)
    if brief.error is not None:
        log(brief.error)
    catalog = build_catalog(env_url, brief) if brief.providers else {"providers": []}
    if "--list" in argv:
        text = json.dumps(catalog, ensure_ascii=False, separators=(",", ":")) + "\n"
        sys.stdout.write(text)
        sys.stdout.flush()
        return 0
    if not env_url:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    agg = Aggregator(env_url, brief, catalog)
    handler = make_handler(agg)
    server = None
    port = 0
    for _ in range(3):
        candidate = http.server.ThreadingHTTPServer(("0.0.0.0", 0), handler)
        candidate.daemon_threads = True
        address = candidate.server_address
        if address[1] != 8080:
            server = candidate
            port = address[1]
            break
        candidate.server_close()
    if server is None:
        log("could not pick a port other than 8080")
        return 1
    log("listening on 0.0.0.0:%d with %d providers and %d free models"
        % (port, len(catalog.get("providers") or []), len(agg.free_model_ids)))
    if not register(env_url, port, catalog):
        log("register failed after retries; serving anyway")
    import signal

    def _exit(_signum, _frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, _exit)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    poll_work(env_url, agg, server)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
