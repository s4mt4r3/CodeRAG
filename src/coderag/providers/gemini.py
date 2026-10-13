"""Google Gemini ``generateContent`` REST adapter.

The key goes in the ``x-goog-api-key`` header, never the URL. Gemini function calls
carry no ids in older models, so stable ids are synthesized and tool results are
mapped back to function names through the history. ``thoughtSignature`` values on
function-call parts are kept on our ``tool_use`` blocks and echoed back, which
newer thinking models require for multi-step function calling [Unverified live].
"""

from __future__ import annotations

import itertools
import re
import time
from collections.abc import Callable

import httpx

from coderag.llm import LLMResponse, ToolCall, Usage
from coderag.netguard import Policy, Resolver, default_resolver, guarded_client
from coderag.providers.base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT,
    ProviderResponseError,
    blocks_of,
    check_key,
    iter_tool_uses,
    json_body,
    parse_arguments,
    send_with_retries,
    tool_result_text,
)

BASE_URL = "https://generativelanguage.googleapis.com"
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SCHEMA_KEYS = {"type", "description", "properties", "required", "items", "enum", "format",
                "nullable", "minimum", "maximum", "minItems", "maxItems"}
_REFUSALS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
             "IMAGE_SAFETY", "LANGUAGE"}


def gemini_schema(schema: object) -> object:
    """Keep only the OpenAPI subset Gemini accepts (drops additionalProperties, default)."""
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k == "properties" and isinstance(v, dict):
                out[k] = {name: gemini_schema(sub) for name, sub in v.items()}
            elif k in _SCHEMA_KEYS:
                out[k] = gemini_schema(v) if k == "items" else v
        return out
    return schema


def _args_object(value: object) -> dict:
    if isinstance(value, dict):
        return value
    parsed = parse_arguments(value)
    return parsed if isinstance(parsed, dict) else {"_raw": str(value)}


def to_gemini_contents(messages: list[dict]) -> list[dict]:
    """Translate Anthropic-shaped history into Gemini ``contents``."""
    names = {b.get("id"): b.get("name", "") for b in iter_tool_uses(messages)}
    out: list[dict] = []
    for m in messages:
        role = "model" if m.get("role") == "assistant" else "user"
        parts: list[dict] = []
        for b in blocks_of(m.get("content")):
            kind = b.get("type")
            if kind == "text" and b.get("text"):
                parts.append({"text": b["text"]})
            elif kind == "tool_use":
                part: dict = {"functionCall": {"name": b.get("name", ""),
                                               "args": _args_object(b.get("input"))}}
                if b.get("gemini_thought_signature"):
                    part["thoughtSignature"] = b["gemini_thought_signature"]
                parts.append(part)
            elif kind == "tool_result":
                parts.append({"functionResponse": {
                    "name": names.get(b.get("tool_use_id"), "unknown_tool"),
                    "response": {"content": tool_result_text(b),
                                 "is_error": bool(b.get("is_error"))}}})
        if not parts:
            continue
        if out and out[-1]["role"] == role:
            out[-1]["parts"].extend(parts)
        else:
            out.append({"role": role, "parts": parts})
    return out


class GeminiClient:
    """``LLMClient`` for the Gemini API."""

    provider = "gemini"

    def __init__(self, model: str, api_key: str | None, http: httpx.Client | None = None,
                 base_url: str = BASE_URL, max_tokens: int = DEFAULT_MAX_TOKENS,
                 retries: int = 2, sleep: Callable[[float], None] = time.sleep,
                 max_retry_wait: float = 10.0, policy: Policy | None = None,
                 resolver: Resolver = default_resolver) -> None:
        name = model[len("models/"):] if isinstance(model, str) and model.startswith(
            "models/") else model
        if not isinstance(name, str) or not _MODEL_RE.match(name) or ".." in name:
            raise ValueError("invalid Gemini model name")
        self.model = name
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self._key = check_key(api_key, required=True)
        self._http = http or guarded_client(policy, resolver=resolver, timeout=DEFAULT_TIMEOUT)
        self._retries = retries
        self._sleep = sleep
        self._max_wait = max_retry_wait
        self._ids = itertools.count(1)

    def __repr__(self) -> str:
        return f"GeminiClient(model={self.model!r})"

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """One generateContent call, normalized to :class:`LLMResponse`."""
        body: dict = {"contents": to_gemini_contents(messages),
                      "generationConfig": {"maxOutputTokens": self.max_tokens}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = [{"functionDeclarations": [
                {"name": t["name"], "description": t.get("description", ""),
                 "parameters": gemini_schema(t["input_schema"])} for t in tools]}]
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": self._key or "", "content-type": "application/json"}
        secrets = (self._key,)
        resp = send_with_retries(lambda: self._http.post(url, json=body, headers=headers),
                                 self.provider, secrets, self._retries, sleep=self._sleep,
                                 max_wait=self._max_wait)
        return self._parse(json_body(resp, self.provider, secrets), secrets)

    def _parse(self, data: dict, secrets: tuple[str | None, ...]) -> LLMResponse:
        meta = data.get("usageMetadata") or {}
        usage = Usage(int(meta.get("promptTokenCount") or 0),
                      int(meta.get("candidatesTokenCount") or 0)
                      + int(meta.get("thoughtsTokenCount") or 0))
        candidates = data.get("candidates")
        if not candidates:
            if (data.get("promptFeedback") or {}).get("blockReason"):
                return LLMResponse("", [], usage, "refusal", [])
            raise ProviderResponseError("gemini: response has no candidates", secrets=secrets)
        cand = candidates[0] if isinstance(candidates[0], dict) else {}
        texts, calls, content = [], [], []
        for part in (cand.get("content") or {}).get("parts") or []:
            if not isinstance(part, dict) or part.get("thought"):
                continue
            if "text" in part:
                texts.append(str(part["text"]))
            fc = part.get("functionCall")
            if isinstance(fc, dict):
                cid = str(fc.get("id") or f"gemini_call_{next(self._ids)}")
                args = fc.get("args") if isinstance(fc.get("args"), dict) else {}
                calls.append(ToolCall(cid, str(fc.get("name", "")), args))
                block = {"type": "tool_use", "id": cid, "name": str(fc.get("name", "")),
                         "input": args}
                if part.get("thoughtSignature"):
                    block["gemini_thought_signature"] = part["thoughtSignature"]
                content.append(block)
        text = "".join(texts)
        if text:
            content.insert(0, {"type": "text", "text": text})
        finish = cand.get("finishReason") or "STOP"
        if calls:
            stop = "tool_use"
        elif finish == "MAX_TOKENS":
            stop = "max_tokens"
        elif finish in _REFUSALS:
            stop = "refusal"
        else:
            stop = "end_turn"
        return LLMResponse(text=text, tool_calls=calls, usage=usage, stop_reason=stop,
                           content=content)

