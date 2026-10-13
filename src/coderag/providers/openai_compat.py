"""OpenAI Chat Completions adapter, shared by OpenAI and compatible servers
(Groq, Together, OpenRouter, Ollama's ``/v1``, ...)."""

from __future__ import annotations

import itertools
import json
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
    json_body,
    parse_arguments,
    send_with_retries,
    tool_result_text,
)

_FINISH = {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal",
           "tool_calls": "tool_use", "function_call": "tool_use"}


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    """Translate Anthropic-shaped history into chat-completions messages."""
    out: list[dict] = [{"role": "system", "content": system}] if system else []
    for m in messages:
        blocks = blocks_of(m.get("content"))
        if m.get("role") == "assistant":
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            calls = [{"id": b["id"], "type": "function", "function": {
                "name": b["name"],
                "arguments": b["input"] if isinstance(b.get("input"), str)
                else json.dumps(b.get("input") or {})}}
                for b in blocks if b.get("type") == "tool_use"]
            msg: dict = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            continue
        if isinstance(m.get("content"), str):
            out.append({"role": "user", "content": m["content"]})
            continue
        for b in blocks:
            if b.get("type") == "tool_result":
                out.append({"role": "tool", "tool_call_id": b.get("tool_use_id", ""),
                            "content": tool_result_text(b)})
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if text:
            out.append({"role": "user", "content": text})
    return out


def to_openai_tools(tools: list[dict]) -> list[dict]:
    """Anthropic tool specs as OpenAI function tools."""
    return [{"type": "function", "function": {"name": t["name"],
                                              "description": t.get("description", ""),
                                              "parameters": t["input_schema"]}}
            for t in tools]


class OpenAICompatClient:
    """``LLMClient`` for any server speaking ``POST {base_url}/chat/completions``."""

    def __init__(self, base_url: str, model: str, api_key: str | None,
                 http: httpx.Client | None = None, provider: str = "openai_compatible",
                 max_tokens: int = DEFAULT_MAX_TOKENS, max_tokens_field: str = "max_tokens",
                 extra_headers: dict[str, str] | None = None, retries: int = 2,
                 sleep: Callable[[float], None] = time.sleep, max_retry_wait: float = 10.0,
                 policy: Policy | None = None, resolver: Resolver = default_resolver) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider = provider
        self.max_tokens = max_tokens
        self.max_tokens_field = max_tokens_field
        self._key = api_key
        self._headers = dict(extra_headers or {})
        self._http = http or guarded_client(policy, resolver=resolver, timeout=DEFAULT_TIMEOUT)
        self._retries = retries
        self._sleep = sleep
        self._max_wait = max_retry_wait
        self._ids = itertools.count(1)

    def __repr__(self) -> str:
        return f"OpenAICompatClient(provider={self.provider!r}, model={self.model!r})"

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """One chat-completions call, normalized to :class:`LLMResponse`."""
        body: dict = {"model": self.model, "messages": to_openai_messages(system, messages),
                      self.max_tokens_field: self.max_tokens}
        if tools:
            body["tools"] = to_openai_tools(tools)
        headers = {"content-type": "application/json", **self._headers}
        if self._key:
            headers["authorization"] = f"Bearer {self._key}"
        secrets = (self._key,)
        resp = send_with_retries(
            lambda: self._http.post(f"{self.base_url}/chat/completions", json=body,
                                    headers=headers),
            self.provider, secrets, self._retries, sleep=self._sleep, max_wait=self._max_wait)
        return self._parse(json_body(resp, self.provider, secrets), secrets)

    def _parse(self, data: dict, secrets: tuple[str | None, ...]) -> LLMResponse:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderResponseError(f"{self.provider}: response has no choices",
                                        secrets=secrets)
        choice = choices[0]
        msg = choice.get("message") or {}
        raw = msg.get("content")
        if isinstance(raw, list):
            text = "".join(p.get("text", "") for p in raw if isinstance(p, dict))
        else:
            text = raw or ""
        content: list[dict] = [{"type": "text", "text": text}] if text else []
        calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            cid = tc.get("id") or f"call_{self.provider}_{next(self._ids)}"
            args = parse_arguments(fn.get("arguments", "{}"))
            calls.append(ToolCall(cid, str(fn.get("name", "")), args))
            content.append({"type": "tool_use", "id": cid, "name": str(fn.get("name", "")),
                            "input": args})
        stop = _FINISH.get(choice.get("finish_reason") or "stop", "end_turn")
        if calls:
            stop = "tool_use"
        elif stop == "tool_use":
            stop = "end_turn"
        if msg.get("refusal") and not text and not calls:
            stop = "refusal"
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text, tool_calls=calls,
            usage=Usage(int(usage.get("prompt_tokens") or 0),
                        int(usage.get("completion_tokens") or 0)),
            stop_reason=stop, content=content)
