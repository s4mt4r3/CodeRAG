"""LLM client protocol, the Anthropic implementation, and a scripted fake for tests.

Messages use the Anthropic content-block shape: ``{"role", "content": [blocks]}`` with
``text``, ``tool_use`` and ``tool_result`` blocks. Assistant ``content`` is kept
verbatim (including thinking blocks) so it can be appended back to the history.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Usage:
    """Token usage for one or more calls."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(self.input_tokens + other.input_tokens,
                     self.output_tokens + other.output_tokens)


@dataclass
class ToolCall:
    """A tool invocation requested by the model; ``arguments`` may be malformed."""

    id: str
    name: str
    arguments: Any


@dataclass
class LLMResponse:
    """One model turn."""

    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    stop_reason: str
    content: list[dict] = field(default_factory=list)


class LLMClient(Protocol):
    """Anything that can take a conversation and return the next model turn."""

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """Return the model's next turn."""
        ...


def _block_to_dict(block: Any) -> dict:
    if isinstance(block, dict):
        return dict(block)
    to_dict = getattr(block, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return dict(vars(block))


class AnthropicClient:
    """Messages API client. The model id comes from ``CODERAG_MODEL``, never a default."""

    provider = "anthropic"

    def __init__(self, model: str | None = None, max_tokens: int = 16000,
                 client: Any = None) -> None:
        self.model = model or os.environ.get("CODERAG_MODEL")
        if not self.model:
            raise ValueError("set CODERAG_MODEL to the Claude model id to use")
        self.max_tokens = max_tokens
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """One Messages API call; returns text, tool calls, usage and raw blocks."""
        kwargs: dict[str, Any] = {
            "model": self.model, "max_tokens": self.max_tokens,
            "system": system, "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        resp = self._client.messages.create(**kwargs)
        content = [_block_to_dict(b) for b in resp.content]
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        calls = [ToolCall(b["id"], b["name"], b.get("input")) for b in content
                 if b.get("type") == "tool_use"]
        usage = Usage(int(getattr(resp.usage, "input_tokens", 0) or 0),
                      int(getattr(resp.usage, "output_tokens", 0) or 0))
        return LLMResponse(text=text, tool_calls=calls, usage=usage,
                           stop_reason=str(resp.stop_reason), content=content)


class ScriptedLLM:
    """Replays a fixed list of responses and records every request it receives."""

    _ids = itertools.count(1)

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    @classmethod
    def tool(cls, name: str, arguments: Any, usage: tuple[int, int] = (10, 5),
             call_id: str | None = None) -> LLMResponse:
        """A turn that requests one tool call."""
        cid = call_id or f"toolu_{next(cls._ids):04d}"
        block = {"type": "tool_use", "id": cid, "name": name, "input": arguments}
        return LLMResponse(text="", tool_calls=[ToolCall(cid, name, arguments)],
                           usage=Usage(*usage), stop_reason="tool_use", content=[block])

    @classmethod
    def final(cls, text: str, usage: tuple[int, int] = (10, 5)) -> LLMResponse:
        """A turn that ends with plain text."""
        return LLMResponse(text=text, tool_calls=[], usage=Usage(*usage),
                           stop_reason="end_turn", content=[{"type": "text", "text": text}])

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """Pop the next scripted response."""
        import copy

        self.calls.append({"system": system, "messages": copy.deepcopy(messages),
                           "tools": tools})
        if not self._responses:
            raise RuntimeError("ScriptedLLM script exhausted")
        return self._responses.pop(0)
