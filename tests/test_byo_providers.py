"""BYO-LLM: provider adapters, registry, errors, and retries (offline, MockTransport)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import httpx
import pytest

from coderag.agent import Agent
from coderag.llm import AnthropicClient, LLMResponse
from coderag.netguard import Policy
from coderag.providers.base import (
    ProviderAuthError,
    ProviderBlockedError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
)
from coderag.providers.gemini import GeminiClient
from coderag.providers.openai_compat import OpenAICompatClient
from coderag.providers.registry import build_client, list_providers
from coderag.store import Store, index_repo
from coderag.tools import Toolbox
from conftest import FIXTURE_ROOT

KEY = "sk-test-0123456789abcdefSECRETVALUE"
TOOLS = [{"name": "search_code", "description": "search",
          "input_schema": {"type": "object",
                           "properties": {"query": {"type": "string"},
                                          "k": {"type": "integer", "default": 5}},
                           "required": ["query"], "additionalProperties": False}}]
HISTORY = [
    {"role": "user", "content": "Where is slugify?"},
    {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "", "signature": "x"},
        {"type": "text", "text": "Let me search."},
        {"type": "tool_use", "id": "call_1", "name": "search_code", "input": {"query": "slugify"}},
        {"type": "tool_use", "id": "call_2", "name": "search_code", "input": "{bad json"},
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "<tool_output>A</tool_output>"},
        {"type": "tool_result", "tool_use_id": "call_2", "content": "ERROR: bad", "is_error": True},
    ]},
]
FINAL = json.dumps({"answer": "slugify makes slugs.", "citations": [
    {"path": "shopcore/utils.py", "start_line": 9, "end_line": 13, "symbol": "slugify"}]})


class Recorder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0)
        return item if isinstance(item, httpx.Response) else httpx.Response(200, json=item)

    def body(self, i=-1) -> dict:
        return json.loads(self.requests[i].content)


def mock(responses) -> tuple[httpx.Client, Recorder]:
    rec = Recorder(responses)
    return httpx.Client(transport=httpx.MockTransport(rec)), rec


def oa_resp(content=None, tool_calls=None, finish="stop", usage=(12, 7), refusal=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    if refusal is not None:
        msg["refusal"] = refusal
    return {"choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}


def oa_call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def openai(http, **kw):
    return OpenAICompatClient(base_url="https://api.example.com/v1", model="m-1", api_key=KEY,
                              http=http, **kw)


# ------------------------------------------------------------------ OpenAI-compatible


def test_openai_request_shape():
    http, rec = mock([oa_resp("hi")])
    openai(http).complete("SYS", HISTORY, TOOLS)
    req = rec.requests[0]
    assert str(req.url) == "https://api.example.com/v1/chat/completions"
    assert req.headers["authorization"] == f"Bearer {KEY}"
    body = rec.body()
    assert body["model"] == "m-1" and body["max_tokens"] > 0
    assert body["tools"] == [{"type": "function", "function": {
        "name": "search_code", "description": "search",
        "parameters": TOOLS[0]["input_schema"]}}]
    msgs = body["messages"]
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1] == {"role": "user", "content": "Where is slugify?"}
    assert msgs[2]["role"] == "assistant" and msgs[2]["content"] == "Let me search."
    assert [c["id"] for c in msgs[2]["tool_calls"]] == ["call_1", "call_2"]
    assert json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"]) == {"query": "slugify"}
    assert msgs[2]["tool_calls"][1]["function"]["arguments"] == "{bad json"
    assert msgs[3] == {"role": "tool", "tool_call_id": "call_1",
                       "content": "<tool_output>A</tool_output>"}
    assert msgs[4]["role"] == "tool" and msgs[4]["tool_call_id"] == "call_2"
    assert "error" in msgs[4]["content"].lower()
    assert len(msgs) == 5


def test_openai_max_tokens_field_is_configurable():
    http, rec = mock([oa_resp("hi")])
    openai(http, max_tokens_field="max_completion_tokens").complete("s", HISTORY[:1], [])
    body = rec.body()
    assert "max_completion_tokens" in body and "max_tokens" not in body and "tools" not in body


def test_openai_parses_text_and_usage():
    http, _ = mock([oa_resp("Hello there", usage=(30, 4))])
    r = openai(http).complete("s", HISTORY[:1], TOOLS)
    assert isinstance(r, LLMResponse)
    assert (r.text, r.stop_reason, r.usage.input_tokens, r.usage.output_tokens) == (
        "Hello there", "end_turn", 30, 4)
    assert r.content == [{"type": "text", "text": "Hello there"}]
    assert r.tool_calls == []


def test_openai_parses_multiple_tool_calls_and_malformed_args():
    http, _ = mock([oa_resp(None, [oa_call("a", "search_code", '{"query": "x"}'),
                                    oa_call("b", "read_file", '{oops')], finish="tool_calls")])
    r = openai(http).complete("s", HISTORY[:1], TOOLS)
    assert r.stop_reason == "tool_use"
    assert [(c.id, c.name) for c in r.tool_calls] == [("a", "search_code"), ("b", "read_file")]
    assert r.tool_calls[0].arguments == {"query": "x"}
    assert r.tool_calls[1].arguments == "{oops"  # raw string -> agent rejects as malformed
    assert [b["type"] for b in r.content] == ["tool_use", "tool_use"]


def test_openai_missing_tool_call_id_is_synthesized():
    http, _ = mock([oa_resp(None, [{"type": "function", "function": {
        "name": "search_code", "arguments": {"query": "x"}}}], finish="tool_calls")])
    r = openai(http).complete("s", HISTORY[:1], TOOLS)
    assert r.tool_calls[0].id and r.tool_calls[0].arguments == {"query": "x"}


@pytest.mark.parametrize(("finish", "expected"), [
    ("stop", "end_turn"), ("length", "max_tokens"), ("content_filter", "refusal"),
    ("tool_calls", "end_turn"), (None, "end_turn")])
def test_openai_finish_reasons(finish, expected):
    http, _ = mock([oa_resp("x", finish=finish)])
    assert openai(http).complete("s", HISTORY[:1], []).stop_reason == expected


def test_openai_refusal_field_and_empty_content():
    http, _ = mock([oa_resp(None, refusal="I can't help with that.")])
    assert openai(http).complete("s", HISTORY[:1], []).stop_reason == "refusal"
    http, _ = mock([oa_resp(None)])
    r = openai(http).complete("s", HISTORY[:1], [])
    assert r.text == "" and r.content == [] and r.stop_reason == "end_turn"


def test_openai_content_parts_list_and_missing_usage():
    resp = {"choices": [{"message": {"role": "assistant", "content": [
        {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}, "finish_reason": "stop"}]}
    http, _ = mock([resp])
    r = openai(http).complete("s", HISTORY[:1], [])
    assert r.text == "ab" and r.usage.input_tokens == 0


def test_openai_malformed_response_is_provider_error():
    for bad in [{"choices": []}, {"nope": 1}, httpx.Response(200, text="<html>")]:
        http, _ = mock([bad])
        with pytest.raises(ProviderResponseError):
            openai(http).complete("s", HISTORY[:1], [])


# ------------------------------------------------------------------ errors and retries


@pytest.mark.parametrize(("status", "exc"), [
    (401, ProviderAuthError), (403, ProviderAuthError), (429, ProviderRateLimitError),
    (500, ProviderServerError), (503, ProviderServerError), (400, ProviderResponseError),
    (404, ProviderResponseError), (302, ProviderResponseError)])
def test_http_errors_map_to_hierarchy_and_never_contain_key(status, exc):
    echo = {"error": {"message": f"Incorrect API key provided: {KEY}. Bearer {KEY}"}}
    http, _ = mock([httpx.Response(status, json=echo) for _ in range(5)])
    with pytest.raises(exc) as info:
        openai(http, retries=2, sleep=lambda s: None).complete("s", HISTORY[:1], [])
    text = str(info.value) + repr(info.value) + str(info.value.args)
    assert KEY not in text and "SECRETVALUE" not in text
    assert isinstance(info.value, ProviderError)


def test_retry_on_429_and_5xx_then_success():
    sleeps = []
    http, rec = mock([httpx.Response(429, headers={"retry-after": "2"}, json={}),
                      httpx.Response(503, json={}), oa_resp("ok")])
    r = openai(http, retries=2, sleep=sleeps.append).complete("s", HISTORY[:1], [])
    assert r.text == "ok" and len(rec.requests) == 3
    assert sleeps[0] == 2.0 and len(sleeps) == 2


def test_retries_are_bounded_and_auth_errors_are_not_retried():
    http, rec = mock([httpx.Response(500, json={}) for _ in range(10)])
    with pytest.raises(ProviderServerError):
        openai(http, retries=2, sleep=lambda s: None).complete("s", HISTORY[:1], [])
    assert len(rec.requests) == 3
    http, rec = mock([httpx.Response(401, json={}) for _ in range(10)])
    with pytest.raises(ProviderAuthError):
        openai(http, retries=2, sleep=lambda s: None).complete("s", HISTORY[:1], [])
    assert len(rec.requests) == 1


def test_retry_after_is_capped():
    sleeps = []
    http, _ = mock([httpx.Response(429, headers={"retry-after": "9999"}, json={}), oa_resp("ok")])
    openai(http, retries=1, sleep=sleeps.append, max_retry_wait=5.0).complete("s", HISTORY[:1], [])
    assert sleeps == [5.0]


def test_transport_failures_map_to_errors():
    def timeout(req):
        raise httpx.ReadTimeout("slow", request=req)

    def refused(req):
        raise httpx.ConnectError("refused", request=req)

    for fn, exc in [(timeout, ProviderTimeoutError), (refused, ProviderConnectionError)]:
        c = OpenAICompatClient(base_url="https://api.example.com/v1", model="m", api_key=KEY,
                               http=httpx.Client(transport=httpx.MockTransport(fn)),
                               sleep=lambda s: None)
        with pytest.raises(exc):
            c.complete("s", HISTORY[:1], [])


def test_blocked_destination_maps_to_provider_blocked_error():
    def resolve(host, port):
        return ["10.0.0.1"]

    c = OpenAICompatClient(base_url="https://internal.example.com/v1", model="m", api_key=KEY,
                           policy=Policy(), resolver=resolve)
    with pytest.raises(ProviderBlockedError):
        c.complete("s", HISTORY[:1], [])


# ------------------------------------------------------------------ Gemini


def gem_resp(parts, finish="STOP", usage=(20, 6, 3)):
    return {"candidates": [{"content": {"role": "model", "parts": parts},
                            "finishReason": finish}],
            "usageMetadata": {"promptTokenCount": usage[0], "candidatesTokenCount": usage[1],
                              "thoughtsTokenCount": usage[2]}}


def gemini(http, **kw):
    return GeminiClient(model="gemini-test", api_key=KEY, http=http, **kw)


def test_gemini_request_shape():
    http, rec = mock([gem_resp([{"text": "hi"}])])
    gemini(http).complete("SYS", HISTORY, TOOLS)
    req = rec.requests[0]
    assert req.url.path == "/v1beta/models/gemini-test:generateContent"
    assert req.url.query == b""  # the key never goes in the URL
    assert req.headers["x-goog-api-key"] == KEY
    body = rec.body()
    assert body["systemInstruction"] == {"parts": [{"text": "SYS"}]}
    decl = body["tools"][0]["functionDeclarations"][0]
    assert decl["name"] == "search_code"
    assert "additionalProperties" not in json.dumps(decl) and "default" not in json.dumps(decl)
    assert decl["parameters"]["required"] == ["query"]
    contents = body["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    model_parts = contents[1]["parts"]
    assert model_parts[0] == {"text": "Let me search."}
    assert model_parts[1]["functionCall"] == {"name": "search_code", "args": {"query": "slugify"}}
    assert model_parts[2]["functionCall"]["name"] == "search_code"
    responses = [p["functionResponse"] for p in contents[2]["parts"]]
    assert [r["name"] for r in responses] == ["search_code", "search_code"]
    assert responses[0]["response"]["content"] == "<tool_output>A</tool_output>"
    assert responses[1]["response"]["is_error"] is True
    assert body["generationConfig"]["maxOutputTokens"] > 0


def test_gemini_parses_calls_text_usage_and_skips_thoughts():
    http, _ = mock([gem_resp([
        {"text": "thinking...", "thought": True},
        {"text": "Searching."},
        {"functionCall": {"name": "search_code", "args": {"query": "x"}},
         "thoughtSignature": "sig-abc"},
        {"functionCall": {"name": "read_file", "args": {"path": "a.py"}}},
    ])])
    r = gemini(http).complete("s", HISTORY[:1], TOOLS)
    assert r.text == "Searching." and r.stop_reason == "tool_use"
    assert [c.name for c in r.tool_calls] == ["search_code", "read_file"]
    assert len({c.id for c in r.tool_calls}) == 2
    assert (r.usage.input_tokens, r.usage.output_tokens) == (20, 9)


def test_gemini_thought_signature_round_trips():
    http, rec = mock([
        gem_resp([{"functionCall": {"name": "search_code", "args": {"query": "x"}},
                   "thoughtSignature": "sig-abc"}]),
        gem_resp([{"text": "done"}])])
    client = gemini(http)
    first = client.complete("s", HISTORY[:1], TOOLS)
    history = [HISTORY[0], {"role": "assistant", "content": first.content},
               {"role": "user", "content": [{"type": "tool_result",
                                             "tool_use_id": first.tool_calls[0].id,
                                             "content": "R"}]}]
    client.complete("s", history, TOOLS)
    sent = rec.body(1)["contents"][1]["parts"][0]
    assert sent["thoughtSignature"] == "sig-abc"
    assert rec.body(1)["contents"][2]["parts"][0]["functionResponse"]["name"] == "search_code"


@pytest.mark.parametrize(("finish", "expected"), [
    ("STOP", "end_turn"), ("MAX_TOKENS", "max_tokens"), ("SAFETY", "refusal"),
    ("RECITATION", "refusal"), ("PROHIBITED_CONTENT", "refusal"),
    ("MALFORMED_FUNCTION_CALL", "end_turn")])
def test_gemini_finish_reasons(finish, expected):
    http, _ = mock([gem_resp([{"text": "x"}], finish=finish)])
    assert gemini(http).complete("s", HISTORY[:1], []).stop_reason == expected


def test_gemini_blocked_prompt_and_empty_candidates():
    http, _ = mock([{"promptFeedback": {"blockReason": "SAFETY"}}])
    assert gemini(http).complete("s", HISTORY[:1], []).stop_reason == "refusal"
    http, _ = mock([{"candidates": []}])
    with pytest.raises(ProviderResponseError):
        gemini(http).complete("s", HISTORY[:1], [])


@pytest.mark.parametrize("model", ["../../v1/files", "a/b", "x?y", "", "m" * 300, "models/../x"])
def test_gemini_model_name_cannot_inject_path(model):
    with pytest.raises(ValueError):
        GeminiClient(model=model, api_key=KEY, http=httpx.Client())


def test_gemini_accepts_models_prefix():
    http, rec = mock([gem_resp([{"text": "ok"}])])
    GeminiClient(model="models/gemini-2.5-flash", api_key=KEY, http=http).complete(
        "s", HISTORY[:1], [])
    assert rec.requests[0].url.path == "/v1beta/models/gemini-2.5-flash:generateContent"


# ------------------------------------------------------------------ full agent passes


@pytest.fixture
def toolbox(tmp_path: Path) -> Toolbox:
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE_ROOT, repo, ignore=shutil.ignore_patterns("__pycache__"))
    db = tmp_path / "i.db"
    index_repo(repo, db)
    return Toolbox(Store(db))


def test_agent_end_to_end_through_openai_adapter(toolbox):
    http, rec = mock([
        oa_resp(None, [oa_call("c1", "search_code", '{"query": "slugify", "k": 3}')],
                finish="tool_calls"),
        oa_resp(None, [oa_call("c2", "read_file",
                               '{"path": "shopcore/utils.py", "start": 9, "end": 13}')],
                finish="tool_calls"),
        oa_resp(FINAL)])
    res = Agent(openai(http), toolbox).run("What does slugify do?")
    assert res.ok and not res.unsupported, res.failure
    assert res.citations[0]["verified"] is True
    assert [t["name"] for t in res.trace] == ["search_code", "read_file"]
    last = rec.body()["messages"]
    tool_msgs = [m for m in last if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]
    assert "def slugify" in tool_msgs[1]["content"]
    assert res.usage.input_tokens == 36


def test_agent_end_to_end_through_gemini_adapter(toolbox):
    http, rec = mock([
        gem_resp([{"functionCall": {"name": "search_code", "args": {"query": "slugify"}}}]),
        gem_resp([{"functionCall": {"name": "read_file",
                                    "args": {"path": "shopcore/utils.py", "start": 9,
                                             "end": 13}}}]),
        gem_resp([{"text": FINAL}])])
    res = Agent(gemini(http), toolbox).run("What does slugify do?")
    assert res.ok and not res.unsupported, res.failure
    assert res.citations[0]["verified"] is True
    responses = [p["functionResponse"] for c in rec.body()["contents"] for p in c["parts"]
                 if "functionResponse" in p]
    assert [r["name"] for r in responses] == ["search_code", "read_file"]


def test_agent_reports_provider_error_without_key(toolbox):
    http, _ = mock([httpx.Response(401, json={"error": {"message": f"bad key {KEY}"}})])
    res = Agent(openai(http), toolbox).run("q")
    assert not res.ok and res.failure.startswith("llm_error")
    assert KEY not in res.failure


# ------------------------------------------------------------------ registry


def test_list_providers():
    names = {p["name"] for p in list_providers()}
    assert {"anthropic", "openai", "gemini", "openai_compatible", "groq", "together",
            "openrouter", "ollama"} <= names
    for p in list_providers():
        assert set(p) >= {"name", "kind", "base_url", "requires_base_url", "local_only",
                          "key_env"}
        assert "api_key" not in p and KEY not in json.dumps(p)


def test_build_presets_and_kinds():
    c = build_client("groq", "llama-x", KEY, http=httpx.Client())
    assert isinstance(c, OpenAICompatClient) and c.base_url == "https://api.groq.com/openai/v1"
    c = build_client("openai", "gpt-x", KEY, http=httpx.Client())
    assert c.max_tokens_field == "max_completion_tokens"
    assert isinstance(build_client("gemini", "gemini-x", KEY, http=httpx.Client()), GeminiClient)
    a = build_client("anthropic", "claude-x", KEY)
    assert isinstance(a, AnthropicClient) and a.model == "claude-x"
    for c in (build_client("together", "t", KEY, http=httpx.Client()),
              build_client("openrouter", "o/r:free", KEY, http=httpx.Client())):
        assert c.provider in ("together", "openrouter")


def test_build_validation_errors():
    with pytest.raises(ValueError):
        build_client("nope", "m", KEY)
    with pytest.raises(ValueError):
        build_client("openai_compatible", "m", KEY)  # base_url required
    with pytest.raises(ValueError):
        build_client("openai", "m", KEY, base_url="https://api.example.com/v1")
    with pytest.raises(ValueError):
        build_client("openai", "", KEY)
    with pytest.raises(ValueError):
        build_client("openai", "m", "")
    with pytest.raises(ValueError):
        build_client("openai", "m", "sk-abc\r\nX-Evil: 1")
    with pytest.raises(ValueError):
        build_client("openai", "bad model\n", KEY)


def test_build_openai_compatible_validates_base_url():
    def resolve(host, port):
        return {"llm.example.com": ["93.184.216.34"], "intranet.example.com": ["10.0.0.9"]}[host]

    c = build_client("openai_compatible", "m", KEY, base_url="https://llm.example.com/v1",
                     resolver=resolve)
    assert c.base_url == "https://llm.example.com/v1"
    for url in ["https://intranet.example.com/v1", "http://llm.example.com/v1",
                "https://169.254.169.254/"]:
        with pytest.raises(ProviderBlockedError):
            build_client("openai_compatible", "m", KEY, base_url=url, resolver=resolve)


def test_ollama_is_local_only(monkeypatch):
    monkeypatch.delenv("CODERAG_ALLOW_INSECURE_LOCAL", raising=False)
    with pytest.raises(ProviderBlockedError):
        build_client("ollama", "llama3", None)
    monkeypatch.setenv("CODERAG_ALLOW_INSECURE_LOCAL", "1")
    c = build_client("ollama", "llama3", None, http=httpx.Client())
    assert c.base_url == "http://localhost:11434/v1"
