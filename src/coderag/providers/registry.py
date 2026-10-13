"""Provider registry: build an ``LLMClient`` from (provider, model, key[, base_url]).

Presets have fixed base URLs; only ``openai_compatible`` accepts a caller-supplied
base URL, which is validated against the SSRF policy before anything is sent.
``ollama`` is local-only and requires ``CODERAG_ALLOW_INSECURE_LOCAL=1``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import httpx

from coderag.llm import AnthropicClient, LLMClient
from coderag.netguard import (
    BlockedDestination,
    Policy,
    Resolver,
    default_resolver,
    validate_base_url,
)
from coderag.providers.base import ProviderBlockedError, check_key
from coderag.providers.gemini import GeminiClient
from coderag.providers.openai_compat import OpenAICompatClient

_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,199}$")


@dataclass(frozen=True)
class ProviderSpec:
    """Static description of a supported provider."""

    name: str
    kind: str
    base_url: str | None
    key_env: str | None
    requires_base_url: bool = False
    local_only: bool = False
    max_tokens_field: str = "max_tokens"


PROVIDERS: dict[str, ProviderSpec] = {p.name: p for p in [
    ProviderSpec("anthropic", "anthropic", "https://api.anthropic.com", "ANTHROPIC_API_KEY"),
    ProviderSpec("openai", "openai_compat", "https://api.openai.com/v1", "OPENAI_API_KEY",
                 max_tokens_field="max_completion_tokens"),
    ProviderSpec("gemini", "gemini", "https://generativelanguage.googleapis.com",
                 "GEMINI_API_KEY"),
    ProviderSpec("groq", "openai_compat", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    ProviderSpec("together", "openai_compat", "https://api.together.xyz/v1",
                 "TOGETHER_API_KEY"),
    ProviderSpec("openrouter", "openai_compat", "https://openrouter.ai/api/v1",
                 "OPENROUTER_API_KEY"),
    ProviderSpec("ollama", "openai_compat", "http://localhost:11434/v1", None,
                 local_only=True),
    ProviderSpec("openai_compatible", "openai_compat", None, None, requires_base_url=True),
]}


def list_providers() -> list[dict]:
    """Public description of every provider (no secrets)."""
    return [{"name": p.name, "kind": p.kind, "base_url": p.base_url,
             "requires_base_url": p.requires_base_url, "local_only": p.local_only,
             "key_env": p.key_env, "requires_key": not p.local_only}
            for p in PROVIDERS.values()]


def key_env_names(provider: str) -> list[str]:
    """Environment variables that may hold the key for ``provider``, in lookup order."""
    spec = PROVIDERS.get((provider or "").strip().lower())
    return ["CODERAG_LLM_KEY"] + ([spec.key_env] if spec and spec.key_env else [])


def key_from_env(provider: str) -> str | None:
    """The key for ``provider`` from ``CODERAG_LLM_KEY`` or the provider's own variable."""
    for name in key_env_names(provider):
        if os.environ.get(name):
            return os.environ[name]
    return None


def build_client(provider: str, model: str, api_key: str | None, base_url: str | None = None,
                 http: httpx.Client | None = None, policy: Policy | None = None,
                 resolver: Resolver = default_resolver) -> LLMClient:
    """Construct a client; raises ``ValueError`` (or ``ProviderBlockedError``) on bad input."""
    spec = PROVIDERS.get((provider or "").strip().lower())
    if spec is None:
        raise ValueError(f"unknown provider; choose one of: {', '.join(PROVIDERS)}")
    if not isinstance(model, str) or not _MODEL_RE.match(model):
        raise ValueError("invalid model name")
    if base_url and not spec.requires_base_url:
        raise ValueError("a base URL is only accepted for provider 'openai_compatible'")
    if spec.requires_base_url and not base_url:
        raise ValueError("provider 'openai_compatible' requires a base URL")
    # Self-hosted endpoints (ollama, openai_compatible) often need no key.
    key = check_key(api_key, required=not (spec.local_only or spec.requires_base_url))
    policy = policy or Policy.from_env()

    if spec.kind == "anthropic":
        import anthropic

        return AnthropicClient(model=model, client=anthropic.Anthropic(api_key=key))
    if spec.kind == "gemini":
        return GeminiClient(model=model, api_key=key, http=http, policy=policy,
                            resolver=resolver)
    url = base_url or spec.base_url or ""
    if spec.requires_base_url or spec.local_only:
        try:
            validate_base_url(url, policy, resolver)
        except BlockedDestination as exc:
            raise ProviderBlockedError(f"base URL not allowed: {exc}") from None
    return OpenAICompatClient(base_url=url, model=model, api_key=key, http=http,
                              provider=spec.name, max_tokens_field=spec.max_tokens_field,
                              policy=policy, resolver=resolver)
