"""SSRF guard for outbound LLM provider requests.

Policy: https only; no userinfo; hosts must resolve exclusively to public addresses
(loopback, private, link-local incl. cloud metadata, CGNAT, multicast, reserved,
unspecified, and IPv6 forms that embed such IPv4 addresses are refused); numeric
host spellings (decimal, octal, hex, short forms) are decoded the way C's
``inet_aton`` would, then checked. Redirects are errors. Every request is
re-validated at send time and the connection is pinned to the validated IP, with
the original name kept for the Host header and TLS SNI/certificate checks, which
closes the DNS-rebinding window. ``CODERAG_ALLOW_INSECURE_LOCAL=1`` permits
loopback (including plain http) for local model servers such as Ollama.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

Resolver = Callable[[str, int], list[str]]

_LOCAL_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}
_NUMERIC_PART = re.compile(r"^(0[xX][0-9a-fA-F]*|[0-9]+)$")
_NAT64 = ipaddress.ip_network("64:ff9b::/96")


class BlockedDestination(ValueError):
    """The URL or the address it resolves to is not allowed by the policy."""


class ResponseTooLarge(Exception):
    """The provider response exceeded the configured byte cap."""


@dataclass(frozen=True)
class Policy:
    """Outbound request policy."""

    allow_insecure_local: bool = False
    allowed_hosts: frozenset[str] | None = None
    max_response_bytes: int = 5_000_000
    total_timeout: float = 120.0

    @classmethod
    def from_env(cls) -> Policy:
        """Read ``CODERAG_ALLOW_INSECURE_LOCAL`` and ``CODERAG_PROVIDER_HOSTS``."""
        hosts = os.environ.get("CODERAG_PROVIDER_HOSTS", "")
        allowed = frozenset(h.strip().lower().rstrip(".") for h in hosts.split(",") if h.strip())
        return cls(
            allow_insecure_local=os.environ.get("CODERAG_ALLOW_INSECURE_LOCAL") == "1",
            allowed_hosts=allowed or None,
        )


@dataclass(frozen=True)
class Target:
    """A validated destination with the IP to connect to."""

    scheme: str
    host: str
    port: int
    ip: str


def default_resolver(host: str, port: int) -> list[str]:
    """All addresses ``host`` resolves to."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as exc:
        raise BlockedDestination("cannot resolve host") from exc
    return sorted({str(info[4][0]) for info in infos})


def parse_ipv4_literal(host: str) -> ipaddress.IPv4Address | None:
    """Decode numeric IPv4 spellings like ``inet_aton`` (``2852039166``, ``0251.0376...``,
    ``0xa9fea9fe``, ``127.1``). Returns None for names; raises on malformed numbers."""
    parts = host.split(".")
    if not 1 <= len(parts) <= 4 or not all(_NUMERIC_PART.match(p) for p in parts):
        return None
    values = []
    for p in parts:
        try:
            if p[:2].lower() == "0x":
                values.append(int(p[2:] or "0", 16))
            elif len(p) > 1 and p.startswith("0"):
                values.append(int(p, 8))
            else:
                values.append(int(p, 10))
        except ValueError as exc:
            raise BlockedDestination("malformed numeric host") from exc
    limits = {1: [2**32], 2: [2**8, 2**24], 3: [2**8, 2**8, 2**16], 4: [2**8] * 4}[len(values)]
    if any(v >= lim for v, lim in zip(values, limits, strict=True)):
        raise BlockedDestination("malformed numeric host")
    number = 0
    for v, lim in zip(values, limits, strict=True):
        number = number * lim + v
    return ipaddress.IPv4Address(number)


def _ip_problem(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.scope_id:
            return "IPv6 zone identifiers are not allowed"
        embedded = ip.ipv4_mapped or ip.sixtofour
        if embedded is None and ip.teredo:
            embedded = ip.teredo[1]
        if embedded is None and ip in _NAT64:
            embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if embedded is not None:
            inner = _ip_problem(embedded)
            if inner:
                return f"embedded IPv4 is {inner}"
    for attr in ("is_loopback", "is_link_local", "is_private", "is_multicast",
                 "is_unspecified", "is_reserved"):
        if getattr(ip, attr):
            return attr[3:].replace("_", "-")
    if not ip.is_global:
        return "not globally routable"
    return None


def _is_local_name(host: str) -> bool:
    return host in _LOCAL_NAMES or host.endswith(".localhost")


def _host_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    if ":" in host:
        try:
            return ipaddress.ip_address(host)
        except ValueError as exc:
            raise BlockedDestination("malformed IPv6 host") from exc
    return parse_ipv4_literal(host)


def validate_url(url: str, policy: Policy, resolver: Resolver = default_resolver) -> Target:
    """Validate a request URL and choose the IP to connect to, or raise."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise BlockedDestination("malformed URL") from exc
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise BlockedDestination("only https URLs are allowed")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise BlockedDestination("credentials in URLs are not allowed")
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise BlockedDestination("URL has no host")
    if "%" in host:
        raise BlockedDestination("IPv6 zone identifiers are not allowed")
    port = port if port is not None else (443 if scheme == "https" else 80)
    if not 1 <= port <= 65535:
        raise BlockedDestination("port out of range")

    literal = _host_ip(host)
    local = _is_local_name(host) or (literal is not None and literal.is_loopback)
    if local:
        if not policy.allow_insecure_local:
            raise BlockedDestination("loopback destinations are not allowed")
        ip = str(literal) if literal is not None else "127.0.0.1"
        return Target(scheme, host, port, ip)
    if scheme != "https":
        raise BlockedDestination("only https URLs are allowed")
    if policy.allowed_hosts is not None and host not in policy.allowed_hosts:
        raise BlockedDestination("host is not in CODERAG_PROVIDER_HOSTS")
    if literal is not None:
        candidates = [literal]
    else:
        candidates = []
        for addr in resolver(host, port):
            try:
                candidates.append(ipaddress.ip_address(addr.split("%")[0]))
            except ValueError as exc:
                raise BlockedDestination("resolver returned a malformed address") from exc
        if not candidates:
            raise BlockedDestination("host did not resolve")
    for ip in candidates:
        problem = _ip_problem(ip)
        if problem:
            raise BlockedDestination(f"destination address is {problem}")
    return Target(scheme, host, port, str(candidates[0]))


def validate_base_url(url: str, policy: Policy, resolver: Resolver = default_resolver) -> Target:
    """Like :func:`validate_url`, and also refuse queries and fragments in a base URL."""
    if not isinstance(url, str) or not url.strip():
        raise BlockedDestination("empty URL")
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise BlockedDestination("malformed URL") from exc
    if parts.query or parts.fragment or "?" in url or "#" in url:
        raise BlockedDestination("base URL must not contain a query or fragment")
    return validate_url(url, policy, resolver)


class GuardedTransport(httpx.BaseTransport):
    """Validates, pins, and caps every request before handing it to ``inner``."""

    def __init__(self, policy: Policy, resolver: Resolver = default_resolver,
                 inner: httpx.BaseTransport | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.policy = policy
        self.resolver = resolver
        self.inner = inner or httpx.HTTPTransport(verify=True, trust_env=False, retries=0)
        self.clock = clock

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Send ``request`` to the validated IP; raise on policy violations."""
        target = validate_url(str(request.url), self.policy, self.resolver)
        default_port = 443 if target.scheme == "https" else 80
        host_header = target.host if target.port == default_port else (
            f"{target.host}:{target.port}")
        headers = httpx.Headers(request.headers)
        headers["host"] = host_header
        headers["accept-encoding"] = "identity"  # caps apply to bytes actually decoded
        extensions = dict(request.extensions)
        if target.scheme == "https":
            extensions["sni_hostname"] = target.host
        pinned = httpx.Request(request.method, request.url.copy_with(host=target.ip),
                               headers=headers, stream=request.stream, extensions=extensions)
        start = self.clock()
        response = self.inner.handle_request(pinned)
        try:
            if 300 <= response.status_code < 400:
                raise BlockedDestination("redirects are not followed")
            chunks, total = [], 0
            for chunk in response.stream:
                total += len(chunk)
                if total > self.policy.max_response_bytes:
                    raise ResponseTooLarge("provider response exceeded the size limit")
                if self.clock() - start > self.policy.total_timeout:
                    raise httpx.ReadTimeout("total request time exceeded", request=request)
                chunks.append(chunk)
            if self.clock() - start > self.policy.total_timeout:
                raise httpx.ReadTimeout("total request time exceeded", request=request)
        finally:
            response.close()
        return httpx.Response(response.status_code, headers=response.headers,
                              stream=httpx.ByteStream(b"".join(chunks)),
                              extensions=response.extensions, request=request)

    def close(self) -> None:
        """Close the inner transport."""
        self.inner.close()


def guarded_client(policy: Policy | None = None, resolver: Resolver = default_resolver,
                   timeout: httpx.Timeout | float | None = None,
                   inner: httpx.BaseTransport | None = None) -> httpx.Client:
    """An httpx client that never follows redirects, ignores proxy env vars, and routes
    every request through :class:`GuardedTransport`."""
    policy = policy or Policy.from_env()
    return httpx.Client(
        transport=GuardedTransport(policy, resolver=resolver, inner=inner),
        follow_redirects=False,
        trust_env=False,
        timeout=timeout if timeout is not None else httpx.Timeout(60.0, connect=10.0),
    )
