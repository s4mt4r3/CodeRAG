"""BYO-LLM: SSRF guard for outbound provider requests (offline, fake resolver)."""

from __future__ import annotations

import httpx
import pytest

from coderag.netguard import (
    BlockedDestination,
    GuardedTransport,
    Policy,
    ResponseTooLarge,
    guarded_client,
    parse_ipv4_literal,
    validate_base_url,
)

PUBLIC = "93.184.216.34"


def resolver_for(mapping: dict[str, list[str]]):
    calls: list[str] = []

    def resolve(host: str, port: int) -> list[str]:
        calls.append(host)
        if host not in mapping:
            raise BlockedDestination(f"cannot resolve {host}")
        return mapping[host]

    resolve.calls = calls  # type: ignore[attr-defined]
    return resolve


PUB = resolver_for({"api.example.com": [PUBLIC]})


@pytest.mark.parametrize("url", [
    "https://127.0.0.1/v1", "https://10.0.0.5/v1", "https://172.16.3.4/", "https://192.168.1.1/",
    "https://169.254.169.254/latest/meta-data", "https://0.0.0.0/", "https://100.64.0.1/",
    "https://224.0.0.1/", "https://240.0.0.1/", "https://[::1]/", "https://[::]/",
    "https://[fe80::1]/", "https://[fc00::1]/", "https://[::ffff:127.0.0.1]/",
    "https://[::ffff:169.254.169.254]/", "https://[64:ff9b::a9fe:a9fe]/",
    "https://[2002:a9fe:a9fe::1]/",
    "https://2852039166/", "https://0251.0376.0251.0376/", "https://0xa9fea9fe/",
    "https://169.254.43518/", "https://127.1/", "https://0/",
    "https://localhost/", "https://LOCALHOST./", "https://foo.localhost/",
    "https://ip6-localhost/",
])
def test_internal_destinations_are_blocked(url):
    with pytest.raises(BlockedDestination):
        validate_base_url(url, Policy(), resolver=PUB)


@pytest.mark.parametrize("url", [
    "https://evil@169.254.169.254/", "https://user:pw@api.example.com/",
    "ftp://api.example.com/", "file:///etc/passwd", "gopher://api.example.com/",
    "http://api.example.com/v1", "https:///v1", "https://api.example.com:0/",
    "https://api.example.com:70000/", "https://api.example.com/v1?x=1",
    "https://api.example.com/v1#frag", "https://[fe80::1%25en0]/", "", "not a url",
])
def test_malformed_or_unsafe_urls_are_rejected(url):
    with pytest.raises(BlockedDestination):
        validate_base_url(url, Policy(), resolver=PUB)


def test_public_host_passes_and_is_pinned():
    target = validate_base_url("https://api.example.com/v1", Policy(), resolver=PUB)
    assert (target.host, target.port, target.ip, target.scheme) == (
        "api.example.com", 443, PUBLIC, "https")


def test_dns_name_resolving_to_private_ip_is_blocked():
    res = resolver_for({"sneaky.example.com": ["10.1.2.3"],
                        "mixed.example.com": [PUBLIC, "127.0.0.1"]})
    for host in ("sneaky.example.com", "mixed.example.com"):
        with pytest.raises(BlockedDestination):
            validate_base_url(f"https://{host}/v1", Policy(), resolver=res)


def test_allowlist():
    res = resolver_for({"api.example.com": [PUBLIC], "other.example.com": [PUBLIC]})
    pol = Policy(allowed_hosts=frozenset({"api.example.com"}))
    assert validate_base_url("https://API.example.com/v1", pol, resolver=res).ip == PUBLIC
    with pytest.raises(BlockedDestination):
        validate_base_url("https://other.example.com/v1", pol, resolver=res)


def test_insecure_local_opt_in_allows_only_localhost_http():
    res = resolver_for({"localhost": ["127.0.0.1"], "api.example.com": [PUBLIC]})
    pol = Policy(allow_insecure_local=True)
    t = validate_base_url("http://localhost:11434/v1", pol, resolver=res)
    assert (t.scheme, t.ip, t.port) == ("http", "127.0.0.1", 11434)
    with pytest.raises(BlockedDestination):
        validate_base_url("http://api.example.com/v1", pol, resolver=res)
    with pytest.raises(BlockedDestination):
        validate_base_url("https://10.0.0.1/v1", pol, resolver=res)
    with pytest.raises(BlockedDestination):
        validate_base_url("http://localhost:11434/v1", Policy(), resolver=res)


def test_policy_from_env(monkeypatch):
    monkeypatch.setenv("CODERAG_ALLOW_INSECURE_LOCAL", "1")
    monkeypatch.setenv("CODERAG_PROVIDER_HOSTS", "a.example.com, B.example.com")
    pol = Policy.from_env()
    assert pol.allow_insecure_local is True
    assert pol.allowed_hosts == frozenset({"a.example.com", "b.example.com"})
    monkeypatch.delenv("CODERAG_ALLOW_INSECURE_LOCAL")
    monkeypatch.delenv("CODERAG_PROVIDER_HOSTS")
    assert Policy.from_env() == Policy()


def test_ipv4_literal_parsing():
    assert str(parse_ipv4_literal("2852039166")) == "169.254.169.254"
    assert str(parse_ipv4_literal("0251.0376.0251.0376")) == "169.254.169.254"
    assert str(parse_ipv4_literal("0xA9.0xFE.0xA9.0xFE")) == "169.254.169.254"
    assert str(parse_ipv4_literal("127.1")) == "127.0.0.1"
    assert parse_ipv4_literal("api.example.com") is None
    assert parse_ipv4_literal("dead.beef") is None
    with pytest.raises(BlockedDestination):
        parse_ipv4_literal("1.2.3.999")
    with pytest.raises(BlockedDestination):
        parse_ipv4_literal("08.1.1.1")


# ------------------------------------------------------------------ transport


def _transport(handler, resolver, policy=None, **kw):
    return GuardedTransport(policy or Policy(), resolver=resolver,
                            inner=httpx.MockTransport(handler), **kw)


def test_transport_pins_validated_ip_and_keeps_host_and_sni():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["host"] = req.url.host
        seen["Host"] = req.headers["host"]
        seen["sni"] = req.extensions.get("sni_hostname")
        seen["enc"] = req.headers.get("accept-encoding")
        return httpx.Response(200, json={"ok": True})

    res = resolver_for({"api.example.com": [PUBLIC]})
    with httpx.Client(transport=_transport(handler, res)) as c:
        r = c.post("https://api.example.com/v1/chat/completions", json={})
    assert r.json() == {"ok": True}
    assert seen == {"host": PUBLIC, "Host": "api.example.com", "sni": "api.example.com",
                    "enc": "identity"}
    assert res.calls == ["api.example.com"]  # one resolution per request, then pinned


def test_rebinding_between_check_and_send_is_caught():
    answers = iter([[PUBLIC], ["127.0.0.1"]])

    def resolve(host, port):
        return next(answers)

    validate_base_url("https://rebind.example.com/v1", Policy(), resolver=resolve)
    hit = []
    t = _transport(lambda r: hit.append(r) or httpx.Response(200), resolve)
    with httpx.Client(transport=t) as c, pytest.raises(BlockedDestination):
        c.get("https://rebind.example.com/v1/models")
    assert hit == []  # never connected


def test_redirects_are_errors_even_to_public_hosts():
    def handler(req):
        return httpx.Response(302, headers={"location": "http://169.254.169.254/"})

    with httpx.Client(transport=_transport(handler, PUB)) as c, pytest.raises(BlockedDestination):
        c.get("https://api.example.com/v1")


def test_response_size_cap():
    def handler(req):
        return httpx.Response(200, content=b"x" * 5000)

    t = _transport(handler, PUB, policy=Policy(max_response_bytes=1000))
    with httpx.Client(transport=t) as c, pytest.raises(ResponseTooLarge):
        c.get("https://api.example.com/v1")


def test_total_time_cap():
    ticks = iter([0.0, 0.0, 999.0, 999.0, 999.0])

    def handler(req):
        return httpx.Response(200, content=b"abc")

    t = _transport(handler, PUB, policy=Policy(total_timeout=10.0), clock=lambda: next(ticks))
    with httpx.Client(transport=t) as c, pytest.raises(httpx.TimeoutException):
        c.get("https://api.example.com/v1")


def test_guarded_client_does_not_follow_redirects_or_use_env_proxies(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    client = guarded_client(Policy(), resolver=PUB)
    assert client.follow_redirects is False
    assert client._trust_env is False
    client.close()
