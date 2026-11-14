"""Tests for the live provider-wall probe (``hermes_cli.kanban_provider_wall``).

The probe is the LIVE half of the kanban rate-limit cooldown: the cooldown says
"a worker died on a quota wall an hour ago", the probe says "the provider is
refusing calls right now". It must be cheap, it must be shareable across the
per-board dispatchers, and it must FAIL OPEN — a broken probe may never strand
the fleet.
"""

from __future__ import annotations

import json
import time

import pytest

from hermes_cli import kanban_provider_wall as kpw


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_http_429_is_a_wall_whatever_the_body_says():
    assert kpw.classify_wall_response(429, "") == kpw.WALL_UP
    assert kpw.classify_wall_response(429, "quota exceeded") == kpw.WALL_UP


def test_token_plan_wall_body_is_a_wall_even_on_an_odd_status():
    body = (
        "Error code: 429 - {'type': 'error', 'error': {'type': 'rate_limit_error', "
        "'message': 'The Token Plan usage limit has been reached. To avoid "
        "interruption, please upgrade your plan or buy credits and turn on auto "
        "consumption. (2067)'}}"
    )
    assert kpw.classify_wall_response(400, body) == kpw.WALL_UP


def test_a_served_2xx_means_the_wall_is_down():
    assert kpw.classify_wall_response(200, '{"id": "abc"}') == kpw.WALL_DOWN


def test_anything_unclassifiable_is_unknown_not_up():
    # Unknown must never read as "up": a 500 or a timeout is not evidence of a wall.
    assert kpw.classify_wall_response(500, "boom") == kpw.WALL_UNKNOWN
    assert kpw.classify_wall_response(None, "") == kpw.WALL_UNKNOWN
    assert kpw.classify_wall_response(404, "not found") == kpw.WALL_UNKNOWN


def test_token_counts_never_masquerade_as_a_wall():
    """``total=206745`` in a successful call is not a 2067."""
    assert kpw.classify_wall_response(200, '{"usage": {"total_tokens": 206745}}') == kpw.WALL_DOWN


# ---------------------------------------------------------------------------
# Shared state file
# ---------------------------------------------------------------------------


def test_state_file_round_trip(tmp_path):
    root = tmp_path / ".hermes"
    assert kpw.read_wall_state(root) == {}
    kpw.write_wall_state(root, {"state": kpw.WALL_UP, "checked_at": 1234})
    # The file is keyed by provider so a second provider can share it.
    assert kpw.read_wall_state(root) == {
        "minimax-oauth": {"state": kpw.WALL_UP, "checked_at": 1234}
    }


def test_an_unreadable_state_file_reads_as_empty(tmp_path):
    root = tmp_path / ".hermes"
    kpw.wall_state_path(root).parent.mkdir(parents=True, exist_ok=True)
    kpw.wall_state_path(root).write_text("{not json", encoding="utf-8")
    assert kpw.read_wall_state(root) == {}


# ---------------------------------------------------------------------------
# live_wall_state: fresh reading vs cached reading
# ---------------------------------------------------------------------------


def test_a_fresh_cached_reading_is_reused_without_a_probe(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    now = int(time.time())
    kpw.write_wall_state(
        root, {"state": kpw.WALL_UP, "checked_at": now - 5, "provider": "minimax-oauth"}
    )

    def _boom(*a, **kw):
        raise AssertionError("a fresh reading must not trigger a probe")

    monkeypatch.setattr(kpw, "probe_minimax_oauth_wall", _boom)
    reading = kpw.live_wall_state(root=root, now=now)
    assert reading["state"] == kpw.WALL_UP
    assert reading["source"] == "cached"
    assert reading["age_seconds"] == 5


def test_a_stale_reading_is_re_probed(tmp_path, monkeypatch):
    """A wall-down measurement taken too early is not evidence about now."""
    root = tmp_path / ".hermes"
    now = int(time.time())
    kpw.write_wall_state(
        root, {"state": kpw.WALL_DOWN, "checked_at": now - 600, "provider": "minimax-oauth"}
    )

    seen = {}

    def _fake_probe(**kwargs):
        seen["called"] = True
        return {"state": kpw.WALL_UP, "checked_at": now, "detail": "429"}

    monkeypatch.setattr(kpw, "probe_minimax_oauth_wall", _fake_probe)
    reading = kpw.live_wall_state(root=root, now=now)
    assert seen.get("called") is True
    assert reading["state"] == kpw.WALL_UP
    assert reading["source"] == "live"
    # The fresh reading is published for the other per-board dispatchers.
    assert kpw.read_wall_state(root)["minimax-oauth"]["state"] == kpw.WALL_UP


def test_an_unknown_reading_is_not_published(tmp_path, monkeypatch):
    """Unknown fails open and must not poison the shared state file."""
    root = tmp_path / ".hermes"
    monkeypatch.setattr(
        kpw, "probe_minimax_oauth_wall",
        lambda **kw: {"state": kpw.WALL_UNKNOWN, "checked_at": int(time.time())},
    )
    reading = kpw.live_wall_state(root=root)
    assert reading["state"] == kpw.WALL_UNKNOWN
    assert kpw.read_wall_state(root) == {}


def test_the_probe_never_raises(tmp_path, monkeypatch):
    """A broken credential is a reading, not a crash."""
    from hermes_cli import auth_minimax

    def _boom(**kwargs):
        raise RuntimeError("no credential")

    monkeypatch.setattr(auth_minimax, "resolve_minimax_oauth_runtime_credentials", _boom)
    reading = kpw.probe_minimax_oauth_wall()
    assert reading["state"] == kpw.WALL_UNKNOWN
    assert "credential_unavailable" in reading["detail"]


def test_the_probe_posts_one_minimal_message(monkeypatch):
    """The probe must be cheap: one user message, one output token."""
    from hermes_cli import auth_minimax

    captured = {}

    class _FakeResponse:
        status = 200

        def read(self, _n=None):
            return b'{"id": "x"}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = dict(request.header_items())
        return _FakeResponse()

    monkeypatch.setattr(
        auth_minimax, "resolve_minimax_oauth_runtime_credentials",
        lambda **kw: {"provider": "minimax-oauth", "api_key": "tok", "base_url": "https://x/anthropic"},
    )
    monkeypatch.setattr(kpw.urllib.request, "urlopen", _fake_urlopen)
    reading = kpw.probe_minimax_oauth_wall(model="MiniMax-M3.1-Flash")
    assert reading["state"] == kpw.WALL_DOWN
    assert captured["url"].endswith("/v1/messages")
    assert captured["payload"]["max_tokens"] == 1
    assert captured["payload"]["messages"] == [{"role": "user", "content": "ping"}]
    # The bearer token must never leak into the reading.
    assert "tok" not in json.dumps(reading)


def test_a_429_from_the_probe_is_classified_as_a_wall(monkeypatch):
    from hermes_cli import auth_minimax
    import urllib.error

    monkeypatch.setattr(
        auth_minimax, "resolve_minimax_oauth_runtime_credentials",
        lambda **kw: {"provider": "minimax-oauth", "api_key": "tok", "base_url": "https://x/anthropic"},
    )

    def _raise(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 429, "Too Many Requests",
            {}, __import__("io").BytesIO(
                b"The Token Plan usage limit has been reached. (2067)"
            ),
        )

    monkeypatch.setattr(kpw.urllib.request, "urlopen", _raise)
    reading = kpw.probe_minimax_oauth_wall()
    assert reading["state"] == kpw.WALL_UP
    assert reading["status"] == 429


def test_the_freshness_window_is_configurable(monkeypatch):
    """The operator can tighten (or widen) the window without a code change."""
    monkeypatch.setenv("HERMES_KANBAN_PROVIDER_WALL_MAX_AGE_SECONDS", "45")
    assert kpw.resolve_max_age_seconds() == 45
    monkeypatch.setenv("HERMES_KANBAN_PROVIDER_WALL_MAX_AGE_SECONDS", "not-a-number")
    assert kpw.resolve_max_age_seconds() == kpw.DEFAULT_MAX_AGE_SECONDS
    monkeypatch.delenv("HERMES_KANBAN_PROVIDER_WALL_MAX_AGE_SECONDS")
    assert kpw.resolve_max_age_seconds() == kpw.DEFAULT_MAX_AGE_SECONDS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
