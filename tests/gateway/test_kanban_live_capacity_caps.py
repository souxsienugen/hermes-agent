"""The embedded dispatcher must enforce the LIVE concurrency caps (#38534647).

The gateway's embedded dispatcher resolved ``kanban.max_in_progress`` ONCE at
boot and passed that frozen copy into ``dispatch_once`` on every tick, while the
CLI dispatch path re-read the same config key on every invocation. The two
dispatchers therefore disagreed about the host-wide budget, and the embedded one
-- the only path that admits without an operator in the loop -- ran above the
operator's ceiling.

These tests pin the fix: the caps are re-read every tick, both dispatch paths
resolve them through one shared helper, and a config read failure keeps the caps
in force instead of dropping to "uncapped".
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from gateway import kanban_watchers_common as kwc
from gateway import kanban_watchers_dispatcher as kwd
from hermes_cli import kanban_db_dispatch as kbd


def _caps(**overrides):
    """A CapacityCaps with sane defaults, overridable per-test."""
    base = {"max_in_progress": 5, "max_in_progress_per_profile": 2, "max_spawn": None}
    base.update(overrides)
    return kbd.CapacityCaps(**base)


# ---------------------------------------------------------------------------
# 1. The live re-read binds a ceiling edit on the next tick
# ---------------------------------------------------------------------------


def test_live_caps_pick_up_a_config_edit():
    """A ceiling edit in config binds immediately, without a gateway restart."""
    cfg = {"kanban": {"max_in_progress": 5}}
    current = _caps(max_in_progress=5)

    caps = kwc._resolve_live_capacity_caps(lambda: cfg, current)

    assert caps.max_in_progress == 5

    # The operator lowers the ceiling.
    cfg["kanban"]["max_in_progress"] = 2
    caps = kwc._resolve_live_capacity_caps(lambda: cfg, caps)

    assert caps.max_in_progress == 2


def test_live_caps_track_a_raised_ceiling_too():
    """The live value is the operator's number in both directions."""
    cfg = {"kanban": {"max_in_progress": 2}}
    caps = kwc._resolve_live_capacity_caps(lambda: cfg, _caps(max_in_progress=2))

    cfg["kanban"]["max_in_progress"] = 7
    caps = kwc._resolve_live_capacity_caps(lambda: cfg, caps)

    assert caps.max_in_progress == 7


def test_live_caps_rebind_is_logged(caplog):
    """A re-bind is visible in gateway.log instead of being silently absorbed."""
    import logging

    cfg = {"kanban": {"max_in_progress": 5}}
    caps = kwc._resolve_live_capacity_caps(lambda: cfg, _caps(max_in_progress=5))

    cfg["kanban"]["max_in_progress"] = 2
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        kwc._resolve_live_capacity_caps(lambda: cfg, caps)

    messages = [r.getMessage() for r in caplog.records]
    assert any("max_in_progress=2" in m and "re-bound live" in m for m in messages)


# ---------------------------------------------------------------------------
# 2. A config read failure keeps the caps in force (fails SAFE)
# ---------------------------------------------------------------------------


def test_config_read_error_keeps_the_caps_in_force():
    """An unreadable config must not silently uncap the host.

    Dropping the caps would mean "no limit" -- the exact opposite of what a
    ceiling must do when the config cannot be read.
    """
    current = _caps(max_in_progress=2)

    def _boom():
        raise RuntimeError("config unreadable")

    caps = kwc._resolve_live_capacity_caps(_boom, current)

    assert caps == current
    assert caps.max_in_progress == 2


def test_unparseable_config_does_not_uncap_the_host():
    """A config that loads to a non-dict still keeps the caps in force."""
    current = _caps(max_in_progress=2)

    caps = kwc._resolve_live_capacity_caps(lambda: "not a dict", current)

    assert caps == current


def test_degraded_config_read_does_not_uncap_the_host():
    """A DEGRADED read (config.yaml unparseable) must not widen the ceiling.

    ``load_config`` serves defaults or last-known-good under the
    ``FailedConfigRead`` marker when the file cannot be parsed. Defaults carry
    no ``kanban.max_in_progress``, so honouring the degraded read would resolve
    the host cap to the memory-derived default -- i.e. a WIDER ceiling than the
    operator set, exactly when their config is broken.
    """
    from hermes_cli.config_read_errors import FailedConfigRead

    current = _caps(max_in_progress=2, max_in_progress_per_profile=2)
    degraded = FailedConfigRead({"kanban": {}}, error=ValueError("bad yaml"))

    caps = kwc._resolve_live_capacity_caps(lambda: degraded, current)

    assert caps == current
    assert caps.max_in_progress == 2
    assert caps.max_in_progress_per_profile == 2


# ---------------------------------------------------------------------------
# 3. Both dispatch paths resolve the cap through ONE helper
# ---------------------------------------------------------------------------


def test_gateway_and_cli_resolve_the_same_caps():
    """The embedded dispatcher and the CLI path agree on the cap.

    They disagreed before the fix: the gateway froze its copy at boot while the
    CLI re-read it live, so each filled to its own ceiling.
    """
    kanban_cfg = {"max_in_progress": 2, "max_in_progress_per_profile": 2}

    shared = kbd.resolve_capacity_caps(kanban_cfg)

    # The gateway's boot-time resolution goes through the same helper.
    gateway_settings = kwd._resolve_dispatcher_settings(kanban_cfg, kbd)
    assert gateway_settings.max_in_progress == shared.max_in_progress
    assert gateway_settings.max_in_progress_per_profile == shared.max_in_progress_per_profile

    # And the CLI path too.
    import hermes_cli.kanban_ops as kops

    assert kops.kbd.resolve_capacity_caps is kbd.resolve_capacity_caps
    assert shared.max_in_progress == 2
    assert shared.max_in_progress_per_profile == 2


def test_resolve_capacity_caps_uses_the_memory_derived_default_when_unset(monkeypatch):
    """An unset host cap falls back to the memory-derived default, not to "none"."""
    monkeypatch.setattr(kbd, "derive_default_max_in_progress", lambda sample=None: 4)

    caps = kbd.resolve_capacity_caps({})

    assert caps.max_in_progress == 4
    # An unset per-profile / per-board cap means "no limit", not "cap of zero".
    assert caps.max_in_progress_per_profile is None
    assert caps.max_spawn is None


@pytest.mark.parametrize(
    "raw,expected",
    [(2, 2), ("2", 2), (0, None), (-1, None), ("lots", None), (None, None)],
)
def test_resolve_capacity_caps_rejects_invalid_values(raw, expected):
    """A non-positive or unparseable cap is "unset", never a cap of zero."""
    caps = kbd.resolve_capacity_caps({"max_in_progress": raw})

    assert caps.max_in_progress == expected


# ---------------------------------------------------------------------------
# 4. The tick actually passes the LIVE caps to dispatch_once
# ---------------------------------------------------------------------------


def test_tick_passes_the_live_caps_to_dispatch_once(monkeypatch):
    """The boot-frozen snapshot must not reach dispatch_once.

    This is the actual breach: the frozen copy was passed to dispatch_once on
    every tick, so the dispatcher enforced the cap the gateway booted with
    rather than the one the operator set.
    """
    from hermes_cli import kanban_db as kb

    settings = kwd._DispatcherSettings(60.0, None, 5, 2, 0, True, None, 2)
    dispatcher = kwd._KanbanDispatcher(kb, settings)

    captured = {}

    def _fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(spawned=[], capacity_hold=None)

    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["default"])
    monkeypatch.setattr(kbd, "dispatch_once", _fake_dispatch_once)
    monkeypatch.setattr(
        kwd._kbc(), "connect", lambda board=None: SimpleNamespace(close=lambda: None)
    )

    live = _caps(max_in_progress=2, max_in_progress_per_profile=1)
    dispatcher.tick_once_for_board("default", live)

    assert captured["max_in_progress"] == 2
    assert captured["max_in_progress_per_profile"] == 1


def test_tick_without_live_caps_keeps_the_boot_snapshot(monkeypatch):
    """Without a live override the boot snapshot is still honoured."""
    from hermes_cli import kanban_db as kb

    settings = kwd._DispatcherSettings(60.0, None, 5, 2, 0, True, None, 2)
    dispatcher = kwd._KanbanDispatcher(kb, settings)

    captured = {}

    def _fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(spawned=[], capacity_hold=None)

    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["default"])
    monkeypatch.setattr(kbd, "dispatch_once", _fake_dispatch_once)
    monkeypatch.setattr(
        kwd._kbc(), "connect", lambda board=None: SimpleNamespace(close=lambda: None)
    )

    dispatcher.tick_once_for_board("default")

    assert captured["max_in_progress"] == 5
    assert captured["max_in_progress_per_profile"] == 2


# ---------------------------------------------------------------------------
# 5. End-to-end: the WATCHER LOOP passes the live cap to dispatch_once
# ---------------------------------------------------------------------------


def test_watcher_loop_passes_the_live_cap_to_dispatch_once(monkeypatch, tmp_path, caplog):
    """The whole loop must honour the live cap -- this is the actual breach.

    Drives the real ``_kanban_dispatcher_watcher`` with a config whose
    ``kanban.max_in_progress`` is edited between ticks. Before the fix the
    boot-frozen copy was passed to ``dispatch_once`` on every tick, so the
    dispatcher enforced the ceiling the gateway booted with rather than the
    one the operator set.
    """
    import asyncio
    import logging

    from gateway.kanban_watchers import GatewayKanbanWatchersMixin
    from hermes_cli import kanban_db as _kb
    from hermes_cli import kanban_db_connect as _kbc

    # The mixin carries the watcher loop; a bare instance is enough and avoids
    # importing gateway.run (whose bootstrap touches the real install's git dir).
    runner = GatewayKanbanWatchersMixin()
    runner._running = True  # type: ignore[attr-defined]

    # The boot-time config says 5; the operator lowers it to 2 afterwards.
    config = {"kanban": {"dispatch_in_gateway": True, "dispatch_interval_seconds": 1,
                          "max_in_progress": 5, "max_in_progress_per_profile": 2}}

    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda *a, **k: config
    )
    monkeypatch.setattr(
        _kb, "list_boards", lambda include_archived=False: [{"slug": _kb.DEFAULT_BOARD}]
    )
    monkeypatch.setattr(_kb, "read_board_metadata", lambda slug: {"slug": slug})
    monkeypatch.setattr(_kb, "kanban_db_path", lambda board=None: tmp_path / "kanban.db")
    monkeypatch.setattr(_kb, "kanban_home", lambda: tmp_path)
    monkeypatch.setattr(
        _kb, "pin_first_board_resolution", lambda: contextlib.nullcontext()
    )

    seen_caps: list = []

    def _fake_dispatch_once(conn, **kwargs):
        seen_caps.append(kwargs.get("max_in_progress"))
        # The operator lowers the ceiling after the first tick.
        if len(seen_caps) == 1:
            config["kanban"]["max_in_progress"] = 2
        return SimpleNamespace(spawned=[], capacity_hold=None)

    monkeypatch.setattr(kbd, "dispatch_once", _fake_dispatch_once)
    monkeypatch.setattr(
        _kbc, "connect", lambda *a, **k: SimpleNamespace(close=lambda: None)
    )

    ticks = {"n": 0}

    async def _to_thread(fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        ticks["n"] += 1
        # reap + auto-decompose + tick_once + ready_nonempty per tick; stop
        # after two ticks so the second one sees the lowered ceiling.
        if ticks["n"] >= 8:
            runner._running = False
        return result

    async def _sleep(_delay):
        return None

    monkeypatch.setattr("gateway.kanban_watchers.asyncio.to_thread", _to_thread)
    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", _sleep)

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=5.0))

    # The first tick used the boot value; a later tick used the live one.
    assert seen_caps, "the dispatcher never ran"
    assert seen_caps[0] == 5
    assert 2 in seen_caps, f"the live cap never reached dispatch_once: {seen_caps}"
    assert seen_caps[-1] == 2
