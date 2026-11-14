"""Tests for the live-wall gate on kanban re-admission, and the archive guard.

Two guards, one defect class (the MiniMax 2067 token-plan wall):

1. The rate-limit cooldown is a LOCAL timer. When it expires into a live wall
   the card is re-admitted, the worker dies on the same 2067, and the fleet
   burns quota re-learning the fact. The dispatcher therefore re-checks the LIVE
   wall before letting a cooldown-expired card back in.
2. A card whose worker run died on the wall minutes earlier is not resolved,
   and a provider tracker may not be archived while the wall is measurably up.
   The 2026-11-13 06:29:43 ET archive of ``t_4b4735f7`` is what this refuses.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from hermes_cli import kanban as kb_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_provider_wall as kpw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    kb.init_db()
    return home


def _rate_limited_run(conn: sqlite3.Connection, task_id: str, ended_at: int) -> None:
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
            "VALUES (?, 'mark', 'rate_limited', 'rate_limited', ?, ?)",
            (task_id, ended_at - 5, ended_at),
        )


def _card(conn: sqlite3.Connection, title: str, body: str = "") -> str:
    return kb.create_task(conn, title=title, body=body, assignee="mark")


# ---------------------------------------------------------------------------
# 1. The live-wall gate on re-admission
# ---------------------------------------------------------------------------


def test_a_cooldown_expired_card_is_held_while_the_wall_is_up(kanban_home, monkeypatch):
    """The cooldown expired, but the provider is still refusing calls."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "3600")
    with kbc.connect() as conn:
        task_id = _card(conn, "worker task")
        _rate_limited_run(conn, task_id, int(time.time()) - 3601)

        monkeypatch.setattr(kbd, "_provider_wall_is_up", lambda: True)
        assert kbd.check_respawn_guard(conn, task_id) == "provider_wall"


def test_a_cooldown_expired_card_is_admitted_once_the_wall_is_down(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "3600")
    with kbc.connect() as conn:
        task_id = _card(conn, "worker task")
        _rate_limited_run(conn, task_id, int(time.time()) - 3601)

        monkeypatch.setattr(kbd, "_provider_wall_is_up", lambda: False)
        assert kbd.check_respawn_guard(conn, task_id) is None


def test_the_wall_is_not_consulted_while_the_cooldown_is_still_running(kanban_home, monkeypatch):
    """Inside the cooldown the timer already holds the card — no probe needed."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "3600")
    with kbc.connect() as conn:
        task_id = _card(conn, "worker task")
        _rate_limited_run(conn, task_id, int(time.time()) - 60)

        def _must_not_probe():
            raise AssertionError("the live probe must not run inside the cooldown")

        monkeypatch.setattr(kbd, "_provider_wall_is_up", _must_not_probe)
        assert kbd.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"


def test_a_broken_probe_fails_open(kanban_home, monkeypatch):
    """A probe that raises must never strand a card."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "3600")
    with kbc.connect() as conn:
        task_id = _card(conn, "worker task")
        _rate_limited_run(conn, task_id, int(time.time()) - 3601)

        def _boom():
            raise RuntimeError("probe exploded")

        monkeypatch.setattr(kbd, "_provider_wall_is_up", _boom)
        # The gate itself is fail-open: the helper swallows probe errors.
        monkeypatch.undo()
        assert kbd._provider_wall_is_up() is False


def test_the_gate_reads_the_live_wall_not_the_timer(kanban_home, monkeypatch):
    """The gate's own contract: it asks the shared live reading."""
    calls = []

    def _fake_live_state(**kwargs):
        calls.append(kwargs)
        return {"state": kpw.WALL_UP, "source": "live"}

    monkeypatch.setattr(kpw, "live_wall_state", _fake_live_state)
    monkeypatch.setattr(kpw, "provider_wall_root", lambda: kanban_home)
    assert kbd._provider_wall_is_up() is True
    assert calls and "root" in calls[0]


# ---------------------------------------------------------------------------
# 2. The archive guard
# ---------------------------------------------------------------------------


def test_a_card_that_died_on_the_wall_may_not_be_archived(kanban_home):
    """Its work never happened — archiving it as resolved is a lie."""
    with kbc.connect() as conn:
        task_id = _card(conn, "PROVIDER: MiniMax 2067 wall tracker")
        _rate_limited_run(conn, task_id, int(time.time()) - 120)
        refusal = kb_cli._provider_wall_archive_refusal(conn, task_id)
        assert refusal is not None
        assert "quota wall" in refusal

        assert kb_cli._archive_one(conn, task_id) is False
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assert row["status"] != "archived"


def test_the_same_card_is_archivable_once_its_cooldown_has_elapsed(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "3600")
    with kbc.connect() as conn:
        task_id = _card(conn, "PROVIDER: MiniMax 2067 wall tracker")
        _rate_limited_run(conn, task_id, int(time.time()) - 3601)
        # The wall is down, so the class rule does not refuse either.
        monkeypatch.setattr(
            kpw, "live_wall_state",
            lambda **kw: {"state": kpw.WALL_DOWN, "source": "live"},
        )
        assert kb_cli._provider_wall_archive_refusal(conn, task_id) is None
        assert kb_cli._archive_one(conn, task_id) is True


def test_a_provider_tracker_may_not_be_archived_while_the_wall_is_up(kanban_home, monkeypatch):
    """Archiving the only tracker of a live defect is what created this card."""
    monkeypatch.setattr(
        kpw, "live_wall_state",
        lambda **kw: {"state": kpw.WALL_UP, "source": "live", "detail": "429"},
    )
    with kbc.connect() as conn:
        task_id = _card(conn, "PROVIDER: MiniMax 2067 wall re-armed and blinded the sweep")
        refusal = kb_cli._provider_wall_archive_refusal(conn, task_id)
        assert refusal is not None
        assert "measurably UP" in refusal
        assert kb_cli._archive_one(conn, task_id) is False


def test_an_ordinary_card_is_unaffected(kanban_home, monkeypatch):
    monkeypatch.setattr(
        kpw, "live_wall_state",
        lambda **kw: {"state": kpw.WALL_UP, "source": "live"},
    )
    with kbc.connect() as conn:
        task_id = _card(conn, "ordinary work card", body="implement the thing")
        assert kb_cli._provider_wall_archive_refusal(conn, task_id) is None
        assert kb_cli._archive_one(conn, task_id) is True


def test_the_db_layer_guard_covers_every_archive_path(kanban_home, monkeypatch):
    """The dashboard's PATCH/bulk routes call ``kb.archive_task`` directly."""
    monkeypatch.setattr(
        kpw, "live_wall_state",
        lambda **kw: {"state": kpw.WALL_UP, "source": "live"},
    )
    with kbc.connect() as conn:
        task_id = _card(conn, "PROVIDER: MiniMax 2067 wall tracker")
        assert kb.archive_task(conn, task_id) is False
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assert row["status"] != "archived"


def test_a_card_that_died_on_the_wall_is_not_closable_at_all(kanban_home):
    """Not even as an ordinary card: its work never happened."""
    with kbc.connect() as conn:
        task_id = _card(conn, "ordinary work card")
        _rate_limited_run(conn, task_id, int(time.time()) - 60)
        assert kb.archive_task(conn, task_id) is False
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assert row["status"] != "archived"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
