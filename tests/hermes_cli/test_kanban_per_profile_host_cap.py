"""Host-wide accounting for the per-profile concurrency cap + capacity holds.

Two gaps in the dispatcher's capacity accounting, both measured live on a
multi-board host:

1. ``kanban.max_in_progress_per_profile`` was counted from the DISPATCHING
   board's rows only. The cap protects ONE profile's local model / browser
   pool / quota — host resources shared by every board — so a board-local
   count let N boards each grant the profile its own slot and a cap of 1
   became N. Measured live: 2 concurrent ``mark`` workers and 2 concurrent
   ``tom`` workers while ``max_in_progress_per_profile=1``.
   :func:`count_running_tasks_other_boards` already solves the same problem
   for the GLOBAL cap; the per-profile cap now uses the same host-wide rule.

2. A tick refused by a concurrency cap recorded no reason, so
   :func:`describe_suppression` returned ``""`` and the gateway's "dispatcher
   stuck" warning named no cause. A ceiling-bound backlog was therefore
   indistinguishable from a quota-parked one in the logs — the exact
   ambiguity that made a capacity-bound backlog look like a hung dispatcher.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_spawn_factory(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42
    return fake_spawn


def _claim(conn: sqlite3.Connection, title: str, assignee: str) -> str:
    tid = kb.create_task(conn, title=title, assignee=assignee)
    assert kb.claim_task(conn, tid) is not None
    return tid


# ---------------------------------------------------------------------------
# 1. The per-profile cap counts a profile's workers on EVERY board
# ---------------------------------------------------------------------------


def test_per_profile_cap_counts_other_boards(
    kanban_home, all_assignees_spawnable,
):
    """A profile already running on another board holds its slot there too.

    Regression guard: with a board-local count, ``max_in_progress_per_profile=1``
    was satisfied by each board independently, so the same profile ran one
    worker per board (measured live: 2 concurrent ``mark`` workers under a
    cap of 1).
    """
    kb.create_board("second")
    with kbc.connect(board="second") as conn:
        _claim(conn, "busy-on-other-board", "alice")

    spawns: list = []
    with kbc.connect() as conn:
        kb.create_task(conn, title="wants-to-run", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns),
            max_in_progress=10, max_in_progress_per_profile=1,
        )

    # The cap is 1 and alice already runs on the other board → nothing spawns.
    assert not spawns
    assert not res.spawned
    assert len(res.skipped_per_profile_capped) == 1


def test_per_profile_cap_defers_task_when_other_board_busy(
    kanban_home, all_assignees_spawnable,
):
    """The deferral is recorded as a per-profile cap, not a silent skip."""
    kb.create_board("second")
    with kbc.connect(board="second") as conn:
        _claim(conn, "busy-on-other-board", "alice")

    spawns: list = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="wants-to-run", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns),
            max_in_progress=10, max_in_progress_per_profile=1,
        )

    assert not spawns
    assert not res.spawned
    # The ready task is deferred because alice is at its cap — the operator
    # needs this bucket to tell "profile busy" apart from "stuck".
    assert res.skipped_per_profile_capped == [(tid, "alice", 1)]


def test_per_profile_cap_allows_other_profiles_across_boards(
    kanban_home, all_assignees_spawnable,
):
    """A DIFFERENT profile is unaffected by another profile's cross-board use."""
    kb.create_board("second")
    with kbc.connect(board="second") as conn:
        _claim(conn, "busy-on-other-board", "alice")

    spawns: list = []
    with kbc.connect() as conn:
        kb.create_task(conn, title="other-profile", assignee="bob")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns),
            max_in_progress=10, max_in_progress_per_profile=1,
        )

    assert len(spawns) == 1
    assert len(res.spawned) == 1


def test_per_profile_cap_still_counts_same_board(
    kanban_home, all_assignees_spawnable,
):
    """Same-board accounting is unchanged: the cap still binds within a board."""
    spawns: list = []
    with kbc.connect() as conn:
        _claim(conn, "busy-here", "alice")
        kb.create_task(conn, title="wants-to-run", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns),
            max_in_progress=10, max_in_progress_per_profile=1,
        )

    assert not spawns
    assert len(res.skipped_per_profile_capped) == 1


def test_count_running_by_assignee_other_boards_excludes_current_board(
    kanban_home, all_assignees_spawnable,
):
    """The helper counts OTHER boards only — the current board is added separately."""
    kb.create_board("second")
    with kbc.connect(board="second") as conn:
        _claim(conn, "a", "alice")
        _claim(conn, "b", "alice")
        _claim(conn, "c", "bob")

    counts = kbd.count_running_tasks_by_assignee_other_boards("second")
    # "second" IS the current board → its own running rows are excluded (the
    # caller adds them separately from the board-local query).
    assert counts == {}

    # From the default board, "second" counts as another board.
    assert kbd.count_running_tasks_by_assignee_other_boards("default") == {
        "alice": 2,
        "bob": 1,
    }


def test_count_running_by_assignee_other_boards_fails_open(
    kanban_home, monkeypatch,
):
    """A broken board enumeration must not brick dispatch (returns {})."""
    monkeypatch.setattr(
        kb, "list_boards",
        lambda **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert kbd.count_running_tasks_by_assignee_other_boards() == {}


# ---------------------------------------------------------------------------
# 2. A ceiling-bound tick names its cause in the suppression line
# ---------------------------------------------------------------------------


def test_describe_suppression_reports_max_in_progress_hold(
    kanban_home, all_assignees_spawnable,
):
    """The host-wide cap refusing a tick must be visible in the log line."""
    kb.create_board("second")
    with kbc.connect(board="second") as conn:
        _claim(conn, "busy-1", "alice")
        _claim(conn, "busy-2", "alice")

    with kbc.connect() as conn:
        kb.create_task(conn, title="wants-to-run", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory([]), max_in_progress=2,
        )

    assert res.capacity_hold == "max_in_progress"
    held = kbd.describe_suppression([res])
    assert "capacity_hold=max_in_progress" in held


def test_describe_suppression_reports_max_spawn_hold(
    kanban_home, all_assignees_spawnable,
):
    """The per-board cap refusing a tick must be visible too."""
    with kbc.connect() as conn:
        _claim(conn, "busy", "alice")
        kb.create_task(conn, title="wants-to-run", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory([]), max_spawn=1,
        )

    assert res.capacity_hold == "max_spawn"
    assert "capacity_hold=max_spawn" in kbd.describe_suppression([res])


def test_describe_suppression_counts_per_profile_capped(
    kanban_home, all_assignees_spawnable,
):
    """Per-profile deferrals are counted so "profile busy" is distinguishable."""
    with kbc.connect() as conn:
        _claim(conn, "busy", "alice")
        kb.create_task(conn, title="wants-to-run", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory([]),
            max_in_progress=10, max_in_progress_per_profile=1,
        )

    held = kbd.describe_suppression([res])
    assert "per_profile_capped=1" in held


def test_describe_suppression_empty_when_nothing_held_back(
    kanban_home, all_assignees_spawnable,
):
    """A healthy tick still reports nothing — no spurious capacity noise."""
    spawns: list = []
    with kbc.connect() as conn:
        kb.create_task(conn, title="runs-fine", assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns))

    assert len(spawns) == 1
    assert res.capacity_hold is None
    assert kbd.describe_suppression([res]) == ""
