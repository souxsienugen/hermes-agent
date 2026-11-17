"""Tests for hermes_cli.kanban_diagnostics — rule-engine that produces
structured distress signals (diagnostics) for kanban tasks.

These tests exercise each rule in isolation using minimal in-memory
task/event/run fixtures (no DB) plus a few integration-style cases
that round-trip through the real kanban_db to make sure the rule
engine works on sqlite3.Row objects as well as dataclasses.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_diagnostics as kd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _task(**overrides):
    base = {
        "id": "t_demo00",
        "title": "demo task",
        "assignee": "demo",
        "status": "ready",
        "consecutive_failures": 0,
        "last_failure_error": None,
    }
    base.update(overrides)
    return base


def _event(kind, ts=None, **payload):
    return {
        "kind": kind,
        "created_at": int(ts if ts is not None else time.time()),
        "payload": payload or None,
    }


def _run(outcome="completed", run_id=1, error=None):
    return {
        "id": run_id,
        "outcome": outcome,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Each rule — positive + negative + clearing
# ---------------------------------------------------------------------------
















def test_running_with_open_parents_fires_only_while_running():
    """A running card whose parent is not terminal is flagged; the same graph
    on a ready/todo card (the gate is holding it) and a done parent are not."""
    graph = {"parents": [{"id": "t_parent", "title": "p", "status": "todo"}], "children": []}
    diags = kd.compute_task_diagnostics(_task(status="running", started_at=100), [], [], graph=graph)
    assert [d.kind for d in diags] == ["running_with_open_parents"]
    assert diags[0].data["open_parents"] == [{"id": "t_parent", "status": "todo"}]
    assert "hermes kanban unlink t_parent t_demo00" in diags[0].actions[0].payload["command"]
    assert kd.compute_task_diagnostics(_task(status="todo"), [], [], graph=graph) == []
    done_graph = {"parents": [{"id": "t_parent", "title": "p", "status": "done"}], "children": []}
    assert kd.compute_task_diagnostics(_task(status="running"), [], [], graph=done_graph) == []


def test_text_started_at_does_not_silence_running_with_open_parents():
    """A TEXT ``tasks.started_at`` must not cost the card its diagnostic.

    ``tasks.started_at`` has INTEGER affinity, but SQLite keeps a non-numeric value as
    TEXT and outside writers do leave text timestamps behind. The rule read the raw
    column with a bare ``int()``, which raised on that text; the engine swallows a
    raising rule, so the ``running_with_open_parents`` warning vanished and the board
    showed a card running concurrently with an open parent as healthy.
    """
    graph = {"parents": [{"id": "t_parent", "title": "p", "status": "todo"}], "children": []}
    now = int(time.time())
    for started_at in ("2026-11-12T10:30:00Z", "not-a-timestamp"):
        diags = kd.compute_task_diagnostics(
            _task(status="running", started_at=started_at), [], [], now=now, graph=graph,
        )
        assert [d.kind for d in diags] == ["running_with_open_parents"], (
            f"started_at={started_at!r} silenced the diagnostic"
        )
        # A parseable text timestamp keeps its moment; an unparseable one falls back
        # to "now" so the diagnostic still carries a first_seen_at.
        assert diags[0].first_seen_at > 0


def test_text_timestamp_columns_do_not_silence_diagnostics():
    """The same TEXT-timestamp trap must not silence the other rules either.

    ``tasks.created_at`` and ``task_events.created_at`` are read the same way, so a
    text value there used to drop ``stranded_in_ready`` (and every rule that reads
    event timestamps) the same silent way.
    """
    now = int(time.time())
    stranded = kd.compute_task_diagnostics(
        _task(status="ready", claim_lock=None, created_at=now - 4000), [], [], now=now,
    )
    assert [d.kind for d in stranded] == ["stranded_in_ready"]

    text_created = kd.compute_task_diagnostics(
        _task(status="ready", claim_lock=None, created_at="2026-11-12T10:30:00Z"), [], [], now=now,
    )
    assert [d.kind for d in text_created] == ["stranded_in_ready"]

    text_event = kd.compute_task_diagnostics(
        _task(status="ready", claim_lock=None, created_at=now - 4000),
        [_event("created", ts=now - 4000)], [], now=now,
    )
    assert [d.kind for d in text_event] == ["stranded_in_ready"]


def test_a_rule_that_raises_is_logged_not_swallowed_silently(caplog):
    """A rule that raises must not 500 the board, but it must not vanish either.

    The engine swallows every rule exception so one broken rule cannot take down a
    whole /board request — but a swallowed exception is indistinguishable from a
    rule that simply did not fire, which is how a dropped diagnostic stayed invisible.
    A text ``worker_pid`` makes ``_rule_stale_running`` raise for real.
    """
    now = int(time.time())
    task = _task(
        status="running",
        claim_expires=now - 7200,
        claim_lock=kb._host_prefix() + "abc",
        last_heartbeat_at=None,
        worker_pid="not-a-pid",
        worker_started_at=None,
    )
    with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_diagnostics"):
        diags = kd.compute_task_diagnostics(task, [], [], now=now)
    assert diags == []
    assert any("_rule_stale_running" in rec.getMessage() for rec in caplog.records)


def test_stuck_in_blocked_fires_past_threshold():
    now = int(time.time())
    task = _task(status="blocked")
    events = [
        _event("blocked", ts=now - 3600 * 48, reason="needs approval"),
    ]
    diags = kd.compute_task_diagnostics(
        task, events, [], now=now,
    )
    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "stuck_in_blocked"
    assert d.severity == "warning"
    assert d.data["age_hours"] >= 48








# ---------------------------------------------------------------------------
# Severity sorting
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Integration — runs through real kanban_db so sqlite.Row fields work
# ---------------------------------------------------------------------------


def test_engine_works_on_sqlite_row_objects(kanban_home):
    """Regression: the rule functions must handle sqlite3.Row (which
    supports mapping access but not attribute access and isn't a dict)
    as well as dataclass Task / plain dict. The API layer passes Row
    objects directly.
    """
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="p", assignee="w")
        real = kb.create_task(conn, title="r", assignee="x", created_by="w")
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent,
                summary="with phantom", created_cards=[real, "t_deadbeef1"],
            )
        # Pull Row objects the way the API helper does.
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (parent,),
        ).fetchone()
        events = list(conn.execute(
            "SELECT * FROM task_events WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        runs = list(conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        diags = kd.compute_task_diagnostics(row, events, runs)
        assert len(diags) == 1
        assert diags[0].kind == "hallucinated_cards"
        assert "t_deadbeef1" in diags[0].data["phantom_ids"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Error-tolerance: a broken rule shouldn't 500 the whole compute call
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# stranded_in_ready
#
# Surfaces ready tasks that nobody has claimed within the threshold.
# Identity-agnostic by design: catches typo'd assignees, deleted profiles,
# down external worker pools, and misconfigured dispatchers in one rule.
# ---------------------------------------------------------------------------


def test_stranded_in_ready_fires_when_age_exceeds_threshold():
    """Default threshold = 30 min. A ready task promoted 45 min ago
    with no claim should fire as a warning."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    # 45 min = 2700s, threshold = 1800s.
    events = [_event("created", ts=now - 45 * 60)]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = [d for d in diags if d.kind == "stranded_in_ready"]
    assert len(stranded) == 1
    assert stranded[0].severity == "warning"
    assert stranded[0].data["age_seconds"] == 45 * 60
    assert stranded[0].data["assignee"] == "demo"




# ---------------------------------------------------------------------------
# triage_aux_unavailable rule — auto-decompose aware
# ---------------------------------------------------------------------------


def _triage_task():
    return _task(id="t_triage1", status="triage")








def test_severity_at_or_above_uses_threshold_semantics():
    assert kd.severity_at_or_above("warning", "warning") is True
    assert kd.severity_at_or_above("error", "warning") is True
    assert kd.severity_at_or_above("critical", "warning") is True
    assert kd.severity_at_or_above("critical", "error") is True
    assert kd.severity_at_or_above("warning", "error") is False
    assert kd.severity_at_or_above("error", "critical") is False
    assert kd.severity_at_or_above("mystery", "warning") is False
    assert kd.severity_at_or_above("warning", None) is True
