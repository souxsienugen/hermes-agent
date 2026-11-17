"""Retiring the gateway-owned managed worker of a card that reached a terminal state.

A Kanban card runs TWO processes. The dispatcher spawns the launcher
(``hermes_cli.kanban_worker_client``) and records its pid on ``task_runs.worker_pid``;
the agent itself runs in a SECOND process that the GATEWAY parents
(``agent.managed_worker``), whose pid the gateway records only in the run's
``worker_bound`` event. The terminal reap ended the launcher alone, so the
gateway-parented worker outlived the card and kept spending provider quota on a card
that was already DONE (2026-11-16, twice: the launcher pid and the bound worker pid
differ in every one of the 631 recorded ``worker_bound`` events).

Identity, never a pid number, authorises a signal here:

* the bound worker's spawn fingerprint (``<boot epoch>|<start time>``) comes from the
  ``worker_bound`` payload, so a recycled pid is never signalled;
* a descendant's kernel start time is captured before the tree is torn down and
  re-checked before the signal, so a descendant pid that was recycled meanwhile is
  never signalled either; and
* a process whose start time cannot be read is REFUSED, never signalled on a bare pid.

The gateway is the worker's PARENT. It is never a descendant of the worker, so walking
the worker's descendants can never reach it, and no process group is ever signalled:
the worker leads a group of its own but its kernel runner and process supervisor lead
groups of their own, so the tree is walked explicitly instead of by group.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import time
from typing import Any, Callable, Optional

#: The ``worker_bound`` event the gateway writes when it hands a card to its managed worker.
WORKER_BOUND_EVENT = "worker_bound"

#: Written once per closed run whose bound worker has been retired (or was already gone).
#: It is the marker that keeps a finished run from being swept again.
BOUND_WORKER_REAPED_EVENT = "terminal_bound_worker_reaped"


def bound_worker_records(conn: sqlite3.Connection, *, ended_before: int) -> list[dict[str, Any]]:
    """The closed runs whose bound worker has not been retired yet.

    One record per run: the run id, the task id, and the parsed ``worker_bound`` payload
    (``pid`` / ``claim_lock`` / ``started_at``) of the worker the gateway bound to it.
    A run that already carries :data:`BOUND_WORKER_REAPED_EVENT` is left out, so a
    finished run is examined once and then never again.
    """
    rows = conn.execute(
        "SELECT r.id AS run_id, r.task_id,"
        "  (SELECT e.payload FROM task_events e"
        "     WHERE e.run_id = r.id AND e.kind = 'worker_bound' ORDER BY e.id DESC LIMIT 1) AS payload"
        " FROM task_runs r"
        " WHERE r.ended_at IS NOT NULL AND r.ended_at <= ?"
        "   AND EXISTS (SELECT 1 FROM task_events e"
        "                WHERE e.run_id = r.id AND e.kind = 'worker_bound')"
        "   AND NOT EXISTS (SELECT 1 FROM task_events e"
        "                   WHERE e.run_id = r.id AND e.kind = ?)",
        (int(ended_before), BOUND_WORKER_REAPED_EVENT),
    ).fetchall()
    records: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row["payload"] or "{}")
        except (TypeError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        records.append({"run_id": int(row["run_id"]), "task_id": row["task_id"], "payload": payload})
    return records


def live_bound_worker_pids(conn: sqlite3.Connection) -> set[int]:
    """The pids bound to runs that have not ended: those workers are still doing live work.

    A bound worker is retired only when no open run still claims it, so a worker that was
    handed to a fresh attempt is never killed out from under that attempt.
    """
    pids: set[int] = set()
    for row in conn.execute(
        "SELECT e.payload FROM task_events e"
        " WHERE e.kind = 'worker_bound'"
        "   AND e.run_id IN (SELECT id FROM task_runs WHERE ended_at IS NULL)"
    ):
        try:
            payload = json.loads(row["payload"] or "{}")
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        pid = payload.get("pid")
        if isinstance(pid, int) and pid > 0:
            pids.add(pid)
    return pids


def _process_start_time(pid: int) -> Optional[int]:
    """The kernel start time of ``pid``, or None when it cannot be read.

    Field 22 of ``/proc/<pid>/stat`` on Linux: clock ticks since THIS boot, so it never
    moves for a live process and never matches another process that inherited the number.
    """
    from gateway.status import get_process_start_time

    return get_process_start_time(int(pid))


def process_alive(pid: int) -> bool:
    """True when ``pid`` is a live, running process.

    A zombie (exited, not yet reaped by its parent) is dead: it holds no fds and runs
    nothing, but its ``/proc`` entry survives, so a start-time probe alone would keep
    reporting it. Mirrors the dispatcher's own liveness rule.
    """
    pid = int(pid)
    if pid <= 0:
        return False
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("State:"):
                    return "Z" not in line.split(":", 1)[1]
    except (FileNotFoundError, PermissionError, OSError):
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OSError):
        return False
    return True


def _same_process(pid: int, start_time: Optional[int]) -> bool:
    """True when ``pid`` is still the process captured with ``start_time``.

    False for a process that is gone and for one whose start time cannot be read: an
    unreadable identity is a refusal to signal, never permission to.
    """
    if start_time is None:
        return False
    current = _process_start_time(pid)
    return current is not None and current == start_time


def _descendant_identities(root_pid: int) -> list[tuple[int, Optional[int]]]:
    """``(pid, start_time)`` of every live descendant of ``root_pid``.

    Snapshotted BEFORE anything is signalled: once the root dies its children are
    reparented and the ppid link that identifies them is gone. A descendant whose start
    time cannot be read is kept with a None identity so it is never signalled.
    """
    try:
        import psutil
    except Exception:
        return []
    try:
        children = psutil.Process(int(root_pid)).children(recursive=True)
    except Exception:
        return []
    identities: list[tuple[int, Optional[int]]] = []
    for child in children:
        try:
            pid = int(child.pid)
        except Exception:
            continue
        identities.append((pid, _process_start_time(pid)))
    return identities


def _signal(kill: Callable[[int, int], None], pid: int, sig: int) -> bool:
    """Deliver ``sig`` to ``pid``; False when the process was already gone."""
    try:
        kill(int(pid), sig)
        return True
    except (ProcessLookupError, OSError):
        return False


def terminate_worker_tree(
    root_pid: int,
    *,
    kill: Callable[[int, int], None],
    root_alive: Callable[[int], bool],
    grace: float = 5.0,
) -> dict[str, Any]:
    """End the worker at ``root_pid`` together with the process tree it owns.

    ``root_alive`` reports whether the root is still running (the caller's identity-aware
    liveness check); ``kill`` delivers a signal. SIGTERM first, then SIGKILL for whatever
    is still there when the grace period expires. A descendant is signalled only while its
    captured start time still matches, so a recycled descendant pid is never hit.

    The DESCENDANTS go first, deepest first, and the root last. The worker owns its
    kernel runner and its process supervisor; ending the root first would leave the
    supervisor free to respawn the runner, and would reparent the survivors to init --
    taking with it the parent chain that says which processes belong to this worker.
    """
    root_pid = int(root_pid)
    descendants = _descendant_identities(root_pid)
    result: dict[str, Any] = {
        "pid": root_pid,
        "terminated": False,
        "sigkill": False,
        "descendants": [pid for pid, _ in descendants],
        "descendants_killed": [],
    }
    # Deepest first: a child is gone before the parent that could replace it.
    for pid, start_time in reversed(descendants):
        if _same_process(pid, start_time):
            _signal(kill, pid, signal.SIGTERM)
    # The root last: it is the process still spending provider quota on a card that is
    # already DONE. Its death also releases the owner-death supervisor it owns.
    _signal(kill, root_pid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and root_alive(root_pid):
        time.sleep(0.2)
    if root_alive(root_pid):
        _signal(kill, root_pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        result["sigkill"] = True
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and root_alive(root_pid):
            time.sleep(0.2)
    # A descendant that ignored SIGTERM, or appeared while the tree was being torn down,
    # would keep its kernel runner and its supervisor alive: kill what is still there.
    for pid, start_time in reversed(descendants):
        if _same_process(pid, start_time):
            _signal(kill, pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            result["descendants_killed"].append(pid)
            result["sigkill"] = True
    result["terminated"] = not root_alive(root_pid)
    return result
