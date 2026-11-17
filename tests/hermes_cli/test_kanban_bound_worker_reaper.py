"""Invariant: the GATEWAY-parented managed worker of a DONE card does not survive it.

A card runs TWO processes. The dispatcher spawns the launcher
(``hermes_cli.kanban_worker_client``) and records its pid on ``task_runs.worker_pid``;
the agent itself runs in a second process the GATEWAY parents
(``agent.managed_worker``), whose pid the gateway records only in the run's
``worker_bound`` event. The two pids are different in every one of the 631
``worker_bound`` events on the kom-chaser board.

The terminal reap only ended the launcher. So the launcher was reaped, the card was
DONE, and the managed worker kept running -- still spending provider quota on a card
nobody was waiting for any more (2026-11-16, twice: the launcher pid and the bound
worker pid differed in both incidents).

These tests pin the fix: the terminal sweep must retire the BOUND worker (and the
kernel runner and process supervisor beneath it) by identity, must never signal a
recycled pid, must never touch the gateway that parents the worker, and must leave a
worker that is still bound to an open run alone.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_bound_worker as kbw
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _sleeper():
    """A live process standing in for a managed worker."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.2)
    return proc


def _child_sleeper():
    """A live process standing in for a managed worker's kernel runner."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.2)
    return proc


def _fingerprint(pid: int) -> str:
    """The spawn fingerprint the gateway records for a worker it bound."""
    return kbd._process_fingerprint(pid)


def _completed_run_with_bound_worker(conn, *, launcher_pid=None, worker=None, ended_ago=600):
    """A card that reached DONE, whose agent ran in the gateway-parented worker.

    The launcher and the managed worker are DIFFERENT processes -- the mismatch this
    whole sweep exists for. The launcher pid is recorded the way the dispatcher records
    it (``task_runs.worker_pid``); the managed worker's pid and fingerprint are recorded
    the way the gateway records them (the ``worker_bound`` event).
    """
    launcher = launcher_pid
    if launcher is None:
        launcher = _sleeper()
    tid = kb.create_task(conn, title="finished", assignee="coder")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    run_id = kb._current_run_id(conn, tid)
    kbd._set_worker_pid(conn, tid, launcher.pid if hasattr(launcher, "pid") else launcher)
    if worker is not None:
        kb._append_event(conn, tid, kbw.WORKER_BOUND_EVENT, {
            "pid": worker.pid,
            "claim_lock": kb._claimer_id(),
            "started_at": _fingerprint(worker.pid),
        }, run_id=run_id)
    assert kb.complete_task(conn, tid, result="done", expected_run_id=run_id) is True
    conn.execute("UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?", (ended_ago, run_id))
    return tid, run_id, launcher


def test_launcher_reap_alone_left_the_managed_worker_running(conn):
    """The gap this fixes: the launcher is a different pid from the managed worker, so
    reaping the launcher says nothing about the worker that was still spending quota."""
    launcher = _sleeper()
    worker = _sleeper()
    try:
        tid, run_id, _ = _completed_run_with_bound_worker(conn, launcher_pid=launcher, worker=worker)
        run = conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (run_id,)).fetchone()
        bound = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='worker_bound'",
            (tid, run_id),
        ).fetchone()
        assert run["worker_pid"] == launcher.pid
        assert json.loads(bound["payload"])["pid"] == worker.pid
        assert run["worker_pid"] != json.loads(bound["payload"])["pid"]

        # The launcher reap ends the launcher and nothing else.
        assert kbd.reap_terminal_workers(conn) == [tid]
        assert launcher.poll() is not None
        assert worker.poll() is None  # the managed worker survived the launcher reap
    finally:
        for p in (launcher, worker):
            if p.poll() is None:
                p.kill()
            p.wait()


def test_bound_worker_of_a_done_card_is_reaped_on_the_dispatch_tick(conn):
    worker = _sleeper()
    try:
        tid, _, _ = _completed_run_with_bound_worker(conn, worker=worker)

        result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0, dry_run=True, max_spawn=0)

        assert tid in result.reaped_terminal_bound_workers
        assert worker.wait(timeout=15) is not None
        kinds = [r["kind"] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id=?", (tid,))]
        assert kbw.BOUND_WORKER_REAPED_EVENT in kinds
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()


def test_bound_worker_reap_ends_the_workers_descendants(conn):
    """The managed worker's kernel runner and process supervisor are separate process
    groups; ending only the worker leaves them holding the card's workspace."""
    parent = _sleeper()
    child = None
    try:
        # A real tree: the "managed worker" owns a child process of its own.
        child = subprocess.Popen(
            [sys.executable, "-c",
             "import subprocess,sys,time;"
             "c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);"
             "print(c.pid,flush=True);time.sleep(120)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        grandchild_pid = int(child.stdout.readline().strip())
        time.sleep(0.2)

        tid = kb.create_task(conn, title="finished", assignee="coder")
        kb.claim_task(conn, tid, claimer=kb._claimer_id())
        run_id = kb._current_run_id(conn, tid)
        # The bound worker is the PARENT of the grandchild, so reaping it must take
        # the grandchild with it.
        kb._append_event(conn, tid, kbw.WORKER_BOUND_EVENT, {
            "pid": child.pid,
            "claim_lock": kb._claimer_id(),
            "started_at": _fingerprint(child.pid),
        }, run_id=run_id)
        assert kb.complete_task(conn, tid, result="done", expected_run_id=run_id) is True
        conn.execute("UPDATE task_runs SET ended_at = ended_at - 600 WHERE id=?", (run_id,))

        assert kbd.reap_terminal_bound_workers(conn) == [tid]

        assert child.wait(timeout=15) is not None
        # The grandchild was in the tree and is gone with it.
        deadline = time.time() + 15
        while time.time() < deadline and kbw.process_alive(grandchild_pid):
            time.sleep(0.2)
        assert not kbw.process_alive(grandchild_pid)
    finally:
        for p in (parent, child):
            if p is not None and p.poll() is None:
                p.kill()
            if p is not None:
                p.wait()


def test_the_gateway_that_parents_the_worker_is_never_signalled(conn):
    """The gateway is the managed worker's PARENT. Signalling it would take the whole
    gateway down with the worker, so it must be out of reach of the reap."""
    gateway = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess,sys,time;"
         "c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);"
         "print(c.pid,flush=True);time.sleep(120)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    worker_pid = int(gateway.stdout.readline().strip())
    time.sleep(0.2)
    try:
        tid = kb.create_task(conn, title="finished", assignee="coder")
        kb.claim_task(conn, tid, claimer=kb._claimer_id())
        run_id = kb._current_run_id(conn, tid)
        kb._append_event(conn, tid, kbw.WORKER_BOUND_EVENT, {
            "pid": worker_pid,
            "claim_lock": kb._claimer_id(),
            "started_at": _fingerprint(worker_pid),
        }, run_id=run_id)
        assert kb.complete_task(conn, tid, result="done", expected_run_id=run_id) is True
        conn.execute("UPDATE task_runs SET ended_at = ended_at - 600 WHERE id=?", (run_id,))

        assert kbd.reap_terminal_bound_workers(conn) == [tid]

        # The worker is gone; the gateway that parents it is still alive.
        deadline = time.time() + 15
        while time.time() < deadline and kbw.process_alive(worker_pid):
            time.sleep(0.2)
        assert not kbw.process_alive(worker_pid)
        assert gateway.poll() is None
    finally:
        if gateway.poll() is None:
            gateway.kill()
        gateway.wait()


def test_recycled_bound_pid_is_never_signalled(conn):
    """A pid that now belongs to an unrelated process must not be signalled."""
    stranger = _sleeper()
    try:
        tid, run_id, _ = _completed_run_with_bound_worker(conn, worker=stranger)
        # The recorded fingerprint belongs to a process that no longer exists.
        conn.execute(
            "UPDATE task_events SET payload = json_set(payload, '$.started_at', 'bogus-epoch|1')"
            " WHERE task_id=? AND run_id=? AND kind='worker_bound'", (tid, run_id))

        assert kbd.reap_terminal_bound_workers(conn) == []

        assert stranger.poll() is None  # the stranger is untouched
    finally:
        stranger.kill()
        stranger.wait()


def test_worker_bound_to_an_open_run_is_left_alone(conn):
    """A worker still bound to a live run is doing live work, not leaking."""
    worker = _sleeper()
    try:
        tid = kb.create_task(conn, title="running", assignee="coder")
        kb.claim_task(conn, tid, claimer=kb._claimer_id())
        run_id = kb._current_run_id(conn, tid)
        kb._append_event(conn, tid, kbw.WORKER_BOUND_EVENT, {
            "pid": worker.pid,
            "claim_lock": kb._claimer_id(),
            "started_at": _fingerprint(worker.pid),
        }, run_id=run_id)

        assert kbd.reap_terminal_bound_workers(conn) == []

        assert worker.poll() is None
    finally:
        worker.kill()
        worker.wait()


def test_a_done_card_is_not_swept_before_the_grace_window(conn):
    """A worker still finalising after its own transition must not be killed."""
    worker = _sleeper()
    signals = []

    def signal_fn(pid, sig):
        signals.append((pid, sig))
        os.kill(pid, sig)

    try:
        tid, run_id, _ = _completed_run_with_bound_worker(conn, worker=worker, ended_ago=0)

        assert kbd.reap_terminal_bound_workers(conn, signal_fn=signal_fn) == []

        assert signals == [] and worker.poll() is None
        conn.execute(
            "UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?",
            (kbd.TERMINAL_WORKER_REAP_GRACE_SECONDS, run_id),
        )
        assert kbd.reap_terminal_bound_workers(conn, signal_fn=signal_fn) == [tid]
        assert worker.pid in [pid for pid, _ in signals]
    finally:
        if worker.poll() is None:
            worker.kill()
        worker.wait()


def test_a_finished_run_is_examined_once(conn):
    """The reap leaves a marker so a closed run is not re-examined on every tick."""
    worker = _sleeper()
    try:
        tid, _, _ = _completed_run_with_bound_worker(conn, worker=worker)

        assert kbd.reap_terminal_bound_workers(conn) == [tid]
        assert worker.wait(timeout=15) is not None

        # A second sweep finds nothing left to do for that run.
        assert kbd.reap_terminal_bound_workers(conn) == []
        markers = conn.execute(
            "SELECT COUNT(*) c FROM task_events WHERE task_id=? AND kind=?",
            (tid, kbw.BOUND_WORKER_REAPED_EVENT),
        ).fetchone()
        assert markers["c"] == 1
    finally:
        if worker.poll() is None:
            worker.kill()
        worker.wait()


def test_one_failing_row_does_not_abort_the_bound_worker_sweep(conn):
    """A signal failure on one run is logged and skipped; the other rows are still reaped."""
    broken, healthy = _sleeper(), _sleeper()
    try:
        _completed_run_with_bound_worker(conn, worker=broken)
        tid, _, _ = _completed_run_with_bound_worker(conn, worker=healthy)

        def signal_fn(pid, sig):
            if pid == broken.pid:
                raise RuntimeError("boom")
            os.kill(pid, sig)

        assert kbd.reap_terminal_bound_workers(conn, signal_fn=signal_fn) == [tid]
    finally:
        for p in (broken, healthy):
            if p.poll() is None:
                p.kill()
            p.wait()
