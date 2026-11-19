"""Pin the PR-acceptance probe's board isolation.

The probe WRITES Kanban rows, so it must never resolve the LIVE board. It used
to: it cleared HERMES_DELEGATED_CHILD_CONTEXT (the write fence that makes
``connect()`` open a board read-only) and then called ``connect()`` with no
``db_path``, so ``kanban_db_path()`` fell through to the HERMES_KANBAN_DB pin a
worker session injects.

What that cost on the live kom-chaser board, attributed exactly:

  * the 2 rows named "PR acceptance live" (2026-11-15) ARE this probe's. A
    pre-fix run wrote its first row and then died at the first complete_task
    with EmptyCompletionError, so one run per row: 2 runs = 2 rows.
  * the 24 rows named "qa-parent"/"qa-child-running" (2026-11-14) are NOT this
    probe's. They came from a different probe -- daniel's qa3r/diag_control.py,
    recorded on t_339c2a3b, which names that script and its titles. This probe
    only ever creates rows named "PR acceptance live" and "Concurrent
    ownership".

These tests drive the REAL probe module (not a re-implementation) and assert the
properties that keep that from happening again:

  * the board the probe opens is never the one the environment pinned;
  * a board inside the live tree, inside the delegation fence root, the board
    the environment pinned, or any board the fence marker covers is REFUSED
    outright, before any write and before any directory is created;
  * the delegation fence marker is left in place, so a mis-resolved board stays
    read-only rather than silently accepting rows;
  * every board the probe opens is named explicitly, so env resolution cannot
    win;
  * the rows the probe completes carry result evidence.

The probe is a script, not a library: it reads sys.argv and runs at import. The
tests therefore drive it the way a user would -- through sys.argv -- so the
board it opens is the one the assertions inspect.
"""
import importlib.util
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "evals" / "kanban_pr_acceptance_live.py"


def _run_probe(monkeypatch, tmp_path, board_dir, env=None):
    """Import (and therefore run) the probe with a controlled argv and env.

    Every key in ``env`` is applied, including keys the base environment does
    not define. An override for a key the base dict happens not to mention used
    to be dropped silently, which is what disarmed the live-tree refusal: the
    test named for the live tree never actually pinned the live tree, and the
    refusal that fired there was the pinned-board one instead.

    Returns the imported module, whose ``board_path`` is the board it opened.
    """
    settings = {
        # A worker session pins the LIVE board here; the probe must not use it.
        "HERMES_KANBAN_DB": str(tmp_path / "pinned" / "kanban.db"),
        "HERMES_KANBAN_BOARD": "kom-chaser",
        "HERMES_HOME": str(tmp_path / "home"),
        "HERMES_KANBAN_WORKSPACES_ROOT": str(tmp_path / "ws"),
        "HERMES_KANBAN_ATTACHMENTS_ROOT": str(tmp_path / "att"),
        # A normal session carries no fence marker; the probe must not add one
        # by popping it, and it must not need one to write its own board.
        "HERMES_DELEGATED_CHILD_CONTEXT": "",
    }
    settings.update(env or {})
    for key, value in settings.items():
        if value == "":
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, "argv", [str(PROBE), str(ROOT), str(ROOT), str(board_dir)])
    spec = importlib.util.spec_from_file_location("pr_acceptance_live_probe", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_board(path: Path) -> None:
    """A real board DB with the kanban schema, created through the real init."""
    from hermes_cli.kanban_db_connect import init_db

    init_db(path)


def _count(path: Path) -> int:
    if not path.exists():
        return 0
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT count(*) FROM tasks").fetchone()[0]
    finally:
        conn.close()


def _rows(path: Path, sql: str):
    """The rows of ``sql`` on the board at ``path`` (empty if there is no board)."""
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def test_probe_never_writes_the_board_the_environment_pinned(monkeypatch, tmp_path):
    """The pinned board stays empty; the probe's rows land on its own board."""
    pinned = tmp_path / "pinned" / "kanban.db"
    pinned.parent.mkdir(parents=True)
    _make_board(pinned)
    before = _count(pinned)

    board_dir = tmp_path / "probe-board"
    module = _run_probe(
        monkeypatch, tmp_path, board_dir,
        env={"HERMES_KANBAN_HOME": str(tmp_path / "live-tree")},
    )

    assert Path(module.board_path) != pinned, "the probe opened the pinned board"
    assert Path(module.board_path).parent == board_dir.resolve()
    assert _count(pinned) == before, "the probe wrote rows into the board the environment pinned"
    assert _count(Path(module.board_path)) > 0, "the probe wrote nothing at all"


def test_probe_refuses_a_board_inside_the_live_tree(monkeypatch, tmp_path):
    """A board inside the live tree is refused, before any write.

    The live tree here is the probe's own HERMES_KANBAN_HOME, and the pinned
    board points somewhere else entirely, so the live-tree refusal is the only
    guard that can fire: the pinned-board refusal cannot shadow it.
    """
    live_tree = tmp_path / "live-tree"
    live_board_dir = live_tree / "kanban" / "boards" / "kom-chaser"

    with pytest.raises(SystemExit) as excinfo:
        _run_probe(
            monkeypatch, tmp_path, live_board_dir,
            env={"HERMES_KANBAN_HOME": str(live_tree)},
        )
    assert "live board tree" in str(excinfo.value)
    assert not live_board_dir.exists(), (
        "the refused probe left a board behind in the live tree it refused"
    )


def test_probe_refuses_a_board_inside_the_fence_root(monkeypatch, tmp_path):
    """A board inside the delegation fence root is refused, before any write.

    The fence root is the tree the delegation fence marker covers. The refusal
    has to come from the live-tree guard -- the fence-marker guard would also
    stop the probe, but it fires later and means something else.
    """
    fence_root = tmp_path / "fence-root"
    fence_root.mkdir()
    fence_board_dir = fence_root / "kanban" / "boards" / "kom-chaser"

    with pytest.raises(SystemExit) as excinfo:
        _run_probe(
            monkeypatch, tmp_path, fence_board_dir,
            env={
                "HERMES_KANBAN_HOME": str(tmp_path / "live-tree"),
                "HERMES_DELEGATED_CHILD_CONTEXT": str(fence_root),
            },
        )
    assert "live board tree" in str(excinfo.value)
    assert "delegation fence marker" not in str(excinfo.value), (
        "the live-tree refusal did not fire -- the probe fell through to the "
        "fence-marker refusal instead"
    )
    assert not fence_board_dir.exists(), (
        "the refused probe left a board behind in the fence root it refused"
    )


def test_probe_refuses_the_board_the_environment_pinned(monkeypatch, tmp_path):
    """The board the environment pinned is refused, before any write.

    The pinned board is the only guard that can fire here: the live tree and the
    fence root both point elsewhere.
    """
    pinned = tmp_path / "pinned" / "kanban.db"
    pinned.parent.mkdir(parents=True)
    _make_board(pinned)
    before = _count(pinned)
    board_dir = tmp_path / "pinned" / "board-under-the-pinned-board"

    with pytest.raises(SystemExit) as excinfo:
        _run_probe(
            monkeypatch, tmp_path, board_dir,
            env={"HERMES_KANBAN_HOME": str(tmp_path / "live-tree")},
        )
    assert "pinned live board" in str(excinfo.value)
    assert _count(pinned) == before, "the refused probe still wrote to the pinned board"


def test_probe_refuses_a_board_the_fence_marker_covers(monkeypatch, tmp_path):
    """A board the delegation fence marker covers is refused, before any write.

    The legacy "1" marker fences every board in the process, the probe's own
    scratch board included, so the probe must refuse to run at all.
    """
    board_dir = tmp_path / "probe-board"

    with pytest.raises(SystemExit) as excinfo:
        _run_probe(
            monkeypatch, tmp_path, board_dir,
            env={
                "HERMES_KANBAN_HOME": str(tmp_path / "live-tree"),
                "HERMES_DELEGATED_CHILD_CONTEXT": "1",
            },
        )
    assert "delegation fence marker" in str(excinfo.value)
    assert not board_dir.exists(), (
        "the refused probe left a board behind for a board it refused to touch"
    )


def test_probe_keeps_the_delegation_fence_marker(monkeypatch, tmp_path):
    """The probe must NOT clear HERMES_DELEGATED_CHILD_CONTEXT.

    That marker is the write fence: with it in place a board the probe did not
    intend to write opens read-only, instead of accepting rows. Popping it is
    what let the pre-fix probe write into the live board.

    Asserted BEHAVIOURALLY -- the marker is still set after the probe ran. A pop
    spelled any way at all (a different literal, a concatenation, a ``del``)
    fails here, and so does a pop that dies on a missing key: the property
    under test is that the marker SURVIVES, not how it was spelled.
    """
    fence_root = tmp_path / "fence-root"
    fence_root.mkdir()
    board_dir = tmp_path / "probe-board"

    module = _run_probe(
        monkeypatch, tmp_path, board_dir,
        env={
            "HERMES_KANBAN_HOME": str(tmp_path / "live-tree"),
            "HERMES_DELEGATED_CHILD_CONTEXT": str(fence_root),
        },
    )

    assert os.environ.get("HERMES_DELEGATED_CHILD_CONTEXT") == str(fence_root), (
        "the probe cleared the delegation fence marker, so a mis-resolved board "
        "becomes writable"
    )
    assert _count(Path(module.board_path)) > 0, (
        "the probe wrote nothing, so the marker check proved nothing"
    )


def test_probe_opens_every_board_with_an_explicit_path(monkeypatch, tmp_path):
    """Every board the probe opens is named explicitly, so env resolution cannot win.

    Recorded at the seam the probe actually uses -- the kanban connector -- so
    this observes what the probe DID rather than what its source looks like. A
    bare ``connect()`` resolves the board through HERMES_KANBAN_DB and can land
    on the live board; an explicit path cannot.
    """
    from hermes_cli import kanban_db_connect

    opened = []
    real_connect = kanban_db_connect.connect

    def spy_connect(db_path=None, *, board=None):
        opened.append(db_path)
        return real_connect(db_path=db_path, board=board)

    monkeypatch.setattr(kanban_db_connect, "connect", spy_connect)

    board_dir = tmp_path / "probe-board"
    module = _run_probe(
        monkeypatch, tmp_path, board_dir,
        env={"HERMES_KANBAN_HOME": str(tmp_path / "live-tree")},
    )

    assert opened, "the probe opened no board at all"
    unnamed = [db_path for db_path in opened if db_path is None]
    assert not unnamed, "the probe opened a board without an explicit path"
    for db_path in opened:
        assert Path(db_path).resolve() == Path(module.board_path).resolve(), (
            "the probe opened a board other than the one it prepared"
        )


def test_probe_completes_its_tasks_with_result_evidence(monkeypatch, tmp_path):
    """The rows the probe completes carry result evidence.

    complete_task refuses a completion with no result or summary, so a probe
    that dropped its result= aborted with EmptyCompletionError instead of
    finishing. This is the positive half of that pin: the finished rows carry
    the evidence, they do not merely avoid the crash.
    """
    board_dir = tmp_path / "probe-board"
    module = _run_probe(
        monkeypatch, tmp_path, board_dir,
        env={"HERMES_KANBAN_HOME": str(tmp_path / "live-tree")},
    )

    done = _rows(Path(module.board_path), "SELECT status, result FROM tasks WHERE status='done'")
    assert done, "the probe completed no task"
    for _status, result in done:
        assert (result or "").strip(), "a completed task carries no result evidence"


@pytest.mark.skipif(not os.environ.get("HERMES_KANBAN_LIVE_PROBE_E2E"), reason="opt-in: runs the real probe")
def test_probe_end_to_end_leaves_the_pinned_board_untouched(tmp_path):
    """Run the probe for real in a subprocess; the pinned board stays empty."""
    pinned = tmp_path / "pinned" / "kanban.db"
    pinned.parent.mkdir(parents=True)
    _make_board(pinned)
    board_dir = tmp_path / "probe-board"
    env = dict(os.environ)
    env.update({
        "HERMES_KANBAN_DB": str(pinned),
        "HERMES_KANBAN_BOARD": "kom-chaser",
        "HERMES_KANBAN_HOME": str(tmp_path / "live-tree"),
        "HERMES_HOME": str(tmp_path / "home"),
    })
    # A normal session carries no fence marker; the probe must not need one.
    env.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    proc = subprocess.run(
        [sys.executable, str(PROBE), str(ROOT), str(ROOT), str(board_dir)],
        env=env, capture_output=True, text=True, timeout=300, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert _count(pinned) == 0
    assert _count(board_dir / "kanban.db") > 0
