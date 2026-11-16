"""Pin the PR-acceptance probe's board isolation.

The probe WRITES Kanban rows, so it must never resolve the LIVE board. It used
to: it cleared HERMES_DELEGATED_CHILD_CONTEXT (the write fence that makes
``connect()`` open a board read-only) and then called ``connect()`` with no
``db_path``, so ``kanban_db_path()`` fell through to the HERMES_KANBAN_DB pin a
worker session injects. Every fixture row then landed on the live board --
2 rows named "PR acceptance live" on 2026-11-15 and 24 named
"qa-parent"/"qa-child-running" on 2026-11-14.

These tests drive the REAL probe module (not a re-implementation) and assert the
properties that keep that from happening again:

  * the board the probe opens is never the one the environment pinned;
  * a board inside the live tree is REFUSED outright, before any write;
  * the delegation fence marker is left in place, so a mis-resolved board stays
    read-only rather than silently accepting rows;
  * every connect() passes an explicit db_path, so env resolution cannot win.

The probe is a script, not a library: it reads sys.argv and runs at import. The
tests therefore drive it the way a user would -- through sys.argv -- so the
board it opens is the one the assertions inspect.
"""
import ast
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

    Returns the imported module, whose ``board_path`` is the board it opened.
    """
    for key, value in {
        # A worker session pins the LIVE board here; the probe must not use it.
        "HERMES_KANBAN_DB": str(tmp_path / "pinned" / "kanban.db"),
        "HERMES_KANBAN_BOARD": "kom-chaser",
        "HERMES_HOME": str(tmp_path / "home"),
        "HERMES_KANBAN_WORKSPACES_ROOT": str(tmp_path / "ws"),
        "HERMES_KANBAN_ATTACHMENTS_ROOT": str(tmp_path / "att"),
        # A normal session carries no fence marker; the probe must not add one
        # by popping it, and it must not need one to write its own board.
        "HERMES_DELEGATED_CHILD_CONTEXT": "",
    }.items():
        if env and key in env:
            monkeypatch.setenv(key, env[key])
        elif value == "":
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


def test_probe_never_writes_the_board_the_environment_pinned(monkeypatch, tmp_path):
    """The pinned board stays empty; the probe's rows land on its own board."""
    pinned = tmp_path / "pinned" / "kanban.db"
    pinned.parent.mkdir(parents=True)
    _make_board(pinned)
    before = _count(pinned)

    board_dir = tmp_path / "probe-board"
    module = _run_probe(monkeypatch, tmp_path, board_dir)

    assert Path(module.board_path) != pinned, "the probe opened the pinned board"
    assert Path(module.board_path).parent == board_dir
    assert _count(pinned) == before, "the probe wrote rows into the board the environment pinned"
    assert _count(Path(module.board_path)) > 0, "the probe wrote nothing at all"


def test_probe_refuses_a_board_inside_the_live_tree(monkeypatch, tmp_path):
    """A board handed to the probe inside the live tree is refused, not opened."""
    live_tree = tmp_path / "live"
    live_board_dir = live_tree / "kanban" / "boards" / "kom-chaser"
    live_board_dir.mkdir(parents=True)
    pinned = live_board_dir / "kanban.db"
    _make_board(pinned)
    before = _count(pinned)

    with pytest.raises(SystemExit) as excinfo:
        _run_probe(
            monkeypatch, tmp_path, live_board_dir,
            env={"HERMES_KANBAN_HOME": str(live_tree), "HERMES_KANBAN_DB": str(pinned)},
        )
    assert "refusing" in str(excinfo.value)
    assert _count(pinned) == before, "the refused probe still wrote to the live board"


def test_probe_keeps_the_delegation_fence_marker():
    """The probe must NOT clear HERMES_DELEGATED_CHILD_CONTEXT.

    That marker is the write fence: with it in place a board the probe did not
    intend to write opens read-only, instead of accepting rows. Popping it is
    what let the 2026-11-14 probe write 24 rows into the live board.
    """
    source = PROBE.read_text(encoding="utf-8")
    for quote in ("'", '"'):
        assert f"pop({quote}HERMES_DELEGATED_CHILD_CONTEXT{quote}" not in source, (
            "the probe pops the delegation fence marker, so a mis-resolved board "
            "becomes writable"
        )


def test_probe_opens_its_board_with_an_explicit_path():
    """Every connect() in the probe passes db_path, so env resolution can't win."""
    tree = ast.parse(PROBE.read_text(encoding="utf-8"))
    bare = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "connect"
        and not any(kw.arg == "db_path" for kw in node.keywords)
    ]
    assert not bare, f"the probe calls connect() without an explicit db_path at lines {bare}"


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
        "HERMES_HOME": str(tmp_path / "home"),
    })
    proc = subprocess.run(
        [sys.executable, str(PROBE), str(ROOT), str(ROOT), str(board_dir)],
        env=env, capture_output=True, text=True, timeout=300, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert _count(pinned) == 0
    assert _count(board_dir / "kanban.db") > 0
