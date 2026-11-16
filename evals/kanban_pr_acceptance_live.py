"""Local HTTP/lifecycle probe; no external GitHub writes or inference.
Run: python evals/kanban_pr_acceptance_live.py ROOT [FIXTURE_ROOT] [BOARD_DIR]

This probe WRITES Kanban rows, so it must never resolve the live board.
Board resolution honours ``HERMES_KANBAN_DB`` (a worker session pins it at
the live board) ahead of anything the probe sets, and the write fence
(``HERMES_DELEGATED_CHILD_CONTEXT``) is what turns a mistaken live board
read-only instead of writable. So the probe:
  * KEEPS the fence marker, so a board it did not intend to write stays
    read-only instead of silently accepting rows;
  * redirects ``HERMES_KANBAN_HOME`` at its own scratch board and clears the
    ``HERMES_KANBAN_DB`` / ``HERMES_KANBAN_BOARD`` pins that outrank it;
  * passes an explicit ``db_path`` to every ``connect()``;
  * keeps the scratch board OUTSIDE the live board tree, because the fence
    marker covers everything under it (a board inside would be read-only).
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import pytest
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect_closing
import inspect


def _under(path, root_path):
    """Whether *path* is *root_path* or lives under it."""
    try:
        Path(path).resolve().relative_to(Path(root_path).resolve())
        return True
    except (ValueError, OSError):
        return False


def _live_board_tree():
    """The board tree this process would write to if the probe did nothing."""
    override = os.environ.get('HERMES_KANBAN_HOME', '').strip()
    if override:
        return Path(override).expanduser()
    from hermes_constants import get_default_hermes_root
    return Path(get_default_hermes_root())


def _fence_root():
    """The root the delegation fence marker covers, or ``None`` for no fence."""
    marker = os.environ.get('HERMES_DELEGATED_CHILD_CONTEXT', '').strip()
    if not marker or marker == '1':
        return None
    return Path(marker).expanduser()


def _scratch_base(excluded):
    """A temp base outside every excluded root (the fence marker covers its own)."""
    for base in (Path(tempfile.gettempdir()), Path('/tmp'), Path('/var/tmp')):
        if not base.is_dir():
            continue
        if any(_under(base, root) for root in excluded):
            continue
        return base
    raise SystemExit('no scratch base outside the live board tree')


def _prepare_board(board_dir_arg):
    """Point every board resolution at a scratch board; return its DB path.

    Refuses (loudly, before any write) to target a board inside the live tree
    or the board a worker session pinned.
    """
    live_tree = _live_board_tree()
    fence_root = _fence_root()
    excluded = [live_tree] + ([fence_root] if fence_root else [])
    if board_dir_arg:
        board_dir = Path(board_dir_arg).expanduser().resolve()
        board_dir.mkdir(parents=True, exist_ok=True)
    else:
        board_dir = Path(tempfile.mkdtemp(
            prefix='hermes-pr-acceptance-board-', dir=str(_scratch_base(excluded))))
    if any(_under(board_dir, root) for root in excluded):
        raise SystemExit(f'refusing to run the probe against a live board tree: {board_dir}')
    pinned = os.environ.get('HERMES_KANBAN_DB', '').strip()
    if pinned and (_under(board_dir, Path(pinned).parent) or _under(Path(pinned), board_dir)):
        raise SystemExit(f'refusing to run the probe against the pinned live board: {pinned}')
    os.environ['HERMES_KANBAN_HOME'] = str(board_dir)
    os.environ.pop('HERMES_KANBAN_DB', None)
    os.environ.pop('HERMES_KANBAN_BOARD', None)
    for var in ('HERMES_KANBAN_WORKSPACES_ROOT', 'HERMES_KANBAN_ATTACHMENTS_ROOT'):
        os.environ.pop(var, None)
    board_path = board_dir / 'kanban.db'
    from agent.delegation_context import kanban_path_is_fenced
    if kanban_path_is_fenced(board_path):
        raise SystemExit(
            'the delegation fence marker covers every board in this process '
            f'(including {board_path}); run the probe outside a delegated-child context'
        )
    return board_path


board_path = _prepare_board(sys.argv[3] if len(sys.argv) > 3 else None)

def create(conn, title):
    if 'completion_contract' in inspect.signature(kb.create_task).parameters:
        return kb.create_task(conn, title=title, completion_contract='acme/repo')
    tid = kb.create_task(conn, title=title)
    if 'completion_contract' not in {r[1] for r in conn.execute('PRAGMA table_info(tasks)')}:
        conn.execute('ALTER TABLE tasks ADD COLUMN completion_contract TEXT')
    conn.execute('UPDATE tasks SET completion_contract=? WHERE id=?', ('acme/repo', tid))
    conn.commit()
    return tid

fixture_root = Path(sys.argv[2]) if len(sys.argv) > 2 else root
spec = importlib.util.spec_from_file_location('http_fixture', fixture_root / 'tests/hermes_cli/test_kanban_pr_acceptance.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
patch = pytest.MonkeyPatch()
with tempfile.TemporaryDirectory(prefix='hermes-pr-live-') as home:
    setup = fixture.github.__wrapped__(Path(home), patch)
    state = next(setup)
    reports = []
    try:
        with connect_closing(db_path=board_path) as conn:
            for outcome in ('failure', 'cancelled', 'timed_out', 'success'):
                state.update(conclusion=outcome)
                tid = create(conn, 'PR acceptance live')
                accepted = kb.complete_task(conn, tid, result='done', metadata={'published_pr': 'https://github.com/acme/repo/pull/7'})
                row = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' ORDER BY id DESC", (tid,)).fetchone()
                receipt = json.loads(row[0]) if row else None
                reports.append({'case': outcome, 'accepted': accepted, 'status': kb.get_task(conn, tid).status, 'receipt': receipt})
            for outcome in (('success', 'failure') if 'completion_contract' in inspect.signature(kb.create_task).parameters else ()):
                tid = kb.create_task(conn, title='Concurrent ownership', completion_contract='acme/repo')
                owner = kb.claim_task(conn, tid)
                run_id = owner.current_run_id
                def reclaim():
                    with connect_closing(db_path=board_path) as other:
                        kb.block_task(other, tid, reason='Reassigned during acceptance')
                        kb.unblock_task(other, tid)
                        state['replacement'] = kb.claim_task(other, tid).current_run_id
                state.update(conclusion=outcome, race=reclaim)
                accepted = kb.complete_task(conn, tid, expected_run_id=run_id, result='done', metadata={'published_pr': 'https://github.com/acme/repo/pull/7'})
                reports.append({'case': 'CAS-'+outcome, 'accepted': accepted, 'run_id': kb.get_task(conn, tid).current_run_id,
                    'receipts': conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0]})
                del state['race']
        print(json.dumps({'reports': reports, 'requests': state['requests']}, indent=2))
    finally:
        try:
            next(setup)
        except StopIteration:
            pass
        patch.undo()
