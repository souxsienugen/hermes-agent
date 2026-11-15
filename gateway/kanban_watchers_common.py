"""Plumbing shared by the kanban notifier and dispatcher loops.

Thread offload, board enumeration, live-config coercers and the singleton
dispatcher lock live here so the notifier, dispatcher and mixin modules read
them from one place.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextvars import Context
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only; the runtime import is deferred
    from hermes_cli.kanban_db_dispatch import CapacityCaps

# Keep the logger name run.py used so extracted log records are unchanged.
logger = logging.getLogger("gateway.run")

# The capacity caps that bound how much work may run at once, in the order the
# dispatcher reports them.
_CAP_KEYS = ("max_in_progress", "max_in_progress_per_profile", "max_spawn")


def _run_in_fresh_context(func: Callable[..., Any], /, *args: Any) -> Any:
    """Run *func* in an empty ``Context`` so request-local ContextVars stay behind.

    ``asyncio.to_thread`` copies the caller's context; a lingering
    ``delegate_task`` child marker would make ``write_txn`` false-trip for
    these process-owned writers. An empty Context keeps the DB guard intact
    for real children without exempting dispatcher writes.
    """
    return Context().run(func, *args)


async def _to_thread_process_service(func: Callable[..., Any], /, *args: Any) -> Any:
    """Offload blocking process-service work without inheriting request ContextVars."""
    return await asyncio.to_thread(_run_in_fresh_context, func, *args)


def _list_boards(kb: Any) -> list:
    """Enumerate live boards; fall back to the default board when listing fails."""
    try:
        return kb.list_boards(include_archived=False)
    except Exception:
        return [kb.read_board_metadata(kb.DEFAULT_BOARD)]


def _board_slugs(kb: Any) -> list:
    return [b.get("slug") or kb.DEFAULT_BOARD for b in _list_boards(kb)]


def _positive_int_setting(kanban_cfg: dict, key: str) -> Optional[int]:
    """Parse an optional ``kanban.<key>`` int cap; None when unset or invalid (< 1 is invalid)."""
    raw = kanban_cfg.get(key)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("kanban dispatcher: invalid kanban.%s=%r; ignoring", key, raw)
        return None
    if value < 1:
        logger.warning("kanban dispatcher: kanban.%s=%r is below 1; ignoring", key, raw)
        return None
    logger.info("kanban dispatcher: %s=%d", key, value)
    return value


def _resolve_live_capacity_caps(
    load_config: Callable[[], Any],
    current: CapacityCaps,
) -> CapacityCaps:
    """Live capacity caps, re-read every dispatcher tick.

    The embedded dispatcher used to freeze ``kanban.max_in_progress`` (and the
    per-profile / per-board caps) at BOOT: ``_resolve_dispatcher_settings`` ran
    once and its frozen copy was passed to ``dispatch_once`` on every tick. An
    operator edit to the concurrency ceiling therefore only bound after a gateway
    restart, while the CLI dispatch path re-read the same keys on every
    invocation. The two dispatchers disagreed about the host-wide budget and the
    embedded one -- the only path that admits without an operator in the loop --
    ran above the operator's ceiling (#38534647).

    Re-read live so a ceiling change binds on the next tick, exactly as it does
    for the CLI path. Both paths resolve through
    ``kanban_db_dispatch.resolve_capacity_caps``, so they cannot disagree about
    what the cap IS.

    ``current`` is the caps in force; it is returned unchanged when the config
    cannot be read. Fails SAFE: dropping the caps would mean "uncapped", which
    is the opposite of what a ceiling must do when the config is unreadable.
    """
    from hermes_cli.config_read_errors import FailedConfigRead
    from hermes_cli.kanban_db_dispatch import resolve_capacity_caps

    try:
        cfg = load_config()
    except Exception:
        logger.warning(
            "kanban dispatcher: cannot re-read the capacity caps; keeping the "
            "caps in force (max_in_progress=%s)",
            current.max_in_progress,
        )
        return current
    if not isinstance(cfg, dict) or isinstance(cfg, FailedConfigRead):
        # A degraded read (config.yaml unparseable) serves defaults or
        # last-known-good under the FailedConfigRead marker. Honouring it could
        # widen the ceiling (defaults carry no kanban.max_in_progress), so keep
        # the caps in force until the file is fixed.
        logger.warning(
            "kanban dispatcher: the config read was degraded; keeping the caps in "
            "force (max_in_progress=%s)",
            current.max_in_progress,
        )
        return current
    kanban_cfg = cfg.get("kanban", {})
    if not isinstance(kanban_cfg, dict):
        kanban_cfg = {}
    caps = resolve_capacity_caps(kanban_cfg)
    if caps != current:
        for key in _CAP_KEYS:
            new = getattr(caps, key)
            old = getattr(current, key)
            if new != old:
                logger.info(
                    "kanban dispatcher: %s=%s (re-bound live; was %s)", key, new, old
                )
    return caps


def _capacity_caps_from_settings(settings: Any) -> CapacityCaps:
    """The boot-snapshot caps, as the shared ``CapacityCaps`` value."""
    from hermes_cli.kanban_db_dispatch import CapacityCaps

    return CapacityCaps(
        max_in_progress=settings.max_in_progress,
        max_in_progress_per_profile=settings.max_in_progress_per_profile,
        max_spawn=settings.max_spawn,
    )


def _resolve_auto_decompose_settings(load_config: Callable[[], Any]) -> tuple[bool, int]:
    """Live (enabled, per_tick) auto-decompose settings, re-read every dispatcher tick.

    Fails safe: a config read error returns ``(False, 3)`` rather than
    re-enabling a feature the user turned off. ``per_tick`` is clamped to ``>= 1``.

    Read fresh from config on every dispatcher tick (#49638) so that flipping ``kanban.auto_decompose:
    false`` to STOP runaway fan-out takes effect on the next tick instead of requiring a gateway restart.
    Auto-decompose is a safety toggle — a user who sees it create and launch tasks they didn't intend
    reaches for this flag to halt it, and a stale boot-captured value silently ignoring that change is the
    bug reported in #49638.
    """
    try:
        cfg = load_config()
    except Exception:
        return False, 3
    kcfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    try:
        per_tick = int(kcfg.get("auto_decompose_per_tick", 3) or 3)
    except (TypeError, ValueError):
        per_tick = 3
    return bool(kcfg.get("auto_decompose", True)), max(per_tick, 1)


def _gc_retention_days() -> int:
    """``kanban.done_sub_retention_days`` (default 30; 0 disables), re-read per sweep; fails safe to 30."""
    try:
        from hermes_cli.config import load_config

        return int(((load_config() or {}).get("kanban") or {}).get("done_sub_retention_days", 30))
    except Exception:
        return 30


def _kanban_dispatch_allowed() -> bool:
    """False while the global emergency stop (`hermes pause`) is engaged.

    Checked every tick before spawning, so a pause applies on the next tick;
    in-flight workers are never touched. Fails open if estop is unimportable.
    """
    try:
        from agent.estop import check_paused
    except ImportError:
        return True
    return not check_paused("kanban", logger)


def _acquire_singleton_lock(lock_path) -> tuple[Optional[object], str]:
    """Take the exclusive, non-blocking advisory lock for the sole dispatcher.

    Only one gateway machine-wide may run the embedded dispatcher: concurrent
    dispatchers double reclaim frequency and claim events, and with
    ``wal_autocheckpoint=0`` concurrent manual checkpoints can corrupt index
    pages. ``dispatch_in_gateway`` is the primary control; this is the backstop.

    Returns ``(handle, "held")`` (release via :func:`_release_singleton_lock`),
    ``(None, "contended")`` when another process holds it (caller must NOT
    dispatch), or ``(None, "unavailable")`` when locking cannot be performed
    (caller falls back to config control).
    """
    try:
        from gateway.status import _try_acquire_file_lock  # deferred; same package
    except ImportError:
        return None, "unavailable"
    try:
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        handle = open(str(lock_path), "a+", encoding="utf-8")  # windows-footgun: ok (append-mode lock handle, write not read)
    except OSError:
        return None, "unavailable"
    if not _try_acquire_file_lock(handle):
        handle.close()
        return None, "contended"
    return handle, "held"


def _release_singleton_lock(handle) -> None:
    """Release a lock acquired via :func:`_acquire_singleton_lock`."""
    if handle is None:
        return
    with contextlib.suppress(Exception):
        from gateway.status import _release_file_lock

        _release_file_lock(handle)
    with contextlib.suppress(Exception):
        handle.close()
