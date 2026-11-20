"""Turn-end guard for kanban workers, which must end with a terminal board tool that hands
the card to whoever owns it next (``kanban_complete``, ``kanban_block``,
``kanban_request_review``, ``kanban_request_changes``). Some models narrate the next step
and stop with no tool calls; Hermes treats that as a clean exit → ``rc=0`` → dispatcher
``protocol_violation``. Policy-only: return a bounded synthetic nudge so the loop continues
instead of exiting.

The same exit-without-a-handoff happens when a TOOL GUARDRAIL halts the turn (the
identical-result guard on a read-only call). That path ends the turn from inside the tool
round, so it never reaches the text-response stop gates below, and the worker exits with
the harness's "I stopped retrying ..." question and no terminal board call — the dispatcher
books a ``protocol_violation`` and respawns it onto the same read, forever. This module
supplies the guardrail-specific nudge for that path so the worker is told to either read a
DIFFERENT path/offset or park the card with ``kanban_block(kind=needs_input)``.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

from agent.delegation_context import owned_kanban_task


# Every tool that ends this worker's responsibility for the card, not just the two that
# close it out: ``kanban_request_review`` moves it to ``review`` (goals.py's continuation /
# finalize prompts tell builders to call it) and ``kanban_request_changes`` returns it to
# ``ready`` (the sdlc-review skill tells reviewers to). Nudging after either asks a worker
# that did the right thing to ``kanban_complete`` a card it must not close.
_TERMINAL_KANBAN_TOOLS = frozenset({
    "kanban_complete",
    "kanban_block",
    "kanban_schedule",
    "kanban_request_review",
    "kanban_request_changes",
})

_DEFAULT_MAX_ATTEMPTS = 2


def kanban_stop_nudge_enabled() -> bool:
    """On when ``HERMES_KANBAN_TASK`` is set for the dispatcher-owned worker, unless
    ``HERMES_KANBAN_STOP_NUDGE`` disables it. In-process delegate_task children and cron runs
    inherit the env var but own no board task and carry no kanban toolset."""
    if (os.environ.get("HERMES_KANBAN_STOP_NUDGE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(owned_kanban_task())


def _tool_call_name(tc: Any) -> str:
    """Tool name from a dict or object tool call (``function.name`` first, then ``name``)."""
    if isinstance(tc, dict):
        fn = tc.get("function")
        return str((fn.get("name") if isinstance(fn, dict) else tc.get("name")) or "")
    fn = getattr(tc, "function", None)
    return str((getattr(fn, "name", "") if fn is not None else getattr(tc, "name", "")) or "")


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True if this conversation already invoked a terminal kanban tool."""
    for msg in filter(lambda m: isinstance(m, dict), messages or ()):
        role = msg.get("role")
        if role == "assistant" and any(
            _tool_call_name(tc) in _TERMINAL_KANBAN_TOOLS for tc in msg.get("tool_calls") or []
        ):
            return True
        if role == "tool" and str(msg.get("name") or "") in _TERMINAL_KANBAN_TOOLS:
            return True
    return False


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a terminal tool; ``None`` when
    the guard should not fire (not a kanban worker, already completed/blocked, budget exhausted)."""
    if (
        not kanban_stop_nudge_enabled()
        or attempts >= max_attempts
        or session_called_kanban_terminal(messages)
    ):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    # The transcript is the status source: this text is only reached when the session made no
    # handoff call, so it never tells a worker to close a card it already sent to review.
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` has not been handed off: this session made no terminal board "
        "call (`kanban_complete` / `kanban_request_review` / `kanban_block`). Ending now "
        "causes a protocol violation (clean exit with the card still `running`).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work is done "
        "and needs no review, `kanban_request_review(summary=...)` if it is a code "
        "change that needs same-card review, OR `kanban_block(reason=...)` if you are "
        "blocked. Reviewers approve with `kanban_complete` or send the card back with "
        "`kanban_request_changes(reason=...)`.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


def build_kanban_guardrail_halt_nudge(
    *,
    messages: Iterable[dict] | None = None,
    tool_name: str = "",
    code: str = "",
    count: int = 0,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Synthetic follow-up for a kanban worker whose turn was ended by a TOOL GUARDRAIL.

    A guardrail halt (the identical-result guard on a read-only call) ends the turn from
    inside the tool round, so the text-response stop gates never run and the worker has
    no terminal board call when it exits. The dispatcher books that as a
    ``protocol_violation`` and respawns the card onto the same read — the respawn loop
    this card tracks.

    The nudge gives the worker the two ways out the guard itself names: read a DIFFERENT
    path/offset, or park the card for a human with ``kanban_block(kind=needs_input)``
    naming the blocked path. It never asks the worker to repeat the blocked call.

    ``None`` when the guard should not fire: not a kanban worker, a terminal board call
    already made, or the nudge budget spent.
    """
    if (
        not kanban_stop_nudge_enabled()
        or attempts >= max_attempts
        or session_called_kanban_terminal(messages)
    ):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    tool = (tool_name or "").strip() or "that tool"
    what = f"`{tool}`" if tool_name else "the tool"
    detail = f" after {count} identical calls" if count else ""
    code_txt = f" (`{code}`)" if code else ""
    return (
        "[System: You are a Hermes kanban worker. A guardrail halt is NOT a terminal "
        "state for the board.\n\n"
        f"Task `{tid}` is still `running`: the tool guardrail stopped {what}{detail}"
        f"{code_txt} because it kept returning the same result. Ending the turn now "
        "causes a protocol violation (clean exit with the card still `running`), and the "
        "card will be respawned onto the same blocked call.\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        f"1. Do NOT repeat the blocked call. Read a DIFFERENT path or offset (the "
        f"result you already have is enough for the blocked {what}); or\n"
        "2. If you genuinely cannot proceed without that exact read, park the card for "
        "a human with `kanban_block(kind=needs_input, reason=...)` naming the blocked "
        "path and what you needed from it. A blocked card waits for a human — it does "
        "not respawn.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


__all__ = [
    "build_kanban_guardrail_halt_nudge",
    "build_kanban_stop_nudge",
    "kanban_stop_nudge_enabled",
    "session_called_kanban_terminal",
]
