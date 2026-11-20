"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.kanban_stop import (
    build_kanban_guardrail_halt_nudge,
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    session_called_kanban_terminal,
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_STOP_NUDGE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_env_can_disable(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


def test_nudge_disabled_inside_delegated_child(clear_kanban_env):
    from agent.delegation_context import delegated_child_context

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_parent")

    assert kanban_stop_nudge_enabled() is True
    with delegated_child_context():
        assert kanban_stop_nudge_enabled() is False
        assert build_kanban_stop_nudge(messages=[]) is None
    assert kanban_stop_nudge_enabled() is True


def test_nudge_disabled_inside_non_dispatcher_context(clear_kanban_env):
    from agent.delegation_context import non_dispatcher_owned_context

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_parent")

    assert kanban_stop_nudge_enabled() is True
    with non_dispatcher_owned_context():
        assert kanban_stop_nudge_enabled() is False
        assert build_kanban_stop_nudge(messages=[]) is None
    assert kanban_stop_nudge_enabled() is True


def test_nudge_when_no_terminal_tool(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    messages = [
        {"role": "user", "content": "work kanban task"},
        {
            "role": "assistant",
            "content": "Let me write the comprehensive recipe.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_heartbeat", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_heartbeat", "tool_call_id": "1", "content": "ok"},
    ]
    nudge = build_kanban_stop_nudge(messages=messages, attempts=0)
    assert nudge is not None
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "t_46be8aa5" in nudge


def test_no_nudge_after_kanban_complete(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": "done"},
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.


@pytest.mark.parametrize(
    "tool_name,who",
    [
        ("kanban_request_review", "build worker handing off for same-card review"),
        ("kanban_request_changes", "review agent sending the card back"),
        ("kanban_schedule", "worker parking the card on a timed wait"),
    ],
)
def test_no_nudge_after_handoff_tool(clear_kanban_env, tool_name, who):
    """Handoff tools end the worker's turn just like complete/block.

    Both move the card out of ``running``, and the worker is told to call
    them — goals.py's continuation/finalize prompts name
    ``kanban_request_review``; the force-loaded sdlc-review skill names
    ``kanban_request_changes``. Nudging afterwards asks a worker that did
    the right thing to close a card it must not close.
    """
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_handoff")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": tool_name, "tool_call_id": "1", "content": "ok"},
    ]
    assert session_called_kanban_terminal(messages) is True, who
    assert build_kanban_stop_nudge(messages=messages) is None


def test_nudge_still_fires_for_non_terminal_kanban_tool(clear_kanban_env):
    """Widening the set must not swallow the case the guard exists for."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "Let me open the review next.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_comment", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_comment", "tool_call_id": "1", "content": "ok"},
    ]
    assert session_called_kanban_terminal(messages) is False
    nudge = build_kanban_stop_nudge(messages=messages)
    assert nudge is not None
    # The nudge offers every worker exit, not just close-out; a card that must go
    # through review must never be steered to ``kanban_complete`` alone.
    assert "kanban_request_review" in nudge and "kanban_block" in nudge


# ── Guardrail-halt nudge (the tool-guardrail path) ───────────────────
# A tool guardrail halt ends the turn from INSIDE the tool round, so the
# text-response stop gates above never run: without its own nudge the worker
# exits rc=0 with no terminal board call, the dispatcher books that as a
# protocol_violation, and the card respawns onto the same blocked read.
# These tests pin the nudge's contract and its no-nudge edges.


def _halt_nudge(clear_kanban_env, **kwargs):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    params = {
        "messages": [{"role": "user", "content": "work kanban task"}],
        "tool_name": "read_file",
        "code": "idempotent_no_progress_block",
        "count": 5,
    }
    params.update(kwargs)
    return build_kanban_guardrail_halt_nudge(**params)


def test_guardrail_halt_nudge_fires_for_kanban_worker(clear_kanban_env):
    """The guardrail path gets its OWN nudge: the text-response gates never run there."""
    nudge = _halt_nudge(clear_kanban_env)
    assert nudge is not None
    # It must name the card, the blocked tool and the structured decision code,
    # so the worker knows WHICH card is stuck and WHY.
    assert "t_46be8aa5" in nudge
    assert "read_file" in nudge
    assert "idempotent_no_progress_block" in nudge
    # Both ways out, and the parking route is a HUMAN-gated one: a blocked card
    # waits for a human instead of respawning onto the same read.
    assert "kanban_block" in nudge and "needs_input" in nudge
    # It must never ask for the blocked call again — that is the respawn loop.
    assert "Do NOT repeat the blocked call" in nudge


def test_guardrail_halt_nudge_absent_for_non_kanban_session(clear_kanban_env):
    """An interactive chat must still see the plain harness halt text."""
    assert build_kanban_guardrail_halt_nudge(
        messages=[], tool_name="read_file", code="idempotent_no_progress_block",
    ) is None


def test_guardrail_halt_nudge_disabled_by_env(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert build_kanban_guardrail_halt_nudge(messages=[]) is None


def test_guardrail_halt_nudge_absent_after_terminal_call(clear_kanban_env):
    """A worker that already handed the card off must not be told to park it."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_block", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_block", "tool_call_id": "1", "content": "ok"},
    ]
    assert build_kanban_guardrail_halt_nudge(messages=messages, tool_name="read_file") is None


def test_guardrail_halt_nudge_budget_is_bounded(clear_kanban_env):
    """The nudge budget is finite: a worker that ignores it is not nudged forever."""
    assert _halt_nudge(clear_kanban_env, attempts=1) is not None
    assert _halt_nudge(clear_kanban_env, attempts=2) is None
    assert _halt_nudge(clear_kanban_env, attempts=2, max_attempts=3) is not None


def test_guardrail_halt_nudge_helper_returns_nudge_for_kanban_worker(clear_kanban_env):
    """The tool-round helper wires the decision's fields into the nudge."""
    from agent.turn_tool_round import _guardrail_halt_kanban_nudge

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    agent = SimpleNamespace(_kanban_guardrail_halt_nudges=0)
    decision = SimpleNamespace(
        tool_name="read_file", code="idempotent_no_progress_block", count=5,
    )
    nudge = _guardrail_halt_kanban_nudge(agent, [{"role": "user", "content": "go"}], decision)
    assert nudge is not None
    assert "read_file" in nudge
    assert "t_46be8aa5" in nudge


def test_guardrail_halt_nudge_helper_is_none_for_non_kanban_worker(clear_kanban_env):
    """A non-kanban session gets no nudge (the plain halt text is shown instead)."""
    from agent.turn_tool_round import _guardrail_halt_kanban_nudge

    agent = SimpleNamespace(_kanban_guardrail_halt_nudges=0)
    decision = SimpleNamespace(
        tool_name="read_file", code="idempotent_no_progress_block", count=5,
    )
    assert _guardrail_halt_kanban_nudge(agent, [{"role": "user", "content": "go"}], decision) is None


def test_guardrail_halt_nudge_helper_swallows_builder_errors(clear_kanban_env, monkeypatch):
    """A broken nudge builder must never crash the tool round."""
    from agent import kanban_stop
    from agent.turn_tool_round import _guardrail_halt_kanban_nudge

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    agent = SimpleNamespace(_kanban_guardrail_halt_nudges=0)
    decision = SimpleNamespace(
        tool_name="read_file", code="idempotent_no_progress_block", count=5,
    )

    def _boom(**_kwargs):
        raise RuntimeError("nudge builder exploded")

    # The tool round imports the builder lazily, so the SOURCE module is what
    # has to be patched for the helper to see the failure.
    monkeypatch.setattr(kanban_stop, "build_kanban_guardrail_halt_nudge", _boom)
    assert _guardrail_halt_kanban_nudge(
        agent, [{"role": "user", "content": "go"}], decision
    ) is None
