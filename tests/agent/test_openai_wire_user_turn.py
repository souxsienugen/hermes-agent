"""The OpenAI wire must always carry a user-role message.

The Anthropic wire already enforces this invariant
(``agent/anthropic_message_convert.py::_ensure_leading_user_turn``). The
OpenAI wire had no equivalent, and strict chat templates (llama.cpp / vLLM
Jinja templates, Qwen served by Ollama) answer a transcript that contains no
user turn at all with::

    HTTP 500  {"error": {"message": "no user query found in messages"}}

That 500 is classified as a TRANSIENT server error, so the agent retries the
same poisoned request three times and then gives up. When the failing request
is the one that walked the ``fallback_providers`` chain (primary = an
account-level 429, fallback = local Ollama), the escape hatch dies with the
very failure it was installed to survive, and the worker requeues forever.

The invariant is enforced in ``ChatCompletionsTransport.build_kwargs`` — at
payload assembly — so ``convert_messages`` stays a pure sanitizer.

Run: cd /home/ai/.hermes/hermes-agent && .venv/bin/python -m pytest \
    tests/agent/test_openai_wire_user_turn.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.transports.chat_completions import (
    ChatCompletionsTransport,
    _MISSING_USER_TURN_PLACEHOLDER,
    _ensure_user_turn,
)


def _roles(messages):
    return [m.get("role") for m in messages]


class TestEnsureUserTurn:
    def test_returns_the_same_list_when_a_user_turn_is_present(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        out = _ensure_user_turn(messages)
        assert out is messages
        assert _roles(out) == ["system", "user", "assistant"]

    def test_inserts_a_user_turn_when_there_is_none(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "summary"},
            {"role": "tool", "content": "result"},
        ]
        out = _ensure_user_turn(messages)
        assert any(m.get("role") == "user" for m in out)
        # The system prefix stays FIRST so the cache prefix is untouched.
        assert out[0]["role"] == "system"
        # The original rows are preserved.
        assert len(out) == len(messages) + 1

    def test_inserts_at_the_front_when_there_is_no_system_prompt(self):
        out = _ensure_user_turn([{"role": "assistant", "content": "summary"}])
        assert out[0]["role"] == "user"
        assert out[1]["role"] == "assistant"

    def test_placeholder_is_non_whitespace(self):
        # A whitespace-only text block trades the 500 for a different 400
        # ("text content blocks must contain non-whitespace text").
        assert _MISSING_USER_TURN_PLACEHOLDER.strip()
        out = _ensure_user_turn([{"role": "assistant", "content": "x"}])
        assert out[0]["content"].strip()

    def test_does_not_mutate_the_input_list(self):
        messages = [{"role": "assistant", "content": "summary"}]
        _ensure_user_turn(messages)
        assert _roles(messages) == ["assistant"]

    def test_a_developer_system_prompt_is_also_kept_first(self):
        out = _ensure_user_turn([
            {"role": "developer", "content": "dev"},
            {"role": "assistant", "content": "x"},
        ])
        assert out[0]["role"] == "developer"
        assert any(m.get("role") == "user" for m in out)

    def test_empty_list_gets_a_user_turn(self):
        out = _ensure_user_turn([])
        assert _roles(out) == ["user"]


class TestConvertMessagesStaysPure:
    """convert_messages is a sanitizer; callers rely on it NOT inserting."""

    def test_convert_messages_does_not_insert_a_user_turn(self):
        transport = ChatCompletionsTransport()
        messages = [{"role": "system", "content": "sys"}, {"role": "assistant", "content": "summary"}]
        out = transport.convert_messages(messages, model="qwen3.8:27b", base_url="http://localhost:11434/v1")
        assert out is messages
        assert _roles(out) == ["system", "assistant"]


class TestBuildKwargsEnforcesTheInvariant:
    """build_kwargs is where the outbound payload is assembled."""

    def test_build_kwargs_guarantees_a_user_turn(self):
        transport = ChatCompletionsTransport()
        kwargs = transport.build_kwargs(
            model="qwen3.8:27b",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "assistant", "content": "summary"},
                {"role": "tool", "content": "result"},
            ],
            base_url="http://localhost:11434/v1",
        )
        assert any(m.get("role") == "user" for m in kwargs["messages"])

    def test_build_kwargs_leaves_a_normal_request_untouched(self):
        transport = ChatCompletionsTransport()
        kwargs = transport.build_kwargs(
            model="gpt-5",
            messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            base_url="https://api.openai.com/v1",
        )
        # Exactly one user turn — the existing one. (build_kwargs may rewrite
        # the system role to "developer" for some models; that is unrelated.)
        assert [m.get("role") for m in kwargs["messages"]].count("user") == 1
        assert kwargs["messages"][-1]["content"] == "hi"
