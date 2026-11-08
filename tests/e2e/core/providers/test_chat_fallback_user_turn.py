"""Reproduction: the FALLBACK path loses the user turn, Ollama answers 500.

The upstream e2e suite (``test_chat_custom_endpoint.py``) covers the PRIMARY
chat-completions path: a ``provider: custom`` endpoint that must always carry a
user message. This file covers the path that bit the MiniMax token-plan
recurrence (t_04d737b5, 2026-11-01): the primary is a provider that 429s
(account-level quota), Hermes walks the ``fallback_providers`` chain onto a
LOCAL Ollama backend, and the fallback request carries NO user message — so
Ollama's chat renderer refuses it with HTTP 500 ``no user query found in
messages``.

The failure is not cosmetic. The 500 is classified as a transient server error,
so Hermes retries the same poisoned request three times and then gives up: the
turn dies, the sweep cron delivers no report, and the board goes unswept for
the whole window the account-level 429 is up.

Two facts make this a real defect and not a config mistake:

1. The Anthropic-wire path has an explicit guard for exactly this
   (``agent/anthropic_message_convert.py::_ensure_leading_user_turn``), so the
   invariant "the request always carries a user turn" is one the codebase
   already considers mandatory — it just is not enforced on the OpenAI-wire
   path a fallback provider uses.
2. The messages the fallback sends are the SAME list the primary already used.
   If that list carries a user turn, the fallback request carries it too. So
   the fallback request losing the user turn means the message list itself was
   rewritten somewhere between the primary's request and the fallback's.

HOW THE TRANSCRIPT LOSES ITS USER TURN
---------------------------------------
The list is rewritten by the per-turn ``ContextEngine.select_context()`` hook
(``agent/conversation_loop.py::_apply_context_engine_selection``): it may
REPLACE the request outright, and it runs before every request sanitizer, so
a replacement that carries no user-role message reaches the wire as-is. This
test installs a context engine that does exactly that, so the lossy transcript
is produced by a real product path rather than by hand-poking the transport.

The engine is deliberately hostile: it drops EVERY user row, which is the
shape Ollama refuses with HTTP 500 ``no user query found in messages``.
"""

from __future__ import annotations

import sys

import pytest

from tests.e2e.core.providers._openai_helpers import (
    READ_TOOL,
    HarnessError,
    Home,
    bug_assertions,
    chat_messages,
    db_messages,
    db_tool_calls,
    bounded_turn,
    oneshot,
)
from tests.fakes.providers.chat_variants import CText, CTools, FakeChatVariantServer

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="subprocess harness is Linux-gated")

# This test runs ONE turn, but the harness spawns a real `hermes -z` child that
# first prepares an isolated runtime for its throwaway HERMES_HOME ("preparing
# dependencies for this data root"). That setup is a FIXED cost paid before the
# turn starts, and it is environment-dependent: with no PyPI reach it falls back
# to a local sync, which on a loaded host takes tens of seconds. The 60s budget
# used by test_chat_custom_endpoint.py does not fit here — that file runs TWO
# turns, so it can only afford a tighter per-turn slice, while this one gets
# the whole allowance for a single turn. 180s leaves room for the prep plus the
# turn on a host at load ~35 without turning a slow machine into a red test.
STRICT_TURN_BUDGET = 180.0

KNOWN: dict[str, tuple[str, str]] = {
    "fallback_keeps_user_turn": (
        r"the fallback request carried no user message",
        "the fallback onto a local backend lost the user turn, so the escape hatch "
        "died with the account-level 429 it was installed to survive",
    ),
}

# A context engine that replaces the request with a transcript carrying NO
# user-role message. This is the production trigger: select_context() may
# replace the request outright, and it runs BEFORE every request sanitizer, so
# a user-turn-free replacement reaches the provider as-is.
_LOSER_ENGINE_PLUGIN = '''\
"""Test context engine: strips every user turn from the outbound request."""
from agent.context_engine import ContextEngine


class UserTurnStrippingEngine(ContextEngine):
    @property
    def name(self):
        return "user-turn-stripping"

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None):
        return messages

    def update_from_response(self, usage):
        pass

    def select_context(self, request_messages, *, conversation_messages=None,
                       incoming_message=None, budget_tokens=0):
        return [m for m in request_messages if m.get("role") != "user"]


def register(ctx):
    ctx.register_context_engine(UserTurnStrippingEngine())
'''

_ENGINE_NAME = "user-turn-stripping"


def _install_user_turn_stripping_engine(h: Home) -> None:
    """Install the lossy context engine into the child's HERMES_HOME.

    The loader resolves user context engines from ``$HERMES_HOME/plugins/<name>/``
    (``plugins/context_engine/__init__.py::find_engine_dir``), and the harness
    points HERMES_HOME at a throwaway dir, so the plugin is isolated per test.
    """
    plugin_dir = h.hermes_home / "plugins" / _ENGINE_NAME
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "__init__.py").write_text(_LOSER_ENGINE_PLUGIN, encoding="utf-8")


def _fallback_config(primary_url: str, backup_url: str) -> dict:
    """A primary that 429s on the account, with a LOCAL backend as the only escape.

    Mirrors the live shape (t_04d737b5): primary = a cloud account on a token
    plan, fallback = Ollama on localhost. The 429 is served by the fake primary.
    """
    return {
        "model": {"provider": "custom", "base_url": primary_url, "default": "primary-model",
                  "context_length": 128000},
        "providers": {
            "custom": {"base_url": primary_url, "api_key": "loopback-only",
                       "request_timeout_seconds": 60},
            "ollama": {"base_url": backup_url, "key_env": "OLLAMA_API_KEY",
                       "request_timeout_seconds": 300},
        },
        "fallback_providers": [
            {"provider": "ollama", "model": "qwen3.8:27b", "base_url": backup_url,
             "key_env": "OLLAMA_API_KEY"},
        ],
        "auxiliary": {"title_generation": {"enabled": False}},
    }


def test_fallback_request_carries_the_user_turn(tmp_path) -> None:
    """The fallback request must carry a user message, exactly like the primary's.

    The fake enforces Ollama's real rule: no user turn -> HTTP 500 ``no user
    query found in messages``.
    """
    h = Home(tmp_path)
    script = [
        CTools([(READ_TOOL, {"path": "a.txt"})]),   # tool continuation
        CText("FALLBACK-DONE"),
    ]
    overrun: HarnessError | None = None
    run = None
    with FakeChatVariantServer(script, strict_user_turn=True) as backup:
        # The primary 429s on EVERY request, so the very first attempt has to
        # walk the chain onto the backup. The backup is the strict Ollama fake.
        class _Always429:
            def __call__(self, record):
                from tests.fakes.providers.chat_variants import CError
                return CError(status=429, message="The Token Plan usage limit has been reached.",
                              code="rate_limit_error")

        with FakeChatVariantServer(_Always429(), strict_user_turn=False) as primary:
            cfg = _fallback_config(primary.base_url, backup.base_url)
            cfg["agent"] = {"auto_recovery_cycles": 0}
            # Select the lossy context engine: it strips every user turn from
            # the outbound request, which is what the OpenAI wire must survive.
            cfg["context"] = {"engine": _ENGINE_NAME}
            h.write(cfg, dotenv={"OPENAI_API_KEY": "loopback-only", "OLLAMA_API_KEY": "ollama"})
            _install_user_turn_stripping_engine(h)
            (h.project / "a.txt").write_text("CANARY-A\n", encoding="utf-8")
            try:
                run = bounded_turn(h, "read a.txt", STRICT_TURN_BUDGET)
            except HarnessError as exc:
                overrun = exc
            backup_records = backup.main_records()

    # Every request the LOCAL backend received must carry a user message.
    no_user = [
        (i, [m.get("role") for m in chat_messages(r["body"])])
        for i, r in enumerate(backup_records)
        if not chat_messages(r["body"], "user")
    ]
    with bug_assertions(KNOWN, "fallback_keeps_user_turn"):
        assert no_user == [], (
            f"the fallback request carried no user message (Ollama answers 500 "
            f"'no user query found in messages'): requests (index, roles) {no_user}"
        )
    if overrun is not None:
        raise overrun
    assert run is not None, "the turn produced no run record"
    assert run.proc.returncode == 0 and run.stdout.strip() == "FALLBACK-DONE", run.describe()
    persisted = [c["function"]["name"] for c in db_tool_calls(db_messages(h, run.session_id))]
    assert persisted == [READ_TOOL], persisted
