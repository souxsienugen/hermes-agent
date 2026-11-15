"""Tests for the log-based provider-wall reader (``hermes_cli.kanban_wall_log``).

The live probe (``hermes_cli.kanban_provider_wall``) answers "is the provider
refusing calls right now". This module answers the other half: "what do the
logs say, and for WHICH profile". That answer has to survive two measurement
traps that were both hit on this host, in OPPOSITE directions:

  TRAP 2 (false POSITIVE). A bare ``2067`` token match also matches a tool
  error whose QUOTED OUTPUT contains the wall string, so a wall re-arm that
  never happened gets reported.

  TRAP 3 (false NEGATIVE). Hermes writes the log level BARE after the
  millisecond field, so a level regex that requires parentheses matches
  NOTHING and a live wall reads as down for the entire history.

A parse that can lie in both directions must be sanity-checked in both
directions, so the tests pin the probes, not just the happy path.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

import pytest

from hermes_cli import kanban_wall_log as kwl


# ---------------------------------------------------------------------------
# The two traps, as literal lines from this host's logs
# ---------------------------------------------------------------------------

# A REAL wall line (2026-11-14 23:41:08 ET, profile mark).
REAL_WALL_LINE = (
    "2026-11-14 23:41:08,359 ERROR [local-541b781727968a8d6d905301a63374550a8c26b2058"
    "a95b010d67eff014bb10d] agent.conversation_loop: API call failed after 3 retries. "
    "HTTP 429: The Token Plan usage limit has been reached. To avoid interruption, "
    "please upgrade your plan or buy credits and turn on auto consumption. (2067) | "
    "provider=minimax-oauth model=MiniMax-M3.1-Flash msgs=2 tokens=~13,393"
)

# The WARNING form of the same wall — the level is BARE, not parenthesised.
REAL_WALL_WARNING = (
    "2026-11-14 23:41:08,356 WARNING [local-541b781727968a8d6d905301a63374550a8c26b2058"
    "a95b010d67eff014bb10d] agent.conversation_loop: API call failed (attempt 3/3) "
    "error_type=RateLimitError thread=MainThread:138765607077760 provider=minimax-oauth "
    "base_url=https://api.minimax.io/anthropic model=MiniMax-M3.1-Flash summary=HTTP 429: "
    "The Token Plan usage limit has been reached. To avoid interruption, please upgrade "
    "your plan or buy credits and turn on auto consumption. (2067)"
)

# TRAP 2: a tool error whose QUOTED OUTPUT is a grep for the wall string.
FALSE_POSITIVE_TOOL_ECHO = (
    "2026-11-15 01:34:25,567 WARNING [local-abe45d3c508d9ef179dab862d5b59ff2b48570"
    "49d198d6faa03bd13fa7ca091e] agent.tool_executor: Tool terminal returned error "
    "(0.05s): {\"output\": \"=== sweep's 2067 / usage-limit grep ===\\n\\n=== and in "
    "sibling scripts ===\", \"exit_code\": 1}"
)

# TRAP 1 (the original trap): the token-count fields of a SUCCESSFUL call.
FALSE_POSITIVE_TOKEN_COUNT = (
    "2026-11-13 10:34:58,412 INFO [local-abc] agent.conversation_loop: API call #27: "
    "model=MiniMax-M3.1-Flash provider=minimax-oauth in=198810 out=7935 total=206745 "
    "latency=3.1s id=deadbeef"
)

# A served call — INFO level, no error class.
SERVED_LINE = (
    "2026-11-15 02:03:42,568 INFO [local-abc] agent.conversation_loop: API call #154: "
    "model=MiniMax-M3.1-Flash provider=minimax-oauth in=489202 out=280 total=489482 "
    "latency=5.1s id=cafe"
)


# ---------------------------------------------------------------------------
# TRAP 2 — false positives
# ---------------------------------------------------------------------------


def test_a_real_wall_line_is_a_hit():
    assert kwl.is_wall_line(REAL_WALL_LINE)
    assert kwl.is_wall_line(REAL_WALL_WARNING)


def test_a_quoted_tool_output_is_not_a_wall():
    """TRAP 2: the quoted grep output must never read as a provider refusal."""
    assert not kwl.is_wall_line(FALSE_POSITIVE_TOOL_ECHO)


def test_token_counts_of_a_successful_call_are_not_a_wall():
    """TRAP 1: ``total=206745`` / ``out=7935`` are token counts, not a 2067."""
    assert not kwl.is_wall_line(FALSE_POSITIVE_TOKEN_COUNT)


def test_a_served_call_is_not_a_wall():
    assert not kwl.is_wall_line(SERVED_LINE)


def test_a_bare_2067_token_outside_the_error_class_is_not_a_wall():
    """The provider ERROR CLASS is required — a bare token is not enough."""
    line = (
        "2026-11-15 01:34:25,567 WARNING [local-abc] agent.conversation_loop: "
        "retrying, context mentions (2067) somewhere"
    )
    assert not kwl.is_wall_line(line)


# ---------------------------------------------------------------------------
# TRAP 3 — false negatives
# ---------------------------------------------------------------------------


def test_a_bare_level_wall_line_is_a_hit():
    """TRAP 3: the level is written BARE after the millisecond field."""
    hit = kwl.parse_wall_line(REAL_WALL_WARNING)
    assert hit is not None
    assert hit.level == "WARNING"
    assert hit.timestamp == "2026-11-14 23:41:08,356"
    assert hit.module == "agent.conversation_loop"


def test_a_parenthesised_level_regex_would_match_nothing():
    """The regression that produced a false "wall down for the whole history".

    A level regex that requires parentheses matches NOTHING on a real Hermes
    line. The corrected parse must match it. This test fails if the parser
    ever goes back to requiring parentheses.
    """
    import re

    paren_requiring = re.compile(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*\((ERROR|WARNING)\)"
    )
    assert paren_requiring.search(REAL_WALL_LINE) is None
    assert paren_requiring.search(REAL_WALL_WARNING) is None
    assert kwl.is_wall_line(REAL_WALL_LINE)
    assert kwl.is_wall_line(REAL_WALL_WARNING)


def test_an_info_line_is_never_a_wall_hit():
    """A wall refusal is logged at ERROR or WARNING, never INFO."""
    info_wall = (
        "2026-11-14 23:41:08,340 INFO [local-abc] agent.conversation_loop: "
        "HTTP 429: The Token Plan usage limit has been reached. (2067)"
    )
    assert not kwl.is_wall_line(info_wall)


# ---------------------------------------------------------------------------
# The bidirectional sanity check
# ---------------------------------------------------------------------------


def test_the_sanity_check_passes_on_the_shipped_parser():
    result = kwl.sanity_check()
    assert result["ok"]
    assert result["matches_known_wall_line"]
    assert result["rejects_known_false_positive"]


def test_the_sanity_check_fails_when_the_parser_goes_blind(monkeypatch):
    """A parser with a false negative must be reported, not trusted.

    This is the whole point of the check: "zero hits" is only evidence of a
    down wall AFTER the parse has been shown to still match a line you KNOW is
    a wall hit.
    """
    monkeypatch.setattr(kwl, "_PROVIDER_ERROR_CLASS_RE", re.compile(r"(?!x)x"))
    result = kwl.sanity_check()
    assert not result["ok"]
    assert not result["matches_known_wall_line"]
    assert "FALSE NEGATIVE" in result["detail"]


def test_the_sanity_check_fails_when_the_parser_starts_matching_everything(monkeypatch):
    """A parser with a false positive must be reported, not trusted."""
    monkeypatch.setattr(kwl, "_PROVIDER_ERROR_CLASS_RE", re.compile(r".*"))
    result = kwl.sanity_check()
    assert not result["ok"]
    assert not result["rejects_known_false_positive"]


def test_the_sanity_check_fails_when_the_module_exclusion_stops_working(monkeypatch):
    """The quoting-module guard is a third, independent guard."""
    monkeypatch.setattr(kwl, "_QUOTING_MODULES", ())
    result = kwl.sanity_check()
    assert not result["ok"]
    assert not result["rejects_quoted_tool_output"]
    assert "quoted-grep" in result["detail"]


def test_the_sanity_check_rejects_the_quoted_tool_output_on_its_merits():
    assert kwl.parse_wall_line(kwl.SANITY_PROBE_QUOTED_OUTPUT) is None
    assert kwl.parse_wall_line(kwl.SANITY_PROBE_REJECT) is None
    assert kwl.parse_wall_line(kwl.SANITY_PROBE_HIT) is not None


# ---------------------------------------------------------------------------
# Served calls
# ---------------------------------------------------------------------------


def test_a_served_call_is_recognised():
    assert kwl.is_served_call_line(SERVED_LINE)


def test_a_failed_call_is_not_a_served_call():
    assert not kwl.is_served_call_line(REAL_WALL_LINE)
    assert not kwl.is_served_call_line(
        "2026-11-14 23:41:08,356 WARNING [local-abc] agent.conversation_loop: "
        "API call failed (attempt 1/3) error_type=RateLimitError"
    )


# ---------------------------------------------------------------------------
# Profile state
# ---------------------------------------------------------------------------


def _write_log(directory, name, lines):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _recent(line, now, *, minutes_ago=2):
    """The same line, dated ``minutes_ago`` before ``now``.

    The verdict is freshness-based, so a fixture pinned to a fixed date reads
    as stale and the test would depend on the wall clock. Every fixture that
    asserts a VERDICT uses this instead of the literal capture, and passes the
    SAME ``now`` to the verdict so the comparison is clock-consistent.

    Two things this avoids:

    - ``strftime`` re-localises a naive datetime against the process TZ, so
      the stamp must be built from the datetime's own fields.
    - the suite pins ``TZ=UTC`` and a later ``tzset()`` can move
      ``datetime.now()`` by hours mid-test, so the reference instant is
      captured ONCE and reused.
    """
    moment = now - timedelta(minutes=minutes_ago)
    stamp = (
        f"{moment.year:04d}-{moment.month:02d}-{moment.day:02d} "
        f"{moment.hour:02d}:{moment.minute:02d}:{moment.second:02d},"
        f"{moment.microsecond // 1000:03d}"
    )
    return re.sub(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+", stamp, line)


def test_a_profile_with_a_wall_hit_and_no_later_served_call_reads_up(tmp_path):
    now = datetime.now()
    log_dir = tmp_path / "logs"
    _write_log(log_dir, "errors.log", [_recent(REAL_WALL_LINE, now)])
    state = kwl.read_profile_state("mark", log_dir)
    assert state.wall_hit_count == 1
    assert state.last_wall_hit is not None
    assert state.verdict(now=now) == kwl.WALL_UP


def test_a_served_call_after_the_wall_hit_proves_recovery(tmp_path):
    now = datetime.now()
    log_dir = tmp_path / "logs"
    _write_log(log_dir, "errors.log", [_recent(REAL_WALL_LINE, now)])
    _write_log(log_dir, "agent.log", [_recent(SERVED_LINE, now, minutes_ago=1)])
    state = kwl.read_profile_state("mark", log_dir)
    assert state.verdict(now=now) == kwl.WALL_DOWN


def test_a_stale_wall_hit_reads_unknown_not_up(tmp_path):
    """A profile that died on the wall days ago is not evidence about now."""
    log_dir = tmp_path / "logs"
    stale = REAL_WALL_LINE.replace("2026-11-14 23:41:08", "2026-11-01 23:41:08")
    _write_log(log_dir, "errors.log", [stale])
    state = kwl.read_profile_state("mark", log_dir)
    assert state.wall_hit_count == 1
    assert state.verdict(now=datetime(2026, 11, 14, 12, 0, 0)) == kwl.WALL_UNKNOWN
    assert state.verdict(now=datetime(2026, 11, 1, 23, 45, 0)) == kwl.WALL_UP


def test_a_windowed_reading_is_judged_against_the_window_end(tmp_path):
    """A historical burst must not read as "unknown" just because it is past.

    Judged against the wall clock, every hit in a window from yesterday is
    stale, so the window would always answer "unknown" — which says nothing
    about the window. The reference is the end of the window instead.
    """
    log_dir = tmp_path / "logs"
    _write_log(log_dir, "errors.log", [REAL_WALL_LINE])
    state = kwl.read_profile_state("mark", log_dir)
    window_end = datetime(2026, 11, 14, 23, 42, 0)
    assert state.verdict(now=datetime(2026, 11, 15, 2, 0, 0)) == kwl.WALL_UNKNOWN
    assert state.verdict(now=datetime(2026, 11, 15, 2, 0, 0), reference=window_end) == kwl.WALL_UP


def test_a_windowed_host_reading_is_judged_against_the_window_end(tmp_path):
    root = tmp_path / ".hermes"
    _write_log(root / "profiles" / "mark" / "logs", "errors.log", [REAL_WALL_LINE])
    state = kwl.host_wall_state(
        root,
        since=datetime(2026, 11, 14, 23, 28, 0),
        until=datetime(2026, 11, 14, 23, 42, 0),
        now=datetime(2026, 11, 15, 2, 0, 0),
    )
    assert state["state"] == kwl.WALL_UP


def test_the_rotated_log_is_read(tmp_path):
    """``errors.log`` rotates at ~97 KB, so a window can live in ``.1``."""
    log_dir = tmp_path / "logs"
    _write_log(log_dir, "errors.log", ["2026-11-15 01:00:00,000 INFO something"])
    _write_log(log_dir, "errors.log.1", [REAL_WALL_LINE])
    state = kwl.read_profile_state("mark", log_dir)
    assert state.wall_hit_count == 1
    assert state.last_wall_hit is not None


def test_the_window_filters_hits(tmp_path):
    log_dir = tmp_path / "logs"
    _write_log(log_dir, "errors.log", [REAL_WALL_LINE])
    since = datetime(2026, 11, 15)
    state = kwl.read_profile_state("mark", log_dir, since=since)
    assert state.wall_hit_count == 0


def test_log_coverage_is_reported(tmp_path):
    """Coverage separates "the provider refused" from "the logs do not reach back"."""
    log_dir = tmp_path / "logs"
    _write_log(log_dir, "errors.log", ["2026-11-11 06:00:00,000 INFO first"])
    _write_log(log_dir, "agent.log", ["2026-11-15 02:00:00,000 INFO last"])
    state = kwl.read_profile_state("mark", log_dir)
    assert state.coverage_start == "2026-11-11 06:00:00,000"
    assert state.coverage_end == "2026-11-15 02:00:00,000"


# ---------------------------------------------------------------------------
# The root profile is included — the premise this card was built on
# ---------------------------------------------------------------------------


def test_the_root_profile_is_scanned_alongside_the_worker_profiles(tmp_path):
    """The 2026-11-14 reading was "root is served, workers are walled".

    That was a measurement artifact: the root's own refusals live in its
    ``errors.log`` (and the rotated ``errors.log.1``), while ``agent.log``
    records served calls only. A reader that skipped the root could never
    see the blackout as host-wide.
    """
    now = datetime.now()
    root = tmp_path / ".hermes"
    _write_log(root / "logs", "errors.log", [_recent(REAL_WALL_LINE, now)])
    _write_log(root / "profiles" / "mark" / "logs", "errors.log", [_recent(REAL_WALL_LINE, now)])

    state = kwl.host_wall_state(root, now=now)
    names = {profile["profile"] for profile in state["profiles"]}
    assert "default" in names
    assert "mark" in names
    assert state["state"] == kwl.WALL_UP


def test_the_host_verdict_is_up_when_one_profile_is_walled(tmp_path):
    now = datetime.now()
    root = tmp_path / ".hermes"
    _write_log(root / "profiles" / "mark" / "logs", "errors.log", [_recent(REAL_WALL_LINE, now)])
    state = kwl.host_wall_state(root, now=now)
    assert state["state"] == kwl.WALL_UP


def test_the_host_verdict_is_down_when_a_later_served_call_proves_recovery(tmp_path):
    now = datetime.now()
    root = tmp_path / ".hermes"
    _write_log(root / "profiles" / "mark" / "logs", "errors.log", [_recent(REAL_WALL_LINE, now)])
    _write_log(root / "profiles" / "mark" / "logs", "agent.log",
               [_recent(SERVED_LINE, now, minutes_ago=1)])
    state = kwl.host_wall_state(root, now=now)
    assert state["state"] == kwl.WALL_DOWN


def test_a_broken_sanity_check_makes_the_host_verdict_unknown(tmp_path, monkeypatch):
    """A broken parse must never produce a confident "down"."""
    import re

    root = tmp_path / ".hermes"
    _write_log(root / "profiles" / "mark" / "logs", "agent.log", [SERVED_LINE])
    monkeypatch.setattr(kwl, "_PROVIDER_ERROR_CLASS_RE", re.compile(r"(?!x)x"))
    state = kwl.host_wall_state(root)
    assert state["state"] == kwl.WALL_UNKNOWN
    assert not state["sanity"]["ok"]


# ---------------------------------------------------------------------------
# Credential identity (AC#4)
# ---------------------------------------------------------------------------


def _write_auth(root, profile, pool_entry):
    path = root / "auth.json" if profile == "default" else (
        root / "profiles" / profile / "auth.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pool_entry), encoding="utf-8")


def test_profiles_sharing_one_credential_report_a_shared_identity(tmp_path):
    root = tmp_path / ".hermes"
    entry = {
        "credential_pool": {
            "minimax-oauth": [
                {"id": "abc123", "access_token": "secret-token", "refresh_token": "secret-refresh"}
            ]
        }
    }
    _write_auth(root, "mark", entry)
    _write_auth(root, "tom", entry)
    identity = kwl.credential_identity(root)
    assert identity["distinct_access_tokens"] == 1
    assert identity["shared_identity"]


def test_profiles_on_different_credentials_report_distinct_identities(tmp_path):
    root = tmp_path / ".hermes"
    _write_auth(root, "mark", {"credential_pool": {"minimax-oauth": [
        {"id": "a", "access_token": "token-a", "refresh_token": "refresh-a"}]}})
    _write_auth(root, "tom", {"credential_pool": {"minimax-oauth": [
        {"id": "b", "access_token": "token-b", "refresh_token": "refresh-b"}]}})
    identity = kwl.credential_identity(root)
    assert identity["distinct_access_tokens"] == 2
    assert not identity["shared_identity"]


def test_the_credential_identity_never_exposes_the_token(tmp_path):
    root = tmp_path / ".hermes"
    _write_auth(root, "mark", {"credential_pool": {"minimax-oauth": [
        {"id": "a", "access_token": "super-secret-value", "refresh_token": "secret-refresh"}]}})
    identity = kwl.credential_identity(root)
    blob = json.dumps(identity)
    assert "super-secret-value" not in blob
    assert "secret-refresh" not in blob


def test_the_fingerprint_is_stable_and_distinguishing():
    assert kwl._fingerprint("abc") == kwl._fingerprint("abc")
    assert kwl._fingerprint("abc") != kwl._fingerprint("abd")
    assert kwl._fingerprint("") is None


# ---------------------------------------------------------------------------
# Window parsing
# ---------------------------------------------------------------------------


def test_parse_when_reads_an_absolute_time():
    assert kwl.parse_when("2026-11-14 23:41:08") == datetime(2026, 11, 14, 23, 41, 8)


def test_parse_when_reads_minutes_ago():
    parsed = kwl.parse_when("90m")
    assert parsed is not None
    assert datetime.now() - parsed >= timedelta(minutes=89)


def test_parse_when_rejects_nonsense():
    assert kwl.parse_when("not a time") is None
    assert kwl.parse_when("") is None
    assert kwl.parse_when(None) is None
