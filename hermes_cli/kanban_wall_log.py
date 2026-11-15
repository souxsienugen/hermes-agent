"""Provider-wall state read from the provider LOGS, per profile.

``hermes_cli.kanban_provider_wall`` answers "is the wall up RIGHT NOW?" with a live
probe of the provider. This module answers the other half of the question a provider
tracker has to answer: "what does the log say, and for WHICH profile?" — the wall
state re-verified from timestamped logs instead of from card text.

Two hand-rolled grep recipes were measured wrong on this host, in OPPOSITE
directions, and both had to be corrected by hand before a tracker could trust them:

TRAP 2 (false POSITIVE). A bare ``2067`` / ``429`` / ``usage limit`` token match
also matches a tool error whose QUOTED OUTPUT happens to contain the search string:
``agent.tool_executor: Tool terminal returned error (0.05s): {"output": "=== sweep
scripts mentioning 2067 / provider wall ===\\n..."}`` is an exit-code-1 grep, not a
provider 429. Measured 2026-11-15 01:15 ET on profile ``mark``: three such lines
would have fabricated a 00:35 wall hit while the wall had been down since 23:41:08.

TRAP 3 (false NEGATIVE). Hermes writes the log level BARE after the millisecond
field — ``2026-11-14 23:41:08,340 ERROR agent.chat_completion_helpers: ...`` — so a
level regex that requires parentheses around the level matches NOTHING. Measured
2026-11-15 01:31 ET: the paren-requiring parse reported "0 hits" over 5,453 real
wall lines.

A parse that can lie in both directions must be sanity-checked in both directions.
This module therefore ships TWO probes: :data:`SANITY_PROBE_HIT` must MATCH and
:data:`SANITY_PROBE_REJECT` must NOT. :func:`sanity_check` runs them, and the
host-level verdict is ``unknown`` — never ``down`` — when they disagree, so a
broken parser can never report a healthy host.

Readings are per profile AND host-wide, and they include the ROOT profile: the
2026-11-14 sweep concluded that "the root/monitor profile is served while the
worker profiles are walled", which was a measurement artifact — the root's own
429s live in its ``errors.log`` (and in the ROTATED ``errors.log.1``), while
``agent.log`` records served calls only. Comparing the root against the worker
profiles side by side is the only way to answer whether the blackout is
profile-scoped or host-wide, and that comparison is what this module exists to make.

Rotation matters as much as the parse: ``errors.log`` rotates at ~97 KB, so a
same-day wall window can sit entirely in ``errors.log.1``. Every scan here walks
the rotated files too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

WALL_UP = "up"
WALL_DOWN = "down"
WALL_UNKNOWN = "unknown"

# TRAP 3 (false negative). Hermes writes the log level BARE after the
# millisecond field:
#   2026-11-14 23:41:08,340 ERROR agent.chat_completion_helpers: ...
# A pattern written as "^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*\((ERROR|WARNING)\)"
# matches NOTHING on those lines and reports the wall DOWN for the entire history.
_ANCHORED_LEVEL_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\s+(?P<level>ERROR|WARNING)\b"
)

# TRAP 2 (false positive). The provider ERROR CLASS, not a bare token. A bare
# "2067" also matches the token-count fields of successful calls and the quoted
# output of a tool that grepped for the wall string.
_PROVIDER_ERROR_CLASS_RE = re.compile(
    r"(?i)(error_type\s*=\s*RateLimitError"
    r"|\brate_limit_error\b"
    r"|Error code:\s*429"
    r"|HTTP\s*429"
    r"|Token Plan usage limit"
    r"|usage limit has been reached)"
)

# Modules that can QUOTE the wall text without being the provider refusing a
# call: a tool error echoes the output of whatever the agent ran, and a session
# banner can echo the card title.
_QUOTING_MODULES = ("agent.tool_executor", "agent.turn_context", "tools.approval")

# The module field of an anchored log line: "<ts> <LEVEL> [<session>] <module>:".
_MODULE_RE = re.compile(
    r"^(?:[^\s]+\s+){2}(?:ERROR|WARNING)\s+(?:\[[^\]]*\]\s+)?"
    r"(?P<module>[A-Za-z_][A-Za-z0-9_.]*):"
)

# A SERVED provider call. The "API call #N:" shape is what separates a served
# call from a failed one ("API call failed (attempt 1/3)"), and from the ",429"
# millisecond field of an unrelated line.
_SERVED_CALL_RE = re.compile(r"agent\.conversation_loop: API call #\d+:")

# The timestamp of ANY log line, whatever its level. Served calls are INFO, so
# the ERROR/WARNING anchor above does not match them; this one does.
_TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\b")

_TS_FORMAT = "%Y-%m-%d %H:%M:%S,%f"

# --- The bidirectional sanity probes ---------------------------------------
#
# SANITY_PROBE_HIT is a real captured 2067 line (the anchor of the 2026-11-14
# 23:41:08 ET wall on profile ``mark``). If the parser stops matching it, the
# parser has a false negative and NO "zero hits" result may be read as "the wall
# is down".
#
# SANITY_PROBE_REJECT is a real captured false positive: an ``agent.tool_executor``
# error whose quoted output is a grep for the wall string. If the parser starts
# matching it, the parser has a false positive and the count is inflated.
SANITY_PROBE_HIT = (
    "2026-11-14 23:41:08,359 ERROR [local-541b781727968a8d6d905301a63374550a8c26b2058"
    "a95b010d67eff014bb10d] agent.conversation_loop: API call failed after 3 retries. "
    "HTTP 429: The Token Plan usage limit has been reached. To avoid interruption, "
    "please upgrade your plan or buy credits and turn on auto consumption. (2067) | "
    "provider=minimax-oauth model=MiniMax-M3.1-Flash msgs=2 tokens=~13,393"
)

# SANITY_PROBE_REJECT is a generic ERROR from a normal module — a failure that
# is NOT a provider quota refusal. If the parser starts matching it, the
# provider-error-class requirement has stopped doing its work and the hit count
# is inflated with unrelated failures.
SANITY_PROBE_REJECT = (
    "2026-11-14 23:41:08,340 ERROR [local-541b781727968a8d6d905301a63374550a8c26b2058"
    "a95b010d67eff014bb10d] agent.conversation_loop: API call failed: connection reset "
    "by peer after 3 retries. | provider=minimax-oauth model=MiniMax-M3.1-Flash"
)

# The TRAP 2 probe: a tool error whose QUOTED OUTPUT contains the provider's
# own refusal text — a worker that grepped the logs and whose tool error then
# echoed the wall line back. The provider-error class IS present in the quoted
# text, so ONLY the module exclusion rejects it. This is the real captured line
# from 2026-11-13 20:48:29 ET on profile ``mark``.
SANITY_PROBE_QUOTED_OUTPUT = (
    "2026-11-13 20:48:29,415 WARNING [local-ae1efd6729aa9532aa8703d725168f4ea6b00cd8a"
    "d39ede69d32c2a759471021c58] agent.tool_executor: Tool terminal returned error "
    "(0.07s): {\"output\": \"total lines: 235\\n222:Provider said: HTTP 429: The Token "
    "Plan usage limit has been reached. To avoid interruption, please upgrade your plan "
    "or buy credits and turn on auto consumption. (20"
)


@dataclass(frozen=True)
class WallHit:
    """One anchored provider-wall line."""

    timestamp: str
    level: str
    module: str
    source: str
    detail: str

    @property
    def moment(self) -> Optional[datetime]:
        return parse_timestamp(self.timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "module": self.module,
            "source": self.source,
            "detail": self.detail,
        }


@dataclass
class ProfileWallState:
    """The wall evidence for one profile (or for the host root)."""

    name: str
    log_dir: str
    last_wall_hit: Optional[WallHit] = None
    wall_hit_count: int = 0
    last_served_call: Optional[str] = None
    served_call_count: int = 0
    wall_hits: list[WallHit] = field(default_factory=list)
    served_calls: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # The time span the scanned log files actually cover. Without this, "zero
    # served calls" is ambiguous: it may mean "the provider refused" or "the
    # rotated logs do not reach back that far". Both have been mistaken for
    # each other on this host.
    coverage_start: Optional[str] = None
    coverage_end: Optional[str] = None

    def verdict(self, *, now: Optional[datetime] = None, max_age_seconds: int = 3600) -> str:
        """``up`` / ``down`` / ``unknown`` for THIS profile.

        A wall hit is only evidence that the wall is up NOW while it is still
        fresh. An ancient hit — a profile that died on the wall a week ago and
        has not run since — says nothing about the present, so it reads
        ``unknown`` rather than ``up``.
        """
        if self.last_wall_hit is None and self.last_served_call is None:
            return WALL_UNKNOWN
        served = parse_timestamp(self.last_served_call) if self.last_served_call else None
        walled = self.last_wall_hit.moment if self.last_wall_hit else None
        if self.last_wall_hit is None:
            return WALL_DOWN
        if served is not None and walled is not None and served > walled:
            # A served call AFTER the last refusal proves the provider recovered.
            return WALL_DOWN
        if walled is None:
            return WALL_UNKNOWN
        if now is None:
            now = datetime.now()
        if (now - walled).total_seconds() <= max_age_seconds:
            return WALL_UP
        return WALL_UNKNOWN

    def to_dict(self, *, now: Optional[datetime] = None, max_age_seconds: int = 3600) -> dict[str, Any]:
        return {
            "profile": self.name,
            "log_dir": self.log_dir,
            "verdict": self.verdict(now=now, max_age_seconds=max_age_seconds),
            "last_wall_hit": self.last_wall_hit.to_dict() if self.last_wall_hit else None,
            "wall_hit_count": self.wall_hit_count,
            "last_served_call": self.last_served_call,
            "served_call_count": self.served_call_count,
            "log_coverage": {"start": self.coverage_start, "end": self.coverage_end},
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_timestamp(text: str) -> Optional[datetime]:
    """Parse a ``YYYY-MM-DD HH:MM:SS,mmm`` log timestamp."""
    if not text:
        return None
    try:
        return datetime.strptime(text[:23], _TS_FORMAT)
    except ValueError:
        return None


def parse_wall_line(line: str, *, source: str = "") -> Optional[WallHit]:
    """One anchored log line -> a :class:`WallHit`, or None.

    Three conditions, all required:

    1. the line is an ERROR/WARNING anchored at the START of the line with a
       BARE level (TRAP 3 — a parenthesised level matches nothing at all);
    2. the line names the provider ERROR CLASS (TRAP 2 — a bare token is not a
       class, and matches token counts and quoted tool output);
    3. the line's module is not one that merely quotes other text.
    """
    if not line:
        return None
    anchored = _ANCHORED_LEVEL_RE.match(line)
    if anchored is None:
        return None
    if _PROVIDER_ERROR_CLASS_RE.search(line) is None:
        return None
    module_match = _MODULE_RE.match(line)
    module = module_match.group("module") if module_match else ""
    if module in _QUOTING_MODULES:
        return None
    return WallHit(
        timestamp=anchored.group("ts"),
        level=anchored.group("level"),
        module=module,
        source=source,
        detail=line.strip()[:400],
    )


def is_wall_line(line: str) -> bool:
    """True when ``line`` is an anchored provider-wall line."""
    return parse_wall_line(line) is not None


def is_served_call_line(line: str) -> bool:
    """True when ``line`` records a SERVED provider call."""
    if "API call #" not in line:
        return False
    return _SERVED_CALL_RE.search(line) is not None


def sanity_check() -> dict[str, Any]:
    """Run the bidirectional probes.

    ``ok`` is True only when the parser still MATCHES the known wall line AND
    still REJECTS the known false positive. A caller that reports "the wall is
    down" on zero hits MUST consult this first: a parser with a false negative
    reads a wall-down host as wall-down for the entire history.

    Three probes, because the parser has three independent guards: the level
    anchor, the provider-error-class requirement, and the module exclusion.
    """
    hit = parse_wall_line(SANITY_PROBE_HIT, source="<sanity-probe>")
    rejected = parse_wall_line(SANITY_PROBE_REJECT, source="<sanity-probe>")
    quoted = parse_wall_line(SANITY_PROBE_QUOTED_OUTPUT, source="<sanity-probe>")
    matched = hit is not None
    rejected_correctly = rejected is None
    quoted_rejected_correctly = quoted is None
    ok = matched and rejected_correctly and quoted_rejected_correctly
    if ok:
        detail = (
            "parser matches the known 2067 line and rejects both known "
            "false positives"
        )
    elif not matched:
        detail = (
            "parser does NOT match the known 2067 line — a zero-hit result is a "
            "FALSE NEGATIVE and must not be read as 'the wall is down'"
        )
    elif not rejected_correctly:
        detail = (
            "parser matches a non-quota provider failure — hits are inflated by "
            "errors that are not quota refusals"
        )
    else:
        detail = (
            "parser matches the known quoted-grep line — hits are inflated by "
            "tool output that echoes the wall string"
        )
    return {
        "ok": ok,
        "matches_known_wall_line": matched,
        "rejects_known_false_positive": rejected_correctly,
        "rejects_quoted_tool_output": quoted_rejected_correctly,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Log scanning
# ---------------------------------------------------------------------------


def log_files(log_dir: Path, stem: str) -> list[Path]:
    """Every log file for ``stem`` under ``log_dir``, NEWEST first.

    Rotations are included: ``errors.log`` rotates at ~97 KB, so a same-day wall
    window can live entirely in ``errors.log.1``.
    """
    if not log_dir.is_dir():
        return []
    found = [
        path
        for path in log_dir.glob(f"{stem}*")
        if path.is_file() and (path.name == stem or path.name.startswith(stem + "."))
    ]
    return sorted(found, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)


def scan_wall_log(path: Path) -> list[WallHit]:
    """Every anchored wall hit in one log file, oldest first."""
    hits: list[WallHit] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                hit = parse_wall_line(line, source=str(path))
                if hit is not None:
                    hits.append(hit)
    except OSError:
        return hits
    return hits


def scan_served_log(path: Path) -> list[str]:
    """Every served ``API call #`` timestamp in one log file, oldest first."""
    served: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not is_served_call_line(line):
                    continue
                head = _TS_RE.match(line)
                if head is not None:
                    served.append(head.group("ts"))
    except OSError:
        return served
    return served


def scan_log_coverage(path: Path) -> tuple[Optional[str], Optional[str]]:
    """The earliest and latest timestamp in one log file, or (None, None).

    Coverage is what separates "the provider refused" from "the rotated logs
    do not reach back that far" — the ambiguity that produced the 2026-11-14
    "the root profile is served while the workers are walled" reading.
    """
    first: Optional[str] = None
    last: Optional[str] = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                head = _TS_RE.match(line)
                if head is None:
                    continue
                stamp = head.group("ts")
                if first is None:
                    first = stamp
                last = stamp
    except OSError:
        return None, None
    return first, last


def profile_log_dir(root: Path, profile: str) -> Path:
    """Where a profile's logs live: the root profile logs in ``<root>/logs``."""
    if profile == "default":
        return root / "logs"
    return root / "profiles" / profile / "logs"


def read_profile_state(
    profile: str,
    log_dir: Path,
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    count: bool = True,
) -> ProfileWallState:
    """Read one profile's wall evidence, walking the rotated logs.

    ``since`` / ``until`` bound the WINDOW (both wall hits and served calls), so
    a question like "what did the root do during the 18:34-18:44 burst" is
    answered from the same window on every profile.
    """
    state = ProfileWallState(name=profile, log_dir=str(log_dir))

    coverage_starts: list[str] = []
    coverage_ends: list[str] = []

    for path in log_files(log_dir, "errors.log"):
        start, end = scan_log_coverage(path)
        if start:
            coverage_starts.append(start)
        if end:
            coverage_ends.append(end)
        hits = scan_wall_log(path)
        for hit in hits:
            moment = hit.moment
            if moment is None:
                continue
            if since is not None and moment < since:
                continue
            if until is not None and moment > until:
                continue
            state.wall_hits.append(hit)
        if not count and state.wall_hits:
            # A last-hit-only scan can stop at the newest file that has one.
            break

    if state.wall_hits:
        # The timestamp format is fixed-width, so a lexicographic sort is
        # chronological. An unparseable timestamp sorts FIRST so it can never
        # become ``last_wall_hit``: the newest PARSEABLE hit is the one that
        # says anything about the wall.
        state.wall_hits.sort(key=lambda hit: (hit.moment is not None, hit.timestamp))
        state.last_wall_hit = state.wall_hits[-1]
    state.wall_hit_count = len(state.wall_hits)

    for path in log_files(log_dir, "agent.log"):
        start, end = scan_log_coverage(path)
        if start:
            coverage_starts.append(start)
        if end:
            coverage_ends.append(end)
        served = scan_served_log(path)
        for stamp in served:
            moment = parse_timestamp(stamp)
            if moment is None:
                continue
            if since is not None and moment < since:
                continue
            if until is not None and moment > until:
                continue
            state.served_calls.append(stamp)
        if not count and state.served_calls:
            break

    if state.served_calls:
        state.served_calls.sort()
        state.last_served_call = state.served_calls[-1]
    state.served_call_count = len(state.served_calls)

    if coverage_starts:
        state.coverage_start = min(coverage_starts)
    if coverage_ends:
        state.coverage_end = max(coverage_ends)

    if not state.wall_hits and not state.served_calls:
        state.notes.append("no anchored wall hits and no served calls in the scanned window")
    return state


def list_profiles(root: Path) -> list[str]:
    """Every profile whose log directory exists, ``default`` first."""
    names = ["default"]
    profiles_root = root / "profiles"
    if profiles_root.is_dir():
        for entry in sorted(profiles_root.iterdir()):
            if entry.is_dir() and (entry / "logs").is_dir():
                names.append(entry.name)
    return names


def list_profiles_with_auth(root: Path) -> list[str]:
    """Every profile that has an ``auth.json``, ``default`` first.

    The credential comparison walks a different set than the log scan: a
    profile can hold a credential without ever having written a log.
    """
    names = ["default"]
    profiles_root = root / "profiles"
    if profiles_root.is_dir():
        for entry in sorted(profiles_root.iterdir()):
            if entry.is_dir() and (entry / "auth.json").is_file():
                names.append(entry.name)
    return names


def host_wall_state(
    root: Optional[Path] = None,
    *,
    profiles: Optional[Iterable[str]] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    count: bool = True,
    max_age_seconds: int = 3600,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """The wall state of every profile, side by side, plus the host verdict.

    The side-by-side comparison is the point: it is the only way to tell a
    host-wide blackout from a profile-scoped one. A single profile's "served"
    reading is not evidence about the others.

    The host verdict is ``up`` only when a profile's wall hit is FRESH (inside
    ``max_age_seconds``) and no later served call contradicts it. A profile
    that died on the wall days ago and has not run since does not make the host
    "up" — it reads ``unknown``.
    """
    if root is None:
        from hermes_constants import get_default_hermes_root

        root = get_default_hermes_root()

    if now is None:
        now = datetime.now()

    wanted = list(profiles) if profiles is not None else list_profiles(root)
    states: list[ProfileWallState] = []
    for name in wanted:
        log_dir = profile_log_dir(root, name)
        if not log_dir.is_dir():
            continue
        states.append(
            read_profile_state(name, log_dir, since=since, until=until, count=count)
        )

    sanity = sanity_check()
    verdicts = {state.verdict(now=now, max_age_seconds=max_age_seconds) for state in states}
    if not sanity["ok"]:
        verdict = WALL_UNKNOWN
    elif WALL_UP in verdicts:
        verdict = WALL_UP
    elif WALL_DOWN in verdicts:
        verdict = WALL_DOWN
    else:
        verdict = WALL_UNKNOWN

    return {
        "root": str(root),
        "state": verdict,
        "sanity": sanity,
        "max_age_seconds": max_age_seconds,
        "window": {
            "since": since.isoformat() if since else None,
            "until": until.isoformat() if until else None,
        },
        "profiles": [
            state.to_dict(now=now, max_age_seconds=max_age_seconds) for state in states
        ],
    }


def parse_when(value: Optional[str]) -> Optional[datetime]:
    """Parse a window bound: ``YYYY-MM-DD HH:MM:SS`` or minutes-ago (``90m``)."""
    if not value:
        return None
    text = value.strip()
    if text.endswith("m") and text[:-1].isdigit():
        from datetime import timedelta

        return datetime.now() - timedelta(minutes=int(text[:-1]))
    if text.endswith("h") and text[:-1].isdigit():
        from datetime import timedelta

        return datetime.now() - timedelta(hours=int(text[:-1]))
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Credential identity — AC#4: is one profile on a DIFFERENT allowance?
# ---------------------------------------------------------------------------


def _fingerprint(secret: str) -> Optional[str]:
    """A short, non-reversible fingerprint of a credential.

    The point is to COMPARE credentials across profiles without ever putting a
    token in a log, a comment, or a card. A 12-hex-char digest of the secret
    answers "same identity or different identity?" and nothing else.
    """
    if not secret or not isinstance(secret, str):
        return None
    import hashlib

    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12]


def _credential_secrets(root: Path, profile: str) -> list[dict[str, Optional[str]]]:
    """The MiniMax OAuth credentials one profile can use, as fingerprints."""
    import json as _json

    path = root / "auth.json" if profile == "default" else root / "profiles" / profile / "auth.json"
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []

    found: list[dict[str, Optional[str]]] = []
    provider = (data.get("providers") or {}).get("minimax-oauth")
    if isinstance(provider, dict) and provider.get("access_token"):
        found.append(
            {
                "source": "providers.minimax-oauth",
                "access_token": _fingerprint(provider["access_token"]),
                "refresh_token": _fingerprint(provider.get("refresh_token", "")),
            }
        )
    pool = (data.get("credential_pool") or {}).get("minimax-oauth")
    if isinstance(pool, list):
        for entry in pool:
            if not isinstance(entry, dict) or not entry.get("access_token"):
                continue
            found.append(
                {
                    "source": f"credential_pool.minimax-oauth[{entry.get('id', '?')}]",
                    "access_token": _fingerprint(entry["access_token"]),
                    "refresh_token": _fingerprint(entry.get("refresh_token", "")),
                }
            )
    return found


def credential_identity(root: Optional[Path] = None) -> dict[str, Any]:
    """Compare the MiniMax OAuth identity each profile uses.

    This answers AC#4 mechanically: "are the worker profiles on a DIFFERENT
    exhausted allowance than the root?" The answer on this host is NO — every
    profile shadows ONE shared OAuth credential, so there is no separate
    allowance to point the workers at, and no fallback provider to absorb a
    2067.

    Only fingerprints are returned. No token, refresh token, or secret ever
    leaves this function.
    """
    if root is None:
        from hermes_constants import get_default_hermes_root

        root = get_default_hermes_root()

    by_profile: dict[str, list[dict[str, Optional[str]]]] = {}
    for profile in list_profiles_with_auth(root):
        secrets = _credential_secrets(root, profile)
        if secrets:
            by_profile[profile] = secrets

    distinct: set[str] = set()
    for secrets in by_profile.values():
        for entry in secrets:
            if entry.get("access_token"):
                distinct.add(entry["access_token"])

    return {
        "root": str(root),
        "distinct_access_tokens": len(distinct),
        "shared_identity": len(distinct) <= 1,
        "profiles": {
            profile: secrets for profile, secrets in sorted(by_profile.items())
        },
        "note": (
            "fingerprints only — no credential material is exposed"
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_wall_log(args) -> int:
    """``hermes kanban wall-log`` — the wall state, measured from the logs.

    One command replaces the hand-rolled grep that this card's measurement
    recipe kept getting wrong in both directions. It reads the rotated logs,
    includes the ROOT profile, and refuses to report "down" unless its own
    parse has been sanity-checked in both directions.
    """
    import sys

    from hermes_cli.kanban_output import _print_json

    since = parse_when(getattr(args, "since", None))
    until = parse_when(getattr(args, "until", None))
    if getattr(args, "since", None) and since is None:
        print("kanban wall-log: --since is not a time this parser understands "
              "(use 'YYYY-MM-DD HH:MM:SS' or minutes-ago like '90m')", file=sys.stderr)
        return 2
    if getattr(args, "until", None) and until is None:
        print("kanban wall-log: --until is not a time this parser understands "
              "(use 'YYYY-MM-DD HH:MM:SS')", file=sys.stderr)
        return 2

    profiles = getattr(args, "profile", None) or None
    state = host_wall_state(
        profiles=profiles,
        since=since,
        until=until,
        count=not getattr(args, "no_count", False),
        max_age_seconds=int(getattr(args, "max_age", 3600)),
    )
    identity = credential_identity()

    if getattr(args, "json", False):
        payload = dict(state)
        payload["credential_identity"] = identity
        _print_json(payload)
        return 0

    sanity = state["sanity"]
    print(f"Provider wall: {state['state'].upper()} (host-wide)")
    print(f"sanity check: {'ok' if sanity['ok'] else 'FAILED'} — {sanity['detail']}")
    if not sanity["ok"]:
        print("  A 'down' reading is NOT trustworthy while the sanity check fails.",
              file=sys.stderr)
    print()
    header = f"{'profile':14s} {'verdict':9s} {'wall hits':>10s}  {'last wall hit':26s} {'last served call':26s}"
    print(header)
    for profile in state["profiles"]:
        hit = profile["last_wall_hit"]
        print(
            f"{profile['profile']:14s} {profile['verdict']:9s} {profile['wall_hit_count']:10d}  "
            f"{(hit['timestamp'] if hit else '-'):26s} {(profile['last_served_call'] or '-'):26s}"
        )
    print()
    print(f"Credential identity: {identity['distinct_access_tokens']} distinct MiniMax OAuth "
          f"access token(s) across {len(identity['profiles'])} profile(s) — "
          f"shared identity: {'yes' if identity['shared_identity'] else 'NO'}")
    if getattr(args, "credentials", False):
        print()
        for profile, entries in identity["profiles"].items():
            for entry in entries:
                print(f"  {profile:14s} {entry['source']:38s} access={entry['access_token']} "
                      f"refresh={entry['refresh_token']}")
    return 0
