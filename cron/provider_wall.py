"""Provider-wall blindness bookkeeping for cron jobs.

A cron job whose agent turn dies on the provider's quota wall produces NO
report at all: the turn fails before the agent can say anything, so the job's
own output is an error and the monitoring lane goes quiet. On 2026-11-13 the
``blocked-kanban-monitor`` job lost 16 consecutive fires that way (06:19-09:45
ET) and the board sweep produced nothing at all — nobody could see that the
sweep had stopped running, because the sweep is what reports on the sweep.

This module makes the blackout visible. It recognises the wall signature in a
cron failure, counts the consecutive blind fires, and produces the delivery
copy that says so in plain words: "monitoring blind — provider wall".

The counting is per job id and lives in one small JSON file under the cron
directory, so the number survives the gateway restart that a wall can trigger.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

STATE_FILENAME = "provider_wall_blind.json"

# The token-plan wall this card is about: MiniMax's 2067.
_TOKEN_PLAN_WALL_RE = re.compile(
    r"(?i)(token[ -]plan usage limit|usage limit has been reached|\(\s*2067\s*\))"
)

# Any provider quota wall: 429 in the error text.
_QUOTA_WALL_RE = re.compile(
    r"(?i)(error code:\s*429|http\s*429|ratelimiterror|rate_limit_error|too many requests)"
)


def is_provider_wall_failure(text: Optional[str]) -> bool:
    """True when a cron failure text is a provider quota wall."""
    if not text:
        return False
    return bool(_TOKEN_PLAN_WALL_RE.search(text) or _QUOTA_WALL_RE.search(text))


def is_token_plan_wall(text: Optional[str]) -> bool:
    """True when the failure is specifically the token-plan wall (2067)."""
    return bool(text) and bool(_TOKEN_PLAN_WALL_RE.search(text or ""))


def blind_state_path() -> Path:
    """Where the per-job blind counters live."""
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "cron" / STATE_FILENAME


def _read_state() -> dict[str, Any]:
    try:
        data = json.loads(blind_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    jobs = data.get("jobs")
    return jobs if isinstance(jobs, dict) else {}


def _write_state(jobs: dict[str, Any]) -> None:
    path = blind_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({"jobs": jobs}, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def record_blind_fire(job_id: str, error: str = "", *, now: Optional[float] = None) -> dict[str, Any]:
    """Count one blind fire for ``job_id`` and return its counter."""
    now = int(now if now is not None else time.time())
    jobs = _read_state()
    entry = jobs.get(job_id)
    if not isinstance(entry, dict):
        entry = {}
    entry["first_seen"] = int(entry.get("first_seen") or now)
    entry["last_seen"] = now
    entry["count"] = int(entry.get("count") or 0) + 1
    entry["token_plan"] = is_token_plan_wall(error)
    if error:
        entry["last_error"] = error[:300]
    jobs[job_id] = entry
    _write_state(jobs)
    return dict(entry)


def blind_state(job_id: str) -> dict[str, Any]:
    """The blind counter for ``job_id`` (empty when the job was never blind)."""
    entry = _read_state().get(job_id)
    return dict(entry) if isinstance(entry, dict) else {}


def clear_blind_state(job_id: str) -> None:
    """Forget the blind counter for ``job_id`` (the job produced a report)."""
    jobs = _read_state()
    if job_id in jobs:
        jobs.pop(job_id, None)
        _write_state(jobs)


def _fmt_when(epoch: Optional[int]) -> str:
    if not epoch:
        return "an unknown time"
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(epoch))
    except Exception:
        return str(epoch)


def provider_wall_failure_notice(
    job_name: str, job_id: str, error: str, *, now: Optional[float] = None
) -> Optional[str]:
    """The delivery copy for a cron fire that died on the provider wall.

    ``None`` when the failure is not the token-plan wall (the caller keeps its
    ordinary failure copy). The scope is deliberately narrow: a generic
    "HTTP 429: Too Many Requests" from another provider is a transient hiccup
    with an established, correct copy, while the token-plan wall (2067) kills
    the turn outright and leaves the monitoring lane with nothing at all.
    """
    if not is_token_plan_wall(error):
        return None
    now = int(now if now is not None else time.time())
    entry = blind_state(job_id)
    count = int(entry.get("count") or 0)
    first_seen = int(entry.get("first_seen") or now)
    label = "token-plan wall (2067)" if is_token_plan_wall(error) else "provider quota wall"
    if count > 1:
        streak = (
            f"Monitoring has now been blind for {count} consecutive fires since "
            f"{_fmt_when(first_seen)}."
        )
    else:
        streak = f"Monitoring went blind at {_fmt_when(first_seen)}."
    return (
        f"⚠ Cron '{job_name}' is BLIND — provider wall ({label}). The agent turn died "
        "on the provider quota wall before it produced any report, so this run "
        "produced NO output for the operator to read. "
        f"{streak} "
        "The local rate-limit cooldown does not shorten the wall: nothing re-admits a "
        "worker until the provider actually serves again."
    )


def provider_wall_recovery_notice(job_id: str) -> Optional[str]:
    """The note that closes a blind window once the job produces a report.

    ``None`` when the job was never blind.
    """
    entry = blind_state(job_id)
    if not entry:
        return None
    count = int(entry.get("count") or 0)
    first_seen = int(entry.get("first_seen") or 0)
    clear_blind_state(job_id)
    return (
        f"ℹ Monitoring was blind for {count} consecutive fire(s) (provider wall) "
        f"starting {_fmt_when(first_seen)} — the provider is serving again, so this "
        "run's report is real."
    )
