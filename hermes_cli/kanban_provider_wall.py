"""Live provider-wall state for the Kanban dispatcher.

The rate-limit cooldown is a LOCAL timer: it expires one hour after a worker run
died on a quota wall, whatever the provider is doing at that moment. A cooldown
that expires into a live wall re-admits the card, the worker dies on the same
2067, and the fleet burns its quota re-learning the same fact — the 2026-11-13
06:19 ET blackout did exactly that for 16 consecutive sweep fires.

This module answers the only question that matters at re-admission time with a
LIVE measurement instead of a timer: "is the provider wall up right now?" The
measurement is one minimal authenticated request against the provider's own
endpoint (one user message, one output token), classified as up / down /
unknown.

Every worker profile on this host shadows ONE MiniMax OAuth credential (the
global-root ``auth.json``, read through the profile-first / global-root fallback
in ``hermes_cli.auth``), so a single probe speaks for the whole fleet. The
reading is shared across processes through a small state file so the per-board
dispatchers do not each burn a probe for the same credential.

The probe is deliberately FAIL-OPEN: an unreadable state file, a missing
credential, a network error or an unclassifiable answer all read as "not a
measurable wall", so a broken probe can never strand the fleet. Only a positive
"up" verdict holds a card back.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

WALL_UP = "up"
WALL_DOWN = "down"
WALL_UNKNOWN = "unknown"

# A reading older than this is not evidence about "right now": the card warns
# that a wall-down measurement taken too early was true when written and false
# twelve minutes later. Keep the window short enough that a stale DOWN cannot
# re-admit a card into a wall that re-armed in the meantime.
DEFAULT_MAX_AGE_SECONDS = 30

DEFAULT_TIMEOUT_SECONDS = 20.0

# Shared state file (under the global Hermes root) so every per-board
# dispatcher shares one probe instead of one probe per board.
STATE_RELPATH = ("state", "provider_wall.json")

# Last-resort model for the probe. The probe only needs the provider to answer
# (or refuse) one token; an unknown model reads as "unknown" and fails open.
DEFAULT_PROBE_MODEL = "MiniMax-M3.1-Flash"

# The wall signature. HTTP 429 alone is a rate limit; the 2067 / token-plan
# text is the MiniMax plan wall this card is about.
_WALL_BODY_MARKERS = (
    "token plan usage limit",
    "usage limit has been reached",
    "(2067)",
    "rate_limit_error",
)


def classify_wall_response(status: Optional[int], body: str = "") -> str:
    """Classify one provider response: ``up`` / ``down`` / ``unknown``.

    ``429`` is a wall whatever the body says (a rate limit is a rate limit).
    A 2xx answer means the provider served us, so the wall is down. Anything
    else — 5xx, a timeout, a body we cannot read — is ``unknown`` and fails
    open at the call site.
    """
    text = (body or "").lower()
    if status == 429:
        return WALL_UP
    if any(marker in text for marker in _WALL_BODY_MARKERS):
        return WALL_UP
    if status is not None and 200 <= status < 300:
        return WALL_DOWN
    return WALL_UNKNOWN


def resolve_probe_model(model: Optional[str] = None) -> str:
    """The model to probe with: an explicit one, else the configured default."""
    if model:
        return str(model).strip()
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
    except Exception:
        cfg = {}
    block = cfg.get("model") if isinstance(cfg, dict) else None
    if isinstance(block, str):
        return block.strip() or DEFAULT_PROBE_MODEL
    if isinstance(block, dict):
        for key in ("default", "name", "id", "model"):
            value = block.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return DEFAULT_PROBE_MODEL


def probe_minimax_oauth_wall(
    *,
    model: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """One minimal authenticated request against MiniMax's Anthropic endpoint.

    Returns a reading dict with ``state`` (up/down/unknown), ``checked_at`` and a
    short ``detail``. Never raises: a probe failure is a reading, not a crash.
    """
    started = time.time()
    try:
        from hermes_cli.auth_minimax import resolve_minimax_oauth_runtime_credentials

        creds = resolve_minimax_oauth_runtime_credentials()
    except Exception as exc:  # no credential / refresh failed — fail open
        return {
            "state": WALL_UNKNOWN,
            "checked_at": int(started),
            "detail": f"credential_unavailable: {type(exc).__name__}: {exc}"[:300],
        }

    base = str((creds or {}).get("base_url") or "").rstrip("/")
    token = str((creds or {}).get("api_key") or "")
    probe_model = resolve_probe_model(model)
    if not base or not token or not probe_model:
        return {
            "state": WALL_UNKNOWN,
            "checked_at": int(started),
            "detail": "probe_not_configured",
        }

    payload = json.dumps(
        {
            "model": probe_model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base}/v1/messages",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            body = response.read(512).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            body = exc.read(1024).decode("utf-8", "replace")
        except Exception:
            body = ""
    except Exception as exc:
        return {
            "state": WALL_UNKNOWN,
            "checked_at": int(started),
            "detail": f"probe_error: {type(exc).__name__}: {exc}"[:300],
        }

    return {
        "state": classify_wall_response(status, body),
        "status": status,
        "detail": body[:200],
        "checked_at": int(time.time()),
        "elapsed_seconds": round(time.time() - started, 2),
        "model": probe_model,
        "provider": "minimax-oauth",
    }


def wall_state_path(root: Path | str) -> Path:
    """Where the shared reading lives (under the global Hermes root)."""
    return Path(root).joinpath(*STATE_RELPATH)


def read_wall_state(root: Path | str) -> dict[str, Any]:
    """The shared reading file (provider -> reading), or ``{}`` when missing."""
    try:
        data = json.loads(wall_state_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_wall_state(root: Path | str, reading: dict[str, Any]) -> None:
    """Publish a reading for the other dispatchers (best effort).

    The file is keyed by provider so a second provider can share it.
    """
    path = wall_state_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        provider = str(reading.get("provider") or "minimax-oauth")
        published = read_wall_state(root)
        published[provider] = reading
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(published, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def live_wall_state(
    *,
    root: Path | str,
    provider: str = "minimax-oauth",
    model: Optional[str] = None,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    now: Optional[float] = None,
) -> dict[str, Any]:
    """The wall state as of right now: a fresh reading, or a fresh-enough one.

    A cached reading younger than ``max_age_seconds`` is reused (the per-board
    dispatchers are separate processes and would otherwise each probe the same
    credential); anything older is re-probed, so a decision is never made on a
    stale measurement.
    """
    now = int(now if now is not None else time.time())
    cached = read_wall_state(root).get(provider)
    if isinstance(cached, dict):
        try:
            checked_at = int(cached.get("checked_at") or 0)
        except (TypeError, ValueError):
            checked_at = 0
        age = now - checked_at
        if 0 <= age <= max_age_seconds and cached.get("state") in (WALL_UP, WALL_DOWN):
            return dict(cached, age_seconds=age, source="cached")

    reading = probe_minimax_oauth_wall(model=model, timeout=timeout)
    reading["provider"] = provider
    if reading.get("state") in (WALL_UP, WALL_DOWN):
        write_wall_state(root, reading)
    return dict(reading, age_seconds=0, source="live")


def provider_wall_root() -> Optional[Path]:
    """The global Hermes root that owns the shared reading file.

    ``None`` when it cannot be resolved — the caller then skips the gate
    (fail open) rather than guessing a path.
    """
    try:
        from hermes_constants import get_default_hermes_root

        return Path(get_default_hermes_root())
    except Exception:
        return None
