"""Tests for the provider-wall blindness bookkeeping (``cron.provider_wall``).

A cron job whose agent turn dies on the provider quota wall produces NO report:
the turn fails before the agent can say anything, so the monitoring lane goes
blind without a word. The 2026-11-13 ``blocked-kanban-monitor`` blackout lost 16
consecutive fires that way and the board sweep said nothing about it.

These tests pin the two halves of the fix: the wall signature is recognised and
counted, and the delivery copy says "BLIND — provider wall" in plain words.
"""

from __future__ import annotations

import pytest

from cron import provider_wall as cpw


WALL_ERROR = (
    "RuntimeError: HTTP 429: The Token Plan usage limit has been reached. To avoid "
    "interruption, please upgrade your plan or buy credits and turn on auto "
    "consumption. (2067)"
)


@pytest.fixture
def cron_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------


def test_the_token_plan_wall_is_recognised():
    assert cpw.is_provider_wall_failure(WALL_ERROR) is True
    assert cpw.is_token_plan_wall(WALL_ERROR) is True


def test_a_plain_429_is_also_a_provider_wall():
    assert cpw.is_provider_wall_failure("RuntimeError: HTTP 429: Too Many Requests") is True
    assert cpw.is_token_plan_wall("RuntimeError: HTTP 429: Too Many Requests") is False


def test_an_ordinary_failure_is_not_a_wall():
    assert cpw.is_provider_wall_failure("RuntimeError: script timed out after 300s") is False
    assert cpw.is_provider_wall_failure("") is False
    assert cpw.is_provider_wall_failure(None) is False


# ---------------------------------------------------------------------------
# Counting the blind fires
# ---------------------------------------------------------------------------


def test_consecutive_blind_fires_are_counted(cron_home):
    cpw.record_blind_fire("job1", WALL_ERROR)
    cpw.record_blind_fire("job1", WALL_ERROR)
    entry = cpw.blind_state("job1")
    assert entry["count"] == 2
    assert entry["first_seen"] <= entry["last_seen"]


def test_the_counter_is_per_job(cron_home):
    cpw.record_blind_fire("job1", WALL_ERROR)
    assert cpw.blind_state("job2") == {}


def test_a_successful_run_clears_the_counter(cron_home):
    cpw.record_blind_fire("job1", WALL_ERROR)
    cpw.clear_blind_state("job1")
    assert cpw.blind_state("job1") == {}


# ---------------------------------------------------------------------------
# Delivery copy
# ---------------------------------------------------------------------------


def test_the_delivery_copy_says_the_lane_is_blind(cron_home):
    cpw.record_blind_fire("job1", WALL_ERROR)
    notice = cpw.provider_wall_failure_notice("blocked-kanban-monitor", "job1", WALL_ERROR)
    assert notice is not None
    assert "BLIND" in notice
    assert "provider wall" in notice
    assert "blocked-kanban-monitor" in notice


def test_the_delivery_copy_counts_the_blackout(cron_home):
    cpw.record_blind_fire("job1", WALL_ERROR)
    cpw.record_blind_fire("job1", WALL_ERROR)
    cpw.record_blind_fire("job1", WALL_ERROR)
    notice = cpw.provider_wall_failure_notice("blocked-kanban-monitor", "job1", WALL_ERROR)
    assert notice is not None
    assert "3 consecutive fires" in notice


def test_a_non_wall_failure_has_no_wall_notice(cron_home):
    assert cpw.provider_wall_failure_notice("job", "job1", "script timed out") is None


def test_the_recovery_notice_closes_the_blind_window(cron_home):
    cpw.record_blind_fire("job1", WALL_ERROR)
    cpw.record_blind_fire("job1", WALL_ERROR)
    notice = cpw.provider_wall_recovery_notice("job1")
    assert notice is not None
    assert "blind for 2 consecutive fire" in notice
    # The counter is consumed: the next report must not repeat it.
    assert cpw.provider_wall_recovery_notice("job1") is None


def test_a_job_that_was_never_blind_has_no_recovery_notice(cron_home):
    assert cpw.provider_wall_recovery_notice("job1") is None


# ---------------------------------------------------------------------------
# The scheduler's failure copy uses it
# ---------------------------------------------------------------------------


def test_the_scheduler_copy_prefers_the_blind_wording(cron_home):
    from cron.scheduler import _summarize_cron_failure_for_delivery

    cpw.record_blind_fire("job1", WALL_ERROR)
    job = {"id": "job1", "name": "blocked-kanban-monitor", "no_agent": False}
    notice = _summarize_cron_failure_for_delivery(job, WALL_ERROR)
    assert "BLIND" in notice
    assert "provider wall" in notice


def test_a_no_agent_job_is_never_blamed_for_a_provider_wall(cron_home):
    """A script job's own "429" wording is not a provider wall."""
    from cron.scheduler import _summarize_cron_failure_for_delivery

    job = {"id": "job1", "name": "some-script", "no_agent": True}
    notice = _summarize_cron_failure_for_delivery(job, "HTTP 429 in the script output")
    assert "BLIND" not in notice


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
