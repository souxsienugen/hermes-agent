"""Per-job HARD agent-turn cap for cron jobs.

A job's ``max_turns`` must TIGHTEN the fleet-wide ``agent.max_turns`` for that
job only: a noisy schedule (a 15-minute sweep) must not be able to run the
whole provider budget on every fire, and no single job may be used to raise the
budget the operator set for everyone else.

The cap is resolved in ``cron/scheduler.py::_resolve_cron_agent_setup``, the
single place every agent-mode cron run gets its iteration budget.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cron.jobs import _normalize_job_max_turns
from cron.scheduler import _resolve_cron_agent_setup


class _JC:
    """Minimal stand-in for the job-config context the scheduler resolves against."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = "test/model"


def _setup_for(job, cfg=None):
    cfg = cfg if cfg is not None else {"agent": {"max_turns": 300}}
    return _resolve_cron_agent_setup(job, "job-1", "job-name", _JC(cfg))


def test_normalizer_accepts_positive_ints_and_clears_on_empty():
    assert _normalize_job_max_turns(60) == 60
    assert _normalize_job_max_turns("60") == 60
    assert _normalize_job_max_turns(None) is None
    assert _normalize_job_max_turns("") is None
    assert _normalize_job_max_turns("unlimited") is None
    assert _normalize_job_max_turns(0) is None


def test_normalizer_rejects_garbage():
    with pytest.raises(ValueError):
        _normalize_job_max_turns("not-a-number")


def test_job_cap_tightens_the_global_budget():
    setup = _setup_for({"max_turns": 12})
    assert setup.max_iterations == 12


def test_job_without_cap_inherits_global_budget():
    setup = _setup_for({})
    assert setup.max_iterations == 300


def test_job_cap_cannot_raise_the_fleet_budget():
    """A per-job cap is a ceiling, not a raise: no single job may be used to
    lift the budget the operator set for everyone else."""
    setup = _setup_for({"max_turns": 5000})
    assert setup.max_iterations == 300
