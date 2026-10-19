"""Failed alpha=.1 jobs must never silently become accepted sweep controls."""
from unittest.mock import patch

import pytest

from examples.multiagent_imaging.cluster.alpha_sweep import campaign


@pytest.mark.parametrize("mode", ["submit", "run"])
def test_retired_campaign_fails_before_scheduler_or_process_execution(mode):
    with patch.object(campaign.subprocess, "run") as scheduler, patch.object(
        campaign.os, "execvpe"
    ) as execution:
        with pytest.raises(RuntimeError, match="Retired alpha sweep"):
            if mode == "submit":
                campaign.submit({})
            else:
                campaign.run({}, 0)
        scheduler.assert_not_called()
        execution.assert_not_called()


def test_historical_plan_is_explicitly_not_launchable():
    plan = dict(alpha_values=[0.1], new_allocations=16,
                wall_seconds_per_task=86340, concurrency=2,
                maximum_new_allocated_cpu_hours=21489.6)
    result = campaign.plan_text(plan)
    assert result["campaign_status"] == "retired"
    assert result["submission_enabled"] is False
    assert result["existing_pair_passed_terminal_acceptance"] is False
