from pathlib import Path

import pytest

from examples.prospectus_rfi.amos2025_matched_300s_design import (
    METHODS,
    TOTAL_TASKS,
)
from examples.prospectus_rfi.amos2025_matched_300s_mc import task_spec


ROOT = Path(__file__).parents[3] / "examples" / "prospectus_rfi"


def test_task_map_covers_four_methods_and_identical_100_seeds():
    tasks = [task_spec(task_id) for task_id in range(TOTAL_TASKS)]
    assert len(tasks) == 400
    for method in METHODS:
        assert [task.seed for task in tasks if task.method == method] == list(range(100))
    with pytest.raises(ValueError):
        task_spec(-1)
    with pytest.raises(ValueError):
        task_spec(TOTAL_TASKS)


def test_submitter_pins_the_breckenridge_policy_checksum_and_400_tasks():
    contents = (ROOT / "submit_amos2025_matched_300s.sh").read_text()
    assert "0d8033272f14cdd408192d7ab6ee819b18691c9385fca87be24044fc950464d2" in contents
    assert '0-399%${MAX_CONCURRENT}' in contents
    assert "afterok:$VALIDATION_JOB" in contents
    assert "afterok:$MC_JOB" in contents
    validation = (
        ROOT / "slurm" / "validate_amos2025_attention_control.sbatch"
    ).read_text()
    assert "--no-wheel-guard" in validation
