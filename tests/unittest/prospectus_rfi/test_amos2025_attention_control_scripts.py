from pathlib import Path


ROOT = Path(__file__).parents[3] / "examples" / "prospectus_rfi"


def test_attention_control_scripts_pass_the_fixed_candidate_count():
    for name in (
        "stress_amos2025_attention_control.sbatch",
        "train_amos2025_attention_control_segment.sbatch",
    ):
        contents = (ROOT / "slurm" / name).read_text()
        assert "--candidate-count 10" in contents
