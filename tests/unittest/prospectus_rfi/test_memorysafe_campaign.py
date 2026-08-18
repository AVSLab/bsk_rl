from pathlib import Path

from examples.prospectus_rfi.config import load_study_config
from examples.prospectus_rfi.environment import make_environment_args
from examples.prospectus_rfi.train import build_ppo_config


REPOSITORY_ROOT = Path(__file__).parents[3]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "examples" / "prospectus_rfi"
CONFIG_ROOT = EXPERIMENT_ROOT / "configs"


def test_training_runner_physically_instantiates_at_most_200_targets():
    study = load_study_config(
        CONFIG_ROOT / "mlp_selected.yaml",
        CONFIG_ROOT / "base_memorysafe_100_200.yaml",
    )

    env_args = make_environment_args(study.environment)

    assert len(env_args["satellites"]) == 201  # one inspector plus 200 targets
    assert env_args["scenario"].catalog_min == 100
    assert env_args["scenario"].catalog_max == 200


def test_ppo_updates_only_the_inspector_policy(tmp_path):
    study = load_study_config(
        CONFIG_ROOT / "mlp_selected.yaml",
        CONFIG_ROOT / "base_memorysafe_100_200.yaml",
    )

    config = build_ppo_config(study, seed=10_001, n_env_runners=12, temp_dir=tmp_path)

    assert set(config.policies_to_train) == {"inspector"}
    assert config.num_env_runners == 12


def test_memorysafe_slurm_request_and_namespace_are_explicit():
    script = (
        EXPERIMENT_ROOT / "slurm" / "train_candidate_sweep_memorysafe_segment.sbatch"
    ).read_text()

    assert "#SBATCH --mem=230G" in script
    assert "#SBATCH --cpus-per-task=16" in script
    assert "--n-env-runners 12" in script
    assert "base_memorysafe_100_200.yaml" in script
    assert "rfi-alpha0-100s-n100-200-memorysafe-v2" in script
    assert "--signal=B:TERM@1800" in script
    assert "--n-env-runners 28" not in script


def test_stress_gate_requires_one_complete_iteration_before_scheduler_signal():
    script = (
        EXPERIMENT_ROOT / "slurm" / "stress_candidate_sweep_memorysafe_2h.sbatch"
    ).read_text()

    assert "#SBATCH --time=04:00:00" in script
    assert "#SBATCH --signal=B:TERM@900" in script
    assert "--max-iterations 1" in script
    assert "--wall-hours 1.5" not in script
