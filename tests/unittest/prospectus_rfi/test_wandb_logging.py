from pathlib import Path

from examples.prospectus_rfi.wandb_logging import (
    DEFAULT_PROJECT,
    FINAL_GROUP,
    TUNING_GROUP,
    maybe_init_wandb,
    public_wandb_metadata,
    wandb_settings,
)


WANDB_ENVIRONMENT_KEYS = (
    "BSK_RL_USE_WANDB",
    "BSK_RL_REQUIRE_WANDB",
    "BSK_RL_WANDB_PROJECT",
    "BSK_RL_WANDB_GROUP",
    "BSK_RL_WANDB_KEY_PATH",
    "WANDB_DIR",
)


def clear_wandb_environment(monkeypatch):
    for key in WANDB_ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_default_final_run_has_dedicated_stable_namespace(tmp_path, monkeypatch):
    clear_wandb_environment(monkeypatch)
    settings = wandb_settings(
        "attention_k20_seed10001",
        tmp_path,
        tuning=False,
    )

    assert settings["project"] == DEFAULT_PROJECT
    assert settings["group"] == FINAL_GROUP
    assert settings["phase"] == "candidate_sweep"
    assert settings["run_name"].endswith("__attention_k20_seed10001")
    assert settings["run_id"].endswith("-attention-k20-seed10001")
    assert Path(settings["local_dir"]) == tmp_path / "wandb" / DEFAULT_PROJECT
    assert "key_path" not in public_wandb_metadata(settings)


def test_tuning_and_environment_overrides_are_explicit(tmp_path, monkeypatch):
    clear_wandb_environment(monkeypatch)
    key_path = tmp_path / "secret-key.txt"
    local_dir = tmp_path / "wandb-local"
    monkeypatch.setenv("BSK_RL_USE_WANDB", "0")
    monkeypatch.setenv("BSK_RL_REQUIRE_WANDB", "1")
    monkeypatch.setenv("BSK_RL_WANDB_PROJECT", "custom-project")
    monkeypatch.setenv("BSK_RL_WANDB_GROUP", "custom-group")
    monkeypatch.setenv("BSK_RL_WANDB_KEY_PATH", str(key_path))
    monkeypatch.setenv("WANDB_DIR", str(local_dir))

    settings = wandb_settings("mlp_k10_seed50000_tune00", tmp_path, tuning=True)

    assert settings["enabled"] is False
    assert settings["required"] is True
    assert settings["project"] == "custom-project"
    assert settings["group"] == "custom-group"
    assert settings["phase"] == "tuning"
    assert settings["key_path"] == str(key_path)
    assert settings["local_dir"] == str(local_dir)
    assert maybe_init_wandb(settings, {}) is None


def test_tuning_uses_separate_default_group(tmp_path, monkeypatch):
    clear_wandb_environment(monkeypatch)
    settings = wandb_settings("mlp_k10_seed50000_tune00", tmp_path, tuning=True)

    assert settings["group"] == TUNING_GROUP
