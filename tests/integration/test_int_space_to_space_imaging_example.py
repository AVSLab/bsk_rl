"""Public example profiles expose stable mission configuration without a policy."""

import runpy
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = runpy.run_path(
    str(Path(__file__).parents[2] / "examples/space_to_space_rso_imaging_demo.py")
)


def test_catalog_sampling_is_independent_of_global_draw_order():
    first = EXAMPLE["make_catalog"](seed=0)
    np.random.random(100)
    assert EXAMPLE["make_catalog"](seed=0) == first


def test_unpinned_catalog_manifest_records_the_realized_epoch_and_replays():
    definition = EXAMPLE["make_catalog"](seed=0)
    assert definition.utc_init is None
    env = EXAMPLE["build_environment"](definition)
    epoch = "2026 OCT 01 00:00:00.000 (UTC)"
    env.world_args_generator["utc_init"] = lambda: epoch
    try:
        initial, _ = env.reset(seed=0)
        manifest = EXAMPLE["mission_manifest"](env)
        assert (
            manifest["catalog"]["utc_init"]
            == manifest["world_args"]["utc_init"]
            == epoch
        )
        assert definition.utc_init is None
    finally:
        env.close()
    replay = EXAMPLE["replay_environment"](manifest)
    try:
        repeated, _ = replay.reset(seed=999)
        np.testing.assert_array_equal(initial["imager"], repeated["imager"])
        assert replay.scenario.catalog.utc_init == epoch
    finally:
        replay.close()


def test_default_example_rewards_only_complete_delivery():
    env = EXAMPLE["build_environment"]()
    try:
        env.reset(seed=0)
        imager = next(sat for sat in env.satellites if sat.name == "imager")
        for _ in range(30):
            choice = 1 if imager.data_store.products else 3
            _, rewards, terminated, truncated, _ = env.step(
                {
                    sat.name: choice if sat.name == "imager" else 0
                    for sat in env.satellites
                }
            )
            if env.rewarder.data.deliveries:
                record = next(iter(env.rewarder.data.deliveries.values()))
                target = env.scenario.targets_by_id[record.target_id]
                assert rewards["imager"] == pytest.approx(target.priority)
                assert env.rewarder.last_reward_components["acquisition"]["imager"] == 0
                break
            assert rewards["imager"] == 0
            assert not all(terminated.values()) and not all(truncated.values())
        else:
            pytest.fail("Example did not deliver an image within 30 decisions.")
    finally:
        env.close()


def test_amos_profile_configuration_is_explicit():
    env = EXAMPLE["build_environment"](amos_profile=True)
    imager = next(sat for sat in env.satellites if sat.name == "imager")
    assert np.degrees(imager.sat_args_generator["oe"].i) == 45
    assert imager.action_space.n == 13
    assert [action.duration for action in imager.action_builder.action_spec[:3]] == [
        300,
        300,
        150,
    ]
    imaging = imager.action_builder.action_spec[-1]
    assert imaging.min_pointing_hold_s == 10
    assert imaging.hold_mode == "cumulative"
    assert not imaging.require_illumination_during_hold
    assert env.rewarder.example_weights == dict(
        acquisition_weight=0.9, delivery_weight=0.1
    )
    assert imager.sat_args_generator["instrumentBaudRate"] == 4e6
    try:
        env.reset(seed=0)
        assert imager.observation_space.shape == (124,)
    finally:
        env.close()


def test_replay_rejects_an_unverified_catalog():
    with pytest.raises(ValueError, match="hash"):
        EXAMPLE["replay_environment"](
            dict(schema_version=1, catalog={}, catalog_sha256="wrong")
        )


def test_manifest_does_not_attribute_an_untracked_package_to_enclosing_git_head(
    monkeypatch,
):
    import bsk_rl

    env = EXAMPLE["build_environment"]()
    try:
        env.reset(seed=0)
        # The directory is in this repository, but this package file is untracked.
        # This reproduces an installed wheel nested inside a checkout's .venv.
        monkeypatch.setattr(
            bsk_rl, "__file__", str(Path(__file__).parents[2] / "examples/__init__.py")
        )
        assert EXAMPLE["mission_manifest"](env)["bsk_rl_commit"] is None
    finally:
        env.close()
