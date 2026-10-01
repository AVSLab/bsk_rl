"""Catalog replay and explicit spacecraft identity contracts."""

from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from bsk_rl.scene import RSOPriorityEvent, RSOTarget, RSOTargetCatalog, RSOTargets


def catalog():
    return RSOTargetCatalog(
        (RSOTarget("id-83", "debris", (7e6, 0, 0), (0, 7500, 0)),),
        "2026 OCT 01 00:00:00.000 (UTC)",
        (RSOPriorityEvent(0, (("id-83", 3),)), RSOPriorityEvent(10, (("id-83", 7),))),
    )


def test_catalog_round_trip_does_not_consume_rng(tmp_path):
    original = catalog()
    original.save(tmp_path / "catalog.json")
    np.random.seed(52)
    expected = np.random.random()
    np.random.seed(52)
    restored = RSOTargetCatalog.load(tmp_path / "catalog.json")
    assert restored == original
    assert np.random.random() == expected
    assert restored.targets[0].id == "id-83"


def test_legacy_catalog_binding_key_is_read_without_changing_identity():
    original = catalog()
    values = original.to_dict()
    values["targets"][0]["spacecraft_name"] = values["targets"][0].pop("satellite_name")
    assert RSOTargetCatalog.from_dict(values) == original
    values["targets"][0]["satellite_name"] = "different"
    with pytest.raises(ValueError, match="Conflicting"):
        RSOTargetCatalog.from_dict(values)


@pytest.mark.parametrize("omit_epoch_key", [False, True])
def test_catalog_epoch_can_be_omitted_from_constructor_and_json(omit_epoch_key):
    original = RSOTargetCatalog(catalog().targets)
    assert original.utc_init is None
    values = original.to_dict()
    if omit_epoch_key:
        values.pop("utc_init")
    assert RSOTargetCatalog.from_dict(values) == original


@pytest.mark.parametrize("epoch", ["", 83, False])
def test_explicit_catalog_epoch_must_be_a_nonempty_string(epoch):
    with pytest.raises(ValueError, match="epoch"):
        RSOTargetCatalog(catalog().targets, utc_init=epoch)


@pytest.mark.parametrize(
    "order", [("other", "debris", "camera"), ("camera", "debris", "other")]
)
@pytest.mark.parametrize("inherit_epoch", [False, True])
def test_explicit_binding_and_repeat_reset(order, inherit_epoch):
    definition = replace(catalog(), utc_init=None) if inherit_epoch else catalog()
    scene = RSOTargets(definition, ["camera"])
    from bsk_rl.sim.dyn.base import EclipseDynModel

    class FakeDyn(EclipseDynModel):
        def bind_rso_target(self, target):
            pass

    class FakeFSW:
        def action_image_rso(self, target):
            pass

    satellites = [
        NS(
            name=name,
            sat_args={},
            dyn_type=FakeDyn,
            fsw_type=FakeFSW,
        )
        for name in order
    ]
    scene.link_satellites(satellites)
    scene.utc_init = catalog().utc_init
    scene.reset_pre_sim_init()
    first_episode = scene.catalog
    assert first_episode.utc_init == scene.utc_init
    assert first_episode.targets == definition.targets
    assert definition.utc_init == (None if inherit_epoch else catalog().utc_init)
    assert scene.targets_by_id["id-83"].target_spacecraft.name == "debris"
    assert scene.targets_by_id["id-83"].priority == 3
    assert scene.imagers[0].name == "camera"
    assert next(sat for sat in satellites if sat.name == "other").sat_args == {}
    scene.after_step(12)
    assert scene.targets_by_id["id-83"].priority == 7
    assert scene.applied_events == [(0, 0.0), (1, 12.0)]
    scene.pending["id-83"] = {"image"}
    scene.reset_overwrite_previous()
    assert scene.catalog is definition
    if inherit_epoch:
        scene.utc_init = "2026 OCT 02 00:00:00.000 (UTC)"
    scene.reset_pre_sim_init()
    assert scene.catalog.utc_init == scene.utc_init
    assert first_episode.utc_init == catalog().utc_init
    assert definition.utc_init == (None if inherit_epoch else catalog().utc_init)
    assert scene.pending == {}
    assert scene.targets_by_id["id-83"].priority == 3


def test_invalid_participants_and_catalog():
    with pytest.raises(ValueError, match="Missing"):
        RSOTargets(catalog(), ["camera"]).link_satellites([])
    with pytest.raises(ValueError, match="own"):
        RSOTargets(catalog(), ["debris"])
    with pytest.raises(ValueError, match="unique"):
        RSOTargetCatalog(catalog().targets * 2, catalog().utc_init)
    with pytest.raises(ValueError, match="nonempty string"):
        RSOTarget(83, "debris", (7e6, 0, 0), (0, 7500, 0))
    with pytest.raises(ValueError, match="unique"):
        RSOTargets(catalog(), ["camera", "camera"])
    with pytest.raises(ValueError, match="unique"):
        RSOTargets(catalog(), ["camera"]).link_satellites(
            [NS(name="camera"), NS(name="camera")]
        )
    scene = RSOTargets(catalog(), ["camera"])
    scene.utc_init = "wrong epoch"
    with pytest.raises(ValueError, match="epoch"):
        scene.reset_pre_sim_init()


def test_ambiguous_original_names_are_rejected_before_environment_renaming():
    from bsk_rl import GeneralSatelliteTasking

    with pytest.raises(ValueError, match="Satellite names must be unique"):
        GeneralSatelliteTasking(
            satellites=[NS(name="camera"), NS(name="camera"), NS(name="debris")],
            scenario=RSOTargets(catalog(), ["camera_0"]),
        )
