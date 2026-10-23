"""Access IDs are explicit bindings, never simulator array indices."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from bsk_rl.scene import RSOTargets
from bsk_rl.sim.dyn.rso_imaging import SpaceToSpaceImagingDynModel


def test_access_binding_uses_registration_output():
    messages = [object(), object()]
    location = NS(accessOutMsgs=[], addSpacecraftToModel=Mock())
    location.addSpacecraftToModel.side_effect = lambda _: location.accessOutMsgs.append(
        messages[len(location.accessOutMsgs)]
    )
    model = object.__new__(SpaceToSpaceImagingDynModel)
    model.logger = Mock()
    model.targetLocation = location
    model.rso_access_messages = {}
    model.rso_partition_names = {}
    model.satellite = NS(rso_scenario=NS(partition_name=RSOTargets.partition_name))
    for key in ("83", "rso/blue"):
        target = NS(
            id=key,
            target_spacecraft=NS(dynamics=NS(scObject=NS(scStateOutMsg=object()))),
        )
        model.bind_rso_target(target)
    assert model.rso_access_messages == dict(zip(("83", "rso/blue"), messages))
    assert model.rso_partition_names["83"] != model.rso_partition_names["rso/blue"]
    assert len(model.rso_partition_names["rso/blue"]) < 64
    with pytest.raises(ValueError, match="Duplicate"):
        model.bind_rso_target(target)


@pytest.mark.parametrize("value", [0, -2, float("nan")])
def test_invalid_range_is_not_silently_ignored(value):
    model = object.__new__(SpaceToSpaceImagingDynModel)
    model.logger = Mock()
    with pytest.raises(ValueError, match="range"):
        model.setup_imaging_target(imageTargetMaximumRange=value)


@pytest.mark.parametrize("value", [0, -1, float("nan")])
def test_invalid_image_size(value):
    model = object.__new__(SpaceToSpaceImagingDynModel)
    model.logger = Mock()
    with pytest.raises(ValueError, match="size"):
        model.setup_instrument(instrumentBaudRate=value)
