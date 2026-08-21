from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from bsk_rl.utils.amos_vizard import (
    AMOSVizardMonitor,
    DESAT_COLOR,
    DOWNLINK_COLOR,
    HIO_COLOR,
    PRIORITY_TIER_COLORS,
    SHIO_COLOR,
    TARGET_STATUS_RING_RADIUS_M,
    _priority_terciles,
    _promotion_marker_style,
    ground_station_visibility_geometry,
    prepare_amos_vizard_assets,
)


def test_ground_station_visibility_geometry_500_km_10_deg():
    earth_radius_m = 6378.1363e3
    fov, slant_range, footprint_arc = ground_station_visibility_geometry(
        earth_radius_m,
        earth_radius_m + 500e3,
        np.radians(10.0),
    )

    assert np.degrees(fov) == pytest.approx(160.0)
    assert slant_range / 1e3 == pytest.approx(1695.091145, abs=1e-6)
    assert footprint_arc / 1e3 == pytest.approx(1563.602549, abs=1e-6)


def test_priority_terciles_follow_initial_priority_order():
    targets = [
        SimpleNamespace(id=target_id, priority=priority)
        for target_id, priority in enumerate([8.0, 2.0, 9.0, 4.0, 1.0, 7.0])
    ]

    tiers = _priority_terciles(targets)

    assert {target_id for target_id, tier in tiers.items() if tier == "lower"} == {1, 4}
    assert {target_id for target_id, tier in tiers.items() if tier == "middle"} == {3, 5}
    assert {target_id for target_id, tier in tiers.items() if tier == "upper"} == {0, 2}


@pytest.mark.parametrize(
    ("planet_radius", "observer_radius", "elevation"),
    [(0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 2.0, -0.1), (1.0, 2.0, np.pi / 2)],
)
def test_ground_station_visibility_geometry_rejects_invalid_geometry(
    planet_radius, observer_radius, elevation
):
    with pytest.raises(ValueError):
        ground_station_visibility_geometry(planet_radius, observer_radius, elevation)


def test_amos_assets_default_to_priority_circles_without_lifecycle_outlines():
    from Basilisk.architecture import messaging
    from Basilisk.simulation import vizInterface
    from Basilisk.utilities import vizSupport

    transmitter_message = messaging.DataNodeUsageMsg()
    scanner = SimpleNamespace(
        dynamics=SimpleNamespace(
            transmitter=SimpleNamespace(nodeDataOutMsg=transmitter_message)
        )
    )
    targets = []
    for target_id, (priority, kind) in enumerate(
        [(1.0, ""), (2.0, "HIO"), (3.0, "SHIO")]
    ):
        target = SimpleNamespace(
            id=target_id,
            priority=priority,
            priority_event_kind=kind,
        )
        targets.append(SimpleNamespace(rso_target=target))

    assets = prepare_amos_vizard_assets(
        [scanner, *targets],
        vizInterface,
        vizSupport,
        show_text_hud=False,
        rw_display="off",
    )

    assert assets.sprite_list[1].startswith("CIRCLE")
    assert assets.sprite_list[2].startswith("CIRCLE")
    assert assets.sprite_list[3].startswith("CIRCLE")
    assert assets.sprite_list[0] == "bskSat"
    assert assets.promotion_marker_sprites[1].startswith("STAR")
    assert assets.promotion_marker_sprites[2].startswith("TRIANGLE")
    assert set(assets.promotion_marker_messages) == {1, 2}
    assert set(assets.target_proxy_messages) == {1, 2}
    assert assets.promotion_marker_outlines == {}
    assert assets.target_outlines == {}
    assert assets.ellipsoid_list[1] is None
    assert assets.promotion_halos[1].isOn == -1
    assert assets.bars["priority_lower"].currentValue == 1.0
    assert list(assets.bars["priority_lower"].color) == PRIORITY_TIER_COLORS["lower"]
    assert list(assets.downlink_transceiver.color) == list(
        vizSupport.toRGBA255(DOWNLINK_COLOR, alpha=1.0)
    )
    assert assets.downlink_transceiver.animationSpeed == 2
    assert assets.downlink_transceiver.transceiverState == 0
    assert len(assets.downlink_transceiver.transceiverStateInMsgs) == 1
    assert assets.downlink_state_reader.isLinked()
    assert list(assets.desat_transceiver.color) == list(
        vizSupport.toRGBA255(DESAT_COLOR, alpha=1.0)
    )
    assert assets.desat_transceiver.animationSpeed == 2
    assert assets.desat_transceiver.transceiverState == 0
    assert len(assets.desat_transceiver.transceiverStateInMsgs) == 1
    assert assets.desat_state_reader.isLinked()
    assert assets.desat_state_message.read().baudRate == 0.0
    assert assets.transceiver_list[0] == [
        assets.downlink_transceiver,
        assets.desat_transceiver,
    ]


def test_target_lifecycle_outlines_are_opt_in():
    from Basilisk.architecture import messaging
    from Basilisk.simulation import vizInterface
    from Basilisk.utilities import vizSupport

    scanner = SimpleNamespace(
        dynamics=SimpleNamespace(
            transmitter=SimpleNamespace(nodeDataOutMsg=messaging.DataNodeUsageMsg())
        )
    )
    target = SimpleNamespace(id=0, priority=1.0, priority_event_kind="HIO")
    target_satellite = SimpleNamespace(rso_target=target)

    assets = prepare_amos_vizard_assets(
        [scanner, target_satellite],
        vizInterface,
        vizSupport,
        show_text_hud=False,
        show_target_status_outlines=True,
        rw_display="off",
    )

    assert list(assets.target_outlines[0].semiMajorAxes) == [
        TARGET_STATUS_RING_RADIUS_M
    ] * 3
    assert assets.promotion_marker_outlines[0].isOn == -1
    assert len(assets.promotion_marker_ellipsoids[0]) == 2


def test_promotion_marker_shape_and_purple_fill():
    from Basilisk.utilities import vizSupport

    hio = SimpleNamespace(priority_event_kind="HIO")
    shio = SimpleNamespace(priority_event_kind="SHIO")

    assert _promotion_marker_style(hio, vizSupport) == (
        "STAR",
        list(vizSupport.toRGBA255(HIO_COLOR)),
    )
    assert _promotion_marker_style(shio, vizSupport) == (
        "TRIANGLE",
        list(vizSupport.toRGBA255(SHIO_COLOR)),
    )


def test_desat_uses_red_rings_without_enabling_downlink_rings():
    from Basilisk.architecture import messaging

    downlink_message = messaging.DataNodeUsageMsg()
    desat_message = messaging.DataNodeUsageMsg()
    assets = SimpleNamespace(
        downlink_state_reader=messaging.DataNodeUsageMsgReader(),
        desat_state_message=desat_message,
    )
    assets.downlink_state_reader.subscribeTo(downlink_message)
    monitor = AMOSVizardMonitor(
        simulator=SimpleNamespace(),
        scanner=SimpleNamespace(),
        target_satellites=[],
        viz_instance=SimpleNamespace(),
        viz_support=SimpleNamespace(),
        assets=assets,
    )

    monitor._update_action_rings(desat_active=True, message_time_ns=123)

    assert downlink_message.read().baudRate == 0.0
    assert desat_message.read().baudRate == -1.0

    downlink_payload = messaging.DataNodeUsageMsgPayload()
    downlink_payload.baudRate = -2.0
    downlink_message.write(downlink_payload, 456)
    monitor._update_action_rings(desat_active=False, message_time_ns=456)

    assert assets.downlink_state_reader().baudRate == -2.0
    assert desat_message.read().baudRate == 0.0


def test_live_metric_bars_can_be_omitted_entirely():
    from Basilisk.architecture import messaging
    from Basilisk.simulation import vizInterface
    from Basilisk.utilities import vizSupport

    scanner = SimpleNamespace(
        dynamics=SimpleNamespace(
            transmitter=SimpleNamespace(nodeDataOutMsg=messaging.DataNodeUsageMsg())
        )
    )

    assets = prepare_amos_vizard_assets(
        [scanner],
        vizInterface,
        vizSupport,
        show_text_hud=False,
        show_live_metric_bars=False,
        rw_display="off",
    )

    assert assets.show_live_metric_bars is False
    assert assets.bars == {}
    assert assets.generic_storage_list == [None]


def test_action_only_monitor_skips_catalog_and_storage_metric_scans():
    from Basilisk.architecture import messaging

    scanner = SimpleNamespace(
        _current_action_label="Charge",
        data_store=SimpleNamespace(data=SimpleNamespace(known=[])),
        dynamics=SimpleNamespace(
            storageUnit=SimpleNamespace(storageCapacity=1.0),
            battery_charge_fraction=0.5,
            transmitterPowerSink=SimpleNamespace(powerStatus=0),
            thrusterPowerSink=SimpleNamespace(powerStatus=0),
            world=SimpleNamespace(groundStations=[]),
        ),
        fsw=SimpleNamespace(),
    )
    monitor = AMOSVizardMonitor(
        simulator=SimpleNamespace(),
        scanner=scanner,
        target_satellites=[],
        viz_instance=SimpleNamespace(),
        viz_support=SimpleNamespace(targetLineList=[]),
        assets=SimpleNamespace(
            show_live_metric_bars=False,
            dialog=None,
            bars={},
            ground_link_line=None,
            scanner_display_name="SS1 Space Surveillance Inspector",
            desat_state_message=messaging.DataNodeUsageMsg(),
            promotion_marker_messages={},
            target_proxy_messages={},
            target_outlines={},
            promotion_marker_outlines={},
            promotion_halos={},
            show_target_status_outlines=False,
        ),
    )
    monitor._target_state = Mock(side_effect=AssertionError("catalog scan"))
    monitor._capture_counts = Mock(side_effect=AssertionError("capture scan"))
    monitor._storage_split_bits = Mock(side_effect=AssertionError("storage scan"))

    monitor.UpdateState(2_000_000_000)

    monitor._target_state.assert_not_called()
    monitor._capture_counts.assert_not_called()
    monitor._storage_split_bits.assert_not_called()
    assert monitor.latest_metrics["sim_time_s"] == 2.0
    assert monitor.latest_metrics["current_action"] == "Charge"


def test_active_promotion_marker_swaps_with_blue_target_proxy():
    from Basilisk.architecture import messaging

    target_state = messaging.SCStatesMsgPayload()
    target_state.r_BN_N = [7.0e6, 1.0, 2.0]
    target_state.v_BN_N = [3.0, 4.0, 5.0]
    target_state_message = messaging.SCStatesMsg().write(target_state, 0)
    marker_message = messaging.SCStatesMsg()
    target_proxy_message = messaging.SCStatesMsg()
    target_spacecraft = SimpleNamespace(
        dynamics=SimpleNamespace(
            scObject=SimpleNamespace(scStateOutMsg=target_state_message)
        )
    )
    target = SimpleNamespace(
        id=4,
        name="target_4",
        target_spacecraft=target_spacecraft,
        priority_event_kind="HIO",
        priority_event_active=True,
    )
    target_spacecraft.rso_target = target
    scanner = SimpleNamespace(
        data_store=SimpleNamespace(data=SimpleNamespace(known=[target]))
    )
    monitor = AMOSVizardMonitor(
        simulator=SimpleNamespace(satellites=[scanner, target_spacecraft]),
        scanner=scanner,
        target_satellites=[target_spacecraft],
        viz_instance=SimpleNamespace(),
        viz_support=SimpleNamespace(),
        assets=SimpleNamespace(
            priority_tiers={4: "upper"},
            promotion_marker_messages={4: marker_message},
            target_proxy_messages={4: target_proxy_message},
            promotion_marker_outlines={},
            target_outlines={},
            promotion_halos={},
        ),
    )

    monitor._update_target_visuals(123.0)

    assert marker_message.read().r_BN_N == pytest.approx(target_state.r_BN_N)
    assert target_proxy_message.read().r_BN_N == pytest.approx([0.0, 0.0, 0.0])


def test_storage_split_uses_live_illumination_when_partition_grows():
    class MutableReadMessage:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return self.payload

    storage_payload = SimpleNamespace(
        storedDataName=["target_0"],
        storedData=[0.0],
    )
    shadow_payload = SimpleNamespace(shadowFactor=1.0)
    target_spacecraft = SimpleNamespace(
        name="target_0",
        dynamics=SimpleNamespace(eclipse_index=0),
    )
    target = SimpleNamespace(id=0, name="target_0", target_spacecraft=target_spacecraft)
    target_spacecraft.rso_target = target
    scanner = SimpleNamespace(
        dynamics=SimpleNamespace(
            storageUnit=SimpleNamespace(
                storageUnitDataOutMsg=MutableReadMessage(storage_payload)
            ),
            world=SimpleNamespace(
                eclipseObject=SimpleNamespace(
                    eclipseOutMsgs=[MutableReadMessage(shadow_payload)]
                )
            ),
            eclipse_threshold_for_imaging=0.1,
        ),
        data_store=SimpleNamespace(data=SimpleNamespace(known=[target])),
    )
    simulator = SimpleNamespace(satellites=[scanner, target_spacecraft])
    monitor = AMOSVizardMonitor(
        simulator=simulator,
        scanner=scanner,
        target_satellites=[target_spacecraft],
        viz_instance=SimpleNamespace(),
        viz_support=SimpleNamespace(),
        assets=SimpleNamespace(),
    )

    assert monitor._storage_split_bits() == (0.0, 0.0, 0.0)

    storage_payload.storedData = [4.0]
    assert monitor._storage_split_bits() == (4.0, 4.0, 0.0)

    shadow_payload.shadowFactor = 0.0
    storage_payload.storedData = [8.0]
    assert monitor._storage_split_bits() == (8.0, 4.0, 4.0)

    storage_payload.storedData = [4.0]
    assert monitor._storage_split_bits() == (4.0, 2.0, 2.0)
