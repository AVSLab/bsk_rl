"""Physical lifecycle colors and native Vizard stream rewrite regression."""

from examples.multiagent_imaging.visualize_baseline_comparison import (
    AGENT_COLORS,
    DUPLICATE_COLOR,
    color_recording,
    encode_varint,
    frames,
    pending_holders,
    proto_types,
    target_state,
)


def episode():
    return {
        "capture_records": [
            {
                "record_id": "a",
                "target_id": 0,
                "source_sensor": "sensor_0",
                "quality": 1.0,
                "capture_time": 0.5,
                "completion_time": 1.0,
            },
            {
                "record_id": "b",
                "target_id": 0,
                "source_sensor": "sensor_1",
                "quality": 1.0,
                "capture_time": 1.5,
                "completion_time": 2.0,
            },
            # Unqualified exposures must not color a target as serviced.
            {
                "record_id": "c",
                "target_id": 1,
                "source_sensor": "sensor_2",
                "quality": 0.1,
                "capture_time": 0.1,
                "completion_time": 0.2,
            },
        ],
        "ground_delivery_records": [
            {"record_id": "a", "delivery_time": 5.0},
            {"record_id": "b", "delivery_time": 8.0},
        ],
    }


def test_colors_follow_completion_and_full_delivery_half_open_intervals():
    d = episode()
    assert pending_holders(d, 0.75) == {}
    assert target_state(pending_holders(d, 1)[0]) == "sensor_0"
    assert target_state(pending_holders(d, 2)[0]) == "duplicate"
    assert target_state(pending_holders(d, 5)[0]) == "sensor_1"
    assert pending_holders(d, 8) == {}


def test_native_stream_contains_one_visible_sprite_per_target_at_every_state(tmp_path):
    Viz = proto_types()
    raw = tmp_path / "raw.bin"
    colored = tmp_path / "colored.bin"
    with raw.open("wb") as f:
        for ix, t in enumerate([0, 1, 2, 5, 8]):
            m = Viz()
            m.currentTime.frameNumber = ix
            m.currentTime.simTimeElapsed = t * 1e9
            s = m.spacecraft.add()
            s.spacecraftName = "sensor_0"
            s.position.extend([1.0, 2.0, 3.0])
            s = m.spacecraft.add()
            s.spacecraftName = "target_0"
            s.position.extend([7e6, 1.0, 2.0])
            s.velocity.extend([0.0, 7500.0, 0.0])
            s.rotation.extend([0.0, 0.0, 0.0])
            s.spacecraftSprite = "CIRCLE 66 146 198 255"
            line = m.liveSettings.targetLines.add()
            line.fromBodyName = "sensor_0"
            line.toBodyName = "target_0"
            data = m.SerializeToString()
            f.write(encode_varint(len(data)))
            f.write(data)
    original = raw.read_bytes()
    stats = color_recording(raw, episode(), colored)
    assert raw.read_bytes() == original
    assert stats["frames"] == 5 and stats["max_red_targets"] == 1
    for m, state in zip(
        frames(colored), ["base", "sensor_0", "duplicate", "sensor_1", "base"]
    ):
        by_name = {s.spacecraftName: s for s in m.spacecraft}
        visible = [
            s
            for s in m.spacecraft
            if s.spacecraftName.startswith("target_")
            and list(s.position) != [0.0, 0.0, 0.0]
        ]
        assert len(visible) == 1
        expected = "target_0" if state == "base" else f"target_0__{state}"
        assert visible[0].spacecraftName == expected
        assert list(visible[0].position) == [7e6, 1.0, 2.0]
        assert m.liveSettings.targetLines[0].toBodyName == expected
        assert list(m.liveSettings.targetLines[0].lineColor) == AGENT_COLORS["sensor_0"]
        if m.currentTime.frameNumber == 0:
            assert by_name[
                "target_0__duplicate"
            ].spacecraftSprite == "STAR " + " ".join(map(str, DUPLICATE_COLOR))
            for sensor, color in AGENT_COLORS.items():
                assert by_name[
                    f"target_0__{sensor}"
                ].spacecraftSprite == "CIRCLE " + " ".join(map(str, color))
