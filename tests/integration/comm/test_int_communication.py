import gymnasium as gym
import pytest

from bsk_rl import act, data, obs, sats
from bsk_rl.comm import (
    FreeCommunication,
    LOSCommunication,
    LOSMultiCommunication,
    NoCommunication,
)
from bsk_rl.scene import UniformTargets
from bsk_rl.sim import dyn
from bsk_rl.utils.orbital import walker_delta

oes_visible = walker_delta(
    n_spacecraft=3,  # Number of satellites
    n_planes=1,
    rel_phasing=0,
    altitude=500 * 1e3,
    inc=45,
    clustersize=3,  # Cluster all 3 satellites together
    clusterspacing=10,  # Space satellites by a true anomaly of 30 degrees
)

oes_eclipsed = walker_delta(
    n_spacecraft=3,  # Number of satellites
    n_planes=1,
    rel_phasing=0,
    altitude=500 * 1e3,
    inc=45,
    clustersize=3,  # Cluster all 3 satellites together
    clusterspacing=60,  # Space satellites by a true anomaly of 30 degrees
)


class FullFeaturedDynModel(dyn.GroundStationDynModel, dyn.LOSCommDynModel):
    pass


class FullFeaturedSatellite(sats.ImagingSatellite):
    observation_spec = [
        obs.SatProperties(dict(prop="r_BN_P", module="dynamics", norm=6e6)),
        obs.Time(),
    ]
    action_spec = [act.Image(n_ahead_image=10)]
    dyn_type = FullFeaturedDynModel


def make_communication_env(oes, comm_type):
    satellites = [
        FullFeaturedSatellite(
            "EO-1",
            sat_args=FullFeaturedSatellite.default_sat_args(
                oe=oe,
                imageAttErrorRequirement=0.05,
                imageRateErrorRequirement=0.05,
                batteryStorageCapacity=1e6,
                storedCharge_Init=1e6,
            ),
        )
        for oe in oes
    ]
    env = gym.make(
        "GeneralSatelliteTasking-v1",
        satellites=satellites,
        scenario=UniformTargets(n_targets=1000),
        rewarder=data.UniqueImageReward(),
        communicator=comm_type(),
        sim_rate=1.0,
        time_limit=5700.0,
        disable_env_checker=True,
    )
    return env


class TestNoCommunication:
    env = make_communication_env(oes_visible, NoCommunication)

    @pytest.mark.flaky(retries=3, reruns_delay=1)
    def test_comms(self):
        self.env.reset()
        for _ in range(10):
            self.env.step([5, 5, 5])
        imagesets = [
            set(sat.data_store.data.imaged) for sat in self.env.unwrapped.satellites
        ]
        assert imagesets[0] != imagesets[1]
        assert imagesets[0] != imagesets[2]
        assert imagesets[1] != imagesets[2]


class TestFreeCommunication:
    env = make_communication_env(oes_visible, FreeCommunication)

    def test_comms(self):
        self.env.reset()
        for _ in range(10):
            self.env.step([5, 5, 5])
        imagesets = [
            set(sat.data_store.data.imaged) for sat in self.env.unwrapped.satellites
        ]
        assert imagesets[0] == imagesets[1]
        assert imagesets[1] == imagesets[2]


class TestLOSCommunication:
    env = make_communication_env(oes_visible, LOSCommunication)

    def test_comms(self):
        self.env.reset()
        for _ in range(10):
            self.env.step([5, 5, 5])
        imagesets = [
            set(sat.data_store.data.imaged) for sat in self.env.unwrapped.satellites
        ]
        assert imagesets[0].issubset(imagesets[1])
        assert imagesets[2].issubset(imagesets[1])


class TestMultiDegreeCommunication:
    env = make_communication_env(oes_visible, LOSMultiCommunication)

    def test_comms(self):
        self.env.reset()
        for _ in range(10):
            self.env.step([5, 5, 5])
        imagesets = [
            set(sat.data_store.data.imaged) for sat in self.env.unwrapped.satellites
        ]
        assert imagesets[0] == imagesets[1]
        assert imagesets[1] == imagesets[2]


class TestEclipsedLOSCommunication:
    env = make_communication_env(oes_eclipsed, LOSMultiCommunication)

    def test_comms(self):
        self.env.reset()
        for _ in range(10):
            self.env.step([5, 5, 5])
        imagesets = [
            set(sat.data_store.data.imaged) for sat in self.env.unwrapped.satellites
        ]
        assert imagesets[0] != imagesets[1]
        assert imagesets[0] != imagesets[2]
        assert imagesets[1] != imagesets[2]
        assert imagesets[1] != imagesets[2]
