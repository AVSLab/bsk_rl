import gymnasium as gym

from bsk_rl.env.scenario import data
from bsk_rl.env.scenario import satellites as sats
from bsk_rl.env.scenario.communication import (
    FreeCommunication,
    LOSCommunication,
    LOSMultiCommunication,
    NoCommunication,
)
from bsk_rl.env.scenario.environment_features import StaticTargets
from bsk_rl.env.simulation import environment
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


def make_communication_env(oes, comm_type):
    satellites = [
        sats.FullFeaturedSatellite(
            "EO-1",
            n_ahead_act=10,
            sat_args=sats.FullFeaturedSatellite.default_sat_args(
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
        env_features=StaticTargets(n_targets=1000),
        data_manager=data.UniqueImagingManager(),
        communicator=comm_type(satellites),
        sim_rate=1.0,
        time_limit=5700.0,
        disable_env_checker=True,
    )
    return env


class TestNoCommunication:
    env = make_communication_env(oes_visible, NoCommunication)

    def test_comms(self):
        self.env.reset()
        for _ in range(10):
            self.env.step([5, 5, 5])
        imagesets = [set(sat.data_store.data.imaged) for sat in self.env.satellites]
        assert imagesets[0] != imagesets[1]
        assert imagesets[0] != imagesets[2]
        assert imagesets[1] != imagesets[2]


class TestFreeCommunication:
    env = make_communication_env(oes_visible, FreeCommunication)

    def test_comms(self):
        self.env.reset()
        for _ in range(10):
            self.env.step([5, 5, 5])
        imagesets = [set(sat.data_store.data.imaged) for sat in self.env.satellites]
        assert imagesets[0] == imagesets[1]
        assert imagesets[1] == imagesets[2]


class TestLOSCommunication:
    env = make_communication_env(oes_visible, LOSCommunication)

    def test_comms(self):
        self.env.reset()
        for _ in range(10):
            self.env.step([5, 5, 5])
        imagesets = [set(sat.data_store.data.imaged) for sat in self.env.satellites]
        assert imagesets[0].issubset(imagesets[1])
        assert imagesets[2].issubset(imagesets[1])


class TestMultiDegreeCommunication:
    env = make_communication_env(oes_visible, LOSMultiCommunication)

    def test_comms(self):
        self.env.reset()
        for _ in range(10):
            self.env.step([5, 5, 5])
        imagesets = [set(sat.data_store.data.imaged) for sat in self.env.satellites]
        assert imagesets[0] == imagesets[1]
        assert imagesets[1] == imagesets[2]


class TestEclipsedLOSCommunication:
    env = make_communication_env(oes_eclipsed, LOSMultiCommunication)

    def test_comms(self):
        self.env.reset()
        for _ in range(10):
            self.env.step([5, 5, 5])
        imagesets = [set(sat.data_store.data.imaged) for sat in self.env.satellites]
        assert imagesets[0] != imagesets[1]
        assert imagesets[0] != imagesets[2]
        assert imagesets[1] != imagesets[2]
        assert imagesets[1] != imagesets[2]
