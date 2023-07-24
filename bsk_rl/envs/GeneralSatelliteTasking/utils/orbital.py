from typing import Callable, Iterable

import numpy as np
from Basilisk import __path__
from Basilisk.simulation import spacecraft
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody
from Basilisk.utilities.orbitalMotion import ClassicElements, elem2rv
from scipy.interpolate import interp1d

bskPath = __path__[0]


def random_orbit(
    i: float | None = 45.0,
    alt: float = 500,
    r_body: float = 6371,
    e: float = 0,
    Omega: float | None = None,
    omega: float | None = 0,
    f: float | None = None,
) -> ClassicElements:
    """Create a set of orbit elements. Parameters are fixed if specified and randomized if None.

    Args:
        i: inclination [deg], randomized in [-pi, pi]
        alt: altitude above r_body [km]
        r_body: body radius [km]
        e: eccentricity
        Omega: LAN [deg], randomized in [0, 2pi]
        omega: Argument of periapsis [deg], randomized in [0, 2pi]
        f: true anomaly [deg], randomized in [0, 2pi]

    Returns:
        ClassicElements: orbital elements
    """
    oe = ClassicElements()
    oe.a = (6371 + alt) * 1e3
    oe.e = e
    oe.i = np.radians(i) if i is not None else np.random.uniform(-np.pi, np.pi)
    oe.Omega = (
        np.radians(Omega) if Omega is not None else np.random.uniform(0, 2 * np.pi)
    )
    oe.omega = (
        np.radians(omega) if omega is not None else np.random.uniform(0, 2 * np.pi)
    )
    oe.f = np.radians(f) if f is not None else np.random.uniform(0, 2 * np.pi)
    return oe


class TrajectorySimulator(SimulationBaseClass.SimBaseClass):
    def __init__(
        self,
        utc_init: str,
        rN: Iterable[float] = None,
        vN: Iterable[float] = None,
        oe: ClassicElements = None,
        mu: float = None,
        dt: float = 60.0,
    ) -> None:
        """Interpolator for trajectory using a point mass simulation under the
        effect of Earth's gravity, in the PCPF frame. Specify either (rN, vN) or (oe, mu).

        Args:
            utc_init: Simulation start time.
            rN: Initial position [m]
            vN: Initial velocity [m/s]
            oe: Orbital elements.
            mu: Gravitational parameter
            dt: Simulation timestep.
        """
        super().__init__()
        self.utc_init = utc_init
        self.dt = dt
        if rN is not None and vN is not None and oe is None:
            self.rN_init = rN
            self.vN_init = vN
        elif oe is not None and rN is None and vN is None:
            self.rN_init, self.vN_init = elem2rv(mu, oe)
        elif rN is None and vN is None and oe is None:
            raise (ValueError("Orbit is underspecified. Provide either (rN, vN) or oe"))
        else:
            raise (ValueError("Orbit is overspecified. Provide either (rN, vN) or oe"))

        self.init_simulator()

    def init_simulator(self) -> None:
        # Set up spacecraft simulator
        simTaskName = "simTask"
        simProcessName = "simProcess"

        dynProcess = self.CreateNewProcess(simProcessName)
        simulationTimeStep = macros.sec2nano(self.dt)
        dynProcess.addTask(self.CreateNewTask(simTaskName, simulationTimeStep))

        scObject = spacecraft.Spacecraft()
        scObject.ModelTag = "traj-sat"
        self.AddModelToTask(simTaskName, scObject)

        # Setup Gravity Body
        self.gravFactory = simIncludeGravBody.gravBodyFactory()
        planet = self.gravFactory.createEarth()
        self.gravFactory.createSun()
        planet.isCentralBody = True
        planet.useSphericalHarmParams = True
        simIncludeGravBody.loadGravFromFile(
            bskPath + "/supportData/LocalGravData/GGM03S.txt", planet.spherHarm, 10
        )

        # Set up spice with spice time
        UTCInit = self.utc_init
        self.gravFactory.createSpiceInterface(
            bskPath + "/supportData/EphemerisData/", UTCInit, epochInMsg=True
        )
        self.gravFactory.spiceObject.zeroBase = (
            "earth"  # Make sure that the Earth is the zero base
        )
        self.AddModelToTask(simTaskName, self.gravFactory.spiceObject)

        # Finally, the gravitational body must be connected to the spacecraft object.  This is done with
        scObject.gravField.gravBodies = spacecraft.GravBodyVector(
            list(self.gravFactory.gravBodies.values())
        )

        # To set the spacecraft initial conditions, the following initial position and velocity variables are set:
        scObject.hub.r_CN_NInit = self.rN_init  # m   - r_BN_N
        scObject.hub.v_CN_NInit = self.vN_init  # m/s - v_BN_N

        # create a logging task object of the spacecraft output message
        self.sc_state_log = scObject.scStateOutMsg.recorder()
        self.planet_state_log = self.gravFactory.spiceObject.planetStateOutMsgs[
            0
        ].recorder()
        self.AddModelToTask(simTaskName, self.sc_state_log)
        self.AddModelToTask(simTaskName, self.planet_state_log)

        self.InitializeSimulation()

    @property
    def sim_time(self) -> float:
        """Current simulator end time"""
        return macros.NANO2SEC * self.TotalSim.CurrentNanos

    def extend_interpolator(self, end_time) -> Callable:
        if end_time < self.sim_time:
            return self.total_interpolator
        self.ConfigureStopTime(macros.sec2nano(end_time))
        self.ExecuteSimulation()

        times = [macros.NANO2SEC * t for t in self.sc_state_log.times()]
        posData = self.sc_state_log.r_BN_N
        dcm_PN = self.planet_state_log.J20002Pfix

        # Compute the position in the planet-centered, planet-fixed frame
        pcpf_positions = []
        for dcm, pos in zip(dcm_PN, posData):
            pcpf_positions.append(np.matmul(dcm, pos))

        if not hasattr(self, "total_interpolator"):
            self.total_interpolator = self.current_interpolator = interp1d(
                np.array(times),
                np.array(pcpf_positions),
                kind="cubic",
                axis=0,
                fill_value="extrapolate",
            )
        else:
            self.total_interpolator = interp1d(
                np.concatenate((self.total_interpolator.x, np.array(times))),
                np.concatenate((self.total_interpolator.y, np.array(pcpf_positions))),
                kind="cubic",
                axis=0,
                fill_value="extrapolate",
            )

        self.sc_state_log.clear()
        self.planet_state_log.clear()
        return self.total_interpolator

    def __call__(
        self, t: float | np.ndarray
    ) -> Iterable[float] | Iterable[Iterable[float]]:
        """Get position at time(s)

        Args:
            t : Query time(s)

        Returns:
            position(s)
        """
        if np.any(t > self.sim_time):
            self.extend_interpolator(np.max(t))
        return self.total_interpolator(t)

    def interpolation_points(
        self, t_start: float = 0, t_end: float = float("inf")
    ) -> tuple[np.ndarray, np.ndarray]:
        in_range = np.logical_and(
            self.total_interpolator.x >= t_start, self.total_interpolator.x <= t_end
        )
        times = self.total_interpolator.x[in_range]
        positions = self.total_interpolator.y[in_range]
        return times, positions

    def __del__(self) -> None:
        self.gravFactory.unloadSpiceKernels()


def elevation(r_sat: np.ndarray, r_target: np.ndarray) -> np.ndarray:
    """Find the elevation angle from a target to a satellite

    Args:
        r_sat: Satellite position(s)
        r_target: Target position

    Returns:
        Elevation angle(s)
    """
    if r_sat.ndim == 2:
        return np.pi / 2 - np.arccos(
            np.sum(r_target * (r_sat - r_target), 1)
            / (np.linalg.norm(r_target) * np.linalg.norm(r_sat - r_target, axis=1))
        )
    else:
        return np.pi / 2 - np.arccos(
            np.sum(r_target * (r_sat - r_target))
            / (np.linalg.norm(r_target) * np.linalg.norm(r_sat - r_target))
        )
