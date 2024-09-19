"""``bsk_rl.utils.orbital``:Utilities for computing orbital events."""

import logging
from typing import Iterable, Optional

import numpy as np
from Basilisk import __path__
from Basilisk.simulation import eclipse, spacecraft
from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    orbitalMotion,
    simIncludeGravBody,
)
from Basilisk.utilities.orbitalMotion import ClassicElements, elem2rv
from scipy.interpolate import interp1d

bskPath = __path__[0]

logger = logging.getLogger(__name__)


def random_orbit(
    i: Optional[float] = 45.0,
    alt: float = 500,
    r_body: float = 6371,
    e: float = 0,
    Omega: Optional[float] = None,
    omega: Optional[float] = 0,
    f: Optional[float] = None,
) -> ClassicElements:
    """Create a set of orbit elements.

    Parameters are fixed if specified and randomized if ``None``. Defaults to a random
    circular orbit at 500 km altitude and 45 deg inclination.

    Args:
        i: [deg] Inclination, randomized in ``[-pi, pi]``.
        alt: [km] Altitude above r_body.
        r_body: [km] Body radius.
        e: Eccentricity.
        Omega: [deg] LAN, randomized in ``[0, 2pi]``.
        omega: [deg] Argument of periapsis, randomized in ``[0, 2pi]``.
        f: [deg] True anomaly, randomized in ``[0, 2pi]``.

    Returns:
        ClassicElements: orbital elements
    """
    oe = ClassicElements()
    oe.a = (r_body + alt) * 1e3
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


def random_epoch(start: int = 2000, end: int = 2022):
    """Generate a random epoch in a year range.

    Date will always be in the first 28 days of the month.

    Args:
        start: Initial year.
        end: Final year.

    Returns:
        Epoch in ``YYYY MMM DD HH:MM:SS.SSS (UTC)`` format
    """
    year = np.random.randint(start, end)
    month = np.random.choice(
        [
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
        ]
    )
    day = np.random.randint(1, 28)  # Assume 28 days for simplicity
    hours = np.random.randint(0, 23)
    minutes = np.random.randint(0, 59)
    seconds = np.random.randint(0, 59)
    milliseconds = np.random.randint(0, 999)

    # Combine the parts to form the datetime string
    epoch = (
        f"{year} {month} {day:02d} "
        + f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d} (UTC)"
    )

    return epoch


def elevation(r_sat: np.ndarray, r_target: np.ndarray) -> np.ndarray:
    """Find the elevation angle from a target to a satellite.

    Args:
        r_sat: Satellite position(s).
        r_target: Target position.

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


def walker_delta(
    n_spacecraft: int,
    n_planes: int,
    rel_phasing: float,
    altitude: float,
    inc: float,
    clustersize: int = 1,
    clusterspacing: float = 0,
) -> list[orbitalMotion.ClassicElements]:
    """Compute the initial orbit conditions of a Walker-delta constellation.

    Args:
        n_spacecraft: Number of spacecraft.
        n_planes: Number of orbital planes.
        rel_phasing: [deg] Relative phasing between planes.
        altitude: [m] Altitude above Earth's surface.
        inc: [deg] Inclination.
        clustersize: Number of spacecraft in each cluster.
        clusterspacing: [deg] Spacing between spacecraft in a cluster.

    Returns:
        list: List of orbital elements
    """
    oe_all = []

    # Loop through s/c
    for idx in range(0, n_spacecraft):
        # Instantiate orbital elements
        oe = orbitalMotion.ClassicElements()

        # Define altitude, eccentricity, and inclination
        oe.a = 6371 * 1000.0 + altitude
        oe.e = 0.0
        oe.i = np.radians(inc)

        # Define the plane number
        spacecraft_per_plane = float(n_spacecraft / n_planes)
        plane_num = float(int(idx / spacecraft_per_plane))

        # Compute longitude of ascending node
        oe.Omega = np.radians(plane_num * 360.0 / n_planes % 360)

        # Set argument of periapsis using relative phasing
        dPhi_rel = plane_num * rel_phasing * 360.0 / n_planes
        oe.omega = np.radians(dPhi_rel)

        # Define true anomoly using in-plane phasing
        dPhi_inplane = 360.0 / (spacecraft_per_plane / clustersize)
        oe.f = np.radians(
            (int((idx % spacecraft_per_plane) / clustersize) * dPhi_inplane)
            + clusterspacing * (idx % clustersize)
        )

        # Append to oe_all
        oe_all.append(oe)

    return oe_all


def walker_delta_args(
    n_planes: int,
    rel_phasing: float = 0.0,
    altitude: float = 500,
    inc: float = 45.0,
    randomize_true_anomaly=True,
    randomize_lan=True,
    **walker_delta_kwargs,
):
    """Generate a function to generate a randomly phased Walker-delta constellation.

    The output of this function should be used to set the ``sat_arg_randomizer`` of the
    environment.

    Args:
        n_planes: Number of constellation planes.
        rel_phasing: [deg] Relative phasing between planes. Defaults to 0.0.
        altitude: [km] Orbit altitude above Earth's surface.
        inc: [deg] Orbit inclination
        randomize_true_anomaly: Add a random true anomaly phase shift that is the same
            for all constellation members on each call.
        randomize_lan: Add a random LAN phase shift that is the same for all constellation
            members on each call.
        walker_delta_kwargs: Additional arguments to pass to :class:`walker_delta`.
    """

    def walker_delta_arg_setup(satellites):
        oe_all = walker_delta(
            n_spacecraft=len(satellites),
            n_planes=n_planes,
            altitude=altitude * 1e3,
            inc=inc,
            rel_phasing=rel_phasing,
            **walker_delta_kwargs,
        )

        if randomize_true_anomaly:
            f_offset = np.random.uniform(0, 2 * np.pi)
            for oe in oe_all:
                oe.f = np.mod(oe.f + f_offset, 2 * np.pi)

        if randomize_lan:
            Omega_offset = np.random.uniform(0, 2 * np.pi)
            for oe in oe_all:
                oe.Omega = np.mod(oe.Omega + Omega_offset, 2 * np.pi)

        return {satellite: {"oe": oe} for satellite, oe in zip(satellites, oe_all)}

    return walker_delta_arg_setup


class TrajectorySimulator(SimulationBaseClass.SimBaseClass):
    """Class for propagating trajectory using a point mass simulation."""

    def __init__(
        self,
        utc_init: str,
        rN: Optional[Iterable[float]] = None,
        vN: Optional[Iterable[float]] = None,
        oe: Optional[ClassicElements] = None,
        mu: Optional[float] = None,
        dt: float = 30.0,
    ) -> None:
        """Class for propagating trajectory using a point mass simulation.

        Simulated under the effect of Earth's gravity. Returns interpolators for
        position as well as upcoming eclipse predictions. Specify either ``(rN, vN)`` or
        ``(oe, mu)``.

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
        rv_specified = rN is not None and vN is not None
        rv_specified_partial = rN is not None or vN is not None
        oe_specified = oe is not None

        if rv_specified and not oe_specified:
            self.rN_init = rN
            self.vN_init = vN
        elif oe_specified and not rv_specified_partial:
            self.rN_init, self.vN_init = elem2rv(mu, oe)
        else:
            raise (
                ValueError(
                    "Orbit is over or underspecified. "
                    + "Provide either (rN, vN) or (oe, mu)"
                )
            )

        self._eclipse_starts: list[float] = []
        self._eclipse_ends: list[float] = []
        self._eclipse_search_time = 0.0

        self._init_simulator()

    def _init_simulator(self) -> None:
        simTaskName = "simTask"
        simProcessName = "simProcess"
        dynProcess = self.CreateNewProcess(simProcessName)
        simulationTimeStep = macros.sec2nano(self.dt)
        dynProcess.addTask(self.CreateNewTask(simTaskName, simulationTimeStep))
        scObject = spacecraft.Spacecraft()
        scObject.ModelTag = "traj-sat"
        self.AddModelToTask(simTaskName, scObject)

        scObject.hub.r_CN_NInit = self.rN_init  # m
        scObject.hub.v_CN_NInit = self.vN_init  # m/s

        # Set up gravity body
        self.gravFactory = simIncludeGravBody.gravBodyFactory()
        planet = self.gravFactory.createEarth()
        self.gravFactory.createSun()
        planet.isCentralBody = True
        planet.useSphericalHarmonicsGravityModel(
            bskPath + "/supportData/LocalGravData/GGM03S.txt", 10
        )
        UTCInit = self.utc_init
        self.gravFactory.createSpiceInterface(
            bskPath + "/supportData/EphemerisData/", UTCInit
        )
        self.gravFactory.spiceObject.zeroBase = "earth"
        self.AddModelToTask(simTaskName, self.gravFactory.spiceObject)
        scObject.gravField.gravBodies = spacecraft.GravBodyVector(
            list(self.gravFactory.gravBodies.values())
        )

        # Set up eclipse
        self.eclipseObject = eclipse.Eclipse()
        self.eclipseObject.addPlanetToModel(
            self.gravFactory.spiceObject.planetStateOutMsgs[0]
        )
        self.eclipseObject.sunInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[1]
        )
        self.AddModelToTask(simTaskName, self.eclipseObject, ModelPriority=988)
        self.eclipseObject.addSpacecraftToModel(scObject.scStateOutMsg)

        # Log outputs
        self.sc_state_log = scObject.scStateOutMsg.recorder()
        self.planet_state_log = self.gravFactory.spiceObject.planetStateOutMsgs[
            0
        ].recorder()
        self.eclipse_log = self.eclipseObject.eclipseOutMsgs[0].recorder()

        self._time_log = []
        self.AddModelToTask(simTaskName, self.sc_state_log)
        self._r_BN_N_log = []
        self.AddModelToTask(simTaskName, self.planet_state_log)
        self._J20002Pfix_log = []
        self.AddModelToTask(simTaskName, self.eclipse_log)
        self._shadowFactor_log = []

        self.InitializeSimulation()

    @property
    def sim_time(self) -> float:
        """Current simulator end time."""
        return macros.NANO2SEC * self.TotalSim.CurrentNanos

    @property
    def times(self) -> np.ndarray:
        """Recorder times in seconds."""
        return macros.NANO2SEC * np.array(self._time_log)

    def extend_to(self, t: float) -> None:
        """Compute the trajectory of the satellite up to t.

        Args:
            t: Computation end [s]
        """
        if t < self.sim_time:
            return
        self.ConfigureStopTime(macros.sec2nano(t))
        self.ExecuteSimulation()
        self._time_log.extend(self.sc_state_log.times())
        self._r_BN_N_log.extend(self.sc_state_log.r_BN_N)
        self._J20002Pfix_log.extend(self.planet_state_log.J20002Pfix)
        self._shadowFactor_log.extend(self.eclipse_log.shadowFactor)
        self.sc_state_log.clear()
        self.planet_state_log.clear()
        self.eclipse_log.clear()

    def _generate_eclipses(self, t: float) -> None:
        self.extend_to(t + self.dt)
        upcoming_times = self.times[self.times > self._eclipse_search_time]
        upcoming_eclipse = (
            np.array(self._shadowFactor_log)[self.times > self._eclipse_search_time] > 0
        ).astype(float)
        for i in np.where(np.diff(upcoming_eclipse) == -1)[0]:
            self._eclipse_starts.append(upcoming_times[i])
        for i in np.where(np.diff(upcoming_eclipse) == 1)[0]:
            self._eclipse_ends.append(upcoming_times[i])

        self._eclipse_search_time = t

    def next_eclipse(self, t: float, max_tries: int = 100) -> tuple[float, float]:
        """Find the soonest eclipse transitions.

        The returned values are not necessarily from the same eclipse event, such as
        when the search start time is in eclipse.

        Args:
            t: Time to start searching [s]
            max_tries: Maximum number of times to search

        Returns:
            eclipse_start: Nearest upcoming eclipse beginning
            eclipse_end:  Nearest upcoming eclipse end
        """
        for i in range(max_tries):
            if any([t_start > t for t_start in self._eclipse_starts]) and any(
                [t_end > t for t_end in self._eclipse_ends]
            ):
                eclipse_start = min(
                    [t_start for t_start in self._eclipse_starts if t_start > t]
                )
                eclipse_end = min([t_end for t_end in self._eclipse_ends if t_end > t])
                return eclipse_start, eclipse_end

            self._generate_eclipses(t + i * self.dt * 10)

        return 1.0, 1.0

    @property
    def r_BN_N(self) -> interp1d:
        """Interpolator for r_BN_N."""
        if self.sim_time < self.dt * 3:
            self.extend_to(self.dt * 3)
        return interp1d(
            self.times,
            self._r_BN_N_log,
            kind="cubic",
            axis=0,
            fill_value="extrapolate",
        )

    @property
    def r_BP_P(self) -> interp1d:
        """Interpolator for r_BP_P."""
        if self.sim_time < self.dt * 3:
            self.extend_to(self.dt * 3)
        return interp1d(
            self.times,
            [
                np.matmul(dcm, pos)
                for dcm, pos in zip(self._J20002Pfix_log, self._r_BN_N_log)
            ],
            kind="cubic",
            axis=0,
            fill_value="extrapolate",
        )

    def __del__(self) -> None:
        """Unload spice kernels when object is deleted."""
        try:
            self.gravFactory.unloadSpiceKernels()
        except AttributeError:
            pass


def lla2ecef(lat: float, long: float, radius: float):
    """Project LLA to Earth Centered, Earth Fixed location.

    Args:
        lat: [deg]
        long: [deg]
        radius: [any]
    """
    lat = np.radians(lat)
    long = np.radians(long)
    return radius * np.array(
        [np.cos(lat) * np.cos(long), np.cos(lat) * np.sin(long), np.sin(lat)]
    )


def rv2HN(r_N: np.ndarray, v_N: np.ndarray):
    """Find the Hill frame rotation matrix from position and velocity.

    Args:
        r_N: Position vector in the inertial frame
        v_N: Velocity vector in the inertial frame
    Returns:
        Hill frame rotation matrix HN
    """
    o_r_N = r_N / np.linalg.norm(r_N)
    h_N = np.cross(r_N, v_N)
    o_h_N = h_N / np.linalg.norm(h_N)
    o_theta_N = np.cross(o_h_N, o_r_N)
    HN = np.array([o_r_N, o_theta_N, o_h_N])
    return HN


def rv2omega(r_N: np.ndarray, v_N: np.ndarray):
    """Find the Hill frame rotation rate from position and velocity.

    Args:
        r_N: Position vector in the inertial frame
        v_N: Velocity vector in the inertial frame
    Returns:
        omega_HN_N: Angular velocity of the Hill frame in the inertial frame
    """
    o_r_N = r_N / np.linalg.norm(r_N)
    h_N = np.cross(r_N, v_N)
    o_h_N = h_N / np.linalg.norm(h_N)
    o_theta_N = np.cross(o_h_N, o_r_N)
    omega_HN_N = o_h_N * np.dot(v_N, o_theta_N) / np.linalg.norm(r_N)
    return omega_HN_N


__doc_title__ = "Orbital"
__all__ = [
    "random_orbit",
    "random_epoch",
    "lla2ecef",
    "elevation",
    "walker_delta",
    "walker_delta_args",
    "TrajectorySimulator",
    "rv2HN",
    "rv2omega",
]
