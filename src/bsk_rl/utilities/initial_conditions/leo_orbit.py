import math as m
import random

import numpy as np
from Basilisk import __path__
from Basilisk.simulation import spacecraft
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros as mc
from Basilisk.utilities import orbitalMotion, simIncludeGravBody
from numpy.random import uniform

bskPath = __path__[0]

mu = 0.3986004415e15


def inclined_circular_300km():
    """
    Returns an inclined, circular LEO orbit.
    :return:
    """

    oe = orbitalMotion.ClassicElements()
    oe.a = 6371 * 1000.0 + 300.0 * 1000
    oe.e = 0.0
    oe.i = 45.0 * mc.D2R

    oe.Omega = 0.0 * mc.D2R
    oe.omega = 0.0 * mc.D2R
    oe.f = 0.0 * mc.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe)

    return oe, rN, vN


def random_inclined_circular_300km():
    """
    Returns an inclined, circular LEO orbit.
    :return:
    """

    oe = orbitalMotion.ClassicElements()
    oe.a = 6371 * 1000.0 + uniform(290e3, 310e3)
    oe.e = 0.0
    oe.i = 45.0 * mc.D2R

    oe.Omega = 0.0 * mc.D2R
    oe.omega = 0.0 * mc.D2R
    oe.f = 0.0 * mc.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe)

    return oe, rN, vN


def sampled_400km():
    """
    Returns an elliptical, prograde LEO orbit with an SMA of 400km.
    :return:
    """
    oe = orbitalMotion.ClassicElements()
    oe.a = 6371 * 1000.0 + 400.0 * 1000
    oe.e = uniform(0, 0.001, 1)
    oe.i = uniform(-90 * mc.D2R, 90 * mc.D2R, 1)
    oe.Omega = uniform(0 * mc.D2R, 360 * mc.D2R, 1)
    oe.omega = uniform(0 * mc.D2R, 360 * mc.D2R, 1)
    oe.f = uniform(0 * mc.D2R, 360 * mc.D2R, 1)
    rN, vN = orbitalMotion.elem2rv(mu, oe)

    return oe, rN, vN


def sampled_500km_boulder_gs():
    """
    Returns an elliptical, prograde LEO orbit with an SMA of 500km.
    Inclination is bounded so the spacecraft can communicate with Boulder.
    :return:
    """
    mu = 0.3986004415e15
    oe = orbitalMotion.ClassicElements()
    oe.a = 6371 * 1000.0 + 500.0 * 1000
    oe.e = uniform(0, 0.01, 1)
    # oe.i = uniform(40*mc.D2R, 60*mc.D2R,1)
    oe.i = uniform(40 * mc.D2R, 60 * mc.D2R, 1)
    oe.Omega = uniform(0 * mc.D2R, 20 * mc.D2R, 1)
    oe.omega = uniform(0 * mc.D2R, 20 * mc.D2R, 1)
    oe.f = uniform(0 * mc.D2R, 360 * mc.D2R, 1)
    rN, vN = orbitalMotion.elem2rv(mu, oe)

    return oe, rN, vN


def sampled_boulder_gs(nominal_radius):
    """
    Returns an elliptical, prograde LEO orbit with an SMA of 500km.
    Inclination is bounded so the spacecraft can communicate with Boulder.
    :return:
    """
    mu = 0.3986004415e15
    oe = orbitalMotion.ClassicElements()
    oe.a = nominal_radius
    oe.e = uniform(0, 0.01, 1)
    # oe.i = uniform(40*mc.D2R, 60*mc.D2R,1)
    oe.i = uniform(40 * mc.D2R, 60 * mc.D2R, 1)
    oe.Omega = uniform(0 * mc.D2R, 360 * mc.D2R, 1)
    oe.omega = uniform(0 * mc.D2R, 360 * mc.D2R, 1)
    oe.f = uniform(0 * mc.D2R, 360 * mc.D2R, 1)
    rN, vN = orbitalMotion.elem2rv(mu, oe)

    return oe, rN, vN


def coordinated_pass_1():
    r_sc1 = (6378.0 + 500.0) * 1000  # meters
    oe_sc1 = orbitalMotion.ClassicElements()
    oe_sc1.a = r_sc1
    oe_sc1.e = 0.00001
    oe_sc1.i = 70.0 * mc.D2R
    oe_sc1.Omega = 135.0 * mc.D2R
    oe_sc1.omega = 184.8 * mc.D2R
    oe_sc1.f = 85.3 * mc.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe_sc1)

    return oe_sc1, rN, vN


def coordinated_pass_2():
    r_sc2 = (6378.0 + 2000.0) * 1000  # meters
    oe_sc2 = orbitalMotion.ClassicElements()
    oe_sc2.a = r_sc2
    oe_sc2.e = 0.00001
    oe_sc2.i = 53.0 * mc.D2R
    oe_sc2.Omega = 115.0 * mc.D2R
    oe_sc2.omega = 5.0 * mc.D2R
    oe_sc2.f = 240.0 * mc.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe_sc2)

    return oe_sc2, rN, vN


def coordinated_pass_3():
    r_sc2 = (6378.0 + 7500.0) * 1000  # meters
    oe_sc2 = orbitalMotion.ClassicElements()
    oe_sc2.a = r_sc2
    oe_sc2.e = 0.00001
    oe_sc2.i = 53.0 * mc.D2R
    oe_sc2.Omega = 115.0 * mc.D2R
    oe_sc2.omega = 5.0 * mc.D2R
    oe_sc2.f = 240.0 * mc.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe_sc2)

    return oe_sc2, rN, vN


def sso_Boulder():
    r_sc = 42164125  # meters
    oe_sc = orbitalMotion.ClassicElements()
    oe_sc.a = r_sc
    oe_sc.e = 0.00001
    oe_sc.i = 53.0 * mc.D2R
    oe_sc.Omega = 115.0 * mc.D2R
    oe_sc.omega = 5.0 * mc.D2R
    oe_sc.f = 75.0 * mc.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe_sc)

    return oe_sc, rN, vN


def inclined_400km():
    """
    Returns an elliptical, prograde LEO orbit with an SMA of 400km.
    :return:
    """
    mu = 0.3986004415e15
    oe = orbitalMotion.ClassicElements()
    oe.a = 6371 * 1000.0 + 500.0 * 1000
    oe.e = 0.0001
    oe.i = mc.D2R * 45.0
    oe.Omega = mc.D2R * 45.0
    oe.omega = 0.0
    oe.f = 0.0
    rN, vN = orbitalMotion.elem2rv(mu, oe)

    return oe, rN, vN


def create_ground_tgts(n_targets, rN, vN, sim_length, utc_init):
    """
    Returns a set of targets based on the orbital parameters by running a simplified
    BSK scenario
    :param n_targets: number of targets to generate
    :param rN: Initial inertial position
    :param vN: Initial inertial velocity
    :param sim_length: simulation length
    :param uct_init: time initialization string
    :return targets:
    """

    # Create simulation variable names
    simTaskName = "simTask"
    simProcessName = "simProcess"

    #  Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()

    #  create the simulation process
    dynProcess = scSim.CreateNewProcess(simProcessName)

    # Create the dynamics task and specify the integration update time
    simulationTimeStep = mc.sec2nano(10.0)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # Setup the simulation tasks/objects, initialize spacecraft object
    # and set properties
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "bsk-Sat"

    # Add spacecraft object to the simulation process
    scSim.AddModelToTask(simTaskName, scObject)

    # Setup Gravity Body
    gravFactory = simIncludeGravBody.gravBodyFactory()
    planet = gravFactory.createEarth()
    planet.isCentralBody = True

    planet.useSphericalHarmParams = True
    simIncludeGravBody.loadGravFromFile(
        bskPath + "/supportData/LocalGravData/GGM03S.txt", planet.spherHarm, 10
    )

    # Set up spice with spice time
    UTCInit = utc_init
    gravFactory.createSpiceInterface(
        bskPath + "/supportData/EphemerisData/", UTCInit, epochInMsg=True
    )
    gravFactory.spiceObject.zeroBase = (
        "earth"  # Make sure that the Earth is the zero base
    )
    scSim.AddModelToTask(simTaskName, gravFactory.spiceObject)

    # Finally, the gravitational body must be connected to the spacecraft object.
    scObject.gravField.gravBodies = spacecraft.GravBodyVector(
        list(gravFactory.gravBodies.values())
    )

    # To set the spacecraft initial conditions, the following initial position and
    # velocity variables are set:
    scObject.hub.r_CN_NInit = rN  # m   - r_BN_N
    scObject.hub.v_CN_NInit = vN  # m/s - v_BN_N

    # Set the simulation time
    simulationTime = mc.sec2nano(sim_length * 60.0)

    # create a logging task object of the spacecraft output message at the desired
    # down sampling ratio
    dataRec1 = scObject.scStateOutMsg.recorder()
    dataRec2 = gravFactory.spiceObject.planetStateOutMsgs[0].recorder()
    scSim.AddModelToTask(simTaskName, dataRec1)
    scSim.AddModelToTask(simTaskName, dataRec2)

    scSim.InitializeSimulation()

    #   configure a simulation stop time time and execute the simulation run
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()

    posData = dataRec1.r_BN_N
    dcm_PN = dataRec2.J20002Pfix

    pcpf_positions = []

    # Compute the position in the planet-centered, planet-fixed frame
    for idx, pos in enumerate(posData):
        pcpf_positions.append(np.matmul(dcm_PN[idx], pos))

    # Grab random PCPF targets
    candidate_position_indices = random.sample(range(0, len(posData)), k=n_targets)
    candidate_position_indices = np.sort(
        candidate_position_indices
    )  # list of the indices of the ordered targets
    candidate_position_times = [
        10 * idx for idx in candidate_position_indices
    ]  # list of the ordered times of the targets in seconds

    # Initialize targets
    targets = []
    for idx in candidate_position_indices:
        # Add noise to s/c position vector
        temp_pos = np.array(pcpf_positions[idx]) + np.random.uniform(-1.0e5, 1.0e5, 3)
        # Normalize and multiply by Earth's radius
        targets.append(6378.0 * 1000.0 * temp_pos / np.linalg.norm(np.array(temp_pos)))

    return np.array(targets).T, candidate_position_times


def walker_delta(
    n_spacecraft, n_planes, rel_phasing, altitude, inc, clustersize=1, clusterspacing=0
):
    """
    Computes the initial orbit conditions of a constellation of spacecraft in the
    walker delta pattern
    :param n_spacecraft: number of spacecraft in the constellation
    :param n_planes: number of orbital planes
    :param rel_phasing: relative phasing between the planes [0, 1)
    :param altitude: Altitude of the s/c (m)
    :param inc: Inclination of the orbit (deg)
    :param clustersize: Size of satellite groups
    :param clusterspacing: True anomaly spacing within cluster (deg)
    :return oe_all: n_spacecraft x 1 list of BSK oe elements
    """
    oe_all = []

    # Loop through s/c
    for idx in range(0, n_spacecraft):
        # Instantiate orbital elements
        oe = orbitalMotion.ClassicElements()

        # Define altitude, eccentricity, and inclination
        oe.a = 6371 * 1000.0 + altitude
        oe.e = 0.0
        oe.i = mc.D2R * inc

        # Define the plane number
        spacecraft_per_plane = float(n_spacecraft / n_planes)
        plane_num = float(int(idx / spacecraft_per_plane))

        # Compute longitude of ascending node
        oe.Omega = mc.D2R * (plane_num * 360.0 / n_planes % 360)

        # Set argument of periapsis using relative phasing
        dPhi_rel = plane_num * rel_phasing * 360.0 / n_planes
        oe.omega = mc.D2R * dPhi_rel

        # Define true anomoly using in-plane phasing
        dPhi_inplane = 360.0 / (spacecraft_per_plane / clustersize)
        oe.f = mc.D2R * (
            (int((idx % spacecraft_per_plane) / clustersize) * dPhi_inplane)
            + clusterspacing * (idx % clustersize)
        )

        # Append to oe_all
        oe_all.append(oe)

    return oe_all


def distribute_tgts(rN, vN, sim_length, utc_init, global_tgts, dt=60.0):
    """
    :param rN:
    :param vN:
    :param sim_length: [m]
    :param utc_init:
    :param global_tgts: np.array of global tgts in ECEF coordinates
    :param dt: [s]
    :return local_tgts: list of indexes into global tgts
    :return local_tgt_times: local time each global tgt in local_tgts is encountered
    """

    local_tgts = []
    local_tgt_times = []

    # Create simulation variable names
    simTaskName = "simTask"
    simProcessName = "simProcess"

    #  Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()

    #  create the simulation process
    dynProcess = scSim.CreateNewProcess(simProcessName)

    # Create the dynamics task and specify the integration update time
    simulationTimeStep = mc.sec2nano(dt)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # Setup the simulation tasks/objects, initialize spacecraft object and
    # set properties
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "bsk-Sat"

    # Add spacecraft object to the simulation process
    scSim.AddModelToTask(simTaskName, scObject)

    # Setup Gravity Body
    gravFactory = simIncludeGravBody.gravBodyFactory()
    planet = gravFactory.createEarth()
    gravFactory.createSun()
    planet.isCentralBody = True
    planet.useSphericalHarmParams = True
    simIncludeGravBody.loadGravFromFile(
        bskPath + "/supportData/LocalGravData/GGM03S.txt", planet.spherHarm, 10
    )

    # Set up spice with spice time
    UTCInit = utc_init
    gravFactory.createSpiceInterface(
        bskPath + "/supportData/EphemerisData/", UTCInit, epochInMsg=True
    )
    gravFactory.spiceObject.zeroBase = (
        "earth"  # Make sure that the Earth is the zero base
    )
    scSim.AddModelToTask(simTaskName, gravFactory.spiceObject)

    # Finally, the gravitational body must be connected to the spacecraft object.
    scObject.gravField.gravBodies = spacecraft.GravBodyVector(
        list(gravFactory.gravBodies.values())
    )

    # To set the spacecraft initial conditions, the following initial position and
    # velocity variables are set:
    scObject.hub.r_CN_NInit = rN  # m   - r_BN_N
    scObject.hub.v_CN_NInit = vN  # m/s - v_BN_N

    # Set the simulation time
    simulationTime = mc.sec2nano(sim_length * 60.0)

    # create a logging task object of the spacecraft output message
    dataRec1 = scObject.scStateOutMsg.recorder()
    dataRec2 = gravFactory.spiceObject.planetStateOutMsgs[0].recorder()
    scSim.AddModelToTask(simTaskName, dataRec1)
    scSim.AddModelToTask(simTaskName, dataRec2)

    scSim.InitializeSimulation()

    #   configure a simulation stop time time and execute the simulation run
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()

    posData = dataRec1.r_BN_N
    dcm_PN = dataRec2.J20002Pfix

    pcpf_positions = []
    local_tgt_locations = []

    # Compute the position in the planet-centered, planet-fixed frame
    for i, pos in enumerate(posData):
        # Compute the pcpf position of the s/c
        pcpf_pos = np.matmul(dcm_PN[i], pos)
        pcpf_positions.append(pcpf_pos)

        # Loop through every target to check if within azelrange requirements
        for j, tgt in enumerate(global_tgts):
            if (
                np.linalg.norm(pcpf_pos - tgt) < 750e3 and j not in local_tgts
            ):  # Perform a quick check - ensure difference is less than ~500km/cosd(45)
                local_tgts.append(j)  # Append the target idx
                local_tgt_times.append(
                    dt / 2 + dt * i
                )  # Append the time the target is encountered
                local_tgt_locations.append(tgt)

    gravFactory.unloadSpiceKernels()

    return local_tgts, local_tgt_times, pcpf_positions, local_tgt_locations


def elrange_req(sc_pos, tgt_pos):
    """
    Determines if the spacecraft is within the elevation and range requirements of
      a target
    :param sc_pos: spacecraft position expressed in the ECEF frame
    :param tgt_pos: tgt_pos expressed in the ECEF frame
    :return within: T/F - within el, range requirements or not
    """

    # Import relevant library
    import pyproj

    # Set the elevation and range requirements
    el_req = 75  # deg
    range_req = 1e9  # m

    # get the lat, long, and altitude of the target (assumes WGS84 ellipsoid)
    ecef = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")
    lla = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")
    lon, lat, alt = pyproj.transform(
        ecef, lla, tgt_pos[0], tgt_pos[1], tgt_pos[2], radians=True
    )  # rad, rad, m

    # Compute the line of sight vector
    los_ecef = np.array(sc_pos) - np.array(tgt_pos)

    # Normalize the vector
    los_ecef_norm = los_ecef / np.linalg.norm(los_ecef)

    # Compute the transformation from ECEF to ENU
    ECEF2ENU = np.array(
        [
            [-m.sin(lon), m.cos(lon), 0],
            [-m.sin(lat) * m.cos(lon), -m.sin(lat) * m.sin(lon), m.cos(lat)],
            [m.cos(lat) * m.cos(lon), m.cos(lat) * m.sin(lon), m.sin(lat)],
        ]
    )

    # Compute the los vector in ENU coordinates
    los_enu = np.matmul(ECEF2ENU, los_ecef_norm)

    # Compute the elevation
    el = m.degrees(m.asin(los_enu[2]))

    # Compute the range
    range = np.linalg.norm(los_ecef)  # m

    if (el > el_req) and (range < range_req):
        return True
    else:
        return False
