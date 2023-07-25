import numpy as np
from Basilisk.utilities import astroFunctions
from Basilisk.utilities import macros as mc
from Basilisk.utilities import orbitalMotion
from numpy.random import uniform

from bsk_rl.utilities.initial_conditions import leo_orbit, sc_attitudes


def sampled_400km_leo_smallsat_tumble():
    # Sample orbital parameters
    oe, rN, vN = leo_orbit.sampled_400km()

    # Sample attitude and rates
    sigma_init, omega_init = sc_attitudes.random_tumble(maxSpinRate=0.00001)

    # Dict of initial conditions
    initial_conditions = {
        # Mass
        "mass": 330,  # kg
        # Orbital parameters
        "oe": oe,
        "rN": rN,
        "vN": vN,
        # Spacecraft dimensions
        "width": 1.38,
        "depth": 1.04,
        "height": 1.58,
        # Attitude and rate initialization
        "sigma_init": sigma_init,
        "omega_init": omega_init,
        # Atmospheric density
        "planetRadius": orbitalMotion.REQ_EARTH * 1000.0,
        "baseDensity": 1.22,  # kg/m^3
        "scaleHeight": 8e3,  # m
        # Disturbance Torque
        "disturbance_magnitude": 2e-4,
        "disturbance_vector": np.random.standard_normal(3),
        # Reaction Wheel speeds
        "wheelSpeeds": uniform(-800, 800, 3),  # RPM
        # Solar Panel Parameters
        "nHat_B": np.array([0, -1, 0]),
        "panelArea": 0.2 * 0.3,
        "panelEfficiency": 0.20,
        # Power Sink Parameters
        "powerDraw": -5.0,  # W
        # Battery Parameters
        "storageCapacity": 20.0 * 3600.0,
        "storedCharge_Init": np.random.uniform(8.0 * 3600.0, 20.0 * 3600.0, 1)[0],
        # Sun pointing FSW config
        "sigma_R0N": [1, 0, 0],
        # RW motor torque and thruster force mapping FSW config
        "controlAxes_B": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        # Attitude controller FSW config
        "K": 7,
        "Ki": -1.0,  # Note: make value negative to turn off integral feedback
        "P": 35,
        # Momentum dumping config
        "hs_min": 4.0,  # Nms
        # Thruster force mapping FSW module
        "thrForceSign": 1,
        # Thruster momentum dumping FSW config
        "maxCounterValue": 4,
        "thrMinFireTime": 0.002,  #   Seconds
    }
    # print('Orbital elements:')
    # print(f'a: {oe.a}')
    # print(f'e: {oe.e}')
    # print(f'i: {oe.i}')
    # print(f'o: {oe.omega}')
    # print(f'O: {oe.Omega}')
    # print(f'f: {oe.f}')
    return initial_conditions


def reasonable_400km_leo_smallsat_tumble():
    # Sample orbital parameters
    oe, rN, vN = leo_orbit.inclined_400km()

    # Sample attitude and rates
    sigma_init, omega_init = sc_attitudes.static_inertial()

    # Dict of initial conditions
    initial_conditions = {
        # Mass
        "mass": 330,  # kg
        # Orbital parameters
        "oe": oe,
        "rN": rN,
        "vN": vN,
        # Spacecraft dimensions
        "width": 1.38,
        "depth": 1.04,
        "height": 1.58,
        # Attitude and rate initialization
        "sigma_init": sigma_init,
        "omega_init": omega_init,
        # Atmospheric density
        "planetRadius": orbitalMotion.REQ_EARTH * 1000.0,
        "baseDensity": 1.22,  # kg/m^3
        "scaleHeight": 8e3,  # m
        # Disturbance Torque
        "disturbance_magnitude": 2e-4,
        "disturbance_vector": [1, 1, 1]
        / np.sqrt([3]),  #  unit vector in 1,1,1 direction
        # Reaction Wheel speeds
        "wheelSpeeds": [400, 400, 400],  # RPM
        # Solar Panel Parameters
        "nHat_B": np.array([0, -1, 0]),
        "panelArea": 0.2 * 0.3,
        "panelEfficiency": 0.20,
        # Power Sink Parameters
        "powerDraw": -5.0,  # W
        # Battery Parameters
        "storageCapacity": 20.0 * 3600.0,
        "storedCharge_Init": 15.0 * 3600.0,
        # Sun pointing FSW config
        "sigma_R0N": [1, 0, 0],
        # RW motor torque and thruster force mapping FSW config
        "controlAxes_B": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        # Attitude controller FSW config
        "K": 7,
        "Ki": -1.0,  # Note: make value negative to turn off integral feedback
        "P": 35,
        # Momentum dumping config
        "hs_min": 3.0,  # Nms
        # Thruster force mapping FSW module
        "thrForceSign": 1,
        # Thruster momentum dumping FSW config
        "maxCounterValue": 4,
        "thrMinFireTime": 0.02,  #   Seconds
    }

    return initial_conditions


def walker_delta_single_sc_500_km(oe, sim_length, global_tgts, priorities):
    # Get the ECI position and velocity
    rN, vN = orbitalMotion.elem2rv(leo_orbit.mu, oe)

    utc_init = "2021 MAY 04 07:47:48.965 (UTC)"

    # Distribute a set of targets to the spacecraft
    targets, times, pcpf_positions, tgt_positions = leo_orbit.distribute_tgts(
        rN, vN, sim_length, utc_init, global_tgts
    )

    # Sample attitude and rates
    sigma_init, omega_init = sc_attitudes.random_tumble(maxSpinRate=0.00001)

    wheel_speeds = uniform(-1500, 1500, 3)  # RPMs

    # Dict of initial conditions
    initial_conditions = {
        # Initialize the start time of the sim
        "utc_init": utc_init,
        # Mass
        "mass": 330,  # kg
        # Orbital parameters
        "oe": oe,
        "rN": rN,
        "vN": vN,
        # Spacecraft dimensions
        "width": 1.38,
        "depth": 1.04,
        "height": 1.58,
        # Attitude and rate initialization
        "sigma_init": sigma_init,
        "omega_init": omega_init,
        # Disturbance Torque
        # "disturbance_magnitude": 1e-6,
        "disturbance_magnitude": 4e-3,
        "disturbance_vector": np.random.standard_normal(3),
        # Reaction Wheel speeds
        # Reaction Wheel speeds
        "wheelSpeeds": wheel_speeds,  # RPM
        "maxSpeed": 3000,  # RPM
        # Solar Panel Parameters
        "nHat_B": np.array([0, 1, 0]),
        "panelArea": 2 * 1.0 * 0.5,
        "panelEfficiency": 0.20,
        # Power Sink Parameters
        "instrumentPowerDraw": -30.0,  # W, Assuming 30 W imager (Harris Spaceview)
        "transmitterPowerDraw": -15.0,  # W
        "rwBasePower": 0.4,  # W, Note the opposite convention
        "rwMechToElecEfficiency": 0.0,
        "rwElecToMechEfficiency": 0.5,
        # Battery Parameters
        "batteryStorageCapacity": 80.0 * 3600.0,
        "storedCharge_Init": np.random.uniform(30.0 * 3600.0, 70.0 * 3600.0, 1)[0],
        # Sun pointing FSW config
        "sigma_R0N": [1, 0, 0],
        # RW motor torque and thruster force mapping FSW config
        "controlAxes_B": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        # Attitude controller FSW config
        "K": 7,
        "Ki": -1.0,  # Note: make value negative to turn off integral feedback
        "P": 35,
        # Steering controller config
        "K1": 0.25,
        "K3": 3.0,
        "omega_max": 3.0 * mc.D2R,
        "servo_Ki": 5.0,
        "servo_P": 150.0,
        # Momentum dumping config
        "hs_min": 0.0,  # Nms
        # Thruster force mapping FSW module
        "thrForceSign": 1,
        # Thruster momentum dumping FSW config
        "maxCounterValue": 4,
        "thrMinFireTime": 0.02,  # Seconds
        # Imaging Target
        "imageTargetMinimumElevation": np.radians(45.0),
        "imageTargetMaximumRange": -1,
        # Data-generating instrument
        "instrumentBaudRate": 8e6,  # baud, 8e6 = 1 MB = 1 image
        # Transmitter
        "transmitterBaudRate": -8e6,  # 8 Mbits/s
        "transmitterNumBuffers": len(targets),
        # Data Storage Unit
        "dataStorageCapacity": 20 * 8e6,  # 30 images
        # Target locations and priorities
        "targetIndices": targets,  # idx into global tgts
        "targetPositions": tgt_positions,  # locations of targets in ECEF frame, m
        "imageAttErrorRequirement": 0.01,  # normalized MRP (approx. 1/4 rad error)
        "target_times": times,  # seconds
        "pcpf_positions": pcpf_positions,  # ECEF position of the spacecraft, m,
    }

    return initial_conditions


def env_initial_conditions(global_tgts, priorities):
    initial_conditions = {
        # Atmospheric density
        "planetRadius": orbitalMotion.REQ_EARTH * 1000.0,
        "baseDensity": 1.22,  # kg/m^3
        "scaleHeight": 8e3,  # m
        "groundLocationPlanetRadius": astroFunctions.E_radius * 1e3,
        # Ground station - Located in Boulder, CO
        "boulderGroundStationLat": np.radians(40.009971),  # 40.0150 N Latitude
        "boulderGroundStationLong": np.radians(-105.243895),  # 105.2705 W Longitude
        "boulderGroundStationAlt": 1624,  # Altitude, m
        "boulderMinimumElevation": np.radians(10.0),
        "boulderMaximumRange": 1e9,
        # Ground station - Located in Merritt Island, FL
        "merrittGroundStationLat": np.radians(28.3181),  # 28.3181 N Latitude
        "merrittGroundStationLong": np.radians(-80.6660),  # 80.6660 W Longitude
        "merrittGroundStationAlt": 0.9144,  # Altitude, m
        "merrittMinimumElevation": np.radians(10.0),
        "merrittMaximumRange": 1e9,
        # Ground station - Located in Singapore
        "singaporeGroundStationLat": np.radians(1.3521),  # 1.3521 N Latitude
        "singaporeGroundStationLong": np.radians(103.8198),  # 103.8198 E Longitude
        "singaporeGroundStationAlt": 15,  # Altitude, m
        "singaporeMinimumElevation": np.radians(10.0),
        "singaporeMaximumRange": 1e9,
        # Ground station - Located in Weilheim, Germany
        "weilheimGroundStationLat": np.radians(47.8407),  # 47.8407 N Latitude
        "weilheimGroundStationLong": np.radians(11.1421),  # 11.1421 E Longitude
        "weilheimGroundStationAlt": 563,  # Altitude, m
        "weilheimMinimumElevation": np.radians(10.0),
        "weilheimMaximumRange": 1e9,
        # Ground station - Located in Santiago, Chile
        "santiagoGroundStationLat": np.radians(-33.4489),  # 33.4489 S Latitude
        "santiagoGroundStationLong": np.radians(-70.6693),  # 70.6693 W Longitude
        "santiagoGroundStationAlt": 570,  # Altitude, m
        "santiagoMinimumElevation": np.radians(10.0),
        "santiagoMaximumRange": 1e9,
        # Ground station - Located in Dongara, Australia
        "dongaraGroundStationLat": np.radians(-29.2452),  # 29.2452 S Latitude
        "dongaraGroundStationLong": np.radians(114.9326),  # 114.9326 E Longitude
        "dongaraGroundStationAlt": 34,  # Altitude, m
        "dongaraMinimumElevation": np.radians(10.0),
        "dongaraMaximumRange": 1e9,
        # Ground station - Located in Hawaii
        "hawaiiGroundStationLat": np.radians(19.8968),  # 19.8968 N Latitude
        "hawaiiGroundStationLong": np.radians(-155.5828),  # 155.5828 W Longitude
        "hawaiiGroundStationAlt": 9,  # Altitude, m
        "hawaiiMinimumElevation": np.radians(10.0),
        "hawaiiMaximumRange": 1e9,
        "globalTargets": global_tgts,  # global targets in ECEF frame, m
        "globalPriorities": priorities,  # target priorities
    }

    return initial_conditions


def walker_delta_n_spacecraft_500_km(
    n_spacecraft,
    n_planes,
    rel_phasing,
    inc,
    global_tgts,
    priorities,
    sim_length,
    clustersize=1,
    clusterspacing=0,
):
    # Initialize initial conditions
    initial_conditions = {}

    # Generate the orbital elements for each s/c
    oe_all = leo_orbit.walker_delta(
        n_spacecraft,
        n_planes,
        rel_phasing,
        500.0 * 1000.0,
        inc,
        clustersize,
        clusterspacing,
    )

    # Generate initial conditions for each spacecraft
    for idx in range(n_spacecraft):
        initial_conditions[str(idx)] = walker_delta_single_sc_500_km(
            oe_all[idx], sim_length, global_tgts, priorities
        )
        print("Spacecraft " + str(idx) + " ICs completed")

    # Create the initial conditions for the env process
    initial_conditions["env_params"] = env_initial_conditions(global_tgts, priorities)
    initial_conditions["n_spacecraft"] = n_spacecraft

    return initial_conditions
