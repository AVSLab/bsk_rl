import math as m
import random

import numpy as np
from Basilisk import __path__
from Basilisk.architecture import cMsgCInterfacePy as cMsgPy
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import (
    locationPointing,
    mrpFeedback,
    rwMotorTorque,
    simpleInstrumentController,
    smallBodyWaypointFeedback,
)
from Basilisk.simulation import (
    ReactionWheelPower,
    eclipse,
    ephemerisConverter,
    extForceTorque,
    groundLocation,
    groundMapping,
    mappingInstrument,
    partitionedStorageUnit,
    planetEphemeris,
    planetNav,
    radiationPressure,
    simpleBattery,
    simpleInstrument,
    simpleNav,
    simplePowerSink,
    simpleSolarPanel,
    spacecraft,
    spaceToGroundTransmitter,
)

#   Basilisk modules
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros as mc
from Basilisk.utilities import (
    orbitalMotion,
    simIncludeGravBody,
    unitTestSupport,
    vizSupport,
)

from bsk_rl.utilities.effector_primitives import actuator_primitives as ap
from bsk_rl.utilities.initial_conditions import sc_attitudes, small_body

bskPath = __path__[0]


class SmallBodyScienceSimulator(SimulationBaseClass.SimBaseClass):
    """
    Simulates a spacecraft in orbit about a small body.

    Dynamics Components:
    - Forces: Sun, Earth, Asteroid gravity, SRP,
    - Environment: Eclipse, GroundLocation
    - Actuators: ExternalForceTorque, reaction wheels
    - Sensors: SimpleNav, PlanetNav
    - Power System: SimpleBattery, SimplePowerSink, SimpleSolarPanel, RWPower
    - Data Management System: spaceToGroundTransmitter, simpleStorageUnit,
    simpleInstrument

    FSW Components:
    - MRP Feedback controller
    - locationPoint - targets, mapping, sun-pointing
    - SimpleInstrumentController
    - SmallBodyWaypointFeedback
    """

    def __init__(
        self,
        dynRate,
        fswRate,
        mapRate,
        step_duration,
        initial_conditions=None,
        render=False,
        n_targets=100,
        n_map_points=100,
        max_length=1440.0,
        n_states=-1,
        n_maps=3,
        phi_c=None,
        lambda_c=None,
        fidelity="low",
    ):
        self.initialized = False
        self.dynRate = dynRate
        self.fswRate = fswRate
        self.mapRate = mapRate
        self.step_duration = step_duration
        self.fidelity = fidelity

        # Set the durations for each mode
        self.maneuver_duration = 2e3  # s
        self.map_duration = 4e3  # s
        self.image_duration = 2e3  # s
        self.downlink_duration = 2e3  # s
        self.charge_duration = 2e3  # s

        SimulationBaseClass.SimBaseClass.__init__(self)
        # self.TotalSim.terminateSimulation()

        self.attRefMsg = None
        self.attGuidMsg = None

        self.simTime = 0.0

        # Initialize performance metrics
        self.downlinked = 0.0
        self.total_access = 0.0
        self.utilized_access = 0.0
        self.dV = 0.0

        self.curr_step = 0
        self.max_steps = 0
        self.max_length = max_length

        self.n_targets = n_targets
        self.n_states = n_states
        self.obs = np.zeros([self.n_states, 1])
        self.obs_full = np.zeros([self.n_states, 1])

        self.n_map_points = n_map_points
        self.n_maps = n_maps

        self.imaged_targets = np.zeros(self.n_targets)
        self.downlinked_targets = np.zeros(self.n_targets)
        self.imaged_maps = np.zeros((self.n_maps, self.n_map_points), dtype=bool)
        self.downlinked_maps = np.zeros((self.n_maps, self.n_map_points), dtype=bool)

        self.imaged_targets_old = np.zeros(self.n_targets)
        self.downlinked_targets_old = np.zeros(self.n_targets)
        self.imaged_maps_old = np.zeros((self.n_maps, self.n_map_points), dtype=bool)
        self.downlinked_maps_old = np.zeros(
            (self.n_maps, self.n_map_points), dtype=bool
        )

        self.phi_c = phi_c
        self.lambda_c = lambda_c

        # Define the longitude and latitude of the waypoints (these are in the Hill
        # frame, not body frame of asteroid)
        self.waypoint_latitudes = [75, 45, 15, -15, -45, -75]
        self.waypoint_longitudes = [-150, -90, -30, 30, 90, 150]

        self.d_phi = 30
        self.d_lambda = 60

        # Set the desired longitudes for mapping
        self.map_longitudes = [-90.0, -150.0, 150]  # deg

        # Set the tolerance for the acceptable longitude for successful mapping
        self.longitude_tolerance = 1  # deg

        # Define the waypoint deltas for the action space
        self.waypoint_latitude_deltas = [
            self.d_phi,
            self.d_phi,
            0,
            -self.d_phi,
            -self.d_phi,
            -self.d_phi,
            0,
            self.d_phi,
        ]
        self.waypoint_longitude_deltas = [
            0,
            self.d_lambda,
            self.d_lambda,
            self.d_lambda,
            0,
            -self.d_lambda,
            -self.d_lambda,
            -self.d_lambda,
        ]

        self.waypointTime = 0
        self.requiredWaypointTime = 8e3

        self.num_waypoints = len(self.waypoint_latitudes) * len(
            self.waypoint_longitudes
        )
        self.num_waypoint_actions = 8

        self.nominal_radius = 1.5 * 1600.0  # m
        self.body_radius = 800.0  # m

        # Define the cadence at which the DSN is available
        self.dsn_cadence = 3600 * 24  # s
        self.dsn_availability = step_duration

        self.current_tgt_index = 0

        self.storageLevel = 0

        self.collision = False

        self.waypoint_hist = None

        self.modeRequest = None

        # If no initial conditions are defined yet, set ICs
        if initial_conditions is None:
            self.initial_conditions = self.set_ic()
        # If ICs were passed through, use the ones that were passed through
        else:
            self.initial_conditions = initial_conditions
            self.phi_c = self.initial_conditions["phi_0"]
            self.lambda_c = self.initial_conditions["lambda_0"]
            self.last_waypoint = self.initial_conditions["r_BO_O"]
            self.current_waypoint = self.initial_conditions["r_BO_O"]
            self.waypoint_hist = [[self.phi_c, self.lambda_c]]

        # Set rendering
        self.render = render
        if self.render:
            self.setup_viz()
            self.clear_logs = False
        else:
            self.clear_logs = True

        self.DynamicsProcessName = "DynamicsProcess"  # Create Dynamics process name
        self.dynProc = None
        self.dynTaskName = "DynTask"  # Create dynamics task name
        self.dynTask = None
        self.FSWProcessName = "FSWProcess"  # Create a FSW process name
        self.fswProc = None
        self.mapTaskName = "MapTask"  # Create map task name
        self.mapTask = None

        # Define the names of each FSW Task
        self.sunPointTaskName = "sunPointTask"
        self.earthPointTaskName = "earthPointTask"
        self.locPointTaskName = "locPointTask"
        self.mapPointTaskName = "mapPointTask"
        self.mrpControlTaskName = "mrpControlTask"
        self.smallBodyFeedbackControlTaskName = "smallBodyFeedbackControlTask"

        return

    def init_tasks_and_processes(self):
        #   Initialize the dynamics and fsw task groups and modules
        self.init_dynamics_process()
        self.init_fsw_process()
        self.init_map_process()

        self.set_dynamics()
        self.set_gateway_msgs()
        self.set_fsw()
        self.init_obs()

        self.set_logging()

        self.InitializeSimulation()
        self.initialized = True

        return

    def init_dynamics_process(self):
        self.dynProc = self.CreateNewProcess(
            self.DynamicsProcessName, priority=3
        )  # Create the dynamics process
        self.dynTask = self.dynProc.addTask(
            self.CreateNewTask(self.dynTaskName, mc.sec2nano(self.dynRate)),
            taskPriority=4000,
        )

        return

    def init_fsw_process(self):
        self.fswProc = self.CreateNewProcess(
            self.FSWProcessName, priority=1
        )  # Create the fsw process

        return

    def init_map_process(self):
        self.mapTask = self.dynProc.addTask(
            self.CreateNewTask(self.mapTaskName, mc.sec2nano(self.mapRate)),
            taskPriority=3000,
        )

        return

    def set_ic(self):
        # Sample orbital parameters
        # rB, vB = leo_orbit.bennu_waypoint()

        # Create the set of ground targets for imaging
        # targets = leo_orbit.create_bennu_ground_tgts(self.n_targets)
        utc_init = "2018 DECEMBER 31 07:47:48.965 (UTC)"

        # Sample attitude and rates
        sigma_init, omega_init = sc_attitudes.random_tumble(maxSpinRate=0.00001)

        x_0_delta = np.zeros(12)
        x_0_delta[0:3] = np.random.uniform(-50, 50.0, 3)  # Relative s/c position
        x_0_delta[3:6] = np.random.uniform(-0.1, 0.1, 3)  # Relative s/c velocity
        x_0_delta[6:9] = np.random.uniform(-0.1, 0.1, 3)  # Small body attitude
        x_0_delta[9:12] = np.random.uniform(-0.1, 0.1, 3)  # Small body attitude rate

        mapping_points = small_body.generate_mapping_points(
            self.n_map_points, self.body_radius
        )
        imaging_targets = small_body.generate_imaging_points(
            self.n_targets, self.body_radius
        )

        if self.phi_c is None:
            self.phi_c = random.choice([75, 45, 15, -15, -45, -75])
            self.lambda_c = random.choice([-30, 30, 90])

        self.waypoint_hist = [[self.phi_c, self.lambda_c]]

        r_BO_O = np.array(
            [
                self.nominal_radius
                * m.sin(mc.D2R * (90 - self.phi_c))
                * m.cos(mc.D2R * self.lambda_c),
                self.nominal_radius
                * m.sin(mc.D2R * (90 - self.phi_c))
                * m.sin(mc.D2R * self.lambda_c),
                self.nominal_radius * m.cos(mc.D2R * (90 - self.phi_c)),
            ]
        )

        self.last_waypoint = np.copy(r_BO_O)
        self.current_waypoint = np.copy(r_BO_O)

        DSN_prescribed_times = np.zeros((32, 2))
        for dsn_idx in range(0, 32):
            DSN_prescribed_times[dsn_idx, 0] = (dsn_idx + 1) * (3600 * 24) - 1e4
            DSN_prescribed_times[dsn_idx, 1] = (dsn_idx + 1) * (3600 * 24)

        pos_sigma_sc = 40.0
        vel_sigma_sc = 0.05
        att_sigma_sc = 0.02 * m.pi / 180.0
        rate_sigma_sc = 0.001 * m.pi / 180.0
        sun_sigma_sc = 0.0
        dv_sigma_sc = 0.0

        pos_sigma_p = 0.0
        vel_sigma_p = 0.0
        att_sigma_p = 1.0 * m.pi / 180.0
        rate_sigma_p = 0.1 * m.pi / 180.0

        # Dict of initial conditions
        initial_conditions = {
            # Initialize the start time of the sim
            "utc_init": utc_init,
            "mu_bennu": 4.892,  # From Scheeres' 2019 paper
            # Mass
            "mass": 330,  # kg
            # Area
            "srp_area": 2,  # m^2
            # Waypoints
            "phi_0": self.phi_c,
            "lambda_0": self.lambda_c,
            # Spacecraft position and velocity relative to small body in Hill frame
            "r_BO_O": r_BO_O,  # m
            "v_BO_O": np.array([0.0, 0.0, 0.0]),  # m/s
            # Spacecraft dimensions
            "width": 1.38,
            "depth": 1.04,
            "height": 1.58,
            # Disturbance torque
            "disturbance_magnitude": 0.002,
            # "disturbance_vector": np.random.standard_normal(3),
            "disturbance_vector": np.array([-1, 0, 0]),
            # Attitude and rate initialization
            "sigma_init": sigma_init,
            "omega_init": omega_init,
            # Reaction Wheel speeds
            "wheelSpeeds": np.random.uniform(-2000 * mc.RPM, 2000 * mc.RPM, 3),  # rad/s
            "max_dV": 40,  # m/s
            # RW motor torque and thruster force mapping FSW config
            "controlAxes_B": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            # Attitude controller FSW config
            "K": 7,
            "Ki": -1.0,  # Note: make value negative to turn off integral feedback
            "P": 35,
            # Momentum dumping config
            "hs_min": 1.0,  # Nms
            # Thruster force mapping FSW module
            "thrForceSign": 1,
            # Thruster momentum dumping FSW config
            "maxCounterValue": 20,
            # "maxCounterValue": 8,
            "thrMinFireTime": 0.002,  # Seconds
            # "thrMinFireTime": 5.0,   # Seconds
            # Initial guess for the Nav EKF - delta from actual state
            "x_0_delta": x_0_delta,
            # DCM for tracking error data - rotates s/c so panels point at the sun
            "C_R0R": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1, 0.0, 0.0]]).T,
            # Map points in the body-fixed frame
            "mapping_points": mapping_points,
            # Imaging targets in the body-fixed frame
            "imaging_targets": imaging_targets,
            # Imaging Target requirements
            "imageTargetMinimumElevation": np.radians(45.0),
            "imageTargetMaximumRange": 1e9,
            # Transmitter
            "transmitterBaudRate": -120e6,  # 8 Mbits/s
            "mapInstrumentBaudRate": 8e6,  # 8 Mbits/s
            "targetInstrumentBaudRate": 8e6,  # 8 Mbits/s
            "dataStorageCapacity": 1e12,
            "imageAttErrorRequirement": 0.1,  # rad
            "groundLocationPlanetRadius": orbitalMotion.REQ_EARTH * 1000.0,
            "dsnPrescribedTimes": DSN_prescribed_times,
            "canberraGroundStationLat": np.radians(-35.401389),  # deg
            "canberraGroundStationLong": np.radians(148.981667),  # deg
            "canberraGroundStationAlt": 550,  # m
            "canberraMinimumElevation": np.radians(10),  # deg
            "canberraMaximumRange": 1e12,  # m
            "goldstoneGroundStationLat": np.radians(35.426667),  # deg
            "goldstoneGroundStationLong": np.radians(-116.89),  # deg
            "goldstoneGroundStationAlt": 900,  # m
            "goldstoneMinimumElevation": np.radians(10),  # deg
            "goldstoneMaximumRange": 1e12,  # m
            "madridGroundStationLat": np.radians(40.431389),  # deg
            "madridGroundStationLong": np.radians(-4.248056),  # deg
            "madridGroundStationAlt": 720,  # m
            "madridMinimumElevation": np.radians(10),  # deg
            "madridMaximumRange": 1e12,  # m
            # Solar Panel Parameters
            "nHat_B": np.array([0, 0, 1]),
            "panelArea": 2 * 1.0 * 0.5,
            "panelEfficiency": 0.20,
            # Power Sink Parameters
            "instrumentPowerDraw": -30.0,  # W, Assuming 30 W imager (Harris Spaceview)
            "transmitterPowerDraw": -15.0,  # W
            "rwBasePower": 0.4,  # W, Note the opposite convention
            "rwMechToElecEfficiency": 0.0,
            "rwElecToMechEfficiency": 0.5,
            # Battery Parameters
            "batteryStorageCapacity": 100.0 * 3600.0,
            "storedCharge_Init": np.random.uniform(50.0 * 3600.0, 100.0 * 3600.0, 1)[0],
            # Simple Nav Parameters
            "pos_sigma_sc": pos_sigma_sc,
            "vel_sigma_sc": vel_sigma_sc,
            "att_sigma_sc": att_sigma_sc,
            "rate_sigma_sc": rate_sigma_sc,
            "sun_sigma_sc": sun_sigma_sc,
            "dv_sigma_sc": dv_sigma_sc,
            "walk_bounds_sc": [
                [10.0],
                [10.0],
                [10.0],
                [0.01],
                [0.01],
                [0.01],
                [0.0005],
                [0.0005],
                [0.0005],
                [0.0002],
                [0.0002],
                [0.0002],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
            ],
            # Planet Nav Parameters
            "pos_sigma_p": pos_sigma_p,
            "vel_sigma_p": vel_sigma_p,
            "att_sigma_p": att_sigma_p,
            "rate_sigma_p": rate_sigma_p,
            "walk_bounds_p": [
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.005],
                [0.005],
                [0.005],
                [0.002],
                [0.002],
                [0.002],
            ],
            # Navigation parameters
            "R": np.diag(
                [
                    pos_sigma_sc**2,
                    pos_sigma_sc**2,
                    pos_sigma_sc**2,
                    vel_sigma_sc**2,
                    vel_sigma_sc**2,
                    vel_sigma_sc**2,
                    att_sigma_p**2,
                    att_sigma_p**2,
                    att_sigma_p**2,
                    rate_sigma_p**2,
                    rate_sigma_p**2,
                    rate_sigma_p**2,
                ]
            ),
            # "R": np.diag([pos_sigma_sc, pos_sigma_sc, pos_sigma_sc,
            #               vel_sigma_sc, vel_sigma_sc, vel_sigma_sc,
            #              att_sigma_p, att_sigma_p, att_sigma_p,
            #              rate_sigma_p, rate_sigma_p, rate_sigma_p]),
            "Q": np.diag(
                [1e-7, 1e-7, 1e-7, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
            ),
            "P_0": (0.1 * np.identity(12)),
        }

        initial_conditions["IHubPntBc_B"] = [
            1.0
            / 12.0
            * initial_conditions["mass"]
            * (initial_conditions["width"] ** 2.0 + initial_conditions["depth"] ** 2.0),
            0.0,
            0.0,
            0.0,
            1.0
            / 12.0
            * initial_conditions["mass"]
            * (
                initial_conditions["depth"] ** 2.0 + initial_conditions["height"] ** 2.0
            ),
            0.0,
            0.0,
            0.0,
            1.0
            / 12.0
            * initial_conditions["mass"]
            * (
                initial_conditions["width"] ** 2.0 + initial_conditions["height"] ** 2.0
            ),
        ]

        return initial_conditions

    def set_dynamics(self):
        """
        Calls each function to set the dynamics.
        :return:
        """
        self.set_spacecraft()
        self.set_grav_bodies()
        self.set_eclipse()
        self.set_power_system()
        self.set_simple_nav()
        self.set_planet_nav()
        self.set_reaction_wheels()
        self.set_control_force()
        self.set_disturbance_force()
        self.set_srp()
        self.set_ground_maps()
        self.set_imaging_target()
        self.set_data_system()
        self.set_dsn()
        self.set_dyn_models_to_tasks()

    def set_spacecraft(self):
        #   Spacecraft, Planet Setup
        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = "spacecraft"

        # Specify the vehicle configuration message to tell things what the vehicle
        # inertia is
        vehicleConfigOut = messaging.VehicleConfigMsgPayload()
        vehicleConfigOut.ISCPntB_B = self.initial_conditions.get("IHubPntBc_B")
        self.vcConfigMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

        return

    def set_grav_bodies(self):
        # Create the ephemeris data for the bodies
        # setup celestial object ephemeris module
        self.gravBodyEphem = planetEphemeris.PlanetEphemeris()
        self.gravBodyEphem.ModelTag = "planetEphemeris"
        self.gravBodyEphem.setPlanetNames(
            planetEphemeris.StringVector(["custom", "earth"])
        )

        # specify orbits of gravitational bodies
        # https://ssd.jpl.nasa.gov/horizons.cgi#results
        # December 31st, 2018
        self.oeAsteroid = planetEphemeris.ClassicElementsMsgPayload()
        self.oeAsteroid.a = 1.1259 * orbitalMotion.AU * 1000  # meters
        self.oeAsteroid.e = 0.20373
        self.oeAsteroid.i = 6.0343 * mc.D2R
        self.oeAsteroid.Omega = 2.01820 * mc.D2R
        self.oeAsteroid.omega = 66.304 * mc.D2R
        self.oeAsteroid.f = 346.32 * mc.D2R

        self.oeEarth = planetEphemeris.ClassicElementsMsgPayload()
        self.oeEarth.a = orbitalMotion.AU * 1000  # meters
        self.oeEarth.e = 0.016975
        self.oeEarth.i = 0.0027666 * mc.D2R
        self.oeEarth.Omega = 177.42 * mc.D2R
        self.oeEarth.omega = 284.26 * mc.D2R
        self.oeEarth.f = 357.30 * mc.D2R

        # Create a sun spice message, zero it out, required by srp
        self.sunPlanetStateMsgData = messaging.SpicePlanetStateMsgPayload()
        self.sunPlanetStateMsg = messaging.SpicePlanetStateMsg()
        self.sunPlanetStateMsg.write(self.sunPlanetStateMsgData)

        # Create a sun ephemeris message, zero it out, required by nav filter
        self.sunEphemerisMsgData = messaging.EphemerisMsgPayload()
        self.sunEphemerisMsg = messaging.EphemerisMsg()
        self.sunEphemerisMsg.write(self.sunEphemerisMsgData)

        # specify celestial object orbit
        self.gravBodyEphem.planetElements = planetEphemeris.classicElementVector(
            [self.oeAsteroid, self.oeEarth]
        )

        self.gravBodyEphem.rightAscension = planetEphemeris.DoubleVector(
            [0 * mc.D2R, 99.767 * mc.D2R]
        )
        self.gravBodyEphem.declination = planetEphemeris.DoubleVector(
            [90 * mc.D2R, 23.133 * mc.D2R]
        )
        self.gravBodyEphem.lst0 = planetEphemeris.DoubleVector(
            [0.0 * mc.D2R, 0.0 * mc.D2R]
        )
        self.gravBodyEphem.rotRate = planetEphemeris.DoubleVector(
            [360 * mc.D2R / (4.297461 * 3600.0), 360 * mc.D2R / (24.0 * 3600.0)]
        )

        # Create an ephemeris converter
        self.ephemConverter = ephemerisConverter.EphemerisConverter()
        self.ephemConverter.ModelTag = "ephemConverter"
        self.ephemConverter.addSpiceInputMsg(self.gravBodyEphem.planetOutMsgs[0])
        self.ephemConverter.addSpiceInputMsg(self.gravBodyEphem.planetOutMsgs[1])

        # clear prior gravitational body and SPICE setup definitions
        self.gravFactory = simIncludeGravBody.gravBodyFactory()

        self.sun = self.gravFactory.createSun()

        self.asteroid = self.gravFactory.createCustomGravObject(
            "Bennu", self.initial_conditions.get("mu_bennu")
        )
        self.asteroid.planetBodyInMsg.subscribeTo(self.gravBodyEphem.planetOutMsgs[0])

        self.earth = self.gravFactory.createEarth()
        self.earth.planetBodyInMsg.subscribeTo(self.gravBodyEphem.planetOutMsgs[1])

        # attach gravity model to spaceCraftPlus
        self.scObject.gravField.gravBodies = spacecraft.GravBodyVector(
            list(self.gravFactory.gravBodies.values())
        )

        r_ON_N, v_ON_N = orbitalMotion.elem2rv(
            orbitalMotion.MU_SUN * (1000.0**3), self.oeAsteroid
        )

        r_BN_N = np.add(
            r_ON_N,
            np.matmul(
                orbitalMotion.hillFrame(r_ON_N, v_ON_N).T,
                self.initial_conditions.get("r_BO_O"),
            ),
        )
        v_BN_N = np.add(
            v_ON_N,
            np.matmul(
                orbitalMotion.hillFrame(r_ON_N, v_ON_N).T,
                self.initial_conditions.get("v_BO_O"),
            ),
        )

        # Grab the mass for readability in inertia computation
        mass = self.initial_conditions.get("mass")

        self.scObject.hub.mHub = mass  # kg
        self.scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(
            self.initial_conditions.get("IHubPntBc_B")
        )

        self.scObject.hub.r_CN_NInit = unitTestSupport.np2EigenVectorXd(r_BN_N)
        self.scObject.hub.v_CN_NInit = unitTestSupport.np2EigenVectorXd(v_BN_N)

        sigma_init = self.initial_conditions.get("sigma_init")
        omega_init = self.initial_conditions.get("omega_init")

        self.scObject.hub.sigma_BNInit = sigma_init  # sigma_BN_B
        self.scObject.hub.omega_BN_BInit = omega_init

        return

    def set_eclipse(self):
        # Define the eclipse object
        self.eclipseObject = eclipse.Eclipse()
        self.eclipseObject.addSpacecraftToModel(self.scObject.scStateOutMsg)
        self.eclipseObject.addPlanetToModel(self.gravBodyEphem.planetOutMsgs[0])
        self.eclipseObject.rEqCustom = self.body_radius
        self.eclipseObject.sunInMsg.subscribeTo(self.sunPlanetStateMsg)

        return

    def set_power_system(self):
        # Power setup
        self.solarPanel = simpleSolarPanel.SimpleSolarPanel()
        self.solarPanel.ModelTag = "solarPanel"
        self.solarPanel.stateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        self.solarPanel.sunEclipseInMsg.subscribeTo(
            self.eclipseObject.eclipseOutMsgs[0]
        )
        self.solarPanel.sunInMsg.subscribeTo(self.sunPlanetStateMsg)
        self.solarPanel.setPanelParameters(
            unitTestSupport.np2EigenVectorXd(self.initial_conditions.get("nHat_B")),
            self.initial_conditions.get("panelArea"),
            self.initial_conditions.get("panelEfficiency"),
        )

        # Instrument power sink
        self.instrumentPowerSink = simplePowerSink.SimplePowerSink()
        self.instrumentPowerSink.ModelTag = "insPowerSink"
        self.instrumentPowerSink.nodePowerOut = self.initial_conditions.get(
            "instrumentPowerDraw"
        )  # Watts

        # Transmitter power sink
        self.transmitterPowerSink = simplePowerSink.SimplePowerSink()
        self.transmitterPowerSink.ModelTag = "transPowerSink"
        self.transmitterPowerSink.nodePowerOut = self.initial_conditions.get(
            "transmitterPowerDraw"
        )  # Watts

        # Battery
        self.powerMonitor = simpleBattery.SimpleBattery()
        self.powerMonitor.ModelTag = "powerMonitor"
        self.powerMonitor.storageCapacity = self.initial_conditions.get(
            "batteryStorageCapacity"
        )
        self.powerMonitor.storedCharge_Init = self.initial_conditions.get(
            "storedCharge_Init"
        )
        self.powerMonitor.addPowerNodeToModel(self.solarPanel.nodePowerOutMsg)
        self.powerMonitor.addPowerNodeToModel(self.instrumentPowerSink.nodePowerOutMsg)
        self.powerMonitor.addPowerNodeToModel(self.transmitterPowerSink.nodePowerOutMsg)

        return

    def set_simple_nav(self):
        # Set up simpleNav for s/c "measurements"
        self.simpleNavMeas = simpleNav.SimpleNav()
        self.simpleNavMeas.ModelTag = "SimpleNav"
        self.simpleNavMeas.scStateInMsg.subscribeTo(self.scObject.scStateOutMsg)

        return

    def set_planet_nav(self):
        # Set up planetNav for asteroid "measurements"
        self.planetNavMeas = planetNav.PlanetNav()
        self.planetNavMeas.ephemerisInMsg.subscribeTo(
            self.ephemConverter.ephemOutMsgs[0]
        )

        return

    def set_reaction_wheels(self):
        if self.fidelity == "high":
            # Add the reaction wheels to the spacecraft
            (
                self.rwStateEffector,
                self.rwFactory,
                initWheelSpeeds,
            ) = ap.balancedHR16Triad(
                useRandom=False,
                randomBounds=(-800, 800),
                wheelSpeeds=self.initial_conditions.get("wheelSpeeds"),
            )
            self.rwFactory.addToSpacecraft(
                "ReactionWheels", self.rwStateEffector, self.scObject
            )
            self.rwConfigMsg = self.rwFactory.getConfigMessage()

            # Reaction wheel power sinks
            self.rwPowerList = []
            for ind in range(self.rwFactory.getNumOfDevices()):
                powerRW = ReactionWheelPower.ReactionWheelPower()
                powerRW.ModelTag = self.scObject.ModelTag
                powerRW.basePowerNeed = self.initial_conditions.get(
                    "rwBasePower"
                )  # baseline power draw, Watts
                powerRW.rwStateInMsg.subscribeTo(self.rwStateEffector.rwOutMsgs[ind])
                powerRW.mechToElecEfficiency = self.initial_conditions.get(
                    "rwMechToElecEfficiency"
                )
                powerRW.elecToMechEfficiency = self.initial_conditions.get(
                    "rwElecToMechEfficiency"
                )
                self.AddModelToTask(
                    self.dynTaskName, powerRW, ModelPriority=(987 - ind)
                )
                self.rwPowerList.append(powerRW)
            for powerRW in self.rwPowerList:
                self.powerMonitor.addPowerNodeToModel(powerRW.nodePowerOutMsg)

        return

    def set_control_force(self):
        # Add the external force torque module for control
        self.extForceTorqueModule = extForceTorque.ExtForceTorque()
        self.scObject.addDynamicEffector(self.extForceTorqueModule)

        return

    def set_disturbance_force(self):
        # Add the external force torque module for disturbance torque
        disturbance_magnitude = self.initial_conditions.get("disturbance_magnitude")
        disturbance_vector = self.initial_conditions.get("disturbance_vector")
        unit_disturbance = disturbance_vector / np.linalg.norm(disturbance_vector)
        self.extForceTorqueModuleDisturbance = extForceTorque.ExtForceTorque()
        self.extForceTorqueModuleDisturbance.extTorquePntB_B = (
            disturbance_magnitude * unit_disturbance
        )
        self.extForceTorqueModuleDisturbance.ModelTag = "DisturbanceTorque"
        # self.scObject.addDynamicEffector(self.extForceTorqueModuleDisturbance)

        return

    def set_srp(self):
        # Add the SRP module
        self.srpModule = (
            radiationPressure.RadiationPressure()
        )  # default model is the SRP_CANNONBALL_MODEL
        self.srpModule.area = self.initial_conditions["srp_area"]  # m^3
        self.srpModule.coefficientReflection = 1.9
        self.scObject.addDynamicEffector(self.srpModule)
        self.srpModule.sunEphmInMsg.subscribeTo(self.sunPlanetStateMsg)

        return

    def set_ground_maps(self):
        # Add the groundMapping module for the first solar longitude
        self.groundMap1 = groundMapping.GroundMapping()
        self.groundMap1.ModelTag = "groundMapping1"
        for map_idx in range(self.n_map_points):
            self.groundMap1.addPointToModel(
                unitTestSupport.np2EigenVectorXd(
                    self.initial_conditions.get("mapping_points")[map_idx, :]
                )
            )
        self.groundMap1.minimumElevation = np.radians(45.0)
        self.groundMap1.maximumRange = 1e9
        self.groundMap1.cameraPos_B = [0, 0, 0]
        self.groundMap1.nHat_B = [0, 0, 1]
        self.groundMap1.halfFieldOfView = np.radians(22.5)
        self.groundMap1.solarLongitude = np.radians(self.map_longitudes[0])
        self.groundMap1.solarLongitudeTolerance = np.radians(self.longitude_tolerance)
        self.groundMap1.sunInMsg.subscribeTo(self.sunPlanetStateMsg)
        self.groundMap1.scStateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        self.groundMap1.planetInMsg.subscribeTo(self.gravBodyEphem.planetOutMsgs[0])

        # Add the groundMapping module for the second solar longitude
        self.groundMap2 = groundMapping.GroundMapping()
        self.groundMap2.ModelTag = "groundMapping2"
        for map_idx in range(self.n_map_points):
            self.groundMap2.addPointToModel(
                unitTestSupport.np2EigenVectorXd(
                    self.initial_conditions.get("mapping_points")[map_idx, :]
                )
            )
        self.groundMap2.minimumElevation = np.radians(45.0)
        self.groundMap2.maximumRange = 1e9
        self.groundMap2.cameraPos_B = [0, 0, 0]
        self.groundMap2.nHat_B = [0, 0, 1]
        self.groundMap2.halfFieldOfView = np.radians(22.5)
        self.groundMap2.solarLongitude = np.radians(self.map_longitudes[1])
        self.groundMap2.solarLongitudeTolerance = np.radians(self.longitude_tolerance)
        self.groundMap2.sunInMsg.subscribeTo(self.sunPlanetStateMsg)
        self.groundMap2.scStateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        self.groundMap2.planetInMsg.subscribeTo(self.gravBodyEphem.planetOutMsgs[0])

        # Add the groundMapping module for the third solar longitude
        self.groundMap3 = groundMapping.GroundMapping()
        self.groundMap3.ModelTag = "groundMapping3"
        for map_idx in range(self.n_map_points):
            self.groundMap3.addPointToModel(
                unitTestSupport.np2EigenVectorXd(
                    self.initial_conditions.get("mapping_points")[map_idx, :]
                )
            )
        self.groundMap3.minimumElevation = np.radians(45.0)
        self.groundMap3.maximumRange = 1e9
        self.groundMap3.cameraPos_B = [0, 0, 0]
        self.groundMap3.nHat_B = [0, 0, 1]
        self.groundMap3.halfFieldOfView = np.radians(22.5)
        self.groundMap3.solarLongitude = np.radians(self.map_longitudes[2])
        self.groundMap3.solarLongitudeTolerance = np.radians(self.longitude_tolerance)
        self.groundMap3.sunInMsg.subscribeTo(self.sunPlanetStateMsg)
        self.groundMap3.scStateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        self.groundMap3.planetInMsg.subscribeTo(self.gravBodyEphem.planetOutMsgs[0])

        return

    def set_imaging_target(self):
        # Add the groundLocation module
        self.imagingTarget = groundLocation.GroundLocation()
        self.imagingTarget.ModelTag = "ImagingTarget"
        self.imagingTarget.planetRadius = self.body_radius
        self.imagingTarget.specifyLocationPCPF(
            unitTestSupport.np2EigenVectorXd(
                self.initial_conditions.get("imaging_targets")[0, :]
            )
        )
        self.imagingTarget.planetInMsg.subscribeTo(self.gravBodyEphem.planetOutMsgs[0])
        self.imagingTarget.minimumElevation = self.initial_conditions.get(
            "imageTargetMinimumElevation"
        )
        self.imagingTarget.maximumRange = self.initial_conditions.get(
            "imageTargetMaximumRange"
        )
        self.imagingTarget.addSpacecraftToModel(self.scObject.scStateOutMsg)

        return

    def set_data_system(self):
        # Add the transmitter
        self.transmitter = spaceToGroundTransmitter.SpaceToGroundTransmitter()
        self.transmitter.ModelTag = "transmitter"
        self.transmitter.nodeBaudRate = self.initial_conditions.get(
            "transmitterBaudRate"
        )  # baud
        self.transmitter.packetSize = -self.initial_conditions.get(
            "mapInstrumentBaudRate"
        )  # bits
        self.transmitter.numBuffers = 1 + self.n_targets

        # Add the mappingInstrument module
        self.mapProgressInstrument = mappingInstrument.MappingInstrument()
        self.mapProgressInstrument.ModelTag = "mapProgressInstrument"
        self.mapProgressInstrument.nodeBaudRate = 1
        for map_idx in range(self.n_map_points):
            self.mapProgressInstrument.addMappingPoint(
                self.groundMap1.accessOutMsgs[map_idx], str(map_idx)
            )
        for map_idx in range(self.n_map_points, 2 * self.n_map_points):
            self.mapProgressInstrument.addMappingPoint(
                self.groundMap2.accessOutMsgs[map_idx - self.n_map_points], str(map_idx)
            )
        for map_idx in range(2 * self.n_map_points, 3 * self.n_map_points):
            self.mapProgressInstrument.addMappingPoint(
                self.groundMap3.accessOutMsgs[map_idx - 2 * self.n_map_points],
                str(map_idx),
            )

        # Add the simpleInstrument module for mapping
        self.mapInstrument = simpleInstrument.SimpleInstrument()
        self.mapInstrument.ModelTag = "mapInstrument"
        self.mapInstrument.nodeDataName = "mapInstrument"
        self.mapInstrument.nodeBaudRate = self.initial_conditions.get(
            "mapInstrumentBaudRate"
        )  # baud

        # Add the simpleInstrument module for target imaging
        self.targetInstrument = simpleInstrument.SimpleInstrument()
        self.targetInstrument.ModelTag = "targetInstrument"
        self.targetInstrument.nodeDataName = "0"
        self.targetInstrument.nodeBaudRate = self.initial_conditions.get(
            "targetInstrumentBaudRate"
        )  # baud

        # Add the partitionedStorageUnit module to keep track of map progress
        self.mappingStorageUnit = partitionedStorageUnit.PartitionedStorageUnit()
        self.mappingStorageUnit.ModelTag = "mappingStorageUnit"
        self.mappingStorageUnit.storageCapacity = 1e12
        self.mappingStorageUnit.useNameIndex = True
        for map_idx in range(self.n_maps * self.n_map_points):
            self.mappingStorageUnit.addDataNodeToModel(
                self.mapProgressInstrument.dataNodeOutMsgs[map_idx]
            )
            self.mappingStorageUnit.addPartition(str(map_idx))

        # Add the second partitionedStorageUnit module for map data and ground target
        # data
        self.dataStorageUnit = partitionedStorageUnit.PartitionedStorageUnit()
        self.dataStorageUnit.ModelTag = "dataStorageUnit"
        self.dataStorageUnit.storageCapacity = self.initial_conditions.get(
            "dataStorageCapacity"
        )  # bits (1 GB)
        self.dataStorageUnit.addDataNodeToModel(self.mapInstrument.nodeDataOutMsg)
        self.dataStorageUnit.addDataNodeToModel(self.targetInstrument.nodeDataOutMsg)
        self.dataStorageUnit.addDataNodeToModel(self.transmitter.nodeDataOutMsg)
        self.transmitter.addStorageUnitToTransmitter(
            self.dataStorageUnit.storageUnitDataOutMsg
        )
        # Add the partitions
        self.dataStorageUnit.addPartition("mapInstrument")
        for idx in range(self.n_targets):
            self.dataStorageUnit.addPartition(str(idx))

        return

    def set_dsn(self):
        # Create the Canberra DSN station
        self.canberraGroundStation = groundLocation.GroundLocation()
        self.canberraGroundStation.ModelTag = "GroundStation1"
        self.canberraGroundStation.planetRadius = self.initial_conditions.get(
            "groundLocationPlanetRadius"
        )
        self.canberraGroundStation.specifyLocation(
            self.initial_conditions.get("canberraGroundStationLat"),
            self.initial_conditions.get("canberraGroundStationLong"),
            self.initial_conditions.get("canberraGroundStationAlt"),
        )
        self.canberraGroundStation.planetInMsg.subscribeTo(
            self.gravBodyEphem.planetOutMsgs[1]
        )
        self.canberraGroundStation.minimumElevation = self.initial_conditions.get(
            "canberraMinimumElevation"
        )
        self.canberraGroundStation.maximumRange = self.initial_conditions.get(
            "canberraMaximumRange"
        )
        self.canberraGroundStation.addSpacecraftToModel(self.scObject.scStateOutMsg)
        self.canberraGroundStation.accessTimes = (
            self.initial_conditions["dsnPrescribedTimes"]
        ).tolist()
        self.transmitter.addAccessMsgToTransmitter(
            self.canberraGroundStation.accessOutMsgs[-1]
        )

        # Create the Goldstone DSN station
        self.goldstoneGroundStation = groundLocation.GroundLocation()
        self.goldstoneGroundStation.ModelTag = "GroundStation2"
        self.goldstoneGroundStation.planetRadius = self.initial_conditions.get(
            "groundLocationPlanetRadius"
        )
        self.goldstoneGroundStation.specifyLocation(
            self.initial_conditions.get("goldstoneGroundStationLat"),
            self.initial_conditions.get("goldstoneGroundStationLong"),
            self.initial_conditions.get("goldstoneGroundStationAlt"),
        )
        self.goldstoneGroundStation.planetInMsg.subscribeTo(
            self.gravBodyEphem.planetOutMsgs[1]
        )
        self.goldstoneGroundStation.minimumElevation = self.initial_conditions.get(
            "goldstoneMinimumElevation"
        )
        self.goldstoneGroundStation.maximumRange = self.initial_conditions.get(
            "goldstoneMaximumRange"
        )
        self.goldstoneGroundStation.addSpacecraftToModel(self.scObject.scStateOutMsg)
        self.goldstoneGroundStation.accessTimes = self.initial_conditions[
            "dsnPrescribedTimes"
        ].tolist()
        self.transmitter.addAccessMsgToTransmitter(
            self.goldstoneGroundStation.accessOutMsgs[-1]
        )

        # Create the Madrid DSN station
        self.madridGroundStation = groundLocation.GroundLocation()
        self.madridGroundStation.ModelTag = "GroundStation3"
        self.madridGroundStation.planetRadius = self.initial_conditions.get(
            "groundLocationPlanetRadius"
        )
        self.madridGroundStation.specifyLocation(
            self.initial_conditions.get("madridGroundStationLat"),
            self.initial_conditions.get("madridGroundStationLong"),
            self.initial_conditions.get("madridGroundStationAlt"),
        )
        self.madridGroundStation.planetInMsg.subscribeTo(
            self.gravBodyEphem.planetOutMsgs[1]
        )
        self.madridGroundStation.minimumElevation = self.initial_conditions.get(
            "madridMinimumElevation"
        )
        self.madridGroundStation.maximumRange = self.initial_conditions.get(
            "madridMaximumRange"
        )
        self.madridGroundStation.addSpacecraftToModel(self.scObject.scStateOutMsg)
        self.madridGroundStation.accessTimes = self.initial_conditions[
            "dsnPrescribedTimes"
        ].tolist()
        self.transmitter.addAccessMsgToTransmitter(
            self.madridGroundStation.accessOutMsgs[-1]
        )

        return

    def set_dyn_models_to_tasks(self):
        self.AddModelToTask(
            self.dynTaskName, self.extForceTorqueModule, ModelPriority=4000
        )
        self.AddModelToTask(self.dynTaskName, self.gravBodyEphem, ModelPriority=3000)
        self.AddModelToTask(self.dynTaskName, self.srpModule, ModelPriority=3000)
        self.AddModelToTask(
            self.dynTaskName, self.canberraGroundStation, ModelPriority=1000
        )
        self.AddModelToTask(
            self.dynTaskName, self.goldstoneGroundStation, ModelPriority=1000
        )
        self.AddModelToTask(
            self.dynTaskName, self.madridGroundStation, ModelPriority=1000
        )
        self.AddModelToTask(self.dynTaskName, self.scObject, ModelPriority=1000)
        if self.fidelity == "high":
            self.AddModelToTask(
                self.dynTaskName, self.rwStateEffector, ModelPriority=997
            )
        self.AddModelToTask(self.dynTaskName, self.ephemConverter, ModelPriority=996)
        self.AddModelToTask(self.dynTaskName, self.simpleNavMeas, ModelPriority=995)
        self.AddModelToTask(self.dynTaskName, self.planetNavMeas, ModelPriority=995)

        self.AddModelToTask(self.mapTaskName, self.groundMap1, ModelPriority=1)
        self.AddModelToTask(self.mapTaskName, self.groundMap2, ModelPriority=1)
        self.AddModelToTask(self.mapTaskName, self.groundMap3, ModelPriority=1)
        self.AddModelToTask(self.dynTaskName, self.imagingTarget, ModelPriority=100)
        self.AddModelToTask(self.dynTaskName, self.transmitter, ModelPriority=100)
        self.AddModelToTask(
            self.mapTaskName, self.mapProgressInstrument, ModelPriority=100
        )
        self.AddModelToTask(self.dynTaskName, self.mapInstrument, ModelPriority=100)
        self.AddModelToTask(self.dynTaskName, self.targetInstrument, ModelPriority=100)
        self.AddModelToTask(
            self.mapTaskName, self.mappingStorageUnit, ModelPriority=100
        )
        self.AddModelToTask(self.dynTaskName, self.dataStorageUnit, ModelPriority=100)
        self.AddModelToTask(self.dynTaskName, self.eclipseObject, ModelPriority=988)
        self.AddModelToTask(self.dynTaskName, self.solarPanel, ModelPriority=898)
        self.AddModelToTask(
            self.dynTaskName, self.instrumentPowerSink, ModelPriority=897
        )
        self.AddModelToTask(
            self.dynTaskName, self.transmitterPowerSink, ModelPriority=896
        )
        self.AddModelToTask(self.dynTaskName, self.powerMonitor, ModelPriority=799)

        return

    def set_fsw(self):
        self.create_fsw_tasks()
        self.init_fsw_tasks()
        self.set_fsw_tasks()
        self.set_fsw_models_to_tasks()

        return

    def init_fsw_tasks(self):
        self.init_target_pointing()
        self.init_instrument_controller()
        self.init_map_pointing()
        self.init_earth_pointing()
        self.init_sun_pointing()
        self.init_attitude_pointing()
        self.init_waypoint_feedback()

        return

    def set_fsw_tasks(self):
        self.set_target_pointing()
        self.set_instrument_controller()
        self.set_map_pointing()
        self.set_earth_pointing()
        self.set_sun_pointing()
        self.set_attitude_pointing()
        self.set_waypoint_feedback()

        return

    def create_fsw_tasks(self):
        self.processTasksTimeStep = mc.sec2nano(self.fswRate)
        self.fswProc.addTask(
            self.CreateNewTask(self.sunPointTaskName, self.processTasksTimeStep),
            taskPriority=99,
        )
        self.fswProc.addTask(
            self.CreateNewTask(self.earthPointTaskName, self.processTasksTimeStep),
            taskPriority=99,
        )
        self.fswProc.addTask(
            self.CreateNewTask(self.locPointTaskName, self.processTasksTimeStep),
            taskPriority=98,
        )
        self.fswProc.addTask(
            self.CreateNewTask(self.mapPointTaskName, self.processTasksTimeStep),
            taskPriority=98,
        )
        if self.fidelity == "high":
            self.fswProc.addTask(
                self.CreateNewTask(self.mrpControlTaskName, self.processTasksTimeStep),
                taskPriority=96,
            )
        self.fswProc.addTask(
            self.CreateNewTask(
                self.smallBodyFeedbackControlTaskName, self.processTasksTimeStep
            ),
            taskPriority=94,
        )

        return

    def init_target_pointing(self):
        # Location pointing configuration for target imaging
        self.locPointConfig = locationPointing.locationPointingConfig()
        self.locPointWrap = self.setModelDataWrap(self.locPointConfig)
        self.locPointWrap.ModelTag = "locPoint"
        cMsgPy.AttGuidMsg_C_addAuthor(
            self.locPointConfig.attGuidOutMsg, self.attGuidMsg
        )
        cMsgPy.AttRefMsg_C_addAuthor(self.locPointConfig.attRefOutMsg, self.attRefMsg)
        self.locPointConfig.pHat_B = [0, 0, 1]
        self.locPointConfig.useBoresightRateDamping = 1

        return

    def set_target_pointing(self):
        self.locPointConfig.scAttInMsg.subscribeTo(self.simpleNavMeas.attOutMsg)
        self.locPointConfig.scTransInMsg.subscribeTo(self.simpleNavMeas.transOutMsg)
        self.locPointConfig.locationInMsg.subscribeTo(
            self.imagingTarget.currentGroundStateOutMsg
        )

        return

    def init_instrument_controller(self):
        # setup the simpleInstrumentController module
        self.simpleInsControlConfig = (
            simpleInstrumentController.simpleInstrumentControllerConfig()
        )
        self.simpleInsControlConfig.attErrTolerance = self.initial_conditions.get(
            "imageAttErrorRequirement"
        )
        self.simpleInsControlWrap = self.setModelDataWrap(self.simpleInsControlConfig)
        self.simpleInsControlWrap.ModelTag = "instrumentController"

        return

    def set_instrument_controller(self):
        self.simpleInsControlConfig.attGuidInMsg.subscribeTo(self.attGuidMsg)
        self.simpleInsControlConfig.locationAccessInMsg.subscribeTo(
            self.imagingTarget.accessOutMsgs[-1]
        )
        self.targetInstrument.nodeStatusInMsg.subscribeTo(
            self.simpleInsControlConfig.deviceCmdOutMsg
        )

        return

    def init_map_pointing(self):
        # Location pointing configuration for mapping
        self.mapPointConfig = locationPointing.locationPointingConfig()
        self.mapPointWrap = self.setModelDataWrap(self.mapPointConfig)
        self.mapPointWrap.ModelTag = "mapPoint"
        cMsgPy.AttGuidMsg_C_addAuthor(
            self.mapPointConfig.attGuidOutMsg, self.attGuidMsg
        )
        cMsgPy.AttRefMsg_C_addAuthor(self.mapPointConfig.attRefOutMsg, self.attRefMsg)
        self.mapPointConfig.pHat_B = [0, 0, 1]
        self.mapPointConfig.useBoresightRateDamping = 1

        return

    def set_map_pointing(self):
        self.mapPointConfig.scAttInMsg.subscribeTo(self.simpleNavMeas.attOutMsg)
        self.mapPointConfig.scTransInMsg.subscribeTo(self.simpleNavMeas.transOutMsg)
        self.mapPointConfig.celBodyInMsg.subscribeTo(
            self.ephemConverter.ephemOutMsgs[0]
        )

        return

    def init_earth_pointing(self):
        self.earthPointConfig = locationPointing.locationPointingConfig()
        self.earthPointWrap = self.setModelDataWrap(self.earthPointConfig)
        self.earthPointWrap.ModelTag = "earthPoint"
        cMsgPy.AttGuidMsg_C_addAuthor(
            self.earthPointConfig.attGuidOutMsg, self.attGuidMsg
        )
        cMsgPy.AttRefMsg_C_addAuthor(self.earthPointConfig.attRefOutMsg, self.attRefMsg)
        self.earthPointConfig.pHat_B = [0, 0, 1]
        self.earthPointConfig.useBoresightRateDamping = 1

        return

    def set_earth_pointing(self):
        self.earthPointConfig.scAttInMsg.subscribeTo(self.simpleNavMeas.attOutMsg)
        self.earthPointConfig.scTransInMsg.subscribeTo(self.simpleNavMeas.transOutMsg)
        self.earthPointConfig.celBodyInMsg.subscribeTo(
            self.ephemConverter.ephemOutMsgs[1]
        )

        return

    def init_sun_pointing(self):
        self.sunPointConfig = locationPointing.locationPointingConfig()
        self.sunPointWrap = self.setModelDataWrap(self.sunPointConfig)
        self.sunPointWrap.ModelTag = "sunPoint"
        cMsgPy.AttGuidMsg_C_addAuthor(
            self.sunPointConfig.attGuidOutMsg, self.attGuidMsg
        )
        cMsgPy.AttRefMsg_C_addAuthor(self.sunPointConfig.attRefOutMsg, self.attRefMsg)
        self.sunPointConfig.pHat_B = self.initial_conditions.get("nHat_B")
        self.sunPointConfig.useBoresightRateDamping = 1

        return

    def set_sun_pointing(self):
        self.sunPointConfig.scAttInMsg.subscribeTo(self.simpleNavMeas.attOutMsg)
        self.sunPointConfig.scTransInMsg.subscribeTo(self.simpleNavMeas.transOutMsg)
        self.sunPointConfig.celBodyInMsg.subscribeTo(self.sunEphemerisMsg)

        return

    def init_attitude_pointing(self):
        if self.fidelity == "high":
            #   Attitude controller configuration
            self.mrpFeedbackControlData = mrpFeedback.mrpFeedbackConfig()
            self.mrpFeedbackControlWrap = self.setModelDataWrap(
                self.mrpFeedbackControlData
            )
            self.mrpFeedbackControlWrap.ModelTag = "mrpFeedbackControl"
            self.mrpFeedbackControlData.K = self.initial_conditions.get("K")
            self.mrpFeedbackControlData.Ki = self.initial_conditions.get("Ki")
            self.mrpFeedbackControlData.P = self.initial_conditions.get("P")
            self.mrpFeedbackControlData.integralLimit = (
                2.0 / self.mrpFeedbackControlData.Ki * 0.1
            )

            # add module that maps the Lr control torque into the RW motor torques
            self.rwMotorTorqueConfig = rwMotorTorque.rwMotorTorqueConfig()
            self.rwMotorTorqueWrap = self.setModelDataWrap(self.rwMotorTorqueConfig)
            self.rwMotorTorqueWrap.ModelTag = "rwMotorTorque"
            self.rwMotorTorqueConfig.controlAxes_B = self.initial_conditions.get(
                "controlAxes_B"
            )
        else:
            self.scObject.attRefInMsg.subscribeTo(self.attRefMsg)

        return

    def set_attitude_pointing(self):
        if self.fidelity == "high":
            #   Attitude controller configuration
            self.mrpFeedbackControlData.guidInMsg.subscribeTo(self.attGuidMsg)
            self.mrpFeedbackControlData.vehConfigInMsg.subscribeTo(self.vcConfigMsg)

            # add module that maps the Lr control torque into the RW motor torques
            self.rwStateEffector.rwMotorCmdInMsg.subscribeTo(
                self.rwMotorTorqueConfig.rwMotorTorqueOutMsg
            )
            self.rwMotorTorqueConfig.rwParamsInMsg.subscribeTo(self.rwConfigMsg)
            self.rwMotorTorqueConfig.vehControlInMsg.subscribeTo(
                self.mrpFeedbackControlData.cmdTorqueOutMsg
            )
            self.rwStateEffector.rwMotorCmdInMsg.subscribeTo(
                self.rwMotorTorqueConfig.rwMotorTorqueOutMsg
            )

        return

    def init_waypoint_feedback(self):
        # Add the waypoint feedback module
        self.waypointFeedback = smallBodyWaypointFeedback.SmallBodyWaypointFeedback()
        self.waypointFeedback.A_sc = self.initial_conditions[
            "srp_area"
        ]  # Surface area of the spacecraft, m^2
        self.waypointFeedback.M_sc = self.initial_conditions[
            "mass"
        ]  # Mass of the spacecraft, kg
        self.waypointFeedback.IHubPntC_B = unitTestSupport.np2EigenMatrix3d(
            self.initial_conditions["IHubPntBc_B"]
        )  # sc inertia
        self.waypointFeedback.mu_ast = self.initial_conditions[
            "mu_bennu"
        ]  # Gravitational constant of the asteroid
        self.waypointFeedback.x1_ref = self.initial_conditions["r_BO_O"].tolist()
        self.waypointFeedback.x2_ref = [0.0, 0.0, 0.0]
        self.waypointFeedback.K1 = unitTestSupport.np2EigenMatrix3d(
            [5e-4, 0e-5, 0e-5, 0e-5, 5e-4, 0e-5, 0e-5, 0e-5, 5e-4]
        )
        self.waypointFeedback.K2 = unitTestSupport.np2EigenMatrix3d(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        )

    def set_waypoint_feedback(self):
        self.waypointFeedback.asteroidEphemerisInMsg.subscribeTo(
            self.planetNavMeas.ephemerisOutMsg
        )
        self.waypointFeedback.sunEphemerisInMsg.subscribeTo(self.sunEphemerisMsg)
        self.waypointFeedback.navAttInMsg.subscribeTo(self.simpleNavMeas.attOutMsg)
        self.waypointFeedback.navTransInMsg.subscribeTo(self.simpleNavMeas.transOutMsg)
        self.extForceTorqueModule.cmdForceBodyInMsg.subscribeTo(
            self.waypointFeedback.forceOutMsg
        )

    def set_fsw_models_to_tasks(self):
        #   Add models to tasks
        self.AddModelToTask(
            "sunPointTask", self.sunPointWrap, self.sunPointConfig, ModelPriority=1200
        )
        self.AddModelToTask(
            "earthPointTask",
            self.earthPointWrap,
            self.earthPointConfig,
            ModelPriority=1200,
        )
        self.AddModelToTask(
            self.locPointTaskName,
            self.locPointWrap,
            self.locPointConfig,
            ModelPriority=1200,
        )
        self.AddModelToTask(
            self.locPointTaskName,
            self.simpleInsControlWrap,
            self.simpleInsControlConfig,
            ModelPriority=987,
        )
        self.AddModelToTask(
            "mapPointTask", self.mapPointWrap, self.mapPointConfig, ModelPriority=1200
        )
        if self.fidelity == "high":
            self.AddModelToTask(
                "mrpControlTask",
                self.mrpFeedbackControlWrap,
                self.mrpFeedbackControlData,
                ModelPriority=1198,
            )
            self.AddModelToTask(
                "mrpControlTask",
                self.rwMotorTorqueWrap,
                self.rwMotorTorqueConfig,
                ModelPriority=1196,
            )
        self.AddModelToTask(
            "smallBodyFeedbackControlTask", self.waypointFeedback, ModelPriority=1300
        )

        return

    def init_obs(self):
        # Construct the observations
        self.obs[0:3, 0] = (
            self.initial_conditions["r_BO_O"] / self.nominal_radius
        )  # Hill-frame position, normalized
        self.obs[3:6, 0] = self.initial_conditions["v_BO_O"]  # Hill-frame velocity
        self.obs[8, 0] = (
            self.initial_conditions["storedCharge_Init"]
            / self.initial_conditions["batteryStorageCapacity"]
        )  # Power storage level, normalized
        self.obs[11:14, 0] = (
            self.current_waypoint.flatten() / self.nominal_radius
        )  # Current waypoint reference, normalized
        self.obs[14:17, 0] = (
            self.last_waypoint.flatten() / self.nominal_radius
        )  # Last waypoint reference, normalized

        return

    def set_logging(self):
        self.scRec = self.scObject.scStateOutMsg.recorder()
        self.astRec = self.gravBodyEphem.planetOutMsgs[0].recorder()
        self.forceRequestRec = self.waypointFeedback.forceOutMsg.recorder()
        self.canberraRecorder = self.canberraGroundStation.accessOutMsgs[-1].recorder()
        self.goldstoneRecorder = self.goldstoneGroundStation.accessOutMsgs[
            -1
        ].recorder()
        self.madridRecorder = self.madridGroundStation.accessOutMsgs[-1].recorder()
        if self.fidelity == "high":
            self.wheelSpeedRecorder = self.rwStateEffector.rwSpeedOutMsg.recorder()
        self.powerRecorder = self.powerMonitor.batPowerOutMsg.recorder()

        self.AddModelToTask(self.dynTaskName, self.scRec)
        self.AddModelToTask(self.dynTaskName, self.astRec)
        self.AddModelToTask(self.dynTaskName, self.forceRequestRec)
        self.AddModelToTask(self.dynTaskName, self.canberraRecorder)
        self.AddModelToTask(self.dynTaskName, self.goldstoneRecorder)
        self.AddModelToTask(self.dynTaskName, self.madridRecorder)
        if self.fidelity == "high":
            self.AddModelToTask(self.dynTaskName, self.wheelSpeedRecorder)
        self.AddModelToTask(self.dynTaskName, self.powerRecorder)

        return

    def setup_viz(self):
        from datetime import datetime

        fileName = f"small_body_science_env-v1_{datetime.today()}"  # noqa: F841

        self.vizInterface = vizSupport.enableUnityVisualization(
            self,
            self.dynTaskName,
            self.scObject,
            # , saveFile=fileName
        )
        self.vizInterface.settings.showSpacecraftLabels = 1
        # load CAD for custom gravity model
        vizSupport.createCustomModel(
            self.vizInterface,
            modelPath="/Users/adamherrmann/Documents/AVS/bennu/Bennu_v20_200k.obj",
            shader=1,
            simBodiesToModify=["Bennu"],
            scale=[1000.0, 1000.0, 1000.0],
        )

        return

    def set_gateway_msgs(self):
        """create C-wrapped gateway messages such that different modules can write to
        this message
        and provide a common input msg for down-stream modules"""
        self.attGuidMsg = cMsgPy.AttGuidMsg_C()
        self.attRefMsg = cMsgPy.AttRefMsg_C()
        return

    def zero_gateway_msgs(self):
        """Zero all the FSW gateway message payloads"""
        self.attGuidMsg.write(messaging.AttGuidMsgPayload())
        self.attRefMsg.write(messaging.AttRefMsgPayload())
        return

    def run_sim(self, action):
        # Turn mode request into a string
        self.modeRequest = str(action)

        # Set the sim_over param to false
        self.sim_over = False

        currentResetTime = mc.sec2nano(self.simTime)  # noqa F841; Unsure if used

        self.turn_on_off_models()

        # Increment time and the current step
        self.simTime += self.step_duration
        simulation_time = mc.sec2nano(self.simTime)
        self.curr_step += 1

        #   Execute the sim
        self.ConfigureStopTime(simulation_time)
        self.ExecuteSimulation()

        return self.get_obs()

    def turn_on_off_models(self):
        self.dynProc.enableAllTasks()
        if self.modeRequest == "0":
            self.charging_mode()
        elif 1 <= int(self.modeRequest) < 1 + self.num_waypoint_actions:
            self.waypoint_mode()
        elif int(self.modeRequest) == 1 + self.num_waypoint_actions:
            self.mapping_mode()
        elif int(self.modeRequest) == 2 + self.num_waypoint_actions:
            self.communication_mode()
        elif int(self.modeRequest) == 3 + self.num_waypoint_actions:
            self.targeting_mode()

        return

    def charging_mode(self):
        self.step_duration = self.charge_duration
        # Turn off imaging, mapping
        self.fswProc.disableAllTasks()
        self.disable_mapping()
        self.disable_transmitter()
        self.disable_imaging()
        # Turn on sun pointing and MRP control
        self.enableTask(self.sunPointTaskName)
        self.enableTask(self.mrpControlTaskName)
        self.enableTask(self.smallBodyFeedbackControlTaskName)

        return

    def waypoint_mode(self):
        self.step_duration = self.maneuver_duration
        # Turn off imaging, mapping
        self.fswProc.disableAllTasks()
        self.disable_mapping()
        self.disable_transmitter()
        self.disable_imaging()
        # Set the waypoint
        if (self.simTime - self.waypointTime) >= self.requiredWaypointTime:
            self.phi_c = (
                self.phi_c + self.waypoint_latitude_deltas[int(self.modeRequest) - 1]
            )
            self.lambda_c = (
                self.lambda_c
                + self.waypoint_longitude_deltas[int(self.modeRequest) - 1]
            )
            self.wrap_phi()
            self.wrap_lambda()
            self.waypoint_hist.append([self.phi_c, self.lambda_c])
            self.last_waypoint = np.copy(self.waypointFeedback.x1_ref)
            self.waypointFeedback.x1_ref = [
                self.nominal_radius
                * m.sin(mc.D2R * (90 - self.phi_c))
                * m.cos(mc.D2R * self.lambda_c),
                self.nominal_radius
                * m.sin(mc.D2R * (90 - self.phi_c))
                * m.sin(mc.D2R * self.lambda_c),
                self.nominal_radius * m.cos(mc.D2R * (90 - self.phi_c)),
            ]
            self.waypointFeedback.x2_ref = [0.0, 0.0, 0.0]
            self.current_waypoint = np.copy(self.waypointFeedback.x1_ref)
            self.waypointTime = np.copy(self.simTime)
        # Turn on sun pointing, MRP control, and feedback control
        self.enableTask(self.sunPointTaskName)
        self.enableTask(self.mrpControlTaskName)
        self.enableTask(self.smallBodyFeedbackControlTaskName)

        return

    def mapping_mode(self):
        self.step_duration = self.map_duration
        # Turn off target imaging
        self.fswProc.disableAllTasks()
        self.targetInstrument.dataStatus = 0
        self.disable_transmitter()
        # Turn on mapping imaging
        self.enableTask(self.mapPointTaskName)
        self.mapProgressInstrument.nodeBaudRate = 1
        self.mapInstrument.dataStatus = 1
        self.instrumentPowerSink.powerStatus = 1
        # Enable MRP, waypoint feedback control tasks
        self.enableTask(self.mrpControlTaskName)
        self.enableTask(self.smallBodyFeedbackControlTaskName)

        return

    def communication_mode(self):
        self.step_duration = self.downlink_duration
        # Turn off target and mapping imaging
        self.fswProc.disableAllTasks()
        self.disable_mapping()
        self.disable_imaging()
        # Turn on transmitter
        self.transmitter.dataStatus = 1
        self.transmitterPowerSink.powerStatus = 1
        # Enable MRP, waypoint feedback control tasks
        self.enableTask(self.earthPointTaskName)
        self.enableTask(self.mrpControlTaskName)
        self.enableTask(self.smallBodyFeedbackControlTaskName)

        return

    def targeting_mode(self):
        self.step_duration = self.image_duration
        # Turn off mapping, downlink
        self.disable_mapping()
        self.disable_transmitter()
        self.fswProc.disableAllTasks()
        # Turn on target imaging
        self.enableTask(self.locPointTaskName)
        self.targetInstrument.dataStatus = 1
        self.instrumentPowerSink.powerStatus = 1
        # Set up the target
        self.imagingTarget.r_LP_P_Init = self.initial_conditions["imaging_targets"][
            self.current_tgt_index, :
        ]
        self.targetInstrument.nodeDataName = str(int(self.current_tgt_index))
        self.simpleInsControlConfig.imaged = 0
        self.enableTask(self.smallBodyFeedbackControlTaskName)
        self.enableTask(self.mrpControlTaskName)

        return

    def disable_mapping(self):
        self.disableTask(self.mapTaskName)
        self.mapProgressInstrument.nodeBaudRate = 0
        self.mapInstrument.dataStatus = 0

    def disable_transmitter(self):
        self.transmitter.dataStatus = 0
        self.transmitterPowerSink.powerStatus = 0

    def disable_imaging(self):
        self.targetInstrument.dataStatus = 0
        self.instrumentPowerSink.powerStatus = 0

    def get_obs(self):
        self.get_eclipse()
        self.get_delta_v()
        self.get_spacecraft_state()
        self.get_asteroid_state()
        self.get_data_state()
        self.get_power_state()
        self.get_dsn_state()
        self.get_map_state()
        self.get_nearest_target()
        self.check_collision()
        self.get_image_target_state()
        self.check_new_mapping_and_imaging()
        self.get_map_region_states()

        # Construct the observations
        self.obs[0:3, 0] = (
            self.r_BO_O / self.nominal_radius
        )  # Hill-frame position, normalized
        self.obs[3:6, 0] = self.v_BO_O  # Hill-frame velocity
        self.obs[6, 0] = self.eclipse
        self.obs[7, 0] = (
            self.storageLevel / self.initial_conditions["dataStorageCapacity"]
        )  # Data buffer level, normalized
        self.obs[8, 0] = (
            self.powerLevel[-1] / self.initial_conditions["batteryStorageCapacity"]
        )  # Power storage level, normalized
        self.obs[9, 0] = self.dV / self.initial_conditions["max_dV"]  # Fuel consumption
        self.obs[10, 0] = self.downlink_state  # Downlink state
        self.obs[11:14, 0] = (
            self.current_waypoint.flatten() / self.nominal_radius
        )  # Current waypoint reference, normalized
        self.obs[14:17, 0] = (
            self.last_waypoint.flatten() / self.nominal_radius
        )  # Last waypoint reference, normalized
        self.obs[17, 0] = (
            np.sum(self.imaged_targets) / self.n_targets
        )  # Percent of ground targets imaged
        self.obs[18, 0] = (
            np.sum(self.downlinked_targets) / self.n_targets
        )  # Percent of ground targets downlinked
        self.obs[19:22, 0] = (
            self.current_tgt_O / self.body_radius
        )  # Location of the next target for imaging
        self.obs[22:31, 0] = self.map_regions

        self.clear_logging()

        return (
            self.obs,
            self.sim_over,
            self.obs_full,
            self.new_downlinked_images,
            self.new_downlinked_maps,
            self.new_imaged,
            self.new_mapped,
        )

    def get_eclipse(self):
        self.eclipse = (
            self.eclipseObject.eclipseOutMsgs[-1].read().shadowFactor
        )  # Eclipse indicator, normalized

    def get_delta_v(self):
        # Pull the force request messages
        force_request = self.forceRequestRec.forceRequestBody[
            -int(self.step_duration / self.dynRate) :
        ]

        # Loop through each to compute the addition to the deltaV
        # Assumes constant mass - will work out thruster details later
        for force in force_request:
            self.dV += (
                self.dynRate * np.linalg.norm(force) / self.initial_conditions["mass"]
            )

        return

    def get_spacecraft_state(self):
        self.r_BN_N = self.scRec.r_BN_N[-int(self.step_duration / self.dynRate) :]
        self.v_BN_N = self.scRec.v_BN_N[-int(self.step_duration / self.dynRate) :]

        return

    def get_asteroid_state(self):
        self.r_AN_N = self.astRec.PositionVector[
            -int(self.step_duration / self.dynRate) :
        ]
        self.v_AN_N = self.astRec.VelocityVector[
            -int(self.step_duration / self.dynRate) :
        ]
        self.dcms_AN = self.astRec.J20002Pfix[-int(self.step_duration / self.dynRate) :]
        return

    def get_data_state(self):
        self.storageLevelOld = np.copy(self.storageLevel)
        self.storageLevel = (
            self.dataStorageUnit.storageUnitDataOutMsg.read().storageLevel
        )
        self.storedData = self.dataStorageUnit.storageUnitDataOutMsg.read().storedData

        return

    def get_power_state(self):
        self.powerLevel = self.powerRecorder.storageLevel[
            -int(self.step_duration / self.dynRate) :
        ]

        return

    def get_dsn_state(self):
        # Compute the time to next downlink
        self.downlink_state = 0
        for idx in range(0, len(self.initial_conditions["dsnPrescribedTimes"])):
            # Check if there is currently a downlink opportunity
            if (
                self.simTime >= self.initial_conditions["dsnPrescribedTimes"][idx, 0]
            ) and (
                self.simTime <= self.initial_conditions["dsnPrescribedTimes"][idx, 1]
            ):
                self.downlink_state = 1
                break
            # Check if we're before the first downlink opportunity
            elif self.simTime < self.initial_conditions["dsnPrescribedTimes"][0, 0]:
                self.downlink_state = (
                    1
                    - (
                        self.initial_conditions["dsnPrescribedTimes"][0, 0]
                        - self.simTime
                    )
                    / self.initial_conditions["dsnPrescribedTimes"][0, 0]
                )
                break
            # Check if we are in between the last end time and the next start time
            elif (
                self.simTime
                >= self.initial_conditions["dsnPrescribedTimes"][idx - 1, 1]
            ) and (
                self.simTime <= self.initial_conditions["dsnPrescribedTimes"][idx, 0]
            ):
                self.downlink_state = (
                    1
                    - (
                        self.initial_conditions["dsnPrescribedTimes"][idx, 0]
                        - self.simTime
                    )
                    / self.initial_conditions["dsnPrescribedTimes"][0, 0]
                )
                break

        return

    def get_map_state(self):
        # Copy over captured, downlinked images and maps
        self.imaged_targets_old = np.copy(self.imaged_targets)
        self.downlinked_targets_old = np.copy(self.downlinked_targets)
        self.imaged_maps_old = np.copy(self.imaged_maps)
        self.downlinked_maps_old = np.copy(self.downlinked_maps)

        self.storedMap = self.mappingStorageUnit.storageUnitDataOutMsg.read().storedData
        # Loop through the maps
        if int(self.modeRequest) == 1 + self.num_waypoint_actions:
            for idx in range(0, self.n_map_points):
                if self.storedMap[idx] > 0:
                    self.imaged_maps[0, idx] = 1
                if self.storedMap[idx + self.n_map_points] > 0:
                    self.imaged_maps[1, idx] = 1
                if self.storedMap[idx + 2 * self.n_map_points] > 0:
                    self.imaged_maps[2, idx] = 1

        return

    def get_nearest_target(self):
        # Update the nearest target
        current_dist = 1e9
        rc_N = self.r_AN_N[-1, :]
        vc_N = self.v_AN_N[-1, :]
        rd_N = self.r_BN_N[-1, :]
        vd_N = self.v_BN_N[-1, :]
        dcm_AN = self.dcms_AN[-1, :, :].reshape(3, 3)
        self.r_BO_O, self.v_BO_O = orbitalMotion.rv2hill(rc_N, vc_N, rd_N, vd_N)
        for idx2 in range(0, self.n_targets):
            check_tgt_O = self.initial_conditions["imaging_targets"][idx2, :]
            self.current_tgt_O = np.copy(check_tgt_O)
            distance = np.linalg.norm(rd_N - (np.matmul(dcm_AN.T, check_tgt_O) + rc_N))
            if (distance < current_dist) and not self.imaged_targets[idx2]:
                current_dist = distance
                self.current_tgt_O = np.copy(check_tgt_O)
                self.current_tgt_index = idx2

        return

    def check_collision(self):
        collision_indices = np.linspace(0, len(self.r_BN_N) - 1, num=10, dtype=int)
        self.collision = False

        for idx in collision_indices:
            rd_N = self.r_BN_N[idx, :]
            vd_N = self.r_BN_N[idx, :]
            rc_N = self.r_AN_N[idx, :]
            vc_N = self.v_AN_N[idx, :]
            r_BO_O, _ = orbitalMotion.rv2hill(rc_N, vc_N, rd_N, vd_N)
            if np.linalg.norm(r_BO_O) <= self.body_radius:
                self.collision = True
                break

    def check_new_mapping_and_imaging(self):
        self.new_downlinked_images = []
        self.new_downlinked_maps = []
        self.new_imaged = []
        self.new_mapped = []

        # Check which images were takes for the first time
        for idx in range(0, self.n_targets):
            if self.imaged_targets[idx] and not self.imaged_targets_old[idx]:
                self.new_imaged.append(idx)

        # Check which part of the map was mapped for the first time
        for idx in range(0, self.n_maps):
            for idx2 in range(0, self.n_map_points):
                if self.imaged_maps[idx, idx2] and not self.imaged_maps_old[idx, idx2]:
                    self.new_mapped.append([idx, idx2])

        # Compute the map and image downlinks
        maps_to_downlink = []
        images_to_downlink = []
        if int(self.modeRequest) == 2 + self.num_waypoint_actions:
            # Compute the percentage of the buffer emptied
            if self.storageLevelOld == 0:
                percent_downlinked = 0
            else:
                percent_downlinked = (
                    self.storageLevelOld - self.storageLevel
                ) / self.storageLevelOld

            # Loop through and compute which maps have been imaged but not downlinked
            for idx in range(0, self.n_maps):
                for idx2 in range(0, self.n_map_points):
                    if (
                        self.imaged_maps[idx, idx2]
                        and not self.downlinked_maps[idx, idx2]
                    ):
                        maps_to_downlink.append([idx, idx2])

            # Loop through and compute which images have been imaged but not downlinked
            for idx in range(0, self.n_targets):
                if self.imaged_targets[idx] and not self.downlinked_targets[idx]:
                    images_to_downlink.append(idx)

            # Compute which maps are downlinked
            for idx in range(0, int(round(percent_downlinked * len(maps_to_downlink)))):
                self.new_downlinked_maps.append(maps_to_downlink[idx])

            # Compute which images are downlinked
            for idx in range(
                0, int(round(percent_downlinked * len(images_to_downlink)))
            ):
                self.new_downlinked_images.append(images_to_downlink[idx])

            # Set the which have been downlinked
            for entry in self.new_downlinked_maps:
                self.downlinked_maps[entry[0], entry[1]] = 1
            for entry in self.new_downlinked_images:
                self.downlinked_targets[entry] = 1

        return

    def get_map_region_states(self):
        # Compute the mapped regions for each mapping longitude
        self.map_regions = np.zeros(self.n_maps * 3)
        num_regions = 3
        region_points = int(self.n_map_points / num_regions)
        for idx in range(0, self.n_maps):
            for idx2 in range(0, num_regions):
                self.map_regions[idx * 3 + idx2] = (
                    np.sum(
                        self.imaged_maps[
                            idx, (idx2 * region_points) : ((idx2 + 1) * region_points)
                        ]
                    )
                    / region_points
                )

    def get_image_target_state(self):
        if self.storedData[self.current_tgt_index + 1]:
            self.imaged_targets[self.current_tgt_index] = 1

        return

    def close_gracefully(self):
        return

    def clear_logging(self):
        # Clear the logs
        if self.clear_logs:
            self.scRec.clear()
            self.astRec.clear()
            self.forceRequestRec.clear()
            self.canberraRecorder.clear()
            self.goldstoneRecorder.clear()
            self.madridRecorder.clear()
            if self.fidelity == "high":
                self.wheelSpeedRecorder.clear()
            self.powerRecorder.clear()

        return

    def wrap_phi(self):
        if self.phi_c < -90:
            self.phi_c += 180
        elif self.phi_c > 90:
            self.phi_c -= 180

    def wrap_lambda(self):
        if self.lambda_c < -180:
            self.lambda_c += 360
        elif self.lambda_c > 180:
            self.lambda_c -= 360
