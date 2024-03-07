import numpy as np
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import smallBodyNavEKF
from Basilisk.simulation import simpleNav
from Basilisk.utilities import macros as mc
from Basilisk.utilities import orbitalMotion, unitTestSupport

from bsk_rl.env.small_body_science.bsk_sim import SmallBodyScienceSimulator


class SmallBodySciencePOMDPSimulator(SmallBodyScienceSimulator):
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
    - SmallBodyNavEKF
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
        super().__init__(
            dynRate,
            fswRate,
            mapRate,
            step_duration,
            initial_conditions=initial_conditions,
            render=render,
            n_targets=n_targets,
            n_map_points=n_map_points,
            max_length=max_length,
            n_states=n_states,
            n_maps=n_maps,
            phi_c=phi_c,
            lambda_c=lambda_c,
            fidelity=fidelity,
        )

        # Set the duration of the navigation mode
        self.nav_duration = 2e3  # s

        self.NavProcessName = "NavProcess"  # Create a FSW process name
        self.NavProc = None

        # Set the name of the nav mode task
        self.navTaskName = "navTask"

        self.measTaskName = "measTask"
        self.measTask = None

        # Redefine the observations
        self.obs = np.zeros([self.n_states, 1])
        self.obs_full = np.zeros([self.n_states, 1])

        # Re-initialize the observations
        # self.init_obs()

    def init_tasks_and_processes(self):
        #   Initialize the dynamics and fsw task groups and modules
        self.init_dynamics_process()
        # self.init_nav_process()
        self.init_fsw_process()
        self.init_map_process()
        self.init_meas_process()

        self.set_dynamics()
        self.set_meas()
        self.set_gateway_msgs()
        self.set_fsw()
        self.init_obs()

        self.set_logging()

        self.InitializeSimulation()
        self.initialized = True

        return

    # # Initializes the navigation process, similar to the fsw process
    # def init_nav_process(self):
    #     self.NavProc = self.CreateNewProcess(self.NavProcessName)

    def set_dynamics(self):
        """
        Calls each function to set the dynamics.
        :return:
        """
        self.set_spacecraft()
        self.set_grav_bodies()
        self.set_eclipse()
        self.set_power_system()
        self.set_thruster()
        self.set_reaction_wheels()
        self.set_control_force()
        self.set_disturbance_force()
        self.set_simple_nav_dyn()
        self.set_planet_nav()
        self.set_srp()
        self.set_ground_maps()
        self.set_imaging_target()
        self.set_data_system()
        self.set_dsn()
        self.set_dyn_models_to_tasks()

    def set_thruster(self):
        # Create a zero'd out thruster message
        self.thrusterMsgData = messaging.THROutputMsgPayload()
        self.thrusterMsg = messaging.THROutputMsg()
        self.thrusterMsg.write(self.thrusterMsgData)

    def set_simple_nav_dyn(self):
        # Set up simpleNav for s/c "measurements"
        self.simpleNavMeas2 = simpleNav.SimpleNav()
        self.simpleNavMeas2.ModelTag = "SimpleNav"
        self.simpleNavMeas2.scStateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        sp = self.initial_conditions["pos_sigma_sc"]
        sv = self.initial_conditions["vel_sigma_sc"]
        satt = self.initial_conditions["att_sigma_sc"]
        srate = self.initial_conditions["rate_sigma_sc"]
        ssun = self.initial_conditions["sun_sigma_sc"]
        sdv = self.initial_conditions["dv_sigma_sc"]
        p_mat = [
            [
                sp,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                sp,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                sp,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                sv,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                sv,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                sv,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                satt,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                satt,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                satt,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                srate,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                srate,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                srate,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                ssun,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                ssun,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                ssun,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                sdv,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                sdv,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                sdv,
            ],
        ]
        walk_bounds_sc = self.initial_conditions["walk_bounds_sc"]
        self.simpleNavMeas2.PMatrix = p_mat
        self.simpleNavMeas2.walkBounds = walk_bounds_sc
        return

    def set_simple_nav(self):
        super().set_simple_nav()
        sp = self.initial_conditions["pos_sigma_sc"]
        sv = self.initial_conditions["vel_sigma_sc"]
        satt = self.initial_conditions["att_sigma_sc"]
        srate = self.initial_conditions["rate_sigma_sc"]
        ssun = self.initial_conditions["sun_sigma_sc"]
        sdv = self.initial_conditions["dv_sigma_sc"]
        p_mat = [
            [
                sp,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                sp,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                sp,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                sv,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                sv,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                sv,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                satt,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                satt,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                satt,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                srate,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                srate,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                srate,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                ssun,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                ssun,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                ssun,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                sdv,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                sdv,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                sdv,
            ],
        ]
        walk_bounds_sc = self.initial_conditions["walk_bounds_sc"]
        self.simpleNavMeas.PMatrix = p_mat
        self.simpleNavMeas.walkBounds = walk_bounds_sc

        return

    def set_planet_nav(self):
        super().set_planet_nav()

        # Define the Pmatrix for planetNav, no uncertainty on position and velocity of
        # the body
        pos_sigma_p = self.initial_conditions["pos_sigma_p"]
        vel_sigma_p = self.initial_conditions["vel_sigma_p"]
        att_sigma_p = self.initial_conditions["att_sigma_p"]
        rate_sigma_p = self.initial_conditions["rate_sigma_p"]
        p_matrix_p = [
            [pos_sigma_p, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, pos_sigma_p, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, pos_sigma_p, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, vel_sigma_p, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, vel_sigma_p, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, vel_sigma_p, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, att_sigma_p, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, att_sigma_p, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, att_sigma_p, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rate_sigma_p, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rate_sigma_p, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rate_sigma_p],
        ]
        walk_bounds_p = self.initial_conditions["walk_bounds_p"]
        self.planetNavMeas.PMatrix = p_matrix_p
        self.planetNavMeas.walkBounds = walk_bounds_p

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
        self.AddModelToTask(self.dynTaskName, self.simpleNavMeas2, ModelPriority=995)
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

    def set_meas_models_to_tasks(self):
        self.AddModelToTask(self.measTaskName, self.simpleNavMeas, ModelPriority=995)

    def init_meas_process(self):
        self.measTask = self.dynProc.addTask(
            self.CreateNewTask(self.measTaskName, mc.sec2nano(self.dynRate)),
            taskPriority=3000,
        )

        return

    def set_meas(self):
        self.set_simple_nav()

        self.set_meas_models_to_tasks()

        return

    def set_fsw(self):
        self.create_fsw_tasks()
        self.init_fsw_tasks()
        self.set_fsw_tasks()
        self.set_fsw_models_to_tasks()

        return

    def create_fsw_tasks(self):
        super().create_fsw_tasks()

        # Define nav task here
        self.fswProc.addTask(
            self.CreateNewTask(self.navTaskName, self.processTasksTimeStep),
            taskPriority=100,
        )

        return

    def init_fsw_tasks(self):
        super().init_fsw_tasks()
        self.init_ekf_nav()

        return

    def set_fsw_tasks(self):
        super().set_fsw_tasks()
        self.set_ekf_nav()

        return

    def init_ekf_nav(self):
        # Set up the small body EKF
        self.smallBodyNav = smallBodyNavEKF.SmallBodyNavEKF()
        self.smallBodyNav.ModelTag = "smallBodyNavEKF"

        # Set the filter parameters (sc area, mass, gravitational constants, etc.)
        self.smallBodyNav.A_sc = self.initial_conditions[
            "srp_area"
        ]  # Surface area of the spacecraft, m^2
        self.smallBodyNav.M_sc = self.initial_conditions[
            "mass"
        ]  # Mass of the spacecraft, kg
        self.smallBodyNav.mu_ast = self.initial_conditions[
            "mu_bennu"
        ]  # Gravitational constant of the asteroid

        # Set the process noise
        self.smallBodyNav.Q = self.initial_conditions["Q"]

        # Set the measurement noise
        self.smallBodyNav.R = self.initial_conditions["R"]

        # Set the initial guess, x_0
        x_hat_0 = np.zeros(12)
        x_hat_0[0:3] = (
            self.initial_conditions["r_BO_O"]
            + self.initial_conditions["x_0_delta"][0:3]
        )
        x_hat_0[3:6] = (
            self.initial_conditions["v_BO_O"]
            + self.initial_conditions["x_0_delta"][3:6]
        )
        x_hat_0[6:9] = np.array([-0.58, 0.615, 0.125])
        x_hat_0[11] = 0.0004
        self.smallBodyNav.x_hat_k = unitTestSupport.np2EigenVectorXd(x_hat_0)

        # Set the covariance
        self.smallBodyNav.P_k = self.initial_conditions["P_0"]

        return

    def set_ekf_nav(self):
        # Connect the relevant modules to the smallBodyEKF input messages
        self.smallBodyNav.navTransInMsg.subscribeTo(self.simpleNavMeas.transOutMsg)
        self.smallBodyNav.navAttInMsg.subscribeTo(self.simpleNavMeas2.attOutMsg)
        self.smallBodyNav.asteroidEphemerisInMsg.subscribeTo(
            self.planetNavMeas.ephemerisOutMsg
        )
        self.smallBodyNav.sunEphemerisInMsg.subscribeTo(self.sunEphemerisMsg)
        self.smallBodyNav.cmdForceBodyInMsg.subscribeTo(
            self.waypointFeedback.forceOutMsg
        )
        self.smallBodyNav.addThrusterToFilter(self.thrusterMsg)

        # Connect the smallBodyEKF output messages to the relevant modules
        self.waypointFeedback.navTransInMsg.subscribeTo(
            self.smallBodyNav.navTransOutMsg
        )

        return

    def set_fsw_models_to_tasks(self):
        super().set_fsw_models_to_tasks()

        # Add the nav task here
        self.AddModelToTask(self.navTaskName, self.smallBodyNav, ModelPriority=90)

        return

    def set_target_pointing(self):
        self.locPointConfig.scAttInMsg.subscribeTo(self.simpleNavMeas2.attOutMsg)
        self.locPointConfig.scTransInMsg.subscribeTo(self.smallBodyNav.navTransOutMsg)
        self.locPointConfig.locationInMsg.subscribeTo(
            self.imagingTarget.currentGroundStateOutMsg
        )

        return

    def set_map_pointing(self):
        self.mapPointConfig.scAttInMsg.subscribeTo(self.simpleNavMeas2.attOutMsg)
        self.mapPointConfig.scTransInMsg.subscribeTo(self.smallBodyNav.navTransOutMsg)
        self.mapPointConfig.celBodyInMsg.subscribeTo(
            self.ephemConverter.ephemOutMsgs[0]
        )

        return

    def set_earth_pointing(self):
        self.earthPointConfig.scAttInMsg.subscribeTo(self.simpleNavMeas2.attOutMsg)
        self.earthPointConfig.scTransInMsg.subscribeTo(self.smallBodyNav.navTransOutMsg)
        self.earthPointConfig.celBodyInMsg.subscribeTo(
            self.ephemConverter.ephemOutMsgs[1]
        )

        return

    def set_sun_pointing(self):
        self.sunPointConfig.scAttInMsg.subscribeTo(self.simpleNavMeas2.attOutMsg)
        self.sunPointConfig.scTransInMsg.subscribeTo(self.smallBodyNav.navTransOutMsg)
        self.sunPointConfig.celBodyInMsg.subscribeTo(self.sunEphemerisMsg)

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

    def set_waypoint_feedback(self):
        self.waypointFeedback.asteroidEphemerisInMsg.subscribeTo(
            self.planetNavMeas.ephemerisOutMsg
        )
        self.waypointFeedback.sunEphemerisInMsg.subscribeTo(self.sunEphemerisMsg)
        self.waypointFeedback.navAttInMsg.subscribeTo(self.simpleNavMeas2.attOutMsg)
        self.waypointFeedback.navTransInMsg.subscribeTo(
            self.smallBodyNav.navTransOutMsg
        )
        self.extForceTorqueModule.cmdForceBodyInMsg.subscribeTo(
            self.waypointFeedback.forceOutMsg
        )

        return

    def turn_on_off_models(self):
        self.dynProc.enableAllTasks()
        self.disableTask(self.measTaskName)
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
        elif int(self.modeRequest) == 4 + self.num_waypoint_actions:
            self.nav_mode()
        # Turn on nav
        self.enableTask(self.navTaskName)

        return

    def nav_mode(self):
        self.step_duration = self.nav_duration
        # Turn off mapping, downlink, and imaging
        self.disable_mapping()
        self.disable_transmitter()
        self.disable_imaging()
        # Disable all FSW tasks
        self.fswProc.disableAllTasks()
        # Turn on MRP control
        if self.fidelity == "high":
            self.enableTask(self.mrpControlTaskName)
        # Turn on small-body pointing
        self.enableTask(self.mapPointTaskName)
        # Turn on measurements
        self.enableTask(self.measTaskName)
        # Turn on feedback control
        self.enableTask(self.smallBodyFeedbackControlTaskName)

        return

    def waypoint_mode(self):
        super().waypoint_mode()
        # Turn on measurements
        # self.enableTask(self.measTaskName)
        # self.disableTask(self.sunPointTaskName)
        # self.enableTask(self.mapPointTaskName)

    def init_obs(self):
        # Construct the observations
        self.obs[0:3, 0] = (
            self.initial_conditions["r_BO_O"]
            + self.initial_conditions["x_0_delta"][0:3]
        ) / self.nominal_radius  # Hill-frame position, normalized
        self.obs[3:6, 0] = (
            self.initial_conditions["v_BO_O"]
            + self.initial_conditions["x_0_delta"][3:6]
        )  # Hill-frame velocity
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
        self.obs[20:27, 0] = self.initial_conditions["P_0"].diagonal()[0:6]

        return

    def set_logging(self):
        super().set_logging()
        self.ekfRec = self.smallBodyNav.smallBodyNavOutMsg.recorder()
        self.ekfTransStateRec = self.smallBodyNav.navTransOutMsg.recorder()

        self.AddModelToTask(self.navTaskName, self.ekfRec)
        self.AddModelToTask(self.navTaskName, self.ekfTransStateRec)

        return

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
        self.check_new_mapping_and_imaging()
        self.get_map_region_states()
        self.get_image_target_state()

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
        self.obs[17:20, 0] = (
            self.current_tgt_O / self.body_radius
        )  # Location of the next target for imaging
        self.obs[20:27, 0] = np.array(
            self.smallBodyNav.smallBodyNavOutMsg.read().covar
        ).diagonal()[
            0:6
        ]  # TODO: Normalize

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
        """
        Saves the eclipse state. For the POMDP, a proxy for the eclipse state is used,
        which is just the power output of the panels. Zero power should indicate
        either eclipse or an anti-sun pointing mode. Non-zero power means not in
        eclipse.
        """
        self.eclipse = self.solarPanel.nodePowerOutMsg.read().netPower / (
            self.initial_conditions["panelArea"]
            * self.initial_conditions["panelEfficiency"]
        )
        return

    def get_spacecraft_state(self):
        self.r_BN_N = self.scRec.r_BN_N[-int(self.step_duration / self.dynRate) :]
        self.v_BN_N = self.scRec.v_BN_N[-int(self.step_duration / self.dynRate) :]
        self.r_BN_N_est = self.ekfTransStateRec.r_BN_N[
            -int(self.step_duration / self.dynRate) :
        ]
        self.v_BN_N_est = self.ekfTransStateRec.v_BN_N[
            -int(self.step_duration / self.dynRate) :
        ]
        self.state_est = self.ekfRec.state
        self.covar = self.ekfRec.covar
        self.times = self.scRec.times()

        return

    def get_nearest_target(self):
        # Update the nearest target
        current_dist = 1e9
        rc_N = self.r_AN_N[-1, :]
        vc_N = self.v_AN_N[-1, :]
        rd_N = self.r_BN_N_est[-1, :]
        vd_N = self.v_BN_N_est[-1, :]
        dcm_AN = self.dcms_AN[-1, :, :]
        dcm_OA = orbitalMotion.hillFrame(rc_N, vc_N) * (dcm_AN)
        self.r_BO_O, self.v_BO_O = orbitalMotion.rv2hill(rc_N, vc_N, rd_N, vd_N)
        for idx2 in range(0, self.n_targets):
            check_tgt_O = np.matmul(
                dcm_OA, self.initial_conditions["imaging_targets"][idx2, :]
            )
            self.current_tgt_O = np.copy(check_tgt_O)
            distance = np.linalg.norm(self.r_BO_O - check_tgt_O)
            if (distance < current_dist) and not self.imaged_targets[idx2]:
                current_dist = distance
                self.current_tgt_O = np.copy(check_tgt_O)
                self.current_tgt_index = idx2

        return

    def clear_logging(self):
        super().clear_logging()
        if self.clear_logs:
            self.ekfRec.clear()
            self.ekfTransStateRec.clear()


if __name__ == "__main__":
    action_dict = {
        "charge": 0,
        "up": 1,
        "up_right": 2,
        "right": 3,
        "down_right": 4,
        "down": 5,
        "down_left": 6,
        "left": 7,
        "up_left": 8,
        "map": 9,
        "downlink": 10,
        "image": 11,
        "nav": 12,
    }

    phi_c = -75
    lambda_c = -90
    dynRate = 10.0
    fswRate = 50.0
    mapRate = 180.0
    initial_conditions = None
    step_duration = 10000.0
    n_targets = 10
    n_map_points = 500
    max_length = 10000.0
    n_states = 26
    n_maps = 3

    # Create the simulator
    simulator = SmallBodySciencePOMDPSimulator(
        dynRate,
        fswRate,
        mapRate,
        step_duration,
        initial_conditions,
        render=False,
        n_targets=n_targets,
        n_map_points=n_map_points,
        max_length=max_length,
        n_states=n_states,
        n_maps=n_maps,
        phi_c=phi_c,
        lambda_c=lambda_c,
        fidelity="high",
    )
    simulator.init_tasks_and_processes()

    # Create an array of test actions
    actions = [
        "nav",
        "charge",
        "map",
        "map",
        "up",
        "map",
        "map",
        "up",
        "map",
        "map",
        "up",
        "map",
        "map",
        "up",
        "map",
        "nav",
        "up",
        "map",
        "map",
        "charge",
        "charge",
        "left",
        "map",
        "map",
        "charge",
        "downlink",
        "downlink",
        "downlink",
    ]

    reward_sum = 0

    for idx, act in enumerate(actions):
        obs, sim_over, _, _, _, _, _ = simulator.run_sim(action_dict[act])
        print("Step: " + str(idx) + ", sim_over: " + str(sim_over))
