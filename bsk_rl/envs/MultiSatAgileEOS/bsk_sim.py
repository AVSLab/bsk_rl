from copy import copy, deepcopy

import numpy as np
from Basilisk.simulation import vizInterface
from Basilisk.utilities import (
    SimulationBaseClass,
    macros as mc,
    orbitalMotion,
    vizSupport,
)
from scipy.sparse.csgraph import connected_components

from bsk_rl.envs.MultiSatAgileEOS.bsk_models import (
    environment,
    dynamics,
    fsw as fsw_feedback,
    fsw_steering,
)
from bsk_rl.utilities.initial_conditions import leo_initial_conditions


class MultiSatAgileEOSSimulator(SimulationBaseClass.SimBaseClass):
    def __init__(
        self,
        envRate,
        dynRate,
        fswRate,
        step_duration,
        initial_conditions=None,
        render=False,
        n_targets=1,
        max_length=270.0,
        target_tuple_size=4,
        n_spacecraft=2,
        n_planes=2,
        rel_phasing=0,
        inc=45,
        global_tgts=None,
        priorities=None,
        comm_method="free",
        clustersize=1,
        clusterspacing=0,
        control_method="feedback",
        renew_tasks=True,
        target_indexing="local",
    ):
        """
        Simulates multiple spacecraft in LEO with atmospheric drag and J2.

        Dynamics Components
        - Forces: J2, Atmospheric Drag
        - Environment: Exponential density model; eclipse
        - Actuators: ExternalForceTorque, reaction wheels
        - Sensors: SimpleNav
        - Power System: SimpleBattery, SimplePOwerSink, SimpleSolarPanel
        - Data Management System: spaceToGroundTransmitter, simpleStorageUnit,
        simpleInstrument

        FSW Components:
        - MRP Feedback controller
        - locationPoint - targets, sun-pointing
        - Desat
        """
        # Initialize the SimBase object
        SimulationBaseClass.SimBaseClass.__init__(self)

        # Simulation settings
        assert target_indexing in [
            "local",
            "global",
        ], f"target_indexing {target_indexing} not valid!"
        self.target_indexing = (
            target_indexing  # Whether local or global target (action-3) is targeted
        )
        assert comm_method in [
            "free",
            "los",
            "los-multi",
            "none",
        ], f"comm_method {comm_method} not valid!"
        self.comm_method = comm_method
        assert control_method in [
            "fb",
            "feedback",
            "steer",
            "steering",
        ], f"control_method {control_method} not valid!"
        self.control_method = control_method
        self.renew_tasks = (
            renew_tasks  # If false, do not reinitialize task if same task selected
        )
        self.envRate = envRate
        self.dynRate = dynRate
        self.fswRate = fswRate
        self.render = render

        # Simulation properties
        self.initialized = False
        self.step_duration = step_duration
        self.attRefMsg = None
        self.attGuidMsg = None
        self.currentResetTime = 0.0
        self.prev_action = [0.0] * n_spacecraft
        self.prev_target = [-1] * n_spacecraft
        self.curr_step = 0
        self.max_length = max_length  # minutes
        self.nominal_radius = 6371 * 1000.0 + 500.0 * 1000

        # Initialize performance metrics
        self.total_downlinked = [0.0] * n_spacecraft
        self.total_access = [0.0] * n_spacecraft
        self.utilized_access = [0.0] * n_spacecraft

        # Set the constellation and environment parameters
        self.n_spacecraft = n_spacecraft
        self.n_planes = n_planes  # Put one spacecraft in each plane
        self.rel_phasing = rel_phasing  # deg
        self.inc = inc  # deg
        self.clusterspacing = clusterspacing
        self.clustersize = clustersize
        if priorities is None:
            priorities = []
        self.priorities = priorities
        if global_tgts is None:
            global_tgts = []
        self.global_tgts = global_tgts

        # Set the number of targets in the simulator
        self.n_targets = n_targets  # Number of targets in the action space for each s/c
        self.n_target_global = len(global_tgts)  # Maximum number of targets
        self.imaged_targets = {
            sc_idx: [] for sc_idx in range(self.n_spacecraft)
        }  # Per-spacecraft imaged targets, stored by index into global targets
        self.known_imaged_targets = {
            sc_idx: [] for sc_idx in range(self.n_spacecraft)
        }  # Per-spacecraft known imaged targets, stored by index into global targets
        self.downlinked_targets = {
            sc_idx: [] for sc_idx in range(self.n_spacecraft)
        }  # Per-spacecraft downlinked targets, stored by index into global targets
        self.known_downlinked_targets = {
            sc_idx: [] for sc_idx in range(self.n_spacecraft)
        }  # Per-spacecraft known downlinked targets, stored by idx into global targets
        self.global_target_log = {
            target_idx: [] for target_idx in range(len(self.global_tgts))
        }
        self.current_tgt_indices = [list(np.arange(self.n_targets))] * self.n_spacecraft

        # Set the communication method
        self.set_up_comms()

        # Set the size of the observations
        self.target_tuple_size = target_tuple_size
        self.obs = np.zeros(
            [self.n_spacecraft, 22 + self.n_targets * self.target_tuple_size]
        )
        self.obs_full = np.zeros(
            [self.n_spacecraft, 22 + self.n_targets * self.target_tuple_size]
        )
        self.sim_over = False

        # Load ICs
        if (
            initial_conditions is None
        ):  # If no initial conditions are defined yet, set ICs
            self.initial_conditions = self.set_ICs()
        else:  # If ICs were passed through, use the ones that were passed through
            self.initial_conditions = initial_conditions

        #################################################
        # Initialize dynModels, FSWModels, processes here
        #################################################

        # Connect the environment, dynamics and fsw scripts
        self.set_environment(environment)
        self.set_dynamics([dynamics] * n_spacecraft)
        if self.control_method in ["fb", "feedback"]:
            self.set_fsw([fsw_feedback] * n_spacecraft)
        elif self.control_method in ["steer", "steering"]:
            self.set_fsw([fsw_steering] * n_spacecraft)

        if self.render:
            self.vizInterface = None
            self.setup_viz()
            self.clear_logs = False
            self.vizSupport = vizSupport
        else:
            self.clear_logs = True

        self.set_logging()
        self.init_obs()

        self.modeRequests = None
        self.InitializeSimulation()
        self.initialized = True

        return

    def __del__(self):
        self.close_gracefully()
        return

    @property
    def simTime(self):
        """Simulation time in seconds, tied to SimBase integrator."""
        return self.TotalSim.CurrentNanos * mc.NANO2SEC

    def set_ICs(self):
        """
        Sets initial conditions
        :return initial_conditions: dictionary of spacecraft ICs, key is spacecraft num
        """
        initial_conditions = leo_initial_conditions.walker_delta_n_spacecraft_500_km(
            self.n_spacecraft,
            self.n_planes,
            self.rel_phasing,
            self.inc,
            self.global_tgts,
            self.priorities,
            self.max_length,
            clustersize=self.clustersize,
            clusterspacing=self.clusterspacing,
        )

        return initial_conditions

    def set_environment(self, envModel):
        """
        Sets up the environment modules for the sim.
        :return:
        """
        self.EnvProcessName = "EnvironmentProcess"
        self.envProc = self.CreateNewProcess(self.EnvProcessName, 300)

        # Add the environment class
        self.EnvModel = envModel.EnvironmentModel(self, self.envRate)

    def set_dynamics(self, dynModel):
        """
        Sets up the dynamics modules for the sim.
        :return:
        """
        self.DynamicsProcessName = []
        self.dynProc = []
        self.DynModels = []
        # Add the dynamics classes
        for spacecraftIndex in range(self.n_spacecraft):
            self.DynamicsProcessName.append(
                "DynamicsProcess" + str(spacecraftIndex)
            )  # Create simulation process name
            self.dynProc.append(
                self.CreateNewProcess(self.DynamicsProcessName[spacecraftIndex], 200)
            )  # Create process
            self.DynModels.append(
                dynModel[spacecraftIndex].DynamicModel(
                    self,
                    self.dynRate,
                    spacecraftIndex,
                    singleSat=(self.n_spacecraft == 1),
                )
            )
        for model in self.DynModels:
            model.ConnectLosComms([addmodel.scObject for addmodel in self.DynModels])

    def set_fsw(self, fswModel):
        """
        Sets up the fsw modules for the sim.
        :return:
        """
        self.FSWProcessName = []
        self.fswProc = []
        self.FSWModels = []
        # Add the FSW classes
        for spacecraftIndex in range(self.n_spacecraft):
            self.FSWProcessName.append(
                "FSWProcess" + str(spacecraftIndex)
            )  # Create simulation process name
            self.fswProc.append(
                self.CreateNewProcess(self.FSWProcessName[spacecraftIndex], 100)
            )  # Create process
            self.FSWModels.append(
                fswModel[spacecraftIndex].FSWModel(self, self.fswRate, spacecraftIndex)
            )

    def init_obs(self):
        """
        Initializes the observations for each spacecraft
        """
        for sc_idx in range(self.n_spacecraft):
            # Initialize the observations (normed)
            # Inertial position
            self.obs[sc_idx, 0] = (
                self.DynModels[sc_idx].scObject.hub.r_CN_NInit[0][0]
                / self.initial_conditions["env_params"]["planetRadius"]
            )
            self.obs[sc_idx, 1] = (
                self.DynModels[sc_idx].scObject.hub.r_CN_NInit[1][0]
                / self.initial_conditions["env_params"]["planetRadius"]
            )
            self.obs[sc_idx, 2] = (
                self.DynModels[sc_idx].scObject.hub.r_CN_NInit[2][0]
                / self.initial_conditions["env_params"]["planetRadius"]
            )
            # Inertial velocity
            self.obs[sc_idx, 3] = self.DynModels[sc_idx].scObject.hub.v_CN_NInit[
                0
            ] / np.linalg.norm(self.DynModels[sc_idx].scObject.hub.v_CN_NInit)
            self.obs[sc_idx, 4] = self.DynModels[sc_idx].scObject.hub.v_CN_NInit[
                1
            ] / np.linalg.norm(self.DynModels[sc_idx].scObject.hub.v_CN_NInit)
            self.obs[sc_idx, 5] = self.DynModels[sc_idx].scObject.hub.v_CN_NInit[
                2
            ] / np.linalg.norm(self.DynModels[sc_idx].scObject.hub.v_CN_NInit)
            # Attitude error
            self.obs[sc_idx, 6] = np.linalg.norm(
                self.DynModels[sc_idx].scObject.hub.sigma_BNInit
            )
            # Attitude rate
            self.obs[sc_idx, 7] = np.linalg.norm(
                self.DynModels[sc_idx].scObject.hub.omega_BN_BInit
            )
            # Wheel speeds
            self.obs[sc_idx, 8] = self.initial_conditions[str(sc_idx)]["wheelSpeeds"][
                0
            ] / (mc.RPM * self.initial_conditions[str(sc_idx)]["maxSpeed"])
            self.obs[sc_idx, 9] = self.initial_conditions[str(sc_idx)]["wheelSpeeds"][
                1
            ] / (mc.RPM * self.initial_conditions[str(sc_idx)]["maxSpeed"])
            self.obs[sc_idx, 10] = self.initial_conditions[str(sc_idx)]["wheelSpeeds"][
                2
            ] / (mc.RPM * self.initial_conditions[str(sc_idx)]["maxSpeed"])
            # Stored charge
            self.obs[sc_idx, 11] = (
                self.DynModels[sc_idx].powerMonitor.storedCharge_Init
                / self.initial_conditions[str(sc_idx)]["batteryStorageCapacity"]
            )

            # Initialize the full observations
            # Inertial position
            self.obs_full[sc_idx, 0:3] = np.asarray(
                self.DynModels[sc_idx].scObject.hub.r_CN_NInit
            ).flatten()
            # Inertial velocity
            self.obs_full[sc_idx, 3:6] = np.asarray(
                self.DynModels[sc_idx].scObject.hub.v_CN_NInit
            ).flatten()
            # Attitude error
            self.obs_full[sc_idx, 6] = np.linalg.norm(
                self.DynModels[sc_idx].scObject.hub.sigma_BNInit
            )
            # Attitude rate
            self.obs_full[sc_idx, 7] = np.linalg.norm(
                self.DynModels[sc_idx].scObject.hub.omega_BN_BInit
            )
            # Wheel speeds
            self.obs_full[sc_idx, 8:11] = (
                self.initial_conditions[str(sc_idx)]["wheelSpeeds"][0:3] * mc.RPM
            )
            # Stored charge
            self.obs_full[sc_idx, 11] = (
                self.DynModels[sc_idx].powerMonitor.storedCharge_Init / 3600.0
            )

            # Eclipse indicator
            self.obs[sc_idx, 12] = self.obs_full[sc_idx, 12] = 0
            # Stored data
            self.obs[sc_idx, 13] = self.obs_full[sc_idx, 13] = 0
            # Transmitted data
            self.obs[sc_idx, 14] = self.obs_full[sc_idx, 14] = 0
            # Ground Station access indicators
            self.obs[sc_idx, 15:22] = self.obs_full[sc_idx, 15:22] = 0

            # Compute the image tuples, add to observations
            image_tuples, image_tuples_norm = self.compute_image_tuples(
                self.obs_full[sc_idx, 0:3], self.obs_full[sc_idx, 3:6], sc_idx
            )
            self.obs[sc_idx, 22:] = image_tuples_norm[
                0 : self.n_targets * self.target_tuple_size
            ]
            self.obs_full[sc_idx, 22:] = image_tuples[
                0 : self.n_targets * self.target_tuple_size
            ]

            self.obs = np.around(self.obs, decimals=5)

    def setup_viz(self):
        """
        Initializes a vizSupport instance
        """
        from datetime import datetime

        fileName = f"multi_tgt_env-v1_{datetime.today()}"

        # Add the transceivers
        self.transceiverList = [[] for _ in range(self.n_spacecraft)]
        for idx in range(self.n_spacecraft):
            transceiverHUD = vizInterface.Transceiver()
            transceiverHUD.r_SB_B = [0.0, 0.0, 1.0]
            transceiverHUD.fieldOfView = 40.0 * mc.D2R
            transceiverHUD.normalVector = [-1.0, 0.0, 0.0]
            transceiverHUD.color = vizInterface.IntVector(
                vizSupport.toRGBA255("yellow", alpha=0.5)
            )
            transceiverHUD.label = "antenna"
            self.transceiverList[idx].append(transceiverHUD)

        # Add the sensors
        self.sensorList = [[] for _ in range(self.n_spacecraft)]
        for idx in range(self.n_spacecraft):
            genericSensor = vizInterface.GenericSensor()
            genericSensor.r_SB_B = [1.0, 1.0, 1.0]
            genericSensor.fieldOfView.push_back(20.0 * mc.D2R)
            genericSensor.normalVector = self.FSWModels[idx].locPointConfig.pHat_B
            genericSensor.color = vizInterface.IntVector(vizSupport.toRGBA255("red"))
            genericSensor.label = "genSen" + str(idx)
            self.sensorList[idx].append(genericSensor)

        # Define the spacecraft list
        spacecraftList = [dyn.scObject for dyn in self.DynModels]
        rwList = [dyn.rwStateEffector for dyn in self.DynModels]
        thrList = [[dyn.thrusterSet] for dyn in self.DynModels]

        self.vizInterface = vizSupport.enableUnityVisualization(
            self,
            self.DynModels[0].taskName,
            spacecraftList,
            rwEffectorList=rwList,
            thrEffectorList=thrList,
            genericSensorList=self.sensorList,
            transceiverList=self.transceiverList,
            saveFile="test",
        )

        vizSupport.addLocation(
            self.vizInterface,
            stationName="Boulder Station",
            parentBodyName=self.EnvModel.planet.planetName,
            r_GP_P=self.EnvModel.boulderGroundStation.r_LP_P_Init,
            fieldOfView=np.radians(160.0),
            color="pink",
            range=1000.0 * 1000,  # meters
        )

        vizSupport.addLocation(
            self.vizInterface,
            stationName="Merritt Station",
            parentBodyName=self.EnvModel.planet.planetName,
            r_GP_P=self.EnvModel.merrittGroundStation.r_LP_P_Init,
            fieldOfView=np.radians(160.0),
            color="pink",
            range=1000.0 * 1000,  # meters
        )

        vizSupport.addLocation(
            self.vizInterface,
            stationName="Singapore Station",
            parentBodyName=self.EnvModel.planet.planetName,
            r_GP_P=self.EnvModel.singaporeGroundStation.r_LP_P_Init,
            fieldOfView=np.radians(160.0),
            color="pink",
            range=1000.0 * 1000,  # meters
        )

        vizSupport.addLocation(
            self.vizInterface,
            stationName="Weilheim Station",
            parentBodyName=self.EnvModel.planet.planetName,
            r_GP_P=self.EnvModel.weilheimGroundStation.r_LP_P_Init,
            fieldOfView=np.radians(160.0),
            color="pink",
            range=1000.0 * 1000,  # meters
        )

        vizSupport.addLocation(
            self.vizInterface,
            stationName="Santiago Station",
            parentBodyName=self.EnvModel.planet.planetName,
            r_GP_P=self.EnvModel.santiagoGroundStation.r_LP_P_Init,
            fieldOfView=np.radians(160.0),
            color="pink",
            range=1000.0 * 1000,  # meters
        )

        vizSupport.addLocation(
            self.vizInterface,
            stationName="Dongara Station",
            parentBodyName=self.EnvModel.planet.planetName,
            r_GP_P=self.EnvModel.dongaraGroundStation.r_LP_P_Init,
            fieldOfView=np.radians(160.0),
            color="pink",
            range=1000.0 * 1000,  # meters
        )

        vizSupport.addLocation(
            self.vizInterface,
            stationName="Hawaii Station",
            parentBodyName=self.EnvModel.planet.planetName,
            r_GP_P=self.EnvModel.hawaiiGroundStation.r_LP_P_Init,
            fieldOfView=np.radians(160.0),
            color="pink",
            range=1000.0 * 1000,  # meters
        )

        stationID = 0
        for spacecraftIndex in range(self.n_spacecraft):
            targets = self.initial_conditions[str(spacecraftIndex)]["targetPositions"]
            if spacecraftIndex > 1:
                break
            for idx in range(
                self.initial_conditions[str(spacecraftIndex)]["transmitterNumBuffers"]
            ):
                vizSupport.addLocation(
                    self.vizInterface,
                    stationName=str(spacecraftIndex) + "_" + str(stationID),
                    parentBodyName=self.EnvModel.planet.planetName,
                    r_GP_P=targets[:][idx].tolist(),
                    fieldOfView=np.radians(160.0),
                    color="green",
                    range=1000.0 * 1000,  # meters
                )
                stationID = stationID + 1
                if idx > 10:
                    break

        vizSupport.createTargetLine(
            self.vizInterface, toBodyName=str(0), lineColor="blue"
        )

        self.vizInterface.settings.spacecraftSizeMultiplier = 1.5
        self.vizInterface.settings.showLocationCommLines = -1
        self.vizInterface.settings.showLocationCones = -1
        self.vizInterface.settings.showLocationLabels = 1

    def set_logging(self):
        """
        Sets up BSK logging functionality for the:
        - ground stations
        - transmitter
        - storage unit
        - simpleInstrumentControl access
        - sc Log
        - planetLog
        - LOS comms
        """

        self.boulderGSLogs = []
        self.singaporeGSLogs = []
        self.merrittGSLogs = []
        self.weilheimGSLogs = []
        self.santiagoGSLogs = []
        self.dongaraGSLogs = []
        self.hawaiiGSLogs = []
        self.transmitterLogs = []
        self.storageUnitLogs = []
        self.accessLogs = []
        self.scLogs = []
        self.losLogs = []

        for idx in range(self.n_spacecraft):
            # Add the ground station logs
            self.boulderGSLogs.append(
                self.EnvModel.boulderGroundStation.accessOutMsgs[idx].recorder()
            )
            self.singaporeGSLogs.append(
                self.EnvModel.singaporeGroundStation.accessOutMsgs[idx].recorder()
            )
            self.merrittGSLogs.append(
                self.EnvModel.merrittGroundStation.accessOutMsgs[idx].recorder()
            )
            self.weilheimGSLogs.append(
                self.EnvModel.weilheimGroundStation.accessOutMsgs[idx].recorder()
            )
            self.santiagoGSLogs.append(
                self.EnvModel.santiagoGroundStation.accessOutMsgs[idx].recorder()
            )
            self.dongaraGSLogs.append(
                self.EnvModel.dongaraGroundStation.accessOutMsgs[idx].recorder()
            )
            self.hawaiiGSLogs.append(
                self.EnvModel.hawaiiGroundStation.accessOutMsgs[idx].recorder()
            )

            # Add the other logs
            self.transmitterLogs.append(
                self.DynModels[idx].transmitter.nodeDataOutMsg.recorder()
            )
            self.storageUnitLogs.append(
                self.DynModels[idx].storageUnit.storageUnitDataOutMsg.recorder()
            )
            self.accessLogs.append(
                self.FSWModels[idx].simpleInsControlConfig.deviceCmdOutMsg.recorder()
            )
            self.scLogs.append(self.DynModels[idx].scObject.scStateOutMsg.recorder())

            # Add LOS logs
            losLog = []
            for ilog in range(self.n_spacecraft):
                if ilog < idx:
                    losLog.append(
                        self.DynModels[idx].losComms.accessOutMsgs[ilog].recorder()
                    )
                elif ilog > idx:
                    losLog.append(
                        self.DynModels[idx].losComms.accessOutMsgs[ilog - 1].recorder()
                    )
                else:
                    losLog.append(None)
            self.losLogs.append(losLog)

            # Add the ground station logs to the tasks
            self.AddModelToTask(
                self.EnvModel.envTaskName, self.boulderGSLogs[idx], ModelPriority=599
            )
            self.AddModelToTask(
                self.EnvModel.envTaskName, self.singaporeGSLogs[idx], ModelPriority=598
            )
            self.AddModelToTask(
                self.EnvModel.envTaskName, self.merrittGSLogs[idx], ModelPriority=597
            )
            self.AddModelToTask(
                self.EnvModel.envTaskName, self.weilheimGSLogs[idx], ModelPriority=596
            )
            self.AddModelToTask(
                self.EnvModel.envTaskName, self.santiagoGSLogs[idx], ModelPriority=595
            )
            self.AddModelToTask(
                self.EnvModel.envTaskName, self.dongaraGSLogs[idx], ModelPriority=594
            )
            self.AddModelToTask(
                self.EnvModel.envTaskName, self.hawaiiGSLogs[idx], ModelPriority=593
            )

            # Add the other logs to the tasks
            self.AddModelToTask(
                self.DynModels[idx].taskName,
                self.transmitterLogs[idx],
                ModelPriority=592,
            )
            self.AddModelToTask(
                self.DynModels[idx].taskName,
                self.storageUnitLogs[idx],
                ModelPriority=591,
            )
            self.AddModelToTask(
                self.EnvModel.envTaskName, self.accessLogs[idx], ModelPriority=591
            )
            self.AddModelToTask(
                self.DynModels[idx].taskName, self.scLogs[idx], ModelPriority=587
            )

            # Add LOS logs to the tasks
            for logger in self.losLogs[idx]:
                if logger:
                    self.AddModelToTask(
                        self.DynModels[idx].taskName, logger, ModelPriority=586
                    )

        self.planetLog = self.EnvModel.gravFactory.spiceObject.planetStateOutMsgs[
            self.EnvModel.earth
        ].recorder()
        self.AddModelToTask(
            self.EnvModel.envTaskName, self.planetLog, ModelPriority=587
        )

    def run_sim(self, action, return_obs=True):
        """
        Executes the sim for a specified duration given a mode command.
        :param action: list of actions, index by s/c num
        :return observations: n_spacecraft x obs_size nparray of observations
        """
        # Set the sim_over param to false
        self.sim_over = False
        self.currentResetTime = mc.sec2nano(self.simTime)

        if self.render:
            vizSupport.targetLineList.clear()

        # Loop through each spacecraft and set the modeRequests
        for idx in range(self.n_spacecraft):
            self.FSWModels[idx].modeRequest = str(int(action[idx]))
            if action[idx] >= 3:
                if self.target_indexing == "local":
                    self.EnvModel.imagingTargetList[
                        idx
                    ].r_LP_P_Init = self.initial_conditions[str(idx)].get(
                        "targetPositions"
                    )[
                        :
                    ][
                        self.current_tgt_indices[idx][
                            int(self.FSWModels[idx].modeRequest) - 3
                        ]
                    ]
                elif self.target_indexing == "global":
                    self.EnvModel.imagingTargetList[idx].r_LP_P_Init = self.global_tgts[
                        int(action[idx] - 3), 0:3
                    ]
            if self.render:
                if action[idx] >= 3:
                    vizSupport.createTargetLine(
                        self.vizInterface,
                        fromBodyName="sat-" + str(idx),
                        toBodyName=str(idx)
                        + "_"
                        + str(int(self.current_tgt_indices[idx][int(action[idx]) - 3])),
                        lineColor="blue",
                    )
        if self.render:
            vizSupport.updateTargetLineList(self.vizInterface)

        for idx in range(self.n_spacecraft):
            # Charging
            if int(action[idx]) == 0 and (
                self.renew_tasks or self.prev_action[idx] != 0
            ):
                self.fswProc[idx].disableAllTasks()
                self.FSWModels[idx].zeroGateWayMsgs()
                self.FSWModels[idx].sunPointWrap.Reset(self.currentResetTime)
                self.FSWModels[idx].trackingErrorWrap.Reset(self.currentResetTime)
                self.EnvModel.imagingTargetList[idx].Reset(self.currentResetTime)
                self.FSWModels[idx].locPointWrap.Reset(self.currentResetTime)
                self.FSWModels[idx].simpleInsControlConfig.controllerStatus = 0
                self.DynModels[idx].transmitter.dataStatus = 0
                self.DynModels[idx].transmitterPowerSink.powerStatus = 0
                self.DynModels[idx].instrumentPowerSink.powerStatus = 0
                if self.render:
                    self.transceiverList[idx][0].transceiverState = 0
                if self.render:
                    self.sensorList[idx][0].genericSensorCmd = 0
                self.enableTask("sunPointTask" + str(idx))
                self.enableTask("mrpControlTask" + str(idx))
                self.enableTask("trackingErrTask" + str(idx))
                self.disableTask("locPointTask" + str(idx))

            # Desaturation mode
            elif int(action[idx]) == 1 and (
                self.renew_tasks or self.prev_action[idx] != 1
            ):
                self.fswProc[idx].disableAllTasks()
                self.FSWModels[idx].zeroGateWayMsgs()
                self.FSWModels[idx].sunPointWrap.Reset(self.currentResetTime)
                self.FSWModels[idx].trackingErrorWrap.Reset(self.currentResetTime)
                self.EnvModel.imagingTargetList[idx].Reset(self.currentResetTime)
                self.FSWModels[idx].locPointWrap.Reset(self.currentResetTime)
                self.FSWModels[idx].simpleInsControlConfig.controllerStatus = 0
                self.FSWModels[idx].thrDesatControlWrap.Reset(self.currentResetTime)
                self.FSWModels[idx].thrDumpWrap.Reset(self.currentResetTime)
                self.DynModels[idx].transmitter.dataStatus = 0
                self.DynModels[idx].transmitterPowerSink.powerStatus = 0
                self.DynModels[idx].instrumentPowerSink.powerStatus = 0
                if self.render:
                    self.transceiverList[idx][0].transceiverState = 0
                if self.render:
                    self.sensorList[idx][0].genericSensorCmd = 0
                self.enableTask("sunPointTask" + str(idx))
                self.enableTask("mrpControlTask" + str(idx))
                self.enableTask("rwDesatTask" + str(idx))
                self.enableTask("trackingErrTask" + str(idx))
                self.disableTask("locPointTask" + str(idx))

            # Downlink mode
            elif int(action[idx]) == 2 and (
                self.renew_tasks or self.prev_action[idx] != 2
            ):
                self.fswProc[idx].disableAllTasks()
                self.FSWModels[idx].zeroGateWayMsgs()
                self.FSWModels[idx].hillPointWrap.Reset(self.currentResetTime)
                self.FSWModels[idx].trackingErrorWrap.Reset(self.currentResetTime)
                self.EnvModel.imagingTargetList[idx].Reset(self.currentResetTime)
                self.FSWModels[idx].locPointWrap.Reset(self.currentResetTime)
                self.FSWModels[idx].simpleInsControlConfig.controllerStatus = 0
                self.DynModels[idx].instrumentPowerSink.powerStatus = 0
                self.DynModels[idx].transmitter.dataStatus = 1
                self.DynModels[idx].transmitterPowerSink.powerStatus = 1
                if self.render:
                    self.transceiverList[idx][0].transceiverState = 1
                if self.render:
                    self.sensorList[idx][0].genericSensorCmd = 0
                self.enableTask("nadirPointTask" + str(idx))
                self.enableTask("mrpControlTask" + str(idx))
                self.enableTask("trackingErrTask" + str(idx))
                self.disableTask("locPointTask" + str(idx))

            # Imaging mode
            elif int(action[idx]) >= 3:
                global_tgt = (
                    self.initial_conditions[str(idx)]["targetIndices"][
                        int(self.current_tgt_indices[idx][int(action[idx]) - 3])
                    ]
                    if self.target_indexing == "local"
                    else int(action[idx] - 3)
                    if self.target_indexing == "global"
                    else -1
                )

                if self.renew_tasks or self.prev_target[idx] != global_tgt:
                    self.fswProc[idx].disableAllTasks()
                    self.FSWModels[idx].zeroGateWayMsgs()
                    self.FSWModels[idx].locPointWrap.Reset(self.currentResetTime)
                    self.FSWModels[idx].simpleInsControlConfig.controllerStatus = 1
                    self.EnvModel.imagingTargetList[idx].Reset(self.currentResetTime)
                    self.DynModels[idx].transmitter.dataStatus = 0
                    self.DynModels[idx].transmitterPowerSink.powerStatus = 0
                    self.DynModels[idx].instrumentPowerSink.powerStatus = 1
                    self.EnvModel.imagingTargetList[idx].r_LP_P_Init = self.global_tgts[
                        global_tgt, 0:3
                    ]
                    self.DynModels[idx].instrument.nodeDataName = str(
                        self.initial_conditions[str(idx)]["targetIndices"].index(
                            global_tgt
                        )
                    )
                    self.FSWModels[idx].simpleInsControlConfig.imaged = 0
                    if self.render:
                        self.transceiverList[idx][0].transceiverState = 0
                    if self.render:
                        self.sensorList[idx][0].genericSensorCmd = 1
                    self.enableTask("mrpControlTask" + str(idx))
                    self.enableTask("locPointTask" + str(idx))
                    self.prev_target[idx] = global_tgt

            if int(action[idx]) < 3:
                self.prev_target[idx] = -1

        # Configure the stop time
        simulation_time = mc.sec2nano(self.simTime + self.step_duration)
        self.curr_step += 1

        # Execute the simulation
        self.ConfigureStopTime(simulation_time)
        # self.ShowExecutionOrder()
        self.ExecuteSimulation()

        # Check what has been imaged and communicate
        downlinked_all, imaged_all = self.update_imaged()

        # Observe the system
        obs, sim_over, obs_full = self.get_obs()

        self.prev_action = action

        return obs, sim_over, obs_full, downlinked_all, imaged_all

    def update_imaged(self):
        """
        Update the imaging state of all satellites by adding to individual lists then
        communicating
        :return downlinked: n_spacecraft x n_targets np.array of downlinked targets
        :return imaged: n_spacecraft x n_targets np.array of imaged targets
        """
        # Initialize lists for each sc's imaging and downlink
        imaged_all = []
        downlinked_all = []
        self.current_tgt_indices = []

        # Update imaging state
        for idx in range(self.n_spacecraft):
            downlinked, imaged = self.update_imaged_targets(idx)

            # Append to imaged_all and downlinked_all
            imaged_all.append(imaged)
            downlinked_all.append(downlinked)

        # update satellite beliefs
        self.communicate()

        return downlinked_all, imaged_all

    def get_obs(self):
        """
        Pulls all of the message logs or reads messages for each spacecraft
        :return self.obs: n_spacecraft x obs_size nparray of normalized observations
        :return self.sim_over: T/F if sim is over
        :return self.obs_full: n_spacecraft x obs_size nparray of normalized
        observations
        """
        # Initialize the local obs_full variable
        obs_full = np.zeros(
            [self.n_spacecraft, 22 + self.n_targets * self.target_tuple_size]
        )
        obs_norm = np.zeros(
            [self.n_spacecraft, 22 + self.n_targets * self.target_tuple_size]
        )

        # create observation
        for idx in range(self.n_spacecraft):
            # Compute the relevant state variables
            attErr = self.FSWModels[idx].attGuidMsg.read().sigma_BR
            attRate = self.DynModels[idx].simpleNavObject.attOutMsg.read().omega_BN_B
            storedCharge = (
                self.DynModels[idx].powerMonitor.batPowerOutMsg.read().storageLevel
            )
            storageLevel = (
                self.DynModels[idx]
                .storageUnit.storageUnitDataOutMsg.read()
                .storageLevel
            )
            eclipseIndicator = (
                self.EnvModel.eclipseObject.eclipseOutMsgs[0].read().shadowFactor
            )
            wheelSpeeds = (
                self.DynModels[idx].rwStateEffector.rwSpeedOutMsg.read().wheelSpeeds
            )

            # Get the rotation matrix from the inertial to planet frame from SPICE
            dcm_PN = np.array(
                self.EnvModel.gravFactory.spiceObject.planetStateOutMsgs[
                    self.EnvModel.earth
                ]
                .read()
                .J20002Pfix
            ).reshape((3, 3))

            # Get inertial position and velocity, rotate to planet-fixed frame
            inertialPos = self.DynModels[idx].scObject.scStateOutMsg.read().r_BN_N
            inertialVel = self.DynModels[idx].scObject.scStateOutMsg.read().v_BN_N
            planetFixedPos = np.matmul(dcm_PN, inertialPos)
            planetFixedVel = np.matmul(dcm_PN, inertialVel)  # Should fix - T.T.
            # print('Spacecraft position ' + str(idx) + ': ' + str(planetFixedPos))

            # Get the access indicators for each ground station
            accessIndicator1 = self.boulderGSLogs[idx].hasAccess[
                -int(self.step_duration) :
            ]
            accessIndicator2 = self.merrittGSLogs[idx].hasAccess[
                -int(self.step_duration) :
            ]
            accessIndicator3 = self.singaporeGSLogs[idx].hasAccess[
                -int(self.step_duration) :
            ]
            accessIndicator4 = self.weilheimGSLogs[idx].hasAccess[
                -int(self.step_duration) :
            ]
            accessIndicator5 = self.santiagoGSLogs[idx].hasAccess[
                -int(self.step_duration) :
            ]
            accessIndicator6 = self.dongaraGSLogs[idx].hasAccess[
                -int(self.step_duration) :
            ]
            accessIndicator7 = self.hawaiiGSLogs[idx].hasAccess[
                -int(self.step_duration) :
            ]

            transmitterBaud = self.transmitterLogs[idx].baudRate[
                -int(self.step_duration) :
            ]

            # Compute the image tuples and which targets have been downlinked and imaged
            image_tuples, image_tuples_norm = self.compute_image_tuples(
                planetFixedPos, planetFixedVel, idx
            )

            # Full observations (non-normalized), all rounded to five decimals
            obs_full[idx, 0:22] = np.hstack(
                np.around(
                    [
                        planetFixedPos[0],
                        planetFixedPos[1],
                        planetFixedPos[2],
                        planetFixedVel[0],
                        planetFixedVel[1],
                        planetFixedVel[2],
                        np.linalg.norm(attErr),
                        np.linalg.norm(attRate),
                        wheelSpeeds[0],
                        wheelSpeeds[1],
                        wheelSpeeds[2],
                        storedCharge / 3600.0,
                        eclipseIndicator,
                        storageLevel,
                        np.sum(transmitterBaud) * self.dynRate / 8e6,
                        np.sum(accessIndicator1) * self.dynRate,
                        np.sum(accessIndicator2) * self.dynRate,
                        np.sum(accessIndicator3) * self.dynRate,
                        np.sum(accessIndicator4) * self.dynRate,
                        np.sum(accessIndicator5) * self.dynRate,
                        np.sum(accessIndicator6) * self.dynRate,
                        np.sum(accessIndicator7) * self.dynRate,
                    ],
                    decimals=5,
                )
            )

            # Check if there is a switch in targets
            self.current_tgt_indices.append(self.check_target_switch(idx))

            # Add the target tuples
            for idx2 in range(self.n_targets):
                obs_full[
                    idx,
                    (22 + idx2 * self.target_tuple_size) : (
                        22 + (idx2 + 1) * self.target_tuple_size
                    ),
                ] = np.around(
                    image_tuples[
                        self.current_tgt_indices[-1][idx2]
                        * self.target_tuple_size : (
                            self.current_tgt_indices[-1][idx2] * self.target_tuple_size
                            + self.target_tuple_size
                        )
                    ]
                )

            # Normalized observations, pull things from dictionary for readability
            transmitterBaudRate = self.initial_conditions[str(idx)][
                "transmitterBaudRate"
            ]
            batteryStorageCapacity = self.initial_conditions[str(idx)][
                "batteryStorageCapacity"
            ]
            dataStorageCapacity = self.initial_conditions[str(idx)][
                "dataStorageCapacity"
            ]

            # Normalized observations, all rounded to five decimals
            obs_norm[idx, 0:22] = np.hstack(
                np.around(
                    [
                        planetFixedPos[0] / self.nominal_radius,
                        planetFixedPos[1] / self.nominal_radius,
                        planetFixedPos[2] / self.nominal_radius,
                        planetFixedVel[0] / np.linalg.norm(planetFixedVel[0:3]),
                        planetFixedVel[1] / np.linalg.norm(planetFixedVel[0:3]),
                        planetFixedVel[2] / np.linalg.norm(planetFixedVel[0:3]),
                        np.linalg.norm(attErr),
                        np.linalg.norm(attRate),
                        wheelSpeeds[0]
                        / (self.initial_conditions[str(idx)]["maxSpeed"] * mc.RPM),
                        wheelSpeeds[1]
                        / (self.initial_conditions[str(idx)]["maxSpeed"] * mc.RPM),
                        wheelSpeeds[2]
                        / (self.initial_conditions[str(idx)]["maxSpeed"] * mc.RPM),
                        storedCharge / batteryStorageCapacity,
                        eclipseIndicator,
                        storageLevel / dataStorageCapacity,
                        np.sum(transmitterBaud)
                        * self.dynRate
                        / (transmitterBaudRate * self.step_duration),
                        np.sum(accessIndicator1) * self.dynRate / self.step_duration,
                        np.sum(accessIndicator2) * self.dynRate / self.step_duration,
                        np.sum(accessIndicator3) * self.dynRate / self.step_duration,
                        np.sum(accessIndicator4) * self.dynRate / self.step_duration,
                        np.sum(accessIndicator5) * self.dynRate / self.step_duration,
                        np.sum(accessIndicator6) * self.dynRate / self.step_duration,
                        np.sum(accessIndicator7) * self.dynRate / self.step_duration,
                    ],
                    decimals=5,
                )
            )

            # Add the normalized imaged tuples
            for idx2 in range(self.n_targets):
                obs_norm[
                    idx,
                    (22 + idx2 * self.target_tuple_size) : (
                        22 + (idx2 + 1) * self.target_tuple_size
                    ),
                ] = np.around(
                    image_tuples_norm[
                        self.current_tgt_indices[-1][idx2]
                        * self.target_tuple_size : (
                            self.current_tgt_indices[-1][idx2] * self.target_tuple_size
                            + self.target_tuple_size
                        )
                    ],
                    decimals=5,
                )

            # Check if crashed into Earth
            if np.linalg.norm(inertialPos) < (orbitalMotion.REQ_EARTH / 1000.0):
                self.sim_over = True

            # Update performance metrics for the sc
            self.total_downlinked[idx] += -obs_full[idx, 14]
            self.total_access[idx] += max(obs_full[idx, 15:22])
            self.utilized_access[idx] += (
                np.sum(transmitterBaud)
                * self.dynRate
                / self.initial_conditions[str(idx)]["transmitterBaudRate"]
            )

            # Clear the logs
            if self.clear_logs:
                self.boulderGSLogs[idx].clear()
                self.merrittGSLogs[idx].clear()
                self.singaporeGSLogs[idx].clear()
                self.weilheimGSLogs[idx].clear()
                self.santiagoGSLogs[idx].clear()
                self.dongaraGSLogs[idx].clear()
                self.hawaiiGSLogs[idx].clear()
                self.transmitterLogs[idx].clear()
                self.storageUnitLogs[idx].clear()

        self.obs_full = obs_full
        self.obs = obs_norm

        return self.obs, self.sim_over, self.obs_full

    def close_gracefully(self):
        """
        makes sure spice gets shut down right when we close.
        :return:
        """
        self.EnvModel.gravFactory.unloadSpiceKernels()
        self.EnvModel.gravFactory.spiceObject.unloadSpiceKernel(
            self.EnvModel.gravFactory.spiceObject.SPICEDataPath, "de430.bsp"
        )
        self.EnvModel.gravFactory.spiceObject.unloadSpiceKernel(
            self.EnvModel.gravFactory.spiceObject.SPICEDataPath, "naif0012.tls"
        )
        self.EnvModel.gravFactory.spiceObject.unloadSpiceKernel(
            self.EnvModel.gravFactory.spiceObject.SPICEDataPath, "de-403-masses.tpc"
        )
        self.EnvModel.gravFactory.spiceObject.unloadSpiceKernel(
            self.EnvModel.gravFactory.spiceObject.SPICEDataPath, "pck00010.tpc"
        )

        return

    def compute_image_tuples(self, r_BN_N, v_BN_N, sc_idx):
        """
        Computes the self.n_images image state tuples
        0-2: S/c Hill-Frame Position
        3: Priority
        4: Imaged?
        5: Downlinked?
        :return image_tuples: image_tuple_size nparray
        :return image_tuples_norm: image_tuple_size nparray
        """
        # Initialize the image tuple array
        image_tuples = np.zeros(
            self.target_tuple_size
            * self.initial_conditions[str(sc_idx)]["transmitterNumBuffers"]
        )
        image_tuples_norm = np.zeros(
            self.target_tuple_size
            * self.initial_conditions[str(sc_idx)]["transmitterNumBuffers"]
        )

        # Grab the targets, priorities, and update whether or not target has been imaged
        # or downlinked
        targets = np.transpose(
            np.array(self.initial_conditions[str(sc_idx)]["targetPositions"])
        )  # SC-specific target locations
        target_indices = self.initial_conditions[str(sc_idx)][
            "targetIndices"
        ]  # Global target indices
        priorities = self.initial_conditions["env_params"]["globalPriorities"][
            target_indices
        ]

        # Compute the inertial-to-Hill DCM
        dcm_HN = orbitalMotion.hillFrame(r_BN_N, v_BN_N)

        # Loop through each target to construct the state tuple
        for idx2 in range(
            self.initial_conditions[str(sc_idx)]["transmitterNumBuffers"]
        ):
            idx_start = idx2 * self.target_tuple_size

            # Grab the PCPF position of the target
            r_TN_N = targets[:, idx2]

            # Compute the position of the target wrt the spacecraft
            r_TB_N = np.subtract(r_TN_N, r_BN_N)

            # Transform to Hill-frame components
            r_TB_H = np.matmul(dcm_HN, r_TB_N)

            # Normalize
            r_TB_H_norm = r_TB_H / self.initial_conditions["env_params"]["planetRadius"]

            # Add to non-normalized image tuples
            image_tuples[idx_start : idx_start + 3] = r_TB_H

            # Add to normalized image tuples
            image_tuples_norm[idx_start : idx_start + 3] = r_TB_H_norm

            # Add the priorities
            image_tuples[idx_start + 3] = priorities[idx2]
            image_tuples_norm[idx_start + 3] = 1 / priorities[idx2]

        return image_tuples, image_tuples_norm

    def update_imaged_targets(self, sc_idx):
        """
        Updates which targets have been imaged and which have been downlinked
        :return downlinked: 1 x n_targets np.array of downlinked targets
        :return imaged: 1 x n_targets np.array of imaged targets
        """
        # Initialize list of targets that were just downlinked or imaged, helpful for
        # reward computation
        downlinked = []
        imaged = []

        # Pull the data log - check that it pulled correctly
        if self.initialized:
            storedData = self.storageUnitLogs[sc_idx].storedData
            if len(storedData.shape) == 1:
                storedData = None
        else:
            storedData = None

        # Loop through the global target indices for this spacecraft to determine if it
        # has been imaged or not
        for local_idx, target_idx in enumerate(
            self.initial_conditions[str(sc_idx)]["targetIndices"]
        ):
            # Check for an update
            if self.check_image_update(
                local_idx, storedData, target_idx in self.imaged_targets[sc_idx]
            ):
                self.global_target_log[target_idx].append(
                    {"spacecraft": sc_idx, "time": self.simTime}
                )
                self.imaged_targets[sc_idx].append(target_idx)
                self.known_imaged_targets[sc_idx].append(target_idx)
                imaged.append(target_idx)

            # check for update
            if self.check_downlink_update(local_idx, storedData):
                self.downlinked_targets[sc_idx].append(target_idx)
                self.known_downlinked_targets[sc_idx].append(target_idx)
                downlinked.append(target_idx)

        return downlinked, imaged

    def check_image_update(self, idx, storedData, alreadyimaged):
        """
        Checks the storageUnitLog to see if data was added or not
        :param idx: index of target
        :param storedData: pulled log of storage unit
        :return: 1 if data was added since the last decision interval, 0 otherwise
        """
        if storedData is not None:
            if storedData[-1, idx] > 0 and not alreadyimaged:
                return 1
            else:
                return 0
        else:
            return 0

    def check_downlink_update(self, idx, storedData):
        """
        Checks the storageUnitLogs to see if an image was downlinked
        :param idx: index of target
        :param storedData: pulled log of storage unit
        :return: 1 if data was added, 0 otherwise
        """
        # If the partition has downlinked data greater than or equal to an image size,
        # the image has been downlinked
        if storedData is not None:
            if (
                storedData[-int(self.step_duration / self.dynRate), idx]
                - storedData[-1, idx]
            ) > 0:
                return 1
            else:
                return 0
        else:
            return 0

    def check_target_switch(self, sc_idx):
        """
        Grabs the index(s) of the next upcoming target(s)
        :return upcoming_tgts: 1 x n_targets nparray of upcoming local tgt indices
        """
        times = self.initial_conditions[str(sc_idx)]["target_times"]
        global_tgts = self.initial_conditions[str(sc_idx)]["targetIndices"]
        upcoming_tgts = []
        last_tgt = len(times) - 1
        for local_idx, time in enumerate(times):
            # If less than simTime, add to upcoming targets
            if self.simTime < time:
                if global_tgts[local_idx] not in self.known_imaged_targets[sc_idx]:
                    upcoming_tgts.append(local_idx)
                    last_tgt = local_idx

        # Check that the list has at least as many upcoming targets as n_targets
        # (num in action space)
        # If not, backfill with last target
        if (
            len(upcoming_tgts)
            < self.initial_conditions[str(sc_idx)]["transmitterNumBuffers"]
        ):
            for tgt in range(self.n_targets - len(upcoming_tgts)):
                # Append the last target
                upcoming_tgts.append(last_tgt)

        return upcoming_tgts

    def set_up_comms(self):
        """
        Complete any preparatory steps for the comm method.
        - none: none
        - los: none
        - los-multi: none
        - free: link all sats to share same known lists
        """
        if self.comm_method == "none":
            pass  # targets only update own list
        elif self.comm_method == "los":
            pass
        elif self.comm_method == "los-multi":
            pass
        elif self.comm_method == "free":  # make all sats point to same list
            global_imaged = []
            global_downlinked = []
            for sc_idx in range(self.n_spacecraft):
                self.known_imaged_targets[sc_idx] = global_imaged
                self.known_downlinked_targets[sc_idx] = global_downlinked

    def communicate(self):
        """
        Share information between satellites based on the comm method.
        - none: none
        - los: merge lists from any satellites with los communication
        - los-multi: merge lists from any satellites in a los-connected graph
        - free: implicitly shared through shared known lists
        """
        if self.comm_method == "none":
            pass  # targets only update own list
        elif self.comm_method == "los":
            commlinks = 0
            old_know_imaged = deepcopy(self.known_imaged_targets)
            old_know_downlinked = deepcopy(self.known_downlinked_targets)
            for sc_idx in range(self.n_spacecraft):
                for comm_idx in range(self.n_spacecraft):
                    if sc_idx != comm_idx and any(
                        self.losLogs[sc_idx][comm_idx].hasAccess[
                            -int(self.step_duration / self.dynRate) :
                        ]
                    ):
                        self.known_imaged_targets[sc_idx] = list(
                            set(
                                self.known_imaged_targets[sc_idx]
                                + old_know_imaged[comm_idx]
                            )
                        )
                        self.known_downlinked_targets[sc_idx] = list(
                            set(
                                self.known_downlinked_targets[sc_idx]
                                + old_know_downlinked[comm_idx]
                            )
                        )
                        commlinks += 1
                        if self.clear_logs:  # Clear the logs
                            self.losLogs[sc_idx][comm_idx].clear()
        elif self.comm_method == "los-multi":
            connectivity = np.array(
                [
                    [
                        any(
                            self.losLogs[sc_idx][comm_idx].hasAccess[
                                -int(self.step_duration / self.dynRate) :
                            ]
                        )
                        if comm_idx != sc_idx
                        else False
                        for comm_idx in range(self.n_spacecraft)
                    ]
                    for sc_idx in range(self.n_spacecraft)
                ]
            )
            n_components, labels = connected_components(connectivity, directed=False)
            for comp in range(n_components):
                sc_idxs = np.where(labels == comp)[0]
                if sc_idxs.size > 1:
                    combined_imaged = []
                    combined_downlinked = []
                    [
                        combined_imaged.extend(self.known_imaged_targets[sc_idx])
                        for sc_idx in sc_idxs
                    ]
                    [
                        combined_downlinked.extend(
                            self.known_downlinked_targets[sc_idx]
                        )
                        for sc_idx in sc_idxs
                    ]
                    combined_imaged = list(set(combined_imaged))
                    combined_downlinked = list(set(combined_downlinked))
                    for sc_idx in sc_idxs:
                        self.known_imaged_targets[sc_idx] = copy(combined_imaged)
                        self.known_downlinked_targets[sc_idx] = copy(
                            combined_downlinked
                        )
            if self.clear_logs:  # Clear the logs
                for sc_idx in range(self.n_spacecraft):
                    [
                        self.losLogs[sc_idx][comm_idx].clear()
                        for comm_idx in range(self.n_spacecraft)
                        if comm_idx != sc_idx
                    ]
        elif self.comm_method == "free":
            pass  # all sats share same known_imaged list so no sharing method needed
