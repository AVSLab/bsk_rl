import numpy as np
import random

from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import macros as mc
from bsk_rl.utilities.initial_conditions import leo_orbit


class Settings:
    """
    To be used as settings for the MultiSensorEOS gymnasium environment and bsk sim.
    """

    def __init__(self):
        # Instrument Settings
        self.img_modes = 2

        # Simulation parameters
        self.SIM_TIME = int(90)  # number of planning intervals
        self.N_TGTS = 50
        self.STEP_DURATION = 180.0  # [sec] length of planning interval
        self.DYN_STEP = 0.5  # [sec]
        self.FSW_STEP = 1.0  # [sec]
        self.RENDER = False  # will render every episode if used in training

        # Spacecraft Attributes
        self.MASS = 330.0  # [kg]
        self.POWER_DRAW = -5.0  # [W]
        self.WHEEL_LIM = 3000 * mc.RPM  # [rad/s]
        self.POWER_MAX = 50.0 * 3600.0  # J

        # Learning Parameters
        self.REWARD_MULTIPLIER = 1.0

        # Observation Space (each obs element is a scalar value)
        self.OBSERVATIONS = [
            # "rx_target_N",
            # "ry_target_N",
            # "rz_target_N",
            # "rx_sc_N",
            # "ry_sc_N",
            # "rz_sc_N",
            # "sc_az",
            # "sc_el",
            # "sc_az_dot",
            # "sc_el_dot",
            # "rx_BL_L",
            # "ry_BL_L",
            # "rz_BL_L",
            "rx_canonical_BL_L",
            "ry_canonical_BL_L",
            "rz_canonical_BL_L",
            # "rxy_2norm_canonical_BL_L",
            # "xDot_BL_L",
            # "yDot_BL_L",
            # "zDot_BL_L",
            "xDot_canonical_BL_L",
            "yDot_canonical_BL_L",
            "zDot_canonical_BL_L",
            # "rxhat_target_N",
            # "ryhat_target_N",
            # "rzhat_target_N",
            "att_err",
            "att_rate",
            "wheel_speed",
            "stored_charge",
            "sun_indicator",
            "access_indicator",
            # "sc_mode",
            # "data_buffer",
            "img_mode_norm",
        ]

        self.UTC_INIT = "2021 MAY 04 06:47:48.965 (UTC)"

        self.CENTRAL_BODY = "earth"

        # Attitude rate
        self.maxSpinRate = 0.001

    def generate_new_ic(self):
        oe, rN, vN = leo_orbit.sampled_400km()

        targets, times = leo_orbit.create_ground_tgts(
            self.N_TGTS, rN, vN, self.SIM_TIME * self.STEP_DURATION / 60, self.UTC_INIT
        )

        # Sample random values
        sigma_init = np.array([random.uniform(0, 1.0) for _ in range(3)])
        omega_init = np.array(
            [random.uniform(-i, i) for i in self.maxSpinRate * np.ones(3)]
        )
        disturbance_vector = np.array([random.gauss(0, 1) for _ in range(3)])
        wheelSpeeds = np.array(
            [random.uniform(-i, i) for i in 0.5 * self.WHEEL_LIM * np.ones(3)]
        )
        storedCharge_Init = random.uniform(0.3 * self.POWER_MAX, self.POWER_MAX)

        # Dict of initial conditions
        self.INITIAL_CONDITIONS = {
            # Initialize the start time of the sim
            "utc_init": self.UTC_INIT,
            # Mass
            "mass": self.MASS,  # kg
            # Orbital parameters
            "central_body": self.CENTRAL_BODY,
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
            "disturbance_magnitude": 5e-3,
            "disturbance_vector": disturbance_vector
            / np.linalg.norm(disturbance_vector),
            # Reaction Wheel speeds
            "wheelSpeeds": wheelSpeeds,  # rad/s
            # Solar Panel Parameters
            "nHat_B": np.array([0, 0, 1]),
            "panelArea": 2 * 0.15 * 0.3,
            "panelEfficiency": 0.20,
            # Power Sink Parameters
            "powerDraw": self.POWER_DRAW,  # W
            # Battery Parameters
            "storageCapacity": self.POWER_MAX,
            "storedCharge_Init": storedCharge_Init,
            # Pointing Configuration (Solar panels & cameras)
            "C_R0R": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T,
            # RW motor torque and thruster force mapping FSW config
            "controlAxes_B": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            # Attitude controller FSW config
            "K": 7,
            "Ki": -1.0,  # Note: make value negative to turn off integral feedback
            "P": 35,
            # Momentum dumping config
            "hs_min": 0.0,  # Nms
            # Thruster force mapping FSW module
            "thrForceSign": 1,
            # Thruster momentum dumping FSW config
            "maxCounterValue": 4,
            "thrMinFireTime": 0.02,  # Seconds
            # Ground target locations
            "targetLocation": targets,  # m
            "instrumentSpecification": np.random.randint(
                1, 1 + self.img_modes, self.N_TGTS
            ),
            "target_times": times,  # seconds
            "minElev": 60.0,  # [degrees]
            "maxRange": 5.0e7,  # [meters}
            "n_targets": self.N_TGTS,
        }
