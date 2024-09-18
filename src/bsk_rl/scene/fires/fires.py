import logging
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Union

import numpy as np
from Basilisk.utilities import orbitalMotion

import bsk_rl.scene.fires.spatial_temporal_model as stm
from bsk_rl.data.fire_data import FireData
from bsk_rl.scene import Scenario
from bsk_rl.utils.orbital import lla2ecef

logger = logging.getLogger(__name__)


class Fire:

    def __init__(
        self,
        name: str,
        r_LP_P: Iterable[float],
        burning_area: Callable,
        burned_area: Callable,
        start_time: float,
        end_time: float,
    ) -> None:
        """Instance of a fire that evolves over time.

        Args:
            name: Identifier; does not need to be unique
            r_LP_P: [m] Fire location in planet frame.
            burning_area: Function of time (relative to episode start) that returns the
                current burning area of the fire.
            burned_area: Function of time (relative to episode start) that returns the
                total burned area of the fire.
        """
        self.name = name
        self.r_LP_P = np.array(r_LP_P)
        self.burning_area = burning_area
        self.burned_area = burned_area
        self.start_time, self.end_time = start_time, end_time

    # def _find_start_time(self, dt=1.0, t_max=86400):
    #     for t in np.arange(0, t_max, dt):
    #         if self.burned_area(t) > 0:
    #             return t
    #     logger.warning(
    #         f"Fire {self.name} does not start burning in first {t_max} seconds of simulation."
    #    )

    @property
    def id(self) -> str:
        """Get unique, human-readable identifier."""
        try:
            return self._id
        except AttributeError:
            self._id = f"{self.name}_{id(self)}"
            return self._id

    def __repr__(self) -> str:
        """Get string representation of target.

        Use ``target.id`` for a unique string identifier.

        Returns:
            Target string
        """
        return f"Fire({self.name}, t0={self.start_time})"


def sample_area_from_duration(duration):
    mean_params = [0.19349507, 1.78426136, 0.75528355]
    std_params = [0.23377339, 2.23512859, 0.98909288]
    mean = mean_params[0] * duration ** mean_params[1] + mean_params[2]
    std = std_params[0] * duration ** std_params[1] + std_params[2]

    area = 0
    while area <= 0:
        area = np.random.normal(mean, std)
    return area


def burning_area_curve(t, duration, area, b) -> float:
    """https://www.desmos.com/calculator/mmpdvdqoy0"""
    if t > duration:
        return 0
    a = 2**b
    p = np.log(-np.log(0.01 / a)) / np.log(a)
    burning_area = area * a / duration * t * np.exp(-np.power((a * t / duration), p))
    return burning_area


class Fires(Scenario):

    def __init__(self, pre_horizon: float = 86400, horizon: float = 86400) -> None:
        super().__init__()
        self.pre_horizon = pre_horizon
        self.horizon = horizon
        self.radius = orbitalMotion.REQ_EARTH * 1e3

    def reset_overwrite_previous(self) -> None:
        """Overwrite target list from previous episode."""
        self.fires = []

    def reset_pre_sim_init(self) -> None:
        month_lookup = {
            "JAN": 31,
            "FEB": 59,
            "MAR": 90,
            "APR": 120,
            "MAY": 151,
            "JUN": 181,
            "JUL": 212,
            "AUG": 243,
            "SEP": 273,
            "OCT": 304,
            "NOV": 334,
            "DEC": 365,
        }
        sim_start_day = month_lookup[self.utc_init[5:8]] - int(self.utc_init[8:10])
        secperday = 86400
        self.fire_distributor = stm.FireLocationGenerator(
            initial_day=sim_start_day - self.pre_horizon / secperday,
            final_day=sim_start_day + self.horizon / secperday,
        )

        for day, location in self.fire_distributor:
            x = lla2ecef(location[1], location[0], self.radius)
            start_time = (day - sim_start_day) * secperday
            duration = 300 * np.random.pareto(100) * 24 * 3600
            area = sample_area_from_duration(duration / (24 * 3600))
            b = np.random.uniform(1, 5)  # Burn profile param
            self.fires.append(
                Fire(
                    f"{location[1]:.2f},{location[0]:.2f}",
                    x,
                    burning_area=lambda t, start_time=start_time, duration=duration, area=area, b=b: burning_area_curve(
                        t - start_time,
                        duration,
                        area,
                        b,
                    ),
                    burned_area=lambda t: (
                        0
                        if t < start_time
                        else (
                            area
                            if t > start_time + duration
                            else area * (t - start_time) / duration
                        )
                    ),
                    start_time=start_time,
                    end_time=start_time + duration,
                )
            )

        for satellite in self.satellites:
            if hasattr(satellite, "add_location_for_access_checking"):
                for fire in self.fires:
                    satellite.add_location_for_access_checking(
                        object=fire,
                        r_LP_P=fire.r_LP_P,
                        min_elev=satellite.sat_args_generator[
                            "imageTargetMinimumElevation"
                        ],  # Assume not randomized
                        type="fire",
                    )

    def reset_post_sim_init(self) -> None:
        super().reset_post_sim_init()
        for satellite in self.satellites:
            satellite.calculate_additional_windows(satellite.scan_ahead_time)
            fires = satellite.opportunities_dict(
                types="fire",
                filter=lambda opportunity: (
                    opportunity["object"].start_time <= opportunity["window"][0]
                    and opportunity["window"][0] <= satellite.scan_ahead_time
                    and opportunity["object"].end_time > 0  # GOOD?
                ),
            )
            data = {fire: [] for fire in fires}
            # logger.info(f"First visible: {data}")
            satellite.data_store.data += FireData(data)
