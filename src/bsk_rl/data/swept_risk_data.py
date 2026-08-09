"""Data logging and rewarding for sweep imaging over a risk field."""

from bsk_rl.data.base import Data, DataStore, GlobalReward


class SweptRisk(Data):
    """Risk-area collected by the image swath."""

    def __init__(self, swept: float = 0.0) -> None:
        """Data for risk collected by the image swath.

        Args:
            swept: [risk * km^2] Risk value integrated over the newly imaged
                ground area.
        """
        self.swept = swept

    def __add__(self, other: "SweptRisk") -> "SweptRisk":
        """Define the combination of two units of data."""
        return self.__class__(self.swept + other.swept)


class SweptRiskStore(DataStore):
    """DataStore for risk collected by the image swath.

    The coverage bookkeeping lives in
    :class:`~bsk_rl.scene.SweepRiskField.update_coverage`, which sweeps the
    ``swath_km``-wide ground footprint along the logged boresight track,
    collects ``risk * cell_area`` from each newly covered grid cell, and
    zeroes those cells so importance is only ever counted once. The store
    simply reads the cumulative collected value.

    Requires :class:`~bsk_rl.sim.dyn.SweepDynModel` (for the attitude
    recorder) and a :class:`~bsk_rl.scene.SweepRiskField` scenario.
    """

    data_type = SweptRisk

    def get_log_state(self) -> float:
        """Cumulative risk-area collected since episode start."""
        return self.satellite.sweep_scene.update_coverage(self.satellite)

    def compare_log_states(self, old_state: float, new_state: float) -> "SweptRisk":
        """Generate a unit of data from the change in collected risk-area.

        Args:
            old_state: Previous cumulative risk-area.
            new_state: Current cumulative risk-area.

        Returns:
            Data: Data generated.
        """
        return SweptRisk(max(new_state - old_state, 0.0))


class SweptRiskReward(GlobalReward):
    """GlobalReward for the risk-area collected by the image swath."""

    datastore_type = SweptRiskStore

    def __init__(self, reward_per_risk_area: float = 1.0 / 40000.0) -> None:
        """GlobalReward for sweep imaging over a risk field.

        This class should be used with the :class:`~bsk_rl.scene.SweepRiskField`
        scenario and a satellite with :class:`~bsk_rl.sim.fsw.SweepFSWModel`,
        :class:`~bsk_rl.sim.dyn.SweepDynModel`, and the
        :class:`~bsk_rl.act.Sweep` action.

        Each ground cell pays out ``risk * cell_area`` the first time the
        swath covers it and is then worth zero (the scenario zeroes it in
        place, so observations see the reduced importance too).

        Args:
            reward_per_risk_area: [1/km^2] Scale factor on the collected
                risk-area. At 800 km altitude a 100 km swath covers about
                40,000 km^2 per minute, so the default makes one minute of
                fresh, full-risk swath worth about 1.0.
        """
        super().__init__()
        self.reward_per_risk_area = reward_per_risk_area

    def initial_data(self, satellite) -> "SweptRisk":
        """Start each satellite with zero collected risk."""
        return SweptRisk(0.0)

    def calculate_reward(self, new_data_dict) -> dict[str, float]:
        """Reward proportional to newly collected risk-area."""
        return {
            sat: new_data.swept * self.reward_per_risk_area
            for sat, new_data in new_data_dict.items()
        }


__doc_title__ = "Swept Risk"
__all__ = ["SweptRiskReward", "SweptRiskStore", "SweptRisk"]
