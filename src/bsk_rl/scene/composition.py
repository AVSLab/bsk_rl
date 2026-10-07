"""Independent scenarios bound to named groups of participants."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from bsk_rl.scene.scenario import Scenario

if TYPE_CHECKING:
    from bsk_rl.sats import Satellite


class MixedScenario(Scenario):
    """Run scenarios with explicit IDs and separate participant groups.

    Participants include physical targets as well as agents. Reward-producing
    satellites are configured independently by MixedReward.
    """

    def __init__(
        self,
        scenarios: Mapping[str, Scenario],
        satellite_mapping: Mapping[str, Sequence[str]],
    ) -> None:
        """Bind independent scenes to explicit participant names.

        Args:
            scenarios: Scenario ID to scenario instance.
            satellite_mapping: Scenario ID to participant names, including any
                physical target spacecraft. Groups may overlap.
        """
        super().__init__()
        self.scenarios = dict(scenarios)
        self.satellite_mapping = {
            key: tuple(names) for key, names in satellite_mapping.items()
        }
        if self.scenarios.keys() != self.satellite_mapping.keys():
            raise ValueError("Every scenario must have a participant mapping.")

    def _groups(self):
        # An instance referenced by several IDs receives each lifecycle hook once.
        groups = {}
        for key, scenario in self.scenarios.items():
            if id(scenario) not in groups:
                groups[id(scenario)] = (scenario, set())
            groups[id(scenario)][1].update(self.satellite_mapping[key])
        return groups.values()

    def validate_satellite_names(self, satellites: list["Satellite"]) -> None:
        """Reject ambiguous or missing names before environment copying."""
        names = [sat.name for sat in satellites]
        if len(names) != len(set(names)):
            raise ValueError("MixedScenario requires unique satellite names.")
        for scenario, participants in self._groups():
            missing = participants - set(names)
            if missing:
                raise ValueError(f"Unknown scene participants: {sorted(missing)}")
            scenario.validate_satellite_names(
                [sat for sat in satellites if sat.name in participants]
            )

    def link_satellites(self, satellites: list["Satellite"]) -> None:
        """Bind each scenario to its copied participants."""
        self.validate_satellite_names(satellites)
        super().link_satellites(satellites)
        for scenario, participants in self._groups():
            scenario.link_satellites(
                [sat for sat in satellites if sat.name in participants]
            )

    def after_step(self, sim_time: float) -> None:
        """Advance each distinct scenario once after reward calculation."""
        for scenario, _ in self._groups():
            scenario.after_step(sim_time)

    def reset_overwrite_previous(self) -> None:
        """Clear episode state and reset each component once."""
        for scenario, _ in self._groups():
            scenario.reset_overwrite_previous()

    def reset_pre_sim_init(self) -> None:
        """Prepare each component before simulator construction."""
        for scenario, _ in self._groups():
            scenario.utc_init = self.utc_init
            scenario.reset_pre_sim_init()

    def reset_during_sim_init(self) -> None:
        """Prepare each component during simulator construction."""
        for scenario, _ in self._groups():
            scenario.reset_during_sim_init()

    def reset_post_sim_init(self) -> None:
        """Finalize component setup after simulator initialization."""
        for scenario, _ in self._groups():
            scenario.reset_post_sim_init()
