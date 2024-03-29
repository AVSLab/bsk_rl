"""Communication of data between satellites."""

import logging
from abc import ABC, abstractmethod
from itertools import combinations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.envs.general_satellite_tasking.types import Satellite

import numpy as np
from scipy.sparse.csgraph import connected_components

from bsk_rl.envs.general_satellite_tasking.simulation.dynamics import LOSCommDynModel

logger = logging.getLogger(__name__)


class CommunicationMethod(ABC):
    """Base class for defining data sharing between satellites."""

    def __init__(self, satellites: Optional[list["Satellite"]] = None) -> None:
        """Construct base communication class.

        Subclasses implement a way of determining which pairs of satellites share data.
        """
        self.satellites = satellites

    def reset(self) -> None:
        """Reset communication after simulator initialization."""
        pass

    @abstractmethod  # pragma: no cover
    def _communication_pairs(self) -> list[tuple["Satellite", "Satellite"]]:
        """List pair of satellite that should share data."""
        pass

    def communicate(self) -> None:
        """Share data between paired satellites."""
        for sat_1, sat_2 in self._communication_pairs():
            sat_1.data_store.stage_communicated_data(sat_2.data_store.data)
            sat_2.data_store.stage_communicated_data(sat_1.data_store.data)
        for satellite in self.satellites:
            satellite.data_store.communication_update()


class NoCommunication(CommunicationMethod):
    """Implements no communication between satellite."""

    def _communication_pairs(self) -> list[tuple["Satellite", "Satellite"]]:
        return []


class FreeCommunication(CommunicationMethod):
    """Implements communication between satellites at every step."""

    def _communication_pairs(self) -> list[tuple["Satellite", "Satellite"]]:
        return list(combinations(self.satellites, 2))


class LOSCommunication(CommunicationMethod):
    """Implements communication between satellites with a direct line-of-sight.

    # TODO only communicate data from before latest LOS time
    """

    def __init__(self, satellites: list["Satellite"]) -> None:
        """Construct line-of-sigh communication management.

        Args:
            satellites: List of satellites to communicate between.
        """
        super().__init__(satellites)
        for satellite in self.satellites:
            if not issubclass(satellite.dyn_type, LOSCommDynModel):
                raise TypeError(
                    f"Satellite dynamics type {satellite.dyn_type} must be a subclass "
                    + "of LOSCommDynModel to use LOSCommunication"
                )

    def reset(self) -> None:
        """Add loggers to satellites to track line-of-sight communication."""
        super().reset()

        self.los_logs = {}
        for sat_1 in self.satellites:
            assert isinstance(sat_1.dynamics, LOSCommDynModel)
            for sat_2 in self.satellites:
                if sat_1 is not sat_2:
                    if sat_1 not in self.los_logs:
                        self.los_logs[sat_1] = {}

                    msg_index = sat_1.dynamics.los_comms_ids.index(sat_2.id)
                    logger = self.los_logs[sat_1][sat_2] = (
                        sat_1.dynamics.losComms.accessOutMsgs[msg_index].recorder()
                    )

                    sat_1.simulator.AddModelToTask(
                        sat_1.dynamics.task_name, logger, ModelPriority=586
                    )

    def _communication_pairs(self) -> list[tuple["Satellite", "Satellite"]]:
        pairs = []
        for sat_1, logs in self.los_logs.items():
            for sat_2, logger in logs.items():
                if any(logger.hasAccess):
                    pairs.append((sat_1, sat_2))
        return pairs

    def communicate(self) -> None:
        """Clear line-of-sight communication logs once communicated."""
        super().communicate()
        for sat_1, logs in self.los_logs.items():
            for sat_2, logger in logs.items():
                logger.clear()


class MultiDegreeCommunication(CommunicationMethod):
    """Compose with another type to use multi-degree communications.

    For example, if a <-> b and b <-> c, multidegree communication will also communicate
    between a <-> c.
    """

    def _communication_pairs(self) -> list[tuple["Satellite", "Satellite"]]:
        graph = np.zeros((len(self.satellites), len(self.satellites)), dtype=bool)
        for sat_1, sat_2 in super()._communication_pairs():
            graph[self.satellites.index(sat_1), self.satellites.index(sat_2)] = True

        pairs = []
        n_components, labels = connected_components(graph, directed=False)
        for comp in range(n_components):
            for i_sat_1, i_sat_2 in combinations(np.where(labels == comp)[0], 2):
                pairs.append((self.satellites[i_sat_1], self.satellites[i_sat_2]))
        return pairs


class LOSMultiCommunication(MultiDegreeCommunication, LOSCommunication):
    """Multidegree line of sight communication.

    Composes MultiDegreeCommunication with LOSCommunication.
    """

    pass
