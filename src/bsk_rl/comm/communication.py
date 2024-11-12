"""Communication of data between satellites."""

import logging
from abc import ABC, abstractmethod
from copy import copy
from itertools import permutations
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.special import comb

from bsk_rl.sim.dyn import LOSCommDynModel
from bsk_rl.utils.functional import Resetable

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite

logger = logging.getLogger(__name__)


class CommunicationMethod(ABC, Resetable):
    """Base class for defining data sharing between satellites."""

    def __init__(self, min_period: float = 0.0) -> None:
        """The base communication class.

        Subclasses implement a way of determining which pairs of satellites share data
        at each environment step.

        Args:
            min_period: Minimum time between evaluation of the communication method.
        """
        self.satellites: list["Satellite"]
        self.min_period = min_period

    def reset_overwrite_previous(self) -> None:
        self.last_communication_time = 0.0

    def link_satellites(self, satellites: list["Satellite"]) -> None:
        """Link the environment satellite list to the communication method.

        Args:
            satellites: List of satellites to communicate between.
        """
        self.satellites = satellites

    @abstractmethod  # pragma: no cover
    def communication_pairs(self) -> list[tuple["Satellite", "Satellite"]]:
        """List pairs of satellite that should share data, in the form (sender, recipient).

        To define a new communication type, this method must be implemented.
        """
        pass

    def communicate(self) -> None:
        """Share data between paired satellites."""
        if (
            self.satellites[0].simulator.sim_time - self.last_communication_time
            < self.min_period
        ):
            return

        communication_pairs = self.communication_pairs()

        if len(communication_pairs) > 0:
            logger.info(
                f"Communicating data in {len(communication_pairs)} direction{'s' if len(communication_pairs) > 1 else ''}."
            )

        if len(communication_pairs) == 2 * comb(len(self.satellites), 2):
            self._communicate_all()
        else:
            for sat_sender, sat_receiver in communication_pairs:
                sat_receiver.data_store.stage_communicated_data(
                    sat_sender.data_store.data
                )
            for satellite in self.satellites:
                satellite.data_store.update_with_communicated_data()

        self.last_communication_time = self.satellites[0].simulator.sim_time

    def _communicate_all(self):
        """Optimized communication between all pairs of satellites."""
        logger.info("Optimizing data communication between all pairs of satellites")

        data_type = self.satellites[0].data_store.data.__class__
        final_data = data_type()
        for satellite in self.satellites:
            final_data += satellite.data_store.data
        for satellite in self.satellites:
            satellite.data_store.data = copy(final_data)


class NoCommunication(CommunicationMethod):
    """Implements no communication between satellites."""

    def __init__(self, *args, **kwargs):
        """Implements no communication between satellites.

        This is the default communication method if no other method is specified. Satellites
        will maintain their own :class:`~bsk_rl.data.base.DataStore` and not share data with others.
        """
        super().__init__(*args, **kwargs)

    def communication_pairs(self) -> list[tuple["Satellite", "Satellite"]]:
        """Return no communication pairs."""
        return []


class FreeCommunication(CommunicationMethod):
    """Implements free communication between every satellite at every step."""

    def __init__(self, *args, **kwargs) -> None:
        """Implements free communication between every satellite at every step.."""
        super().__init__(*args, **kwargs)

    def communication_pairs(self) -> list[tuple["Satellite", "Satellite"]]:
        """Return all possible communication pairs."""
        return list(permutations(self.satellites, 2))


class LOSCommunication(CommunicationMethod):
    """Implements communication between satellites with a direct line-of-sight."""

    # TODO only communicate data from before latest LOS time

    def __init__(self, *args, **kwargs) -> None:
        """Implements communication between satellites with a direct line-of-sight.

        At the end of each step, satellites will communicate with each other if they have a
        line-of-sight between them that is not occluded by the Earth.

        Satellites must have a dynamics model that is a subclass of
        :class:`~bsk_rl.sim.dyn.LOSCommDynModel`. to use this communication method.
        """
        super().__init__(*args, **kwargs)

    def link_satellites(self, satellites: list["Satellite"]) -> None:
        """Link the environment satellite list to the communication method.

        Args:
            satellites: List of satellites to communicate between.
        """
        super().link_satellites(satellites)
        for satellite in self.satellites:
            if not issubclass(satellite.dyn_type, LOSCommDynModel):
                raise TypeError(
                    f"Satellite dynamics type {satellite.dyn_type} must be a subclass "
                    + "of LOSCommDynModel to use LOSCommunication"
                )

    def reset_post_sim_init(self) -> None:
        """Add loggers to satellites to track line-of-sight communication."""
        super().reset_post_sim_init()

        self.los_logs = {}
        for sat_1 in self.satellites:
            assert isinstance(sat_1.dynamics, LOSCommDynModel)
            for sat_2 in self.satellites:
                if sat_1 is not sat_2:
                    if sat_1 not in self.los_logs:
                        self.los_logs[sat_1] = {}

                    msg_index = sat_1.dynamics.los_comms_ids.index(sat_2.name)
                    logger = self.los_logs[sat_1][sat_2] = (
                        sat_1.dynamics.losComms.accessOutMsgs[msg_index].recorder()
                    )

                    sat_1.simulator.AddModelToTask(
                        sat_1.dynamics.task_name, logger, ModelPriority=586
                    )

    def communication_pairs(self) -> list[tuple["Satellite", "Satellite"]]:
        """Return pairs of satellites that have line-of-sight visibility."""
        pairs = []
        for sat_1, logs in self.los_logs.items():
            for sat_2, logger in logs.items():
                if any(logger.hasAccess):
                    pairs.append((sat_1, sat_2))
                    pairs.append((sat_2, sat_1))
        return pairs

    def communicate(self) -> None:
        """Clear line-of-sight communication logs once communicated."""
        super().communicate()
        for sat_1, logs in self.los_logs.items():
            for sat_2, logger in logs.items():
                logger.clear()


class MultiDegreeCommunication(CommunicationMethod):
    """Compose with another type to use multi-degree communications."""

    def __init__(self, *args, **kwargs) -> None:
        """Compose with another communication type to propagate multi-degree communication.

        If a communication method allows satellites A and B to communicate and satellites
        B and C to communicate, MultiDegreeCommunication will allow satellites A and C to
        communicate on the same step as well.
        """
        super().__init__(*args, **kwargs)

    def communication_pairs(self) -> list[tuple["Satellite", "Satellite"]]:
        """Return pairs of satellites that are connected by a path of communication through other satellites."""
        graph = np.zeros((len(self.satellites), len(self.satellites)), dtype=bool)
        for sat_1, sat_2 in super().communication_pairs():
            graph[self.satellites.index(sat_1), self.satellites.index(sat_2)] = True

        pairs = []
        n_components, labels = connected_components(graph, directed=False)
        for comp in range(n_components):
            for i_sat_1, i_sat_2 in permutations(np.where(labels == comp)[0], 2):
                pairs.append((self.satellites[i_sat_1], self.satellites[i_sat_2]))
        return pairs


class LOSMultiCommunication(MultiDegreeCommunication, LOSCommunication):
    """Multidegree line-of-sight communication.

    Composes :class:`MultiDegreeCommunication` with :class:`LOSCommunication`.
    """

    pass


class BroadcastCommunication(CommunicationMethod):
    def communication_pairs(self) -> list[tuple["Satellite", "Satellite"]]:
        """Return pairs of satellites that used broadcast action."""
        broadcasters = []
        for satellite in self.satellites:
            if any(
                hasattr(act, "broadcast_pending") and act.broadcast_pending
                for act in satellite.action_builder.action_spec
            ):
                broadcasters.append(satellite)

        access_pairs = super().communication_pairs()
        comm_pairs = [
            (sat_sender, sat_receiver)
            for sat_sender, sat_receiver in access_pairs
            if sat_sender in broadcasters
        ]

        for satellite in broadcasters:
            for act in satellite.action_builder.action_spec:
                if hasattr(act, "broadcast_pending"):
                    act.broadcast_pending = False

        return comm_pairs


__all__ = []
