"""Lightweight passive RSO targets with analytic Earth gravity.

A :class:`LightweightRSOTargetDynModel` target writes the same spacecraft state and
navigation messages as :class:`~bsk_rl.sim.dyn.RSOTargetDynModel`, but has no
spacecraft hub, actuators, or resources. One :class:`PassiveTargetPublisher` per
episode propagates every lightweight target together and publishes their messages
once per world tick: after SPICE (1100), before eclipse (988), and before imager
dynamics and flight software. No future state is published, and the world and imager
gravity objects are never modified.

Central gravity plus J2 is the default. Pure two-body propagation remains available
as a numerical and performance control. Pair these targets with
:class:`~bsk_rl.sim.fsw.PassiveRSOTargetFSWModel` and the ``Drift`` action; target
lifetime is the environment's time limit.
"""

from types import SimpleNamespace
from weakref import ProxyTypes, proxy

import numpy as np
from Basilisk.architecture import messaging, sysModel
from Basilisk.utilities import macros, orbitalMotion

from bsk_rl.sim.dyn.rso_imaging import RSOTargetDynModel
from bsk_rl.utils.functional import default_args
from bsk_rl.utils.passive_orbits import EARTH_MU, EARTH_RADIUS, J2Batch, KeplerBatch


class PassiveTargetPublisher(sysModel.SysModel):
    """Episode-owned catalog. Registration order is satellite order."""

    def __init__(self, simulator):
        """Attach one catalog publisher to the episode world task."""
        super().__init__()
        self.ModelTag = "passiveTargetCatalog"
        self.simulator = (
            simulator if isinstance(simulator, ProxyTypes) else proxy(simulator)
        )
        self.targets = []
        self.groups = []
        self.callback_count = 0
        simulator.AddModelToTask(
            simulator.world.world_task_name, self, ModelPriority=1001
        )

    def register(self, dynamics, r, v):
        """Register before InitializeSimulation, retaining independent initial states."""
        self.targets.append((dynamics, np.array(r, copy=True), np.array(v, copy=True)))

    def Reset(self, current_nanos):
        """Rebuild the propagators from the episode epoch on each native reset."""
        if current_nanos != 0:
            raise ValueError("Passive target reset requires the episode epoch (t=0).")
        self.callback_count = 0
        self.groups = []
        keys = dict.fromkeys(
            (d.target_force_mode, d.max_step_s) for d, _, _ in self.targets
        )
        for mode, step in keys:
            rows = [
                row
                for row in self.targets
                if (row[0].target_force_mode, row[0].max_step_s) == (mode, step)
            ]
            r = np.array([row[1] for row in rows])
            v = np.array([row[2] for row in rows])
            propagator = (
                KeplerBatch(r, v) if mode == "two_body" else J2Batch(r, v, max_step_s=step)
            )
            self.groups.append(([row[0] for row in rows], propagator))
        self._publish(0)

    def UpdateState(self, current_nanos):
        """Publish exactly current_nanos after current SPICE and before eclipse."""
        self.callback_count += 1
        self._publish(current_nanos)

    def _publish(self, current_nanos):
        t = current_nanos * macros.NANO2SEC
        pole = self.simulator.world.PN[2]
        for targets, propagator in self.groups:
            r, v = (
                propagator.at(t)
                if isinstance(propagator, KeplerBatch)
                else propagator.at(t, pole=pole)
            )
            for target, ri, vi in zip(targets, r, v):
                target.publish(ri, vi, current_nanos, self.moduleID)


class LightweightRSOTargetDynModel(RSOTargetDynModel):
    """Message-compatible passive RSO target with no scheduled spacecraft hub.

    Inherited ``sat_args`` defaults keep the same randomizer draws as
    :class:`~bsk_rl.sim.dyn.RSOTargetDynModel`, so swapping target models does not
    perturb other satellites' reset sampling. Inherited resource setup methods are
    never called. Target attitude is a fixed display orientation from
    ``sigma_init`` and is not part of the physical model.
    """

    passive_state_provider = True

    def __init__(self, satellite, dyn_rate, **kwargs) -> None:
        """Register target messages without creating native spacecraft services."""
        self.satellite = satellite
        self.logger = satellite.logger.getChild(type(self).__name__)
        self.dyn_rate = dyn_rate
        for required in self._requires_world():
            if not isinstance(self.world, required):
                raise TypeError(f"Passive targets require {required.__name__}.")
        if not np.isclose(dyn_rate, self.simulator.sim_rate):
            raise ValueError("Passive targets require the world dynamics cadence.")
        self.task_name = self.world.world_task_name
        if not hasattr(self.simulator, "passive_target_publisher"):
            self.simulator.passive_target_publisher = PassiveTargetPublisher(
                self.simulator
            )
        self._setup_dynamics_objects(**kwargs)

    @default_args(target_force_mode="j2", target_max_step_s=1.0)
    def _setup_dynamics_objects(
        self,
        target_force_mode: str,
        target_max_step_s: float,
        rN,
        vN,
        oe,
        mu: float,
        sigma_init,
        min_orbital_radius: float,
        **kwargs,
    ) -> None:
        """Validate the initial state and register it with the episode publisher.

        Args:
            target_force_mode: ``"j2"`` (central gravity plus J2) or ``"two_body"``.
            target_max_step_s: [s] Largest internal J2 integration step.
            rN: [m] Initial inertial position.
            vN: [m/s] Initial inertial velocity.
            oe: Orbital elements, used with ``mu`` when ``rN`` and ``vN`` are not given.
            mu: Gravitational parameter for converting ``oe`` only.
            sigma_init: Fixed display attitude MRP.
            min_orbital_radius: [m] Minimum allowable orbital radius.
            kwargs: Ignored spacecraft-hub and resource arguments.
        """
        if target_force_mode not in ("two_body", "j2"):
            raise ValueError("target_force_mode must be 'two_body' or 'j2'.")
        if not np.isfinite(target_max_step_s) or target_max_step_s <= 0:
            raise ValueError("target_max_step_s must be positive and finite.")
        if oe is not None and rN is None and vN is None:
            rN, vN = orbitalMotion.elem2rv(mu, oe)
        elif not (oe is None and rN is not None and vN is not None):
            raise KeyError("Provide either (rN, vN) or oe.")
        KeplerBatch([rN], [vN])  # Validate before publishing any messages.
        # Propagation matches the actual Earth body, whose mu and radius replace
        # the gravity-file header values when Basilisk initializes the body.
        if (
            self.world.planet.mu != EARTH_MU
            or self.world.planet.radEquator != EARTH_RADIUS
        ):
            raise ValueError("Passive targets require the standard Earth constants.")
        self.mu, self.min_orbital_radius = EARTH_MU, min_orbital_radius
        self.target_force_mode, self.max_step_s = target_force_mode, target_max_step_s
        self.scObject = SimpleNamespace(
            ModelTag=self.satellite.name, scStateOutMsg=messaging.SCStatesMsg()
        )
        self.simpleNavObject = SimpleNamespace(transOutMsg=messaging.NavTransMsg())
        self._sigma = list(sigma_init)
        self.publish(rN, vN, 0, 0)
        self.setup_eclipse_object()
        self.simulator.passive_target_publisher.register(self, rN, vN)

    def publish(self, r, v, time_ns, module_id) -> None:
        """Write noise-free Earth-centered navigation and state messages."""
        if not hasattr(self, "_state_payload"):
            self._state_payload = messaging.SCStatesMsgPayload()
            self._state_payload.sigma_BN = self._sigma
            self._nav_payload = messaging.NavTransMsgPayload()
        state, nav = self._state_payload, self._nav_payload
        position = r.tolist() if isinstance(r, np.ndarray) else list(r)
        velocity = v.tolist() if isinstance(v, np.ndarray) else list(v)
        state.r_BN_N = state.r_CN_N = position
        state.v_BN_N = state.v_CN_N = velocity
        nav.r_BN_N, nav.v_BN_N = position, velocity
        nav.timeTag = time_ns * macros.NANO2SEC
        self.scObject.scStateOutMsg.write(state, time_ns, module_id)
        self.simpleNavObject.transOutMsg.write(nav, time_ns, module_id)

    def is_alive(self, log_failure=False) -> bool:
        """Only orbital altitude applies to a resource-free target."""
        return bool(self.altitude_valid())

    def battery_valid(self) -> bool:
        """Always valid: the target has no battery. Not an aliveness checker."""
        return True

    def rw_speeds_valid(self) -> bool:
        """Always valid: the target has no reaction wheels. Not an aliveness checker."""
        return True


def rso_target_backend(mode: str = "j2"):
    """Return ``(dyn_type, fsw_type, sat_args)`` for an RSO target backend.

    ``"full"`` keeps :class:`~bsk_rl.sim.dyn.RSOTargetDynModel` with all Basilisk
    forces and services. ``"two_body"`` and ``"j2"`` select lightweight targets.
    """
    from bsk_rl.sim.fsw import FSWModel, PassiveRSOTargetFSWModel

    if mode == "full":
        return RSOTargetDynModel, FSWModel, {}
    if mode in ("two_body", "j2"):
        return (
            LightweightRSOTargetDynModel,
            PassiveRSOTargetFSWModel,
            {"target_force_mode": mode},
        )
    raise ValueError("RSO target backend must be 'full', 'two_body', or 'j2'.")


__doc_title__ = "Lightweight RSO Targets"
__all__ = ["LightweightRSOTargetDynModel", "PassiveTargetPublisher", "rso_target_backend"]
