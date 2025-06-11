"""Relative properties between two satellites."""

from typing import Any

import numpy as np
from Basilisk.utilities.RigidBodyKinematics import MRP2C

from bsk_rl.obs import Observation
from bsk_rl.utils.orbital import rv2HN, rv2omega


def r_DC_N(deputy, chief):
    """Relative position of the deputy satellite to the chief satellite in inertial frame."""
    return np.array(deputy.dynamics.r_BN_N) - np.array(chief.dynamics.r_BN_N)


def r_DC_C(deputy, chief):
    """Relative position of the deputy satellite to the chief satellite in chief body frame."""
    raise NotImplementedError()


def r_DC_D(deputy, chief):
    """Relative position of the deputy satellite to the chief satellite in deputy body frame."""
    raise NotImplementedError()


def r_DC_Hc(deputy, chief):
    """Relative position of the deputy satellite to the chief satellite in chief Hill frame."""
    HcN = chief.dynamics.HN
    return HcN @ r_DC_N(deputy, chief)


def r_DC_Hd(deputy, chief):
    """Relative position of the deputy satellite to the chief satellite in deputy Hill frame."""
    raise NotImplementedError()


def v_DC_N(deputy, chief):
    """Relative velocity of the deputy satellite to the chief satellite in inertial frame."""
    return np.array(deputy.dynamics.v_BN_N) - np.array(chief.dynamics.v_BN_N)


def v_DC_C(deputy, chief):
    """Relative velocity of the deputy satellite to the chief satellite in chief body frame."""
    raise NotImplementedError()


def v_DC_D(deputy, chief):
    """Relative velocity of the deputy satellite to the chief satellite in deputy body frame."""
    raise NotImplementedError()


def v_DC_Hc(deputy, chief):
    """Relative velocity of the deputy satellite to the chief satellite in chief Hill frame."""
    HcN = chief.dynamics.HN
    omega_HcN_N = rv2omega(chief.dynamics.r_BN_N, chief.dynamics.v_BN_N)
    return HcN @ (v_DC_N(deputy, chief) - np.cross(omega_HcN_N, r_DC_N(deputy, chief)))


def v_DC_Hd(deputy, chief):
    """Relative velocity of the deputy satellite to the chief satellite in deputy Hill frame."""
    raise NotImplementedError()


def sigma_DC(deputy, chief):
    """Relative attitude of the deputy satellite to the chief satellite."""
    raise NotImplementedError()


def sigma_DHc(deputy, chief):
    """Relative attitude of the deputy satellite to the chief satellite in chief Hill frame."""
    raise NotImplementedError()


def sigma_HdC(deputy, chief):
    """Relative attitude of the deputy satellite Hill frame to the chief satellite."""
    raise NotImplementedError()


def sigma_HdHc(deputy, chief):
    """Relative attitude of the deputy satellite Hill frame to the chief satellite Hill frame."""
    raise NotImplementedError()


# TODO Could probably make some thing that generates these and other relative properties
# (i.e. whether to use body or hill frame for each sat, what
# frame to express in)


def rso_imaged_regions(
    servicer,
    chief,
    region_centers=np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    ),
    frame="chief_hill",
):
    """For use with the RSO tasking scenario.

    Returns the fraction of regions imaged by the servicer, where regions are defined by
    the inspection points closest to each region center.

    Args:
        servicer: The servicer satellite.
        chief: The chief satellite.
        region_centers: The centers of the regions to return inspection metrics for.
            It may be useful to set this as :class:`fibonacci_sphere` points.
        frame: The frame to use for the region centers. Can be ``chief_hill`` or
            ``chief_body``.
    """
    assert frame in ["chief_hill", "chief_body"]
    point_inspect_status = servicer.data_store.data.point_inspect_status

    region_centers_C = []
    for region_center in region_centers:
        if frame == "chief_hill":
            HN = rv2HN(chief.dynamics.r_BN_N, chief.dynamics.v_BN_N)
            CN = MRP2C(chief.dynamics.sigma_BN)
            region_center_C = CN @ HN.T @ region_center
        elif frame == "chief_body":
            region_center_C = region_center

        region_centers_C.append(region_center_C)

    imaged = np.zeros(len(region_centers))
    total = np.zeros(len(region_centers))
    for point, inspected in point_inspect_status.items():
        # find the closest region center
        i_closest = np.argmin(np.linalg.norm(region_centers_C - point.r_PB_B, axis=1))
        total[i_closest] += 1
        if inspected:
            imaged[i_closest] += 1

    return imaged / np.maximum(total, 1)


class RelativeProperties(Observation):
    """Add arbitrary properties relative to some other satellite."""

    def __init__(
        self, *rel_properties: dict[str, Any], chief_name: str, name="rel_props"
    ) -> None:
        """Include properties relative to another satellite.

        Within the observation specification for the deputy satellite, this would look like.

        .. code-block:: python

            obs.RelativeProperties(
                dict(prop="r_DC_N", norm=1e3),
                dict(prop="v_DC_N", norm=1e3),
                chief_name="ChiefSat",
            ),

        Args:
            rel_properties: Property specifications. Properties are optionally
                normalized by some factor. Each observation is a dictionary with the keys:

                * ``prop``: Name of a function in :ref:`bsk_rl.obs.relative_observations`.
                * ``norm`` `optional`: Value to normalize property by. Defaults to 1.0.
                * ``name`` `optional`: Name of the observation element. Defaults to the value of ``prop``.
                * ``fn`` `optional`: Alternatively, call a function that takes the deputy (self) and chief (other)
                  as arguments.
            chief_name: Name of the satellite to compare against.
            name: Name of the observation.
        """
        super().__init__(name=name)

        for rel_property in rel_properties:
            for key in rel_property:
                if key not in ["prop", "norm", "name", "fn"]:
                    raise ValueError(f"Invalid property key: {key}")
            if "norm" not in rel_property:
                rel_property["norm"] = 1.0
            if "fn" not in rel_property:
                try:
                    rel_property["fn"] = globals()[rel_property["prop"]]
                except KeyError:
                    raise ValueError(
                        f"Property prop={rel_property['prop']} is not predefined and no `fn` was provided."
                    )
            if "name" not in rel_property:
                rel_property["name"] = rel_property["prop"]
                if rel_property["norm"] != 1.0:
                    rel_property["name"] += "_normd"

        self.rel_properties = rel_properties
        self.chief_name = chief_name

    def reset_post_sim_init(self) -> None:
        """Connect to the chief satellite.

        :meta private:
        """
        try:
            self.chief = [
                sat
                for sat in self.satellite.simulator.satellites
                if sat.name == self.chief_name
            ][0]
        except IndexError:
            raise ValueError(f"Chief satellite {self.chief_name} not found")

    def get_obs(self) -> dict[str, Any]:
        """Return the observation.

        :meta private:
        """
        obs = {}
        for rel_property in self.rel_properties:
            value = rel_property["fn"](self.satellite, self.chief)
            if isinstance(value, list):
                value = np.array(value)
            norm = rel_property["norm"]
            obs[rel_property["name"]] = value / norm
        return obs


__doc_title__ = "Relative Properties"
__all__ = [
    "RelativeProperties",
    "r_DC_N",
    "r_DC_C",
    "r_DC_D",
    "r_DC_Hc",
    "r_DC_Hd",
    "v_DC_N",
    "v_DC_C",
    "v_DC_D",
    "v_DC_Hc",
    "v_DC_Hd",
    "sigma_DC",
    "sigma_DHc",
    "sigma_HdC",
    "sigma_HdHc",
    "rso_imaged_regions",
]
