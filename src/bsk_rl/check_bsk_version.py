"""Version checking for Basilisk."""

import os
from importlib.metadata import PackageNotFoundError, version
from warnings import warn

from packaging.version import parse as parse_version


def check_bsk_version():
    """Check Basilisk version against requirement."""
    # Don't run check if Basilisk is mocked
    try:
        if os.environ["PYTHON_MOCK_BASILISK"] == "1":
            return
    except KeyError:
        pass

    # Otherwise, check Basilisk version against requirement
    f = open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "bsk_version_req.txt",
        ),
        "r",
    )
    bsk_req = parse_version(f.read().strip())
    # Basilisk's current wheel metadata is named ``bsk`` while released legacy
    # wheels used ``Basilisk``.  The import package remains ``Basilisk`` in both
    # cases.  Accept either distribution name so a source-built current runtime
    # is not rejected after its native modules have loaded successfully.
    bsk_version = None
    for distribution_name in ("bsk", "Basilisk"):
        try:
            bsk_version = parse_version(version(distribution_name))
            break
        except PackageNotFoundError:
            continue
    if bsk_version is None:
        raise ImportError(
            "Neither the 'bsk' nor legacy 'Basilisk' distribution was found. "
            "Install from "
            "http://hanspeterschaub.info/basilisk/."
        )
    if bsk_version < bsk_req:
        warn(
            f"Basilisk>={bsk_req} is required for full functionality. "
            f"Currently installed: {bsk_version}",
        )
