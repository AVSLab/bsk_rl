"""Version checking for Basilisk."""

import os
from importlib.metadata import PackageNotFoundError, version
from warnings import warn

from packaging.version import parse as parse_version


def _get_basilisk_distribution_version() -> str:
    """Return the installed Basilisk distribution version.

    Basilisk was historically published under the distribution name
    "Basilisk" and is now published as "bsk".
    """
    for dist_name in ("Basilisk", "bsk"):
        try:
            return version(dist_name)
        except PackageNotFoundError:
            continue

    raise PackageNotFoundError("Basilisk")


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
    try:
        bsk_version = parse_version(_get_basilisk_distribution_version())
        if not bsk_version >= bsk_req:
            warn(
                f"Basilisk>={bsk_req} is required for full functionality. "
                f"Currently installed: {bsk_version}",
            )
    except PackageNotFoundError:
        raise ImportError(
            "No Basilisk distribution metadata was found (tried 'Basilisk' and 'bsk'). "
            "Install from "
            "http://hanspeterschaub.info/basilisk/."
        )
