"""Package install scripts."""

import io
import zipfile
from pathlib import Path

import requests

from bsk_rl.check_bsk_version import check_bsk_version


def pck_install():
    """Download data dependencies and check package readiness."""
    r = requests.get(
        "https://simplemaps.com/static/data/world-cities/basic/simplemaps_worldcities_basicv1.76.zip"
    )
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(Path(__file__).parent.resolve() / "_dat" / "simplemaps_worldcities")

    check_bsk_version()
