import io
import zipfile
from pathlib import Path

import requests

from bsk_rl._check_bsk_version import _check_bsk_version


def pck_install():
    r = requests.get(
        "https://simplemaps.com/static/data/world-cities/basic/simplemaps_worldcities_basicv1.76.zip"
    )
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(Path(__file__).parent.resolve() / "data" / "simplemaps_worldcities")

    _check_bsk_version()
