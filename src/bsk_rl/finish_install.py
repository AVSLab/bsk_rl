import io
import subprocess
import sys
import zipfile
from pathlib import Path

import requests


def pck_install():
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/chebpy/chebpy.git",
        ]
    )

    r = requests.get(
        "https://simplemaps.com/static/data/world-cities/basic/simplemaps_worldcities_basicv1.76.zip"
    )
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(
        Path(__file__).parent.resolve()
        / "envs"
        / "general_satellite_tasking"
        / "scenario"
        / "simplemaps_worldcities"
    )
