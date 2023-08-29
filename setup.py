import io
import subprocess
import sys
import zipfile
from pathlib import Path

from setuptools import setup

setup(
    name="bsk_rl",
    version="0.0.0",
    install_requires=[
        "deap==1.3.3",
        "Deprecated",
        "gymnasium",
        "matplotlib",
        "numpy",
        "pandas",
        "pytest",
        "requests",
        "scikit-learn",
        "scipy",
        "stable-baselines3",
        "tensorflow",
        "torch",
    ],
)

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "git+https://github.com/chebpy/chebpy.git"]
)

import requests  # noqa: E402

r = requests.get(
    "https://simplemaps.com/static/data/world-cities/basic/simplemaps_worldcities_basicv1.76.zip"
)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(
    Path(__file__).parent.resolve()
    / "bsk_rl"
    / "envs"
    / "general_satellite_tasking"
    / "scenario"
    / "simplemaps_worldcities"
)
