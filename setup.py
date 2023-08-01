import subprocess
import sys

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

import io
import zipfile
from pathlib import Path

import requests

r = requests.get(
    "https://simplemaps.com/static/data/world-cities/basic/simplemaps_worldcities_basicv1.76.zip"
)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(
    Path(__file__).parent.resolve()
    / "bsk_rl"
    / "envs"
    / "GeneralSatelliteTasking"
    / "scenario"
    / "simplemaps_worldcities"
)
