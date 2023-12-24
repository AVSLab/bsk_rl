import pathlib
import runpy

import pytest

examples_path = (
    pathlib.Path(__file__).parent
    / ".."
    / ".."
    / "examples"
    / "general_satellite_tasking"
)
scripts = examples_path.resolve().glob("*.py")
script_names = {script.name: script for script in scripts}


@pytest.mark.parametrize("script", script_names)
def test_example_script(script):
    runpy.run_path(script_names[script])
