import multiprocessing as mp
import os
from itertools import product

import numpy as np


def build_job_array(**kwargs):
    """For a nested dict of iterables, return a list of every combination of iterables.

    For example:
    ```
        build_job_array(a=[1, 2], b=dict(c=[3, 4]))
    ```
    returns
    ```
        [{'a': 1, 'b': {'c': 3}},
         {'a': 1, 'b': {'c': 4}},
         {'a': 2, 'b': {'c': 3}},
         {'a': 2, 'b': {'c': 4}}]
    ```
    """
    keys = kwargs.keys()
    vectors = [
        build_job_array(**value) if isinstance(value, dict) else value
        for value in kwargs.values()
    ]
    combinations = list(product(*vectors))
    job_array = []
    for combination in combinations:
        job_dict = {}
        for key, value in zip(keys, combination):
            job_dict[key] = value
        job_array.append(job_dict)

    return job_array


def sanitize_np(obj):
    """Make a nested dict with numpy arrays into a nested dict with lists."""
    if type(obj).__module__ == "numpy":
        if isinstance(obj, np.ndarray):
            return [sanitize_np(el) for el in obj]
        else:
            return obj.item()
    if isinstance(obj, dict):
        return {k: sanitize_np(v) for k, v in obj.items()}
    else:
        return obj


def unpack_config(env):
    """Create a wrapped version of an env class that unpacks env_config from Ray into kwargs"""

    class UnpackedEnv(env):
        def __init__(self, env_config):
            super().__init__(**env_config)

    UnpackedEnv.__name__ = f"{env.__name__}_Unpacked"

    return UnpackedEnv


def get_available_cores():
    try:
        processes = int(os.environ["SLURM_JOB_CPUS_PER_NODE"].split("(")[0])
    except Exception:
        processes = mp.cpu_count()

    print(f"Process Count: {processes}")
    return processes