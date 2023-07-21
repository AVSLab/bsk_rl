from multiprocessing import cpu_count
import torch.nn as nn
import os
import sys
from bsk_rl.training.sb3.experiments import (
    run_a2c_experiments,
    create_a2c_kwargs_list,
)
from pathlib import Path

SEP = os.path.sep

# Check if any args were passed to the script
if len(sys.argv) > 1:
    # Get the index of the job array
    experiment_index = int(sys.argv[1])
else:
    experiment_index = None

if __name__ == "__main__":
    """
    This script runs a hyperparameter search for the A2C algorithm on the
    MultiSensorEOS-v0 environment. The hyperparameters that are searched are:
        - network width
        - network depth

    The activation function, dropout, alpha (for LeakyReLU), entropy coefficient,
    n_steps, and learning rate are all fixed.

    The hyperparameters may be adjusted as needed.

    A couple of important variables are described below:
        - max_steps: The maximum number of steps for the environment
        - n_its: The number of training iterations to run for each experiment. This
        helps reduce slowdowns from potential memory leaks.
        - base_steps: The number of steps to run for each iteration
        - n_experiments: The number of experiments to run for each set of
        hyperparameters

    The total number of steps for each experiment is calculated as:
        total_steps = n_its * max_steps * base_steps

    The results from this experiment can be plotted using the plotting
    tools in `plotting_tools/`.

    Other environments that can be used:
        - MultiSensorEOS-v0
        - SimpleEOS-v0 (not yet tested)
        - AgileEOS-v0
    """
    max_steps = 90
    n_its = 1
    base_steps = 90
    n_experiments = 1
    num_cores = cpu_count() - 1
    env_name = "MultiSensorEOS-v0"

    activation_functions = [nn.LeakyReLU]
    dropouts = [None]
    alphas = [0.1]
    entropy_coeffs = [0.0]

    widths = [10, 20, 40, 80, 160]
    depths = [1, 2, 4]
    n_steps_ = [int(0.5 * max_steps)]
    learning_rates = [0.007]

    agent_dir = (
        ".."
        + SEP
        + ".."
        + SEP
        + "bsk_rl"
        + SEP
        + "results"
        + SEP
        + "SB3"
        + SEP
        + "A2C"
        + SEP
        + "MultiSensorEOS-v0"
        + SEP
        + "A2C Test"
        + SEP
    )

    agent_dir = str(Path(__file__).parent.resolve() / agent_dir)

    kwargs_list = create_a2c_kwargs_list(
        activation_functions=activation_functions,
        widths=widths,
        depths=depths,
        dropouts=dropouts,
        alphas=alphas,
        learning_rates=learning_rates,
        entropy_coeffs=entropy_coeffs,
        n_steps_=n_steps_,
        n_experiments=n_experiments,
    )

    run_a2c_experiments(
        agent_dir,
        kwargs_list,
        n_its=n_its,
        base_steps=base_steps,
        index=experiment_index,
        env_name=env_name,
        max_steps=max_steps,
        num_cores=num_cores,
    )
