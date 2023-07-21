import os
import sys
from multiprocessing import cpu_count
import torch.nn as nn
from bsk_rl.training.sb3.experiments import (
    run_ppo_experiments,
    create_ppo_kwargs_list,
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
    This script runs a hyperparameter search for the SPPO algorithm on the
    MultiSensorEOS-v0 environment. The hyperparameters that are searched are:
        - network width
        - network depth

    The activation function, dropout, alpha (for LeakyReLU), entropy coefficient,
    n_steps, clip range, and learning rate are all fixed.

    The hyperparameters may be adjusted as needed.

    This example uses shielding. While the environment will not fail
    to manage its resources, the performance of the agent will be lower than
    that of vanilla PPO.

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
    # Define the environment parameters
    max_steps = 90
    env_name = "MultiSensorEOS-v0"

    # Define the experiment parameters
    num_cores = cpu_count() - 2
    n_its = 1
    base_steps = 180

    # Define the hyperparameters
    activation_functions = [nn.LeakyReLU]
    dropouts = [None]
    alphas = [0.1]
    learning_rates = [3e-4]
    clip_ranges = [0.1]
    entropy_coeffs = [0.01]
    shielded = True

    widths = [10, 20, 40, 80, 160]
    depths = [1, 2, 4]
    n_epochs = [50]
    batch_sizes = [int(0.25 * max_steps * num_cores)]

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
        + "SPPO"
        + SEP
        + "MultiSensorEOS-v0"
        + SEP
        + "SPPO Test"
        + SEP
    )

    agent_dir = str(Path(__file__).parent.resolve() / agent_dir)

    # Create the kwargs_list
    kwargs_list = create_ppo_kwargs_list(
        activation_functions=activation_functions,
        widths=widths,
        depths=depths,
        dropouts=dropouts,
        alphas=alphas,
        learning_rates=learning_rates,
        clip_ranges=clip_ranges,
        entropy_coeffs=entropy_coeffs,
        n_epochs=n_epochs,
        batch_sizes=batch_sizes,
    )

    run_ppo_experiments(
        agent_dir,
        kwargs_list,
        n_its=n_its,
        base_steps=base_steps,
        index=experiment_index,
        env_name=env_name,
        n_steps=max_steps,
        num_cores=num_cores,
        shielded=shielded,
    )
