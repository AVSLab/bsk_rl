import os
import sys
from multiprocessing import cpu_count
from pathlib import Path

from bsk_rl.training.sb3.experiments import create_dqn_kwargs_list, run_dqn_experiments

SEP = os.path.sep

# Check if any args were passed to the script
if len(sys.argv) > 1:
    # Get the index of the job array
    experiment_index = int(sys.argv[1])
else:
    experiment_index = None

if __name__ == "__main__":
    """
    This script runs a hyperparameter search for the DQN algorithm on the
    MultiSensorEOS-v0 environment. The hyperparameters that are searched are:
        - network width
        - network depth

    The learning_rate, batch_size, and buffer size are all fixed.

    The hyperparameters may be adjusted as needed.

    A couple of important variables are described below:
        - n_steps: The maximum number of steps for the environment
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

    n_steps = 90
    num_cores = cpu_count() - 2
    n_its = 1
    base_steps = 90
    n_experiments = 1
    env_name = "MultiSensorEOS-v0"

    learning_rates = [1e-4]

    widths = [10, 20, 40, 80, 160]
    depths = [1, 2, 4]
    batch_sizes = [int(n_steps * num_cores)]
    buffer_sizes = [int(5e4)]

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
        + "DQN"
        + SEP
        + "MultiSensorEOS-v0"
        + SEP
        + "DQN Test"
        + SEP
    )

    agent_dir = str(Path(__file__).parent.resolve() / agent_dir)

    kwargs_list = create_dqn_kwargs_list(
        widths=widths,
        depths=depths,
        learning_rates=learning_rates,
        batch_sizes=batch_sizes,
        buffer_sizes=buffer_sizes,
        n_experiments=n_experiments,
    )

    run_dqn_experiments(
        agent_dir,
        kwargs_list,
        n_its=n_its,
        base_steps=base_steps,
        index=experiment_index,
        env_name=env_name,
        max_steps=n_steps,
        num_cores=num_cores,
    )
