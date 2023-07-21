import sys
from bsk_rl.utilities.genetic_algorithm import experiments
import multiprocessing
from pathlib import Path

# Check if any args were passed to the script
if len(sys.argv) > 1:
    # Get the index of the job array
    experiment_index = int(sys.argv[1])
else:
    experiment_index = None

if __name__ == "__main__":
    """
    This script runs a hyperparameter search for the genetic algorithm on the
    MultiSensorEOS-v0 environment. The hyperparameters that are searched are:
        - generations
        - gen_sizes (i.e. population size)

    The hyperparameters may be adjusted as needed.

    n_experiments determines how many times each set of hyperparameters is run.

    The results from this experiment can be plotted using the plotting
    tools in `plotting_tools/`.

    Other Environments that can be used:
        - MultiSensorEOS-v0
        - SimpleEOS-v0
        - AgileEOS-v0
        - MultiSatAgileEOS-v0
    """

    # Set the start method for multiprocessing (required for some Linux systems)
    multiprocessing.set_start_method("spawn")

    # Define the directory
    ga_dir = str(
        Path(__file__).parent.resolve()
        / "../../bsk_rl/results/GA/MultiSensorEOS-v0/GA Test/"
    )

    generations = [10, 20]
    gen_sizes = [10, 20]
    n_experiments = 1

    # Create the hyperparameter list
    ga_kwargs = experiments.create_ga_hp_kwargs_list(
        generations=generations,
        gen_sizes=gen_sizes,
        n_experiments=n_experiments,
    )

    # Run the experiment
    experiments.run_ga_hp_experiment(
        ga_dir,
        ga_kwargs,
        num_cores=4,
        env_name="MultiSensorEOS-v0",
        index=experiment_index,
    )
