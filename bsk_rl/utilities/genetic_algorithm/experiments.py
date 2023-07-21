import os
import numpy as np
from bsk_rl.agents.genetic_algorithm import ga_env_solver

SEP = os.path.sep


def create_ga_hp_kwargs_list(**kwargs):
    generations = kwargs.get("generations", [100, 200])
    gen_sizes = kwargs.get("gen_sizes", [10, 20])
    n_experiments = kwargs.get("n_experiments", 1)

    # Create the hyperparameter list
    ga_kwargs = []

    # Set the experiment ID to -1
    experiment_ID = -1

    # Loop over the number of experiments
    for gens in generations:
        for gen_size in gen_sizes:
            experiment_ID += 1
            for i in range(n_experiments):
                ga_kwargs.append(
                    {
                        "n_gens": gens,
                        "gen_size": gen_size,
                        "experiment_ID": experiment_ID,
                    }
                )

    return ga_kwargs


def run_ga_hp_experiment(
    ga_dir,
    ga_kwargs,
    index=None,
    num_cores=4,
    env_name="AgileEOS-v0",
):
    """
    Runs a genetic algorithm experiment hyperparameter search experiment over the
    number of generations and population size."""

    # Set the results dictionary
    results = {}
    idx = 0

    if index is not None:
        ga_kwargs = [ga_kwargs[index]]
        idx = index

    for kwargs in ga_kwargs:
        n_gens = kwargs["n_gens"]
        gen_size = kwargs["gen_size"]

        # Create the solver
        solver = ga_env_solver(
            env_name,
            n_gens,
            gen_size,
            ga_dir=ga_dir + SEP + "ga_" + str(idx) + SEP,
            num_cores=num_cores,
        )

        # Run the solver
        max_reward, metrics = solver.optimize()

        # Save the results
        results[idx] = {
            "n_gens": n_gens,
            "gen_size": gen_size,
            "max_reward": max_reward,
            "metrics": metrics,
        }

        # Intermittently save the results
        np.save(ga_dir + SEP + "results_" + str(idx) + ".npy", results)

        # Increment the index
        idx += 1
