from multiprocessing import Process, Manager
from bsk_rl.training.mcts.mcts_train import mcts_batch
import numpy as np
from os import cpu_count
import os
from pathlib import Path

SEP = os.path.sep

if __name__ == "__main__":
    """
    This scripts provides an example of how to generate data for supervised learning
    using Monte Carlo tree search. After running this script and generating the training
    data, the networks may be trained with `network_hyperparam_search.py`.
    """

    # Set the number of processes for multi-processing
    num_processes = cpu_count() - 2

    data_directory = (
        ".."
        + SEP
        + ".."
        + SEP
        + "bsk_rl"
        + SEP
        + "results"
        + SEP
        + "mcts"
        + SEP
        + "AgileEOS"
        + SEP
        + "MCTS-Train Test"
        + SEP
    )

    data_directory = str(Path(__file__).parent.resolve() / data_directory)

    # Set the exploration constant
    c = 4

    # Set the number of simulations-per-step
    num_sims = 5

    # Set the amount of data to generate
    data_amt = 5

    # Set the starting index for the data generation
    start_idx = 0
    idx = start_idx

    # Set the max_steps and final time
    max_steps = 5
    final_time = 270.0

    # Initialize a manager to save the results
    manager = Manager()
    results = manager.list()

    # Loop
    while idx < data_amt:
        # Initialize processes
        processes = []
        # Loop through and create each process
        for i in range(num_processes):
            # Append to the processes
            processes.append(
                Process(
                    target=mcts_batch,
                    args=(
                        data_directory,
                        "c_"
                        + str(c)
                        + "_num_sims_"
                        + str(num_sims)
                        + "_IC_"
                        + str(idx),
                        c,
                        num_sims,
                        None,
                        results,
                        False,
                        "AgileEOS-v0",
                        max_steps,
                        final_time,
                    ),
                )
            )
            # Start the process
            processes[i].start()
            idx += 1
        # Join each process
        for process in processes:
            process.join()

    # Save the data
    np.save(
        data_directory
        + "/results_"
        + str(start_idx)
        + "_"
        + str(data_amt)
        + "_sims.npy",
        results,
    )
