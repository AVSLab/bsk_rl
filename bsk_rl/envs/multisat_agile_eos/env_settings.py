import numpy as np


class env_settings:
    def __init__(self):
        # Define the number of spacecraft
        self.n_spacecraft = 6

        # Set the number of planes
        self.n_planes = 3

        # Set the relative phasing
        self.rel_phasing = 0.0

        # Set the inclination
        self.inc = 45.0

        # Set the communication method
        self.comm_method = "free"

        # Set the number of targets
        self.n_targets = 1040

    def generate_global_targets(self):
        tgt_data = np.random.uniform(-1, 1, (1000, 3))
        tgt_data /= np.linalg.norm(tgt_data, axis=1, keepdims=True)
        tgt_data *= 6371.0 * 1000.0
        tgt_data = np.concatenate(
            (tgt_data, np.random.randint(1, 4, size=(1000, 1))), axis=1
        )
        tgt_pos = tgt_data[:, 0:3]
        tgt_priority = tgt_data[:, 3]

        return tgt_pos, tgt_priority
