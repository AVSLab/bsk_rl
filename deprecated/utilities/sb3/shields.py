import os

import torch as th

from bsk_rl.agents.state_machine import StateMachine

abs_path = os.path.dirname(os.path.abspath(__file__))


class AgileEOSShield:
    """Custom shield layer
    :param: size_in -> input size, should be equal to the number of actions
    :param: size_out -> output size, should be equal to the number of actions"""

    def __init__(self):
        self.state_machine = StateMachine()
        self.state_machine.loadTransferConditions(
            os.path.join(abs_path, "../state_machine/agile_eos_ops.adv")
        )

    def shield_actions(self, obs, actions):
        batch_size = obs.shape[0]
        new_actions = th.ones(batch_size, dtype=th.int32, device="cpu")

        for batch_num in range(0, batch_size):
            # Define the state
            s = obs[batch_num, :]

            # Pass the state to the shield
            discretized_state = self.state_machine.SimpleEOSDiscretizer(
                s.cpu().detach().numpy()
            )

            # Grab the action from the shield
            shield_act = self.state_machine.selectAction(discretized_state)

            # If nominal state, only modify if we have a downlink window
            if shield_act == 3:
                # Downlink if data is in the buffer and a ground station is available
                if (s[13] > 5e-5) and any(s[15:22]):
                    # Randomly choose between downlink and imaging
                    shield_act = th.randint(2, 4, [1])

            # If the action is image, let the network do what it wants
            if shield_act >= 3:
                new_actions[batch_num] = actions[batch_num]
            # If the action is not nominal, let the shield override
            else:
                new_actions[batch_num] = shield_act

        return new_actions


class MultiSensorEOSShield:
    """Custom shield layer
    :param: size_in -> input size, should be equal to the number of actions
    :param: size_out -> output size, should be equal to the number of actions"""

    def __init__(self):
        self.state_machine = StateMachine()
        self.state_machine.loadTransferConditions(
            os.path.join(abs_path, "../state_machine/multisensor_eos_ops.adv")
        )

    def shield_actions(self, obs, actions):
        batch_size = obs.shape[0]
        new_actions = th.ones(batch_size, dtype=th.int32, device="cpu")

        for batch_num in range(0, batch_size):
            # Define the state
            s = obs[batch_num, :]

            # Pass the state to the shield
            discretized_state = self.state_machine.earthObsEnvDiscretizer(
                s.cpu().detach().numpy()
            )

            # Grab the action from the shield
            shield_act = self.state_machine.selectAction(discretized_state)

            # If the action is image, let the network do what it wants
            if shield_act >= 2:
                new_actions[batch_num] = actions[batch_num]
            # If the action is not nominal, let the shield override
            else:
                new_actions[batch_num] = shield_act

        return new_actions
