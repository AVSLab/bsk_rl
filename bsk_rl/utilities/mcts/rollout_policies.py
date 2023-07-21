import numpy as np


class RolloutPolicy:
    def __init__(self, env=None, state_machine=None):
        self.env = env
        self.state_machine = state_machine

    def act(self):
        if self.env is None:
            raise Exception("RolloutPolicy: Environment not set")

        if self.state_machine is None:
            raise Exception("RolloutPolicy: State machine not set")


class SimpleEOSRolloutPolicy(RolloutPolicy):
    def __init__(self, env=None, state_machine=None):
        super().__init__(env=env, state_machine=state_machine)

    def act(self, s):
        discretized_state = self.state_machine.SimpleEOSDiscretizer(s)
        act = self.state_machine.selectAction(discretized_state)
        # Modify to downlink data whenever we have data and a ground station is in view
        if (s[13] > 5e-5) and any(s[15:22]) and (act == 0):
            act = 3
        return act


class AgileEOSRolloutPolicy(RolloutPolicy):
    def __init__(self, env=None, state_machine=None):
        super().__init__(env=env, state_machine=state_machine)

    def act(self, s):
        super().act()
        discretized_state = self.state_machine.AgileEOSEnvDiscretizer(s)
        act = self.state_machine.selectAction(discretized_state)
        # Modify to either downlink or attempt to capture the nearest target
        if act == 3:
            # Downlink if data is in the buffer and a ground station is available
            if (s[13] > 5e-5) and any(s[15:22]):
                act = 2
            # Grab the nearest target using the normalized Hill-frame coordinates
            else:
                min_target = np.linalg.norm(s[23 : 23 + 3])
                for idx in range(1, self.env.simulator.n_targets):
                    if (
                        np.linalg.norm(
                            s[
                                23
                                + idx * self.env.simulator.target_tuple_size : 23
                                + idx * self.env.simulator.target_tuple_size
                                + 3
                            ]
                        )
                        < min_target
                    ):
                        min_target = np.linalg.norm(
                            s[
                                23
                                + idx * self.env.simulator.target_tuple_size : 23
                                + idx * self.env.simulator.target_tuple_size
                                + 3
                            ]
                        )
                        act = 3 + idx
        return act


class RandomAgileRolloutPolicy(RolloutPolicy):
    def __init__(self, env=None):
        super().__init__(env=env)

    def act(self, s):
        act = np.random.randint(self.env.action_space.n)
        # Modifying to downlink data whenever we're in view and have data
        if (s[13] > 5e-5) and any(s[15:22]):
            act = 3

        return act


class SmallBodyRolloutPolicy(RolloutPolicy):
    def __init__(self, env=None, state_machine=None):
        super().__init__(env=env, state_machine=state_machine)

    def act(self, s):
        super().act()
        discretized_state = self.state_machine.smallBodyScienceEnvDiscretizer(s)
        act = self.state_machine.smallBodyScienceAct(
            discretized_state,
            self.env.simulator.phi_c,
            self.env.simulator.lambda_c,
            self.env.simulator.simTime,
            self.env.simulator.waypointTime,
            self.env.simulator.requiredWaypointTime,
            s,
            self.env.simulator.waypoint_hist,
        )

        return act


class RandomSmallBodyRolloutPolicy(RolloutPolicy):
    def __init__(self, env=None):
        super().__init__(env=env)

    def act(self, s):
        return np.random.randint(self.env.action_space.n)


class KerasRolloutPolicy(RolloutPolicy):
    def __init__(self, model):
        super().__init__(env=None, state_machine=None)
        self.model = model

    def act(self, s):
        ob_temp = np.concatenate(s)
        net_output = self.model.predict(ob_temp.reshape(1, -1))
        action_value = net_output[0, 0 : int(len(net_output[0]) / 2)]
        return max(action_value)


class MultiSensorEOSRolloutPolicy(RolloutPolicy):
    def __init__(self, env=None, state_machine=None):
        super().__init__(env=None, state_machine=None)
        self.env = env
        self.state_machine = state_machine

    def act(self, s):
        super().act()
        discretized_state = self.state_machine.earthObsEnvDiscretizer(s)
        act = self.state_machine.selectAction(discretized_state)
        # If the action is image
        if act == 2:
            act = int(self.env.simulator.img_modes * s[-1]) + 1
        return act
