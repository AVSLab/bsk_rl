import math

import gymnasium as gym
import numpy as np

from bsk_rl.utilities.mcts.rollout_policies import SmallBodyRolloutPolicy


class MCTS:
    """
    This provides and MCTS class for deterministic environments.

    To use as UCT, set the rollout type to either 'heuristic' or 'random'
    If a heuristic rollout type is used, create the policy using the following two
    lines of code:

    .. code-block:: python

        stateMachineMCTS = state_machine.StateMachine()
        stateMachineMCTS.loadTransferConditions("agile_eos_ops.adv")
        rollout_policy = AgileEOSRolloutPolicy(env=env, state_machine=stateMachineMCTS)

    Then, load the policy as the rollout_policy during initialization:

    .. code-block:: python

        MCTS_Agent = MCTS(c=c, num_sims=num_sims, rollout_policy=rollout_policy)

    The env and initial conditions must be loaded in after initialization. The
    algorithm will automatically restart the sim and step it forward to the last
    state. This is due to limitations in copying Basilisk:

    .. code-block:: python

        MCTS_Agent.setEnv(
            env_name, env.initial_conditions, max_steps=num_steps, max_length=t_final
        )

    Args:
        c: scaling of the exploration bonus
        num_sims: number of simulations per call of selectAction()
    """

    def __init__(
        self,
        c=1.0,
        num_sims=10,
        max_length=270.0,
        max_steps=45,
        rollout_policy=None,
        backup_type="max",
    ):
        # Set the env to none
        self.env = None
        # Set the initial conditions to none
        self.initial_conditions = None
        # Initialize Q and N as empty dicts
        self.Q = {}
        self.N = {}
        # Initialize Q, N, and the trajectory along the main tree
        self.Q_main = {}
        self.N_main = {}
        self.info = {}
        self.trajectory = []  # Make into a list so order is preserved
        # Initialize T as empty (visited states)
        self.T = []
        # Initialize C
        self.c = c
        # Rollout policy initialization
        self.rollout_policy = rollout_policy
        # Number of simulations
        self.num_sims = num_sims
        # set env type
        self.envType = None
        # set max length and max steps
        self.max_length = max_length
        self.max_steps = max_steps
        self.action_history = None
        # Set the backup type
        self.backup_type = backup_type

    # Define the environment
    def setEnv(self, envType, initial_conditions, max_steps=30, max_length=90):
        """Sets the environment and initial conditions MCTS will step through"""
        # Create the environment
        self.envType = envType
        self.env = gym.make(envType)
        self.env.max_steps = max_steps
        self.env.max_length = max_length
        print(max_steps)
        print(max_length)
        # Set the initial conditions for MCTS
        self.initial_conditions = initial_conditions
        # initialize the environment with the initial conditions
        self.env.reset(options={"initial_conditions": self.initial_conditions})

    # @profiler.profile
    def selectAction(self, s, d, actHist):
        """Selects the next action for the true environment to step through"""
        # We make a tuple out of s so it an be used as a dictionary key
        s_tuple = tuple(s.reshape(1, -1)[0])

        # Run simulate for the specified number, defaults to 10
        for i in range(self.num_sims):
            # Before calling simulate each time, reset env and take it to current place
            # initialize the environment with the initial conditions
            ob, info = self.env.reset(
                options={"initial_conditions": self.initial_conditions}
            )

            # Step through env to take it to current place
            self.env.return_obs = False
            for index, act in enumerate(actHist):
                if index != (len(actHist) - 1):
                    _, _, _, _, _ = self.env.step(act)
                else:
                    s_temp, _, _, _, _ = self.env.step(act)
            self.env.return_obs = True

            if actHist != []:
                try:
                    if tuple(s_temp.reshape(1, -1)[0]) != s_tuple:
                        raise Exception("Environment is not deterministic")
                except Exception:
                    self.rewind_sim(actHist)
                    print("Rewound state: ", tuple(s_temp.reshape(1, -1)[0]))
                    print("Passed state: ", s_tuple)

            # Reset the rollout policy
            if self.rollout_policy is type(SmallBodyRolloutPolicy):
                self.rollout_policy.state_machine.setupSmallBodyScience()

            self.action_history = np.copy(actHist).tolist()
            self.simulate(s, d)

        # Copy Q[s] to Q_main
        self.Q_main[s_tuple] = self.Q[s_tuple].copy()
        self.N_main[s_tuple] = self.N[s_tuple].copy()

        # Return the action associated with the max Q at the current state
        try:
            print("Q(s): ", self.Q[s_tuple])
            print("N(s): ", self.N[s_tuple])
        except KeyError:
            print("Raw s: ", s)
            print("Tuple s: ", s_tuple)

        print("s: ", s_tuple)
        return max(self.Q[s_tuple], key=self.Q[s_tuple].get)

    def simulate(self, s, d):
        """Simulates a trajectory through the environment and updates Q_search"""
        # We make a tuple out of s so it an be used as a dictionary key
        try:
            s_tuple = tuple(s.reshape(1, -1)[0])
            if np.isnan(np.sum(s_tuple)):
                raise Exception("State is NaN")
        except Exception:
            s = self.rewind_sim(self.action_history)
            self.simulate(s, d)

        # If depth is zero, return 0
        if d == 0:
            return 0.0

        # If s is not in T
        if s_tuple not in self.T:
            # Initialize empty dicts inside of Q and N
            self.Q[s_tuple] = {}
            self.N[s_tuple] = {}
            # Loop through actions
            for a in range(self.env.action_space.n):
                # Initialize Q and N for action and state to zero
                self.Q[s_tuple][a] = -self.env.failure_penalty
                self.N[s_tuple][a] = 0.0
            # Add s to visited set
            self.T.append(s_tuple)
            # Roll it out
            return self.rollout(s, d)

        function_max = {}
        for action in self.Q[s_tuple]:
            function_max[action] = self.Q[s_tuple][action] + self.c * math.sqrt(
                sum(self.N[s_tuple].values())
            ) / (1 + self.N[s_tuple][action])

        # Get action
        a = max(function_max, key=function_max.get)

        # Take our new action!
        sp, reward, episode_over, _, _ = self.env.step(a)
        self.action_history.append(a)

        # If the episode is over, don't call simulate recursively. Sim is over.
        if episode_over:
            q = reward
        # Otherwise, call simulate recursively using new action history and add to q
        else:
            q = reward + self.simulate(sp, d - 1)

        # Update N
        self.N[s_tuple][a] = self.N[s_tuple][a] + 1

        # Update Q
        if self.backup_type == "incremental_avg":
            self.Q[s_tuple][a] = (
                self.Q[s_tuple][a] + (q - self.Q[s_tuple][a]) / self.N[s_tuple][a]
            )
        elif self.backup_type == "max":
            # Max-q
            if self.Q[s_tuple][a] < q:
                self.Q[s_tuple][a] = q
        else:
            raise Exception("Invalid backup type")

        return q

    def rollout(self, s, d):
        """Executes a rollout to the desired depth or end of the environment"""
        # If we have reached max depth, just return 0
        if d == 0:
            return 0.0
        # If we have not reached max depth, select action using rollout policy
        else:
            # Select action using rollout policy
            act = self.rollout_policy.act(s)

            # Take a step in the environment
            sp, reward, episode_over, _, _ = self.env.step(act)

            # If the episode is over, return the penalty reward only
            if episode_over:
                return reward

            # Recursively call rollout
            return reward + self.rollout(sp, d - 1)

    def backup_tree(self):
        """Backs up the value along the main tree once the sim has terminated"""
        # 'Anti-Sum' is added to at each node to subtract from r_sum at last node
        r_anti_sum = 0
        # Grab the total reward from the last node
        r_sum = self.trajectory[-1][3]
        for idx, node in enumerate(self.trajectory):
            # If we're at the first node
            if idx == 0:
                # Update the state-action value at that node with r_sum
                self.Q_main[node[0]][node[1]] = r_sum
                # Add the reward at the node to r_anti_sum
                r_anti_sum += node[2]
            else:
                self.Q_main[node[0]][node[1]] = r_sum - r_anti_sum
                r_anti_sum += node[2]
            # Add the info
            self.info[node[0]] = node[4]

    def rewind_sim(self, action_history):
        self.env.reset(options={"initial_conditions": self.initial_conditions})
        for action in action_history:
            s, _, _, _, _ = self.env.step(action)

        # Check to make sure it isn't nan again
        try:
            s_tuple = tuple(s.reshape(1, -1)[0])
            if np.isnan(np.sum(s_tuple)):
                raise Exception("State is NaN inside of rewind_sim")
        except Exception:
            print("State is NaN inside of rewind_sim")
            return self.rewind_sim(self.action_history)

        return s
