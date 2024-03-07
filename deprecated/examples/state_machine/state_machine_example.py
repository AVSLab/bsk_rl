import os
from pathlib import Path

import gymnasium as gym

from bsk_rl.agents.state_machine import StateMachine

SEP = os.path.sep

if __name__ == "__main__":
    """
    This script runs a single episode of the AgileEOS-v0 environment using the
    StateMachine agent. The StateMachine agent is at the core of the MCTS rollout
    policies and PPO shields. This script is useful for debugging or tuning the
    StateMachine.
    """
    # Create env
    env = gym.make("AgileEOS-v0")

    # Set num steps
    tFinal = float(270.0)  # number of minutes to conduct sims
    numSteps = int(tFinal / (env.step_duration / 60.0))  # number of steps
    env.max_length = tFinal
    env.max_steps = numSteps

    # Call reset on the environment
    ob, info = env.reset()

    ops_dir = (
        ".."
        + SEP
        + ".."
        + SEP
        + "bsk_rl"
        + SEP
        + "utilities"
        + SEP
        + "state_machine"
        + SEP
        + "agile_eos_ops.adv"
    )

    ops_dir = str(Path(__file__).parent.resolve() / ops_dir)

    # We create a rollout policy using a simple state machine
    StateMachineAgent = StateMachine()
    # Load transfer conditions
    StateMachineAgent.loadTransferConditions(ops_dir)

    actHist = []

    # Total reward
    reward_sum = 0.0
    episode_over = False

    # Loop through each time step
    for ind in range(0, numSteps):
        discretized_state = StateMachineAgent.AgileEOSEnvDiscretizer(ob)
        act = StateMachineAgent.selectAction(discretized_state)

        # Append last action to action list
        actHist.append(act)

        print("Actual Action Taken: ", act)
        print("Real Environment Step: ", ind)

        # Take the step in the environment
        ob, reward, episode_over, _, _ = env.step(act)
        print("Reward: ", reward)

        # Sum the reward, add to rewards array
        reward_sum = reward_sum + reward

        # If the episode is over, end the simulation
        if episode_over:
            print("episode over")
            break

    if not episode_over:
        print("Successfully avoided resource constraint violations")
    else:
        print("Resource constraint violation occurred")

    print("Total Reward: ", reward_sum)
