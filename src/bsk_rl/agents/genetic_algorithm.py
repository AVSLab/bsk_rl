import multiprocessing
import os
import pickle
import random
from datetime import datetime
from os import sep as SEP

import gymnasium as gym
import numpy as np
from deap import algorithms, base, creator, tools
from matplotlib import pyplot as plt

"""
This file is designed to provide generitc interfaces to a DEAP-based genetic algorithm
for solving arbitrary gymnasium environments. It does a few things:
1. Includes the ability to cast arbitrary gymnasium environments with a max_length
    parameter as many-input, single-output optimzation problems
2. Allows a genetic algorithm to be called to optimize said environments in a parellel
    way
"""

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def mutUniformIntList(individual, num_samples=2, low=0, up=5, indpb=0.3):
    """
    This function is designed to mutate a list of integers, rather than a single
    integer.
    :param individual: The individual to be mutated
    :param num_samples: The number of samples to be mutated
    :param low: The lower bound for the mutation
    :param up: The upper bound for the mutation
    :param indpb: The probability of mutation
    :returns: A list of a single individual.
    """

    size = len(individual)
    for i in range(size):
        for j in range(num_samples):
            if random.random() < indpb:
                individual[i][j] = random.randint(low, up)

    return (individual,)


class env2opt_problem(object):
    """
    This class provides a generic interface for recasting a gymnasium environment as a
    total reward maximization problem.
    """

    def __init__(self, env_name, initial_conditions=None):
        self.env_name = env_name
        tmp_env = gym.make(env_name)
        if initial_conditions is None:
            tmp_env.reset()
            self.initial_conditions = tmp_env.simulator.initial_conditions
        else:
            self.initial_conditions = initial_conditions

        try:
            self.n_actions = tmp_env.max_steps
        except AttributeError:
            print("Error - environment must have max_steps attribute.")
            return

        self.action_space = tmp_env.action_space

    def evaluate(self, action_set):
        """
        Evaluates a full run of the environment given a list of actions.
        """
        total_reward = 0

        self.env = gym.make(self.env_name)
        self.env.reset(options={"initial_conditions": self.initial_conditions})

        for ind in range(0, self.n_actions):
            _, r, ep_over, _, info = self.env.step(action_set[ind])
            total_reward = total_reward + r
            if ep_over:
                break

        # Initialize a metrics dictionary
        metrics = {}
        # Add the total reward to the metrics dictionary
        metrics["total_reward"] = total_reward
        # Add the number of steps taken to the metrics dictionary
        metrics["steps"] = ind + 1
        if self.env_name == "AgileEOS-v0" or self.env_name == "MultiSatAgileEOS-v0":
            metrics["imaged_targets"] = np.sum(self.env.simulator.imaged_targets)
            metrics["downlinked_targets"] = np.sum(
                self.env.simulator.downlinked_targets
            )
        elif self.env_name == "MultiSensorEOS-v0":
            metrics["imaged_targets"] = np.sum(self.env.simulator.imaged_targets)
        elif self.env_name == "SmallBodyScience-v0":
            metrics["imaged_targets"] = np.sum(self.env.simulator.imaged_targets)
            metrics["downlinked_targets"] = np.sum(
                self.env.simulator.downlinked_targets
            )
            metrics["imaged_maps"] = self.env.simulator.imaged_maps
            metrics["downlinked_maps"] = self.env.simulator.downlinked_maps
        elif self.env_name == "SimpleEOS-v0":
            metrics["downlinked"] = self.env.simulator.downlinked
            metrics["total_access"] = self.env.simulator.total_access
            metrics["utilized_access"] = self.env.simulator.utilized_access

        if self.env_name == "MultiSatAgileEOS-v0":
            return (total_reward[0, -1], metrics)
        else:
            return (total_reward, metrics)


class ga_env_solver(object):
    def __init__(
        self,
        env_name,
        n_gens,
        gen_size,
        cx_prob=0.25,
        mut_prob=0.25,
        num_cores=4,
        ga_dir="ga_results" + SEP,
    ):
        self.env_name = env_name
        self.n_gens = n_gens
        self.gen_size = gen_size
        self.ga_dir = ga_dir
        self.num_cores = num_cores

        problem = env2opt_problem(env_name)

        #   Define some problem-specific parameters
        IND_SIZE = (
            problem.n_actions
        )  # Each individual is a list of actions to take at each env.step()

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        #   Check to see whether the env takes integer, list of integer, or float inputs
        if type(problem.action_space) == gym.spaces.discrete.Discrete:
            act_min = 0
            act_max = problem.action_space.n - 1
            self.toolbox.register("attr_action", random.randint, act_min, act_max)
            self.toolbox.register(
                "mutate", tools.mutUniformInt, low=act_min, up=act_max, indpb=0.3
            )
        elif type(problem.action_space) == gym.spaces.multi_discrete.MultiDiscrete:
            num_agents = len(problem.action_space.nvec)
            act_min = 0
            act_max = problem.action_space.nvec[0] - 1
            self.toolbox.register(
                "attr_action", random.sample, range(act_min, act_max + 1), num_agents
            )
            self.toolbox.register(
                "mutate",
                mutUniformIntList,
                num_samples=num_agents,
                low=act_min,
                up=act_max,
                indpb=0.3,
            )

        self.MU = gen_size  # Number of individuals per generation
        self.NGEN = n_gens  # Number of generations
        self.CXPB = cx_prob  # Crossover probability.
        self.MUTPB = mut_prob  # Mutation probability.

        # generation functions
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            (self.toolbox.attr_action),
            n=IND_SIZE,
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # evolutionary ops
        self.toolbox.register("mate", tools.cxOnePoint)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        #   Evaluation ops
        self.toolbox.register("evaluate", problem.evaluate)

    def optimize(self, seed=datetime.now(), checkpoint_freq=1, checkpoint=None):
        if not os.path.exists(self.ga_dir):
            os.makedirs(self.ga_dir)

        pool = multiprocessing.Pool(min(self.num_cores, self.MU))
        self.toolbox.register("map", pool.map)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        print("Multiprocessing start method: ", multiprocessing.get_start_method())

        if checkpoint is not None:
            with open(checkpoint, "rb") as checkpoint_file:
                check_dict = pickle.load(checkpoint_file)
                pop = check_dict["population"]
                start_gen = check_dict["generation"]
                logbook = check_dict["logbook"]
        else:
            logbook = tools.Logbook()
            logbook.header = "gen", "evals", "std", "min", "avg", "max"
            pop = self.toolbox.population(n=self.MU)

            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            fitness_list = [list(fitness) for fitness in fitnesses]

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit[0],)

            record = stats.compile(pop)
            logbook.record(gen=0, evals=len(invalid_ind), **record)
            print(logbook.stream)
            start_gen = 1

        for gen in range(start_gen, start_gen + self.NGEN):
            offspring = algorithms.varAnd(pop, self.toolbox, self.CXPB, self.MUTPB)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit[0],)

            fitness_list.extend([list(fitness) for fitness in fitnesses])

            pop = self.toolbox.select(pop + offspring, self.MU)

            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            print(logbook.stream)

            if gen % checkpoint_freq == 0:
                chkpnt = dict(
                    population=pop,
                    generation=gen,
                    logbook=logbook,
                    rndstate=random.getstate(),
                )
                pickle.dump(
                    chkpnt, open(self.ga_dir + self.env_name + f"_{gen}.pkl", "wb")
                )

        fitness_list = np.array(fitness_list)
        index = np.where(fitness_list[:, 0] == np.max(fitness_list[:, 0]))[0]
        max_reward = fitness_list[index[0], 0]
        metrics = fitness_list[index[0], 1]

        results = (max_reward, metrics)

        return results

    def plot_results(self):
        """
        Generates plots of GA convergence, performance, and final population vector
        behavior
        :param checkpoint_names: list of checkpoint names, assumed to be pickle files.
        :return:
        """
        #   Pull data out of the pickles
        gen_list = []
        min_reward = []
        max_reward = []
        mean_reward = []

        checkpoint_names = [
            self.ga_dir + self.env_name + f"_{gen}.pkl" for gen in range(1, self.n_gens)
        ]

        for name in checkpoint_names:
            with open(name, "rb") as file:
                result = pickle.load(file)
            gen_list.append(result["generation"])
            tmp_log = result["logbook"]
            mean_reward.append(tmp_log[-1]["avg"])
            min_reward.append(tmp_log[-1]["min"])
            max_reward.append(tmp_log[-1]["max"])

        fig, ax = plt.subplots(figsize=(7, 5))
        plt.plot(gen_list, mean_reward, label="Mean Reward")
        plt.plot(gen_list, max_reward, "--", label="Max Reward")
        plt.plot(gen_list, min_reward, "--", label="Min Reward")
        plt.grid()
        plt.ylabel("Total Reward", fontsize=16)
        plt.xlabel("Training Generation", fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.savefig(
            self.ga_dir + self.env_name + "_training.png",
            dpi=300,
            pad_inches=0.1,
            bbox_inches="tight",
            transparent=True,
        )

        plt.show()
