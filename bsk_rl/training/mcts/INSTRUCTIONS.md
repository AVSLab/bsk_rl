# INSTRUCTIONS: How to generate state-action value function decision-making agent using MCTS

## Overview
Generating state-action value functions to serve as decision-making agents is a simple three-step process, referred to
as MCTS-Train:

1. Training data generation with MCTS
2. Neural network function approximation
3. Validation of the trained neural networks

This process is described in detail in this paper by [Herrmann and Schaub](https://arc.aiaa.org/doi/10.2514/1.I010992).

An example is provided in the `examples/` directory. Currently, MCTS-Train works for the:
1. SimpleEOS Environment
2. MultiSensorEOS Environment
3. AgileEOS Environment
4. SmallBodyScience Environment (only MCTS has been attempted for this environment, not the full pipeline)

## Generating Training Data
Training data is generated with scripts titled `mcts_data_generator.py`.

## Neural Network Hyperparameter Search
A hyperparameter search over the neural networks is performed with a file titled `network_hyperparam_search.py`. Be
sure to generate the training data before performing the hyperparameter search.

## Validating the Trained Agents
Validation of the trained agents is performed with a file titled `network_validation.py`. If `_multiprocessing` is
appended to the end of the name of this script, that means that the script supports Python multiprocessing.

Be sure to train the neural networks before attempting to validate them.

## Misc.
The `mcts_train.py` file contains the majority of the functionality for the training process.