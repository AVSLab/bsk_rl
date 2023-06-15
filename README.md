# BSK-RL: Environments and Algorithms for Spacecraft Planning and Scheduling
bsk_rl is a python package consisting of various agents, environments, training scripts, and examples for spacecraft planning and scheduling problems, with an emphasis on reinforcement learning. 

This package is the next iteration of the Basilisk-Gym-Interface library found [here](https://bitbucket.org/avslab/basilisk-gym-interface/src/develop/). This package upgrades some of those environments to the [Gymnasium](https://gymnasium.farama.org/) interface. Furthermore, this package also integrates the tools internal to the AVS laboratory that train decision-making agents for these environments.

##	Getting Started
These instructions will give you the information needed to install and use bsk_rl.

### Prerequisites
bsk_rl requires the following Python packages:

- Numerics and plotting: numpy, matplotlib, pandas
- Simulation software: gymnasium and basilisk
- MCTS agents: keras, scikit-learn
- Stable-Baslines Agents: stable-baselines3, torch
- Genetic algorithms: DEAP

Instructions for installing Basilisk may be found [here](https://hanspeterschaub.info/basilisk/index.html).

Ensure that [stable-baselines3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) is also installed (this requirement may drop off as Gymnasium support is integrated into vanilla SB3).

### Installation
To install the package, run:

```
pip install -e .
```

while in the base directory. Test the installation by opening a Python terminal and calling:

```
import bsk_rl
```

### Papers
Several papers utilizing these tools may be found at these links:
- [Generation of Spacecraft Operations Procedures Using Deep Reinforcement Learning](https://hanspeterschaub.info/PapersPrivate/Harris2022a.pdf)
- [Monte Carlo Tree Search Methods for the Earth-Observing Satellite Scheduling Problem](https://hanspeterschaub.info/PapersPrivate/Herrmann2022b.pdf)
- [Reinforcement Learning for the Agile Earth-Observing Satellite Scheduling Problem](https://ieeexplore.ieee.org/document/10058020)

###	Agents
The `agents/` folder provides a number of decision-making agents developed by the AVS laboratory. Some of these decision-making agents, such as the `genetic_algorithm` agent, are derived using open-source Python packages.

### Environments
The `envs/` folder provides a number of environments for spacecraft planning and scheduling problems. Environments include:
- AgileEOS: An agile Earth-observing satellite scheduling environment
- MultiSatAgileEOS: An agile multi-satellite Earth-observing scheduling environment
- MultiSensorEOS: A non-agile EO scheduling environment where the spacecraft has multiple sensors for imaging
- SimpleEOS: A non-agile EO scheduling environment. Similar to the AgileEOS environment in that it includes a simulated data system.
- SmallBodyScience: A small body science operations environment
- SmallBodySciencePOMDP: A small body science operations environment with an EKF

###	Training
The `training/` folder provides training scripts for the MCTS-Train algorithm and several SB3 algorithms.

### Examples
The `examples/` folder provides examples of how to train decision-making or benchmark agents for various scheduling problems. 

### Utilities
Utilities and tools utilized throughout the repository may be found in the `utilities/` directory. This directory also includes some plotting tools. 

### Results
The `results/` folder can be used to store trained agents and results from benchmarking and validation. All files placed in this folder are ignored by .git. 

## Contribution Guidelines
The repository enforces that code is formatted by `black` and `isort`, and that commit messages take the form `Issue #XXX: Commit message`, not exceeding 72 characters.

## Authors
Maintainers: 
Mr. Adam Herrmann (adam.herrmann@colorado.edu)
Mr. Mark Stephenson (mark.a.stephenson@colorado.edu)

Past Contributors:
Dr. Andrew T. Harris (andrew.t.harris@colorado.edu)
Dr. Thibaud Teil (Thibaud.Teil@colorado.edu)
Mr. Islam Nazmy
Mr. Trace Valade

##	License
This project is licensed under the MIT license - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgements
[The Autonomous Vehicle Systems Laboratory](http://hanspeterschaub.info/main.html)
