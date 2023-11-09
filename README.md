# BSK-RL: Environments and Algorithms for Spacecraft Planning and Scheduling
`bsk-rl` is a Python package consisting of various [Gymnasium](https://gymnasium.farama.org/index.html) environments, agents, training scripts, and examples for spacecraft planning and scheduling problems, with an emphasis on reinforcement learning. This package is the next iteration of the [Basilisk-Gym-Interface library](https://bitbucket.org/avslab/basilisk-gym-interface/src/develop/); environments are upgraded from Gym to Gymnasium and some tools for training decision-making agents for these environments are included.

##	Getting Started

### Prerequisites
`bsk-rl` requires [Basilisk](https://hanspeterschaub.info/basilisk), a spacecraft simulation framework package, to be installed. Instructions for installing and compiling Basilisk may be found [here](https://hanspeterschaub.info/basilisk/index.html). 

### Installation
To install the package, run:

```
pip install -e . && finish_install
```

while in the base directory. This will install `pip` dependencies and download data dependencies (see [#51](https://github.com/AVSLab/bsk_rl/issues/51) for issues with `chebpy` installation on Silicon Macs). Test the installation by running
```
pytest .
```
in the base directory.

When using older versions of Stable Baselines, ensure that [stable-baselines3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) is also installed for Gymnasium support.

## Contents

### Environments
The `envs/` folder provides a number of environments for spacecraft planning and scheduling problems. Environments include:
- **AgileEOS**: An agile Earth-observing satellite scheduling environment
- **GeneralSatelliteTasking** and **SingleSatelliteTasking**: A highly modular framework for satellite tasking problems with the ability to define custom satellite configurations, action spaces, and observation spaces. New environments should be developed using this framework. 
- **MultiSatAgileEOS**: An agile multi-satellite Earth-observing scheduling environment
- **MultiSensorEOS**: A non-agile EO scheduling environment where the spacecraft has multiple sensors for imaging
- **SimpleEOS**: A non-agile EO scheduling environment. Similar to the AgileEOS environment in that it includes a simulated data system.
- **SmallBodyScience**: A small body science operations environment
- **SmallBodySciencePOMDP**: A small body science operations environment with an EKF

###	Agents
The `agents/` folder provides a number of decision-making agents developed by the AVS laboratory. Some of these decision-making agents, such as the `genetic_algorithm` agent, are derived using open-source Python packages.

###	Training
The `training/` folder provides training scripts for the MCTS-Train algorithm and several SB3 algorithms.

### Examples
The `examples/` folder provides examples of how to train decision-making or benchmark agents for various scheduling problems. 

### Utilities
Utilities and tools utilized throughout the repository may be found in the `utilities/` directory. This directory also includes some plotting tools. 

### Results
The `results/` folder can be used to store trained agents and results from benchmarking and validation. All files placed in this folder are ignored by .git. 

## Papers
Several papers utilizing these tools may be found at these links:
- [Generation of Spacecraft Operations Procedures Using Deep Reinforcement Learning](https://hanspeterschaub.info/PapersPrivate/Harris2022a.pdf)
- [Monte Carlo Tree Search Methods for the Earth-Observing Satellite Scheduling Problem](https://hanspeterschaub.info/PapersPrivate/Herrmann2022b.pdf)
- [Reinforcement Learning for the Agile Earth-Observing Satellite Scheduling Problem](https://ieeexplore.ieee.org/document/10058020)

##	License
This project is licensed under the MIT license - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgements
Developed in the [Autonomous Vehicle Systems Laboratory](http://hanspeterschaub.info/main.html) at the University of Colorado, Boulder.

### Authors
Maintainers: 
- Dr. Adam Herrmann (adam.herrmann@colorado.edu)
- Mr. Mark Stephenson (mark.a.stephenson@colorado.edu)
- Mr. Lorenzzo Mantovani

Past Contributors:
- Dr. Andrew T. Harris (andrew.t.harris@colorado.edu)
- Dr. Thibaud Teil (Thibaud.Teil@colorado.edu)
- Mr. Islam Nazmy
- Mr. Trace Valade

### Funding
This work has been supported by NASA Space Technology Graduate Research Opportunity
(NSTGRO) grants, 80NSSC20K1162 and 80NSSC23K1182.

This work has also been supported by Air Force Research Lab grant FA9453-22-2-0050.
