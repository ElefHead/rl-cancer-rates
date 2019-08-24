# QuBBD RL
Codebase containing reinforcement learning experiments for SMART-ACT project using the QuBBD data

This repository contains two experiments:
1. _Simulator_: Completely simulated data and environment. Self defined policy based on random vectors.
2. _QuBBD_: Policy defined based on Qubbd v3 data.

## Directory Structure
The structure of the project as it is now is as follows:
* /root
    * /QuBBD
        * /data (find data on Box)
        * constants.py
        * preprocess.py
        * utils.py
    * /simulator
        * dqn.py
        * model.py
        * plotting.py
        * policy.py
        * run_exp.py

## Acknowledgements
* Deep Q-Network model has been adapted from [Denny Britz](https://twitter.com/dennybritz?lang=en)'s [repository](https://github.com/dennybritz/reinforcement-learning/tree/master/DQN) and modified for our purpose.
* Policy definitions have been inspired by UC Berkeley [CS 294-112](http://rail.eecs.berkeley.edu/deeprlcourse/) [homework](https://github.com/berkeleydeeprlcourse/homework) definitions.
---

This is still an ongoing project.

