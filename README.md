# BayesDDQN
This repository contains the code for the paper "A Reinforcement Learning based Framework for Holistic Energy Optimization of Sustainable Cloud Data Centers".

In this work, we propose an energy-efficient framework based on reinforcement learning, where a new joint prediction method named MTL-LSTM is implemented to improve the accuracy of energy consumption and thermal status estimates within the state space, and a novel energy-aware approach named BayesDDQN is designed to avoid IT and cooling resources’ asynchronous allocations in the action space, thereby optimizing the energy consumption.

# Overview
Below is the overall architecture of the proposed holistic energy optimization framework:

<img src="https://github.com/user-attachments/assets/cdeec7ff-90c1-4a64-868c-4cf0d5604e2b" width="30%">

In Analyze module:

* MTL-LSTM based joint prediction method simultaneously modeling both energy consumption and thermal status

In Plan module:

* BayesDDQN based approach leverages Bayesian optimization to synchronize the adjustments of VM migration and cooling parameter within the hybrid action space of the DDQN for achieving the holistic energy optimization
* Pre-cooling technology is employed to proactive adjustment stabilizes temperatures without falling into coordinating extra VM migrations within the hybrid action space to satisfy thermal constraints

# Code Structure

simulator/: code for simulation

* dataloader.py: code for experimental parameters' settings, which can be classified as:
  * PMs and VMs information: server counts and types, VM counts and types
  * Thermal information: Heat Recirculation Matrix (HRM), thresholds of inlet temperature and CPU temperature, default values of thermal resistance and capacitance, supplied CRAC temperature
  * Intial VM distribution: VMs' allocation information on PMs, which can be replaced by referring Example Description file

* utils.py: code for collect PMs and VMs information

* MTL-LSTM.py: code for jointly predicting energy consumption and thermal status

* 

* DDQN.py and run_maze.py: code for training and learning of BayesDDQN

* maze.py: code for evaluate state and action spaces through reward function

* run_bayesddqn.py: code for running and testing BayesDDQN

* match.py: evaluate the VM allocation

* TAS.py: code for TAS approach reproduction

* TA.py: code for TA approach reproduction

* deepee.py: code for DeepEE approach reproduction

* TAWM.py: code for TAWM approach reproduction

requirements.txt: Python dependencies



