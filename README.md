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


