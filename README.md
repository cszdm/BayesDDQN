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

* DDQN.py and run_maze.py: code for training and learning of BayesDDQN

* Maze_BayesDDQN.py: code for evaluate state and action spaces through reward function of BayesDDQN

* Maze_DeepEE.py: code for evaluate state and action spaces through reward function of DeepEE

* run_bayesddqn.py: code for running and testing BayesDDQN

* match.py: evaluate the VM allocation

* TAS.py: code for TAS approach reproduction

* TA.py: code for TA approach reproduction

* deepee.py: code for DeepEE approach reproduction

* TAWM.py: code for TAWM approach reproduction

plot/: code for drawing figures of experimental results

data-traces: dataset for wind speed and PMs workload, the citations of them are also given in data sources.md

requirements.txt: Python dependencies

# Usage

**Requirements**

The repo is tested on:

* Ubuntu 22.04.4 LTS  
* Python 3.9.6  
* Tensorflow 2.6.0  

```pip install -r requirements.txt```

**MTL-LSTM Prediction**

For regression and prediction, MTL-LSTM is used to evaluate energy consumption and thermal status simultaneously:  

```
cd simulator
python MTL-LSTM.py
```
**Analyze Module**

Accuracy of various training methods:

```
cd simulator
python eval_prediction.py
python lstm.py
python xgboost.py
```

Results will be consistent with Table 4 in the paper where prediction RMSEs of energy consumption and inlet temperature are compared.

**Plan Module** 

For VM migrations and supplied CRAC adjustments, BayesDDQN will be executed to achieve the holistic energy optimization:

```
cd simulator
python dataloader.py
python run_bayesddqn.py
```

The BayesDDQN will call Maze_BayesDDQN.py, run_maze.py, utilis.py automatically.

**Comparision Methods**

In this paper, TAS, TA, DeepEE, TAWM are evaluated as baseline methods, they will be executed:

```
cd simulator
python TAS.py
python TA.py
python DeepEE.py
python TAWM.py
```

**Evaluation Metrics**

In this paper, prediction accuracy, energy consumption, temperature, RES utilization, and VM migrations are used as evaluation metrics, results are recorded which are shown in Figure 4 to Figure 7. 

For MTL-LSTM prediction accuracy:

```
cd plot
python training_loss.py
python energy_prediction.py
python inlet_prediciton.py
```

For energy consumption:

```
cd plot
python TAS_energy.py
python TA_energy.py
python DeepEE_energy.py
python TAWM_energy.py
python BayesDDQN_energy.py
```

For evaluating energy savings of BayesDDQN in 95% Confidence Interval：

```
cd plot
python CI_energy.py
```

For thermal status comparison:

```
cd plot
python CPU_temperature.py
python Inlet_temperature.py
python Cooling_temperature.py
python Hotspots.py
```


