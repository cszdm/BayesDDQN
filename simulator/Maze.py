import numpy as np
import pandas as pd
import random
import csv
import heapq
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import itertools
from functools import reduce
from bayes_opt import BayesianOptimization
import operator
import copy
import math
from scipy.linalg import solve
from sklearn.preprocessing import MinMaxScaler

class Maze():
    def __init__(self):
        dim_1 = sum(server_count) * len(Server_VM[sourceRack][sourceServer])    
        dim_2 = (T_red - T_initial) / 0.1 + 1  
        self.n_actions = dim_1
        self.n_features = len(Server_VM[sourceRack][sourceServer])      

    
    def move(self, observation, action, vm_index, server_index, vm_id, Server_VM, bayes_T_sup):
        observation_ = observation.copy()
        
        observation_[vm_index] = action
        action_Rack, action_Server = choose_server(server_index)
        
       
        new_cpu_Utilization, new_inlet_Temperature, new_server_Temperature, new_Rack_Power, new_CoP, \
        new_IT_Power, new_Cooling_Power, new_Total_Power, new_Brown_Energy = updateInformation(t, Server_VM, bayes_T_sup)

        r1 = []
        r1.append(new_IT_Power)
        r1.append(new_Total_Power / new_Cooling_Power)
        r1.append(new_Brown_Energy)
        r1.append(np.mean(new_inlet_Temperature))
        r1.append(np.mean(new_server_Temperature))
        r1.append(new_Total_Power)

        
        Server_VM[sourceRack][sourceServer].remove(vm_id)
        if (sourceRack != action_Rack) or (sourceServer != action_Server):
            Server_VM[action_Rack][action_Server].append(vm_id)
        else:
            
            Server_VM[action_Rack][action_Server].insert(vm_index, vm_id)

        
        new_cpu_Utilization, new_inlet_Temperature, new_server_Temperature, new_Rack_Power, new_CoP, \
        new_IT_Power, new_Cooling_Power, new_Total_Power, new_Brown_Energy = updateInformation(t, Server_VM,
                                                                                               bayes_T_sup)
        


        r2 = []
        r2.append(new_IT_Power)                                 # 0.15
        r2.append(new_Total_Power / new_Cooling_Power)          # 0.4
        r2.append(new_Brown_Energy)                             # 0.05
        r2.append(np.mean(new_inlet_Temperature))               # 0.1
        r2.append(np.mean(new_server_Temperature))              # 0.1
        r2.append(new_Total_Power)                              # 0.2


        if new_cpu_Utilization[action_Rack][action_Server] > U_max or \
                new_inlet_Temperature[action_Rack][action_Server] > T_red or \
                new_server_Temperature[action_Rack][action_Server] > Host_red:
            
            Server_VM[action_Rack][action_Server].remove(vm_id)
            Server_VM[sourceRack][sourceServer].insert(vm_index, vm_id)  # 还原VM迁移操作
            r = -100000
            done = True

        elif new_cpu_Utilization[sourceRack][sourceServer] < U_max and \
                new_inlet_Temperature[sourceRack][sourceServer] < T_red and \
                new_server_Temperature[sourceRack][sourceServer] < Host_red:
            
            r = 0
            normal_r = []
            for i in range(6):
                normal_r.append(r1[i] - r2[i])

           

            scaler = normal_r / np.linalg.norm(normal_r)

            
            for i in range(6):
                r = r + weights[i] * scaler[i]
            done = True
            # TotalMigration += 1

        else:
            
            r = 0
            normal_r = []
            for i in range(6):
                normal_r.append(r1[i] - r2[i])


            scaler = normal_r / np.linalg.norm(normal_r)

            for i in range(6):
                r = r + weights[i] * scaler[i]
            done = False
            

        return observation_, r, done

    def reset(self):
        
        knapsack = np.array([-1] * self.n_features)     
        return knapsack