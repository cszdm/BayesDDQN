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


def run_maze(M, DDQN):
    step = 0
    Server_VM = copy.deepcopy(copyServerVM)
    for episode in range(3):
        observation = M.reset()

        while True:
            step += 1

            action = DDQN.choose_action(observation)
          
          
            TT_sup = findTemperature(action)
            vm_index, server_index = findMigrateVM(action)
            
            vm_id = copyServerVM[sourceRack][sourceServer][vm_index]
           
            
            while vm_id not in Server_VM[sourceRack][sourceServer]:
                
                action = np.random.randint(0, sum(server_count) * M.n_features)
               
                
                vm_index, server_index = findMigrateVM2(action)
                
                vm_id = copyServerVM[sourceRack][sourceServer][vm_index]
                


            observation_, reward, done = M.move(observation, action, vm_index, server_index, vm_id, Server_VM, TT_sup)
            
            DDQN.store_transition(observation, action, reward, observation_)

       
            if step > 200 and step % 5 == 0:
                DDQN.learn()

            # update observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                
                Server_VM = copy.deepcopy(copyServerVM)
                break

    # end of game
    print('Game Over.')

