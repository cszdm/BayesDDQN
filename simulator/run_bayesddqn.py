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

sumMigration = 0
totalPower = 0
brownPower = 0
resPower = 0
SLAV = 0

avgSup = 0

aa = []
bb = []
cc = []
dd = []
ee = []
ff = []
gg = []


for t in range(T):
   
    resPower = resPower + ArizonaWindPower[t]
    cpu_Utilization, inlet_Temperature, server_Temperature, Rack_Power, CoP, \
    IT_Power, Cooling_Power, Total_Power, Brown_Power = updateInformation(t, Server_VM, T_sup)
    TotalMigration = 0
    SLAV = SLAV + slavServer(cpu_Utilization, inlet_Temperature, server_Temperature)
    
    SLAV1 = 0
    SLAV1 = SLAV1 + slavServer1(inlet_Temperature, server_Temperature)
    

    
    if Brown_Power == 0:
        
       
        while migrateServer(cpu_Utilization, inlet_Temperature, server_Temperature) is not True:
            for i in range(len(Rack)):
                
                
                S_List = findServer(cpu_Utilization[i], inlet_Temperature[i], server_Temperature[i])
                
                sourceRack = i
                if len(S_List) == 0:
                   
                    continue
               
                while len(S_List) != 0:
                    maxInlet = adjustTemperature(inlet_Temperature)
                    
                    new_T_sup = T_sup + T_red - maxInlet
                    new_T_sup = round(new_T_sup, 2)
                   
                    sourceServer = S_List[0]
                    if inlet_Temperature[sourceRack][sourceServer] > T_red and \
                            server_Temperature[sourceRack][sourceServer] < Host_red and \
                            cpu_Utilization[sourceRack][sourceServer] < U_max:
                        
                        cpu_Utilization, inlet_Temperature, server_Temperature, Rack_Power, CoP, \
                        IT_Power, Cooling_Power, Total_Power, Brown_Power = updateInformation(t, Server_VM, new_T_sup)
                        if Brown_Power == 0 and inlet_Temperature[sourceRack][sourceServer] < Host_red:
                           
                            T_sup = new_T_sup

                            # adjustFlag = True
                        else:
                           
                            M = Maze()
                            print(M.n_features)
                            print(M.n_actions)
                            # print(type(M.n_actions))
                            observation = M.reset()
                            sess = tf.Session()
                            
                            with tf.variable_scope('Double_DQN'):
                                double_DQN = DoubleDeepQNetwork(
                                    n_actions=M.n_actions, n_features=M.n_features, memory_size=3000,
                                    e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)
                            sess.run(tf.global_variables_initializer())
                            tf.reset_default_graph()
                            q_double = run_maze(M, double_DQN)
                           
                            # ----------------训练好了Q函数（即Q表）---------------- #
                            actionList = []
                            knapsack = M.reset()
                            action = double_DQN.choose_action(knapsack)
                            
                            Server_VM = copy.deepcopy(copyServerVM)
                            for iters in itertools.count():
                                actionList.append(action)
                                T_sup = findTemperature(action)
                                vm_index, server_index = findMigrateVM(action)
                                
                                vm_id = copyServerVM[sourceRack][sourceServer][vm_index]
                                
                                while vm_id not in Server_VM[sourceRack][sourceServer]:
                                   
                                    action = np.random.randint(0, sum(server_count) * M.n_features)
                                   
                                    vm_index, server_index = findMigrateVM2(action)
                                    
                                    vm_id = copyServerVM[sourceRack][sourceServer][vm_index]
                                   
                                
                                knapsack_, reward, done = M.move(knapsack, action, vm_index, server_index, vm_id, Server_VM,
                                                                 T_sup)
                                
                                action_ = double_DQN.choose_action(knapsack_)
                                
                                if done:
                                    if reward == -100000:
                                        
                                        actionList.pop()
                                        break
                                    else:
                                        
                                        action = action_
                                        knapsack = knapsack_
                                        break
                                else:
                                    action = action_
                                    knapsack = knapsack_
                            
                            for kk in range(len(actionList)):
                                if actionList[kk] != -1:
                                    TotalMigration += 1
                            
                            cpu_Utilization, inlet_Temperature, server_Temperature, Rack_Power, CoP, \
                            IT_Power, Cooling_Power, Total_Power, Brown_Power = updateInformation(t, Server_VM, T_sup)

                           
                            copyServerVM = copy.deepcopy(Server_VM)
                        cpu_Utilization, inlet_Temperature, server_Temperature, Rack_Power, CoP, \
                        IT_Power, Cooling_Power, Total_Power, Brown_Power = updateInformation(t, Server_VM, T_sup)
                        S_List = findServer(cpu_Utilization[sourceRack], inlet_Temperature[sourceRack],
                                            server_Temperature[sourceRack])
                        
                    elif cpu_Utilization[sourceRack][sourceServer] > U_max and \
                            inlet_Temperature[sourceRack][sourceServer] < T_red and \
                            server_Temperature[sourceRack][sourceServer] < Host_red:
                        
                        VM_List = sortVM(Server_VM[sourceRack][sourceServer], t)
                        
                       
                        Server_VM[sourceRack][sourceServer] = VM_List
                       
                        len1 = len(Server_VM[sourceRack][sourceServer])
                        availGreen = ArizonaWindPower[t] - Total_Power
                        Server_VM, T_sup = PABFD(sourceRack, sourceServer, Server_VM[sourceRack][sourceServer], t, availGreen, Server_VM, T_sup)
                        TotalMigration = len1 - len(Server_VM[sourceRack][sourceServer])
                        cpu_Utilization, inlet_Temperature, server_Temperature, Rack_Power, CoP, \
                        IT_Power, Cooling_Power, Total_Power, Brown_Power = updateInformation(t, Server_VM, T_sup)
                       
                        copyServerVM = copy.deepcopy(Server_VM)
                        S_List = findServer(cpu_Utilization[sourceRack], inlet_Temperature[sourceRack],
                                            server_Temperature[sourceRack])
                        

                    else:
                       
                        cpu_Utilization, inlet_Temperature, server_Temperature, Rack_Power, CoP, \
                        IT_Power, Cooling_Power, Total_Power, Brown_Power = updateInformation(t, Server_VM, T_sup)
                        

                        M = Maze()
                        print(M.n_features)
                        print(M.n_actions)
                        # print(type(M.n_actions))
                        observation = M.reset()
                        sess = tf.Session()
                        
                        with tf.variable_scope('Double_DQN'):
                            double_DQN = DoubleDeepQNetwork(
                                n_actions=M.n_actions, n_features=M.n_features, memory_size=3000,
                                e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)
                        sess.run(tf.global_variables_initializer())
                        tf.reset_default_graph()
                        q_double = run_maze(M, double_DQN)
                        
                        actionList = []
                        knapsack = M.reset()
                        action = double_DQN.choose_action(knapsack)
                        
                        Server_VM = copy.deepcopy(copyServerVM)
                        for iters in itertools.count():
                            actionList.append(action)
                            T_sup = findTemperature(action)
                            vm_index, server_index = findMigrateVM(action)
                           
                            vm_id = copyServerVM[sourceRack][sourceServer][vm_index]
                            
                            
                            while vm_id not in Server_VM[sourceRack][sourceServer]:
                                action = np.random.randint(0, sum(server_count) * M.n_features)
                                
                                vm_index, server_index = findMigrateVM2(action)
                                
                                vm_id = copyServerVM[sourceRack][sourceServer][vm_index]
                               
                            knapsack_, reward, done = M.move(knapsack, action, vm_index, server_index, vm_id, Server_VM,
                                                             T_sup)
                            print('reward', reward)
                            print('knapsack_', knapsack_)
                            print('done', done)
                            # print(agsdf)
                            action_ = double_DQN.choose_action(knapsack_)
                            print('action_:', action_)
                            if done:
                                if reward == -100000:
                                   
                                    actionList.pop()
                                    break
                                else:
                                   
                                    action = action_
                                    knapsack = knapsack_
                                    break
                            else:
                                action = action_
                                knapsack = knapsack_
                        
                        for kk in range(len(actionList)):
                            if actionList[kk] != -1:
                                TotalMigration += 1
                        
                        cpu_Utilization, inlet_Temperature, server_Temperature, Rack_Power, CoP, \
                        IT_Power, Cooling_Power, Total_Power, Brown_Power = updateInformation(t, Server_VM, T_sup)

                        
                        copyServerVM = copy.deepcopy(Server_VM)
                        S_List = findServer(cpu_Utilization[sourceRack], inlet_Temperature[sourceRack],
                                            server_Temperature[sourceRack])
                       
    else:
        
        maxInlet = adjustTemperature(inlet_Temperature)
       
        T_sup = T_sup + T_red - maxInlet
        T_sup = round(T_sup, 2)
       
        while migrateServer(cpu_Utilization, inlet_Temperature, server_Temperature) is not True:
            for i in range(len(Rack)):
                S_List = findServer(cpu_Utilization[i], inlet_Temperature[i], server_Temperature[i])
                
                sourceRack = i
                if len(S_List) == 0:
                   
                    continue
                
                while len(S_List) != 0:
                    
                    sourceServer = S_List[0]
                    cpu_Utilization, inlet_Temperature, server_Temperature, Rack_Power, CoP, \
                    IT_Power, Cooling_Power, Total_Power, Brown_Power = updateInformation(t, Server_VM, T_sup)
                    
                    M = Maze()
                    print(M.n_features)
                    print(M.n_actions)
                    # print(type(M.n_actions))
                    observation = M.reset()
                    sess = tf.Session()
                   
                    with tf.variable_scope('Double_DQN'):
                        double_DQN = DoubleDeepQNetwork(
                            n_actions=M.n_actions, n_features=M.n_features, memory_size=3000,
                            e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)
                    sess.run(tf.global_variables_initializer())
                    tf.reset_default_graph()
                    q_double = run_maze(M, double_DQN)
                   
                    actionList = []
                    knapsack = M.reset()
                    action = double_DQN.choose_action(knapsack)
                    
                    Server_VM = copy.deepcopy(copyServerVM)
                    for iters in itertools.count():
                        actionList.append(action)
                        vm_index, server_index = findMigrateVM2(action)
                        
                        vm_id = copyServerVM[sourceRack][sourceServer][vm_index]
                     
                        while vm_id not in Server_VM[sourceRack][sourceServer]:
                            action = np.random.randint(0, sum(server_count) * M.n_features)
                           
                            vm_index, server_index = findMigrateVM2(action)
                            
                            vm_id = copyServerVM[sourceRack][sourceServer][vm_index]
                            
                     
                        knapsack_, reward, done = M.move(knapsack, action, vm_index, server_index, vm_id, Server_VM,
                                                         T_sup)
                       
                        action_ = double_DQN.choose_action(knapsack_)
                        
                        if done:
                            if reward == -100000:
                                
                                actionList.pop()
                                break
                            else:
                                
                                action = action_
                                knapsack = knapsack_
                                break
                        else:
                            action = action_
                            knapsack = knapsack_
                   
                    for kk in range(len(actionList)):
                        if actionList[kk] != -1:
                            TotalMigration += 1
                    
                    cpu_Utilization, inlet_Temperature, server_Temperature, Rack_Power, CoP, \
                    IT_Power, Cooling_Power, Total_Power, Brown_Power = updateInformation(t, Server_VM, T_sup)

                    
                    copyServerVM = copy.deepcopy(Server_VM)
                    cpu_Utilization, inlet_Temperature, server_Temperature, Rack_Power, CoP, \
                    IT_Power, Cooling_Power, Total_Power, Brown_Power = updateInformation(t, Server_VM, T_sup)
                    S_List = findServer(cpu_Utilization[sourceRack], inlet_Temperature[sourceRack],
                                        server_Temperature[sourceRack])
                    
    sumMigration = sumMigration + TotalMigration
    totalPower = totalPower + Total_Power
    brownPower = brownPower + Brown_Power
    avgSup = avgSup + T_sup
    aa.append(Total_Power)
    bb.append(Brown_Power)
    cc.append(Cooling_Power)
    dd.append(T_sup)
    ee.append(np.mean(server_Temperature))
    ff.append(np.mean(inlet_Temperature))
    gg.append(SLAV1)
