import numpy as np
import pandas as pd
import random
import csv
import heapq
import tensorflow as tf
import itertools
from functools import reduce
from bayes_opt import BayesianOptimization
import operator
import copy
import math
from scipy.linalg import solve
from sklearn.preprocessing import MinMaxScaler



Rack = [rack for rack in range(10)]    

server_count = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

server0 = {'name': 'G4', 'CPU': 1860, 'Core': 2, 'Memory': 4048, 'Storage': 100}
server1 = {'name': 'G5', 'CPU': 2660, 'Core': 2, 'Memory': 4048, 'Storage': 100}
server_type = [server0, server1]


vm_count = 300



vm0 = {'CPU': 2500, 'Core': 1, 'Memory': 870, 'Storage': 1}
vm1 = {'CPU': 2000, 'Core': 1, 'Memory': 1740, 'Storage': 1}
vm2 = {'CPU': 1000, 'Core': 1, 'Memory': 1740, 'Storage': 1}
vm3 = {'CPU': 500, 'Core': 1, 'Memory': 613, 'Storage': 1}
vm = [vm0, vm1, vm2, vm3]

Time_Slot = 5   
HRM_Slot = server_count[0]
T = 24  

T_red = 25      

T_sup = 20
T_initial = 15  
  
U_max = 1

Host_red = 105
Resistance = 0.34   
Capacitance = 340   

Wind_Turbine = 5


data = pd.read_csv('Dataset/Total.csv')

vm_cpu = []     
for i in range(vm_count):
    temp_CPU = []
    for j in range(T):
        temp_CPU.append(data['VM' + str(i)][j])
    vm_cpu.append(temp_CPU)

data1 = pd.read_csv('Dataset/Exp-Arizona-Wind Speed.csv')
ArizonaWindSpeed = data1['Speed1']


data2 = pd.read_csv('Dataset/Exp-California-Wind Speed.csv')
CaliforniaWindSpeed = data2['Speed']


data3 = pd.read_csv('Dataset/Exp-Oregon-Wind Speed.csv')
OregonWindSpeed = data3['Speed']


data4 = pd.read_csv('Dataset/Exp-Louisiana-Wind Speed.csv')
LouisianaWindSpeed = data4['Speed']


v_in = 2.5
v_out = 35
v_r = 11
Pr = 3000


ArizonaWindPower = []
for i in range(T):
    print('此时的风速:', ArizonaWindSpeed[i])
    if ArizonaWindSpeed[i] < v_in or ArizonaWindSpeed[i] > v_out:
        ArizonaWindPower.append(0)
    elif v_in < ArizonaWindSpeed[i] < v_r:
        ArizonaWindPower.append(round(Pr * (ArizonaWindSpeed[i] - v_in) / (v_r - v_in), 2) * Wind_Turbine)
    elif v_r < ArizonaWindSpeed[i] < v_out:
        ArizonaWindPower.append(round(Pr, 2) * Wind_Turbine)
    
CaliforniaWindPower = []
for i in range(T):
   
    if CaliforniaWindSpeed[i] < v_in or CaliforniaWindSpeed[i] > v_out:
        CaliforniaWindPower.append(0)
    elif v_in < CaliforniaWindSpeed[i] < v_r:
        CaliforniaWindPower.append(round(Pr * (CaliforniaWindSpeed[i] - v_in) / (v_r - v_in), 2) * Wind_Turbine)
    elif v_r < CaliforniaWindSpeed[i] < v_out:
        CaliforniaWindPower.append(round(Pr, 2) * Wind_Turbine)

OregonWindPower = []
for i in range(T):
    
    if OregonWindSpeed[i] < v_in or OregonWindSpeed[i] > v_out:
        OregonWindPower.append(0)
    elif v_in < OregonWindSpeed[i] < v_r:
        OregonWindPower.append(round(Pr * (OregonWindSpeed[i] - v_in) / (v_r - v_in), 2) * Wind_Turbine)
    elif v_r < OregonWindSpeed[i] < v_out:
        OregonWindPower.append(round(Pr, 2) * Wind_Turbine)



LouisianaWindPower = []
for i in range(T):
    
    if LouisianaWindSpeed[i] < v_in or LouisianaWindSpeed[i] > v_out:
        LouisianaWindPower.append(0)
    elif v_in < LouisianaWindSpeed[i] < v_r:
        LouisianaWindPower.append(round(Pr * (LouisianaWindSpeed[i] - v_in) / (v_r - v_in), 2) * Wind_Turbine)
    elif v_r < LouisianaWindSpeed[i] < v_out:
        LouisianaWindPower.append(round(Pr, 2) * Wind_Turbine)




vm_number = []  

vm_type = [0, 1, 0, 1, 2, 1, 2, 1, 2, 1, 0, 2, 0, 0, 2, 3, 3, 1, 0, 3, 0, 1, 0, 2, 2, 1, 3, 0, 0, 3, 0, 2, 1, 3, 2, 1, 3, 0, 1, 3, 1, 1, 1, 2, 3, 3, 2, 3, 0, 1, 2, 1, 0, 3, 1, 2, 0, 3, 2, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 3, 3, 0, 0, 0, 1, 0, 2, 1, 2, 3, 3, 3, 3, 3, 3, 1, 3, 2, 1, 2, 2, 2, 0, 2, 2, 0, 0, 3, 2, 3, 1, 3, 1, 3, 3, 1, 2, 0, 2, 2, 3, 0, 2, 2, 3, 2, 1, 2, 3, 3, 0, 3, 0, 0, 3, 0, 2, 1, 2, 1, 2, 1, 0, 0, 1, 2, 2, 0, 3, 0, 0, 3, 3, 1, 0, 0, 3, 3, 3, 0, 1, 2, 2, 0, 1, 3, 2, 1, 0, 0, 3, 3, 2, 3, 3, 3, 0, 0, 2, 0, 3, 1, 2, 0, 3, 0, 3, 0, 0, 0, 3, 0, 1, 1, 1, 1, 2, 1, 0, 3, 3, 1, 1, 1, 3, 3, 3, 2, 0, 1, 3, 0, 3, 2, 3, 2, 3, 2, 0, 2, 1, 0, 2, 3, 1, 0, 0, 1, 1, 0, 2, 0, 1, 0, 3, 1, 1, 3, 0, 1, 2, 2, 1, 0, 0, 1, 3, 1, 2, 1, 1, 2, 3, 3, 3, 3, 0, 2, 3, 0, 2, 0, 1, 3, 3, 1, 3, 1, 3, 0, 2, 0, 1, 2, 1, 0, 0, 1, 0, 3, 1, 2, 1, 0, 2, 0, 1, 1, 1, 3, 3, 1, 2, 2, 2, 2, 1, 1, 1, 2, 3, 1, 0, 3, 1, 3, 1, 2, 3, 2]


vm_capacity = []  


for i in range(vm_count):
    vm_capacity.append(vm[vm_type[i]]['CPU'])


server_number = [] 
cpu_Capacity = []  
cpu_Available = []  
cpu_Utilization = []  


for i in range(len(Rack)):
    serverNumber = []
    cpuCapacity = []
    cpuAvailable = []  
    cpuUtilization = []
    for j in range(server_count[i]):
        serverNumber.append(j)
        if j < server_count[i] / 2:
            cpuCapacity.append(server_type[0]['CPU'] * 2)
            cpuAvailable.append(server_type[0]['CPU'] * 2)
        else:
            cpuCapacity.append(server_type[1]['CPU'] * 2)
            cpuAvailable.append(server_type[1]['CPU'] * 2)
        cpuUtilization.append(0)
    server_number.append(serverNumber)
    
    cpu_Capacity.append(cpuCapacity)
    cpu_Available.append(cpuAvailable)
    cpu_Utilization.append(cpuUtilization)




Server_VM = [[[45, 178, 277, 292], [130, 247, 125, 163, 75, 213], [269, 36], [244, 161, 55, 228, 149], [299, 205, 227]], [[29, 155, 265, 224, 53, 214], [156, 112], [209, 245, 187, 18, 32, 169, 135], [251, 117, 158, 24], [2, 181, 198, 93, 33]], [[252, 193, 260, 182, 268, 199], [42, 62, 56, 256, 13, 190, 67], [173, 287, 188, 266, 88, 241, 83], [273, 154, 271, 71, 194, 89], [102, 37, 11, 253]], [[168, 267, 65, 126, 281, 289, 4, 218, 1, 298], [238, 40, 285, 25, 48, 94], [57, 14], [148, 185], [274, 272, 261, 127, 222, 282, 63]], [[105, 66, 270, 41, 258], [114, 254, 91, 242, 136], [283, 70, 147, 153, 54, 23], [152, 3, 99, 8], [118, 171, 143, 39, 249, 246, 262, 229]], [[236, 115, 79, 96, 28, 279, 176, 85, 16], [87, 121, 108, 104, 197, 9], [122, 206, 107, 72, 61, 35, 184], [232, 288, 20, 210, 264, 77, 142], [140, 106, 124, 225, 78, 159, 239, 12]], [[49, 167, 44, 31], [19, 157, 120, 200, 166, 174, 191, 59, 30], [86, 101, 144, 215, 74, 15, 27], [52, 141, 219, 134, 38], [110, 208, 151, 189, 231]], [[82, 259, 137, 204, 22, 226, 177, 212], [98, 133, 95, 0, 250], [51, 290, 116, 286, 183, 5, 170, 296], [203, 90, 257, 97, 139, 68, 131, 275], [201, 237, 103, 47, 297, 138, 180, 216, 233]], [[243, 284, 294, 278, 123], [], [221, 172, 7, 207, 17, 111, 50, 280, 220, 109, 255, 129, 291], [46, 146, 293, 113, 195], [10, 162, 132, 119, 164, 230, 60, 84, 175, 160]], [[217, 186, 202, 58, 73, 179, 234, 211], [240, 150, 196, 263, 21, 145, 92], [295, 80, 34, 64], [69, 276, 43, 235], [192, 128, 81, 6, 100, 223, 26, 248, 76, 165]]]




copyServerVM = copy.deepcopy(Server_VM)



Rack_HRM = np.zeros((10, 5, 5))


for i in range(len(Rack)):
    Server_Power = np.zeros((server_count[i], server_count[i]))  
    for t in range(HRM_Slot):  
        for j in range(server_count[i]):  
            Server_Power[t][j] = files[j + i * server_count[i]]['Power'][t]
    print('Server_Power:\n', Server_Power)  

    T_in = np.zeros((server_count[i], server_count[i]))
    for t in range(HRM_Slot):  
        for j in range(server_count[i]):  
            T_in[t][j] = files[j + i * server_count[i]]['Inlet_Temperature'][t] - 20
    print('T_in:\n', T_in)  

    HRM = np.zeros((server_count[i], server_count[i]))
    for j in range(server_count[i]):
        x = solve(Server_Power, T_in[:, j])
       
        HRM[j, :] = x
   

    Rack_HRM[i, :, :] = HRM



def calUtilization(x, y, z):
   
    for i in range(server_count[x]):
        
        temp_VMCPUCapacity = 0
        for j in range(len(z[x][i])):
            
            temp_VMCPU = vm_cpu[z[x][i][j]][y]
            
            temp_VMNumber = z[x][i][j]
            
            temp_VMType = vm_type[z[x][i][j]]
            
            temp_VMCapacity = vm[temp_VMType]['CPU']
            
            temp_VMCPUCapacity = temp_VMCPUCapacity + temp_VMCapacity * temp_VMCPU / 100
        

        
        cpu_Available[x][i] = cpu_Capacity[x][i] - temp_VMCPUCapacity  
        
        cpu_Utilization[x][i] = round(temp_VMCPUCapacity / cpu_Capacity[x][i], 2)

    return cpu_Utilization[x]



def serverPower(x, y, z):
    if y < server_count[x] / 2:  
            return 66
        elif 0 <= z < 0.1:
            return 107
        elif 0.1 <= z < 0.2:
            return 120
        elif 0.2 <= z < 0.3:
            return 131
        elif 0.3 <= z < 0.4:
            return 143
        elif 0.4 <= z < 0.5:
            return 156
        elif 0.5 <= z < 0.6:
            return 173
        elif 0.6 <= z < 0.7:
            return 191
        elif 0.7 <= z < 0.8:
            return 211
        elif 0.8 <= z < 0.9:
            return 229
        else:
            return 247
    else:
        if z == 0:
            return 58.4
        elif 0 <= z < 0.1:
            return 98
        elif 0.1 <= z < 0.2:
            return 109
        elif 0.2 <= z < 0.3:
            return 118
        elif 0.3 <= z < 0.4:
            return 128
        elif 0.4 <= z < 0.5:
            return 140
        elif 0.5 <= z < 0.6:
            return 153
        elif 0.6 <= z < 0.7:
            return 170
        elif 0.7 <= z < 0.8:
            return 189
        elif 0.8 <= z < 0.9:
            return 205
        else:
            return 222



def rackPower(x, y):
    rack_Power = []
    for i in range(len(x)):
        sum = 0
        for j in range(server_count[x[i]]):
            
            sum = sum + serverPower(i, j, y[i][j])
        rack_Power.append(sum)      
    
    return rack_Power



def calInletTemperature(x, y, z, xx):
    
    inlet_Temperature = []
    result = 0
    sum = 0
    if z[x][y] != 0:
        for j in range(server_count[x]):
            
            sum = sum + serverPower(x, j, z[x][j]) * Rack_HRM[x][y][j]
        
        result = round(sum + xx, 2)
    return result



def calServerTemperature(x, y, z, xx):
    return round(serverPower(x, y, z[x][y]) * Resistance + calInletTemperature(x, y, z, xx) +
                 T_initial * math.exp(-Resistance * Capacitance), 2)



def calCoP(x):
    
    return round(0.0068 * x * x + 0.0008 * x + 0.458, 2)



def calBrownPower(x, y):
    if x <= y:
        return 0
    else:
        return x - y



def updateInformation(x, y, z):
    # 1、更新CPU利用率
    for i in range(len(Rack)):
        calUtilization(i, x, y)
    print('更新后%s时刻各主机的CPU利用率%s:' % (x, cpu_Utilization))

    # 2、更新进气温度
    inlet_Temperature = []
    for i in range(len(Rack)):
        temp_InletTemperature = []
        for j in range(server_count[i]):
            temp_InletTemperature.append(calInletTemperature(i, j, cpu_Utilization, z))
        inlet_Temperature.append(temp_InletTemperature)
    print('更新后%s时刻各主机的进气温度%s:' % (x, inlet_Temperature))

    # 3、更新主机温度
    server_Temperature = []
    for i in range(len(Rack)):
        temp_ServerTemperature = []
        for j in range(server_count[i]):
            temp_ServerTemperature.append(calServerTemperature(i, j, cpu_Utilization, z))
        server_Temperature.append(temp_ServerTemperature)
    print('更新后%s时刻各主机的温度%s:' % (x, server_Temperature))

    # 4、更新各机架能耗
    Rack_Power = rackPower(Rack, cpu_Utilization)
    print('更新后%s时刻各机架的IT能耗%s:' % (x, Rack_Power))

    # 5、更新当前CoP
    CoP = calCoP(z)

    # 6、更新总IT能耗
    IT_Power = sum(Rack_Power)
    print('更新后%s时刻，所有机架所产生的IT能耗为%sw' % (x, IT_Power))

    # 7、更新总Cooling能耗
    Cooling_Power = IT_Power / CoP
    print('更新后%s时刻，所有机架所产生的Cooling能耗为%sw' % (x, Cooling_Power))

    # 8、更新总Total能耗
    Total_Power = IT_Power + Cooling_Power
    print('更新后%s时刻，所有机架所产生的Total能耗为%sw' % (x, Total_Power))

    # 9、更新总Brown能耗
    Brown_Power = calBrownPower(Total_Power, ArizonaWindPower[x])
    print('更新后%s时刻，所有机架所产生的Brown能耗为%sw' % (x, Brown_Power))


    return cpu_Utilization, inlet_Temperature, server_Temperature, Rack_Power, CoP, \
           IT_Power, Cooling_Power, Total_Power, Brown_Power



def findServer(x, y, z):
    serverList = []
    for i in range(len(x)):     
        if x[i] > U_max:
            print('机架内第%s个主机过载' % (i))
            serverList.append(i)
        if y[i] > T_red:
            if x[1] != 0:
                print('机架内第%s个主机进气温度过高' % (i))
                serverList.append(i)
        if z[i] > Host_red:
            if x[i] != 0:
                print('机架内第%s个主机温度过高' % (i))
            serverList.append(i)
    return list(set(serverList))


def findTemperature(x):
    tempSum = sum(server_count) * len(Server_VM[sourceRack][sourceServer])
    T_index = x // tempSum
    TT_sup = T_initial + T_index * 0.1 - 0.1
    return TT_sup


def findMigrateVM(x):
    tempSum = sum(server_count) * len(Server_VM[sourceRack][sourceServer])
    vm_index = (x % tempSum) // sum(server_count)
    server_index = (x % tempSum) % sum(server_count)
    return vm_index, server_index


def findMigrateVM2(x):
    vm_index = x // sum(server_count)
    server_index = x % sum(server_count)
    return vm_index, server_index


def choose_server(x):  
    
    sum = 0
    for i in range(len(server_count)):
        if x < sum + server_count[i]:
            index_Rack = i
            index_Server = x - sum
            break
        sum = sum + server_count[i]
    return index_Rack, index_Server


def findMigrateVM3(x):
    vm_index = x // sum(server_count)
    server_index = x % sum(server_count)
    action_Rack, action_Server = choose_server(server_index)
    return vm_index, action_Rack, action_Server



def migrateServer(x, y, z):
    flag = True
    for i in range(len(Rack)):
        for j in range(server_count[i]):
            if x[i][j] > U_max or y[i][j] > T_red or z[i][j] > Host_red:
                flag = False
                break
    return flag


def slavServer(x, y, z):
    sum = 0
    for i in range(len(Rack)):
        for j in range(server_count[i]):
            if x[i][j] > U_max or y[i][j] > T_red or z[i][j] > Host_red:
                sum += 1
    return sum

def slavServer1(y, z):
    sum = 0
    for i in range(len(Rack)):
        for j in range(server_count[i]):
            if y[i][j] > T_red or z[i][j] > Host_red:
                sum += 1
    return sum


def adjustTemperature(x):
    rackInletTemperature = []
    for i in range(len(Rack)):
        rackInletTemperature.append(max(x[i]))
    
    return max(rackInletTemperature)



def sortVM(x, y):
    
    sorted_id = sorted(range(len(x)), key=lambda k: vm_cpu[x[k]][y], reverse=True)
    
    z = []
    for i in range(len(x)):
        z.append(x[sorted_id[i]])
    
    return z


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
    
    while migrateServer(cpu_Utilization, inlet_Temperature, server_Temperature) is not True:    
        maxInlet = adjustTemperature(inlet_Temperature)
       
        T_sup = T_sup + T_red - maxInlet
        T_sup = round(T_sup, 2)
        
        cpu_Utilization, inlet_Temperature, server_Temperature, Rack_Power, CoP, \
        IT_Power, Cooling_Power, Total_Power, Brown_Power = updateInformation(t, Server_VM, T_sup)
        for i in range(len(Rack)):
            
            S_List = findServer(cpu_Utilization[i], inlet_Temperature[i], server_Temperature[i])
           
            sourceRack = i
            if len(S_List) == 0:
                
                continue
         
            while len(S_List) != 0:
                sourceServer = S_List[0]
                cpu_Utilization, inlet_Temperature, server_Temperature, Rack_Power, CoP, \
                IT_Power, Cooling_Power, Total_Power, Brown_Power = updateInformation(t, Server_VM, T_sup)
                
                VM_List = sortVM(Server_VM[sourceRack][sourceServer], t)
              
                Server_VM[sourceRack][sourceServer] = VM_List
                len1 = len(Server_VM[sourceRack][sourceServer])
                Server_VM = TAS(sourceRack, sourceServer, Server_VM[sourceRack][sourceServer], t, Server_VM)
                
                TotalMigration = len1 - len(Server_VM[sourceRack][sourceServer])
                
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








