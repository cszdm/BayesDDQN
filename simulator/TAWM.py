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



files = []

file1 = pd.read_csv('workload/qh2-rcc120.csv')
file2 = pd.read_csv('workload/qh2-rcc121.csv')
file3 = pd.read_csv('workload/qh2-rcc122.csv')
file4 = pd.read_csv('workload/qh2-rcc123.csv')
file5 = pd.read_csv('workload/qh2-rcc124.csv')
file6 = pd.read_csv('workload/qh2-rcc125.csv')
file7 = pd.read_csv('workload/qh2-rcc126.csv')
file8 = pd.read_csv('workload/qh2-rcc127.csv')
file9 = pd.read_csv('workload/qh2-rcc128.csv')
file10 = pd.read_csv('workload/qh2-rcc129.csv')
file11 = pd.read_csv('workload/qh2-rcc130.csv')
file12 = pd.read_csv('workload/qh2-rcc131.csv')
file13 = pd.read_csv('workload/qh2-rcc132.csv')
file14 = pd.read_csv('workload/qh2-rcc133.csv')
file15 = pd.read_csv('workload/qh2-rcc134.csv')
file16 = pd.read_csv('workload/qh2-rcc135.csv')
file17 = pd.read_csv('workload/qh2-rcc136.csv')
file18 = pd.read_csv('workload/qh2-rcc137.csv')
file19 = pd.read_csv('workload/qh2-rcc138.csv')
file20 = pd.read_csv('workload/qh2-rcc139.csv')
file21 = pd.read_csv('workload/qh2-rcc140.csv')
file22 = pd.read_csv('workload/qh2-rcc141.csv')
file23 = pd.read_csv('workload/qh2-rcc142.csv')
file24 = pd.read_csv('workload/qh2-rcc143.csv')
file25 = pd.read_csv('workload/qh2-rcc144.csv')
file26 = pd.read_csv('workload/qh2-rcc145.csv')
file27 = pd.read_csv('workload/qh2-rcc146.csv')
file28 = pd.read_csv('workload/qh2-rcc147.csv')
file29 = pd.read_csv('workload/qh2-rcc148.csv')
file30 = pd.read_csv('workload/qh2-rcc149.csv')
file31 = pd.read_csv('workload/qh2-rcc150.csv')
file32 = pd.read_csv('workload/qh2-rcc151.csv')
file33 = pd.read_csv('workload/qh2-rcc152.csv')
file34 = pd.read_csv('workload/qh2-rcc153.csv')
file35 = pd.read_csv('workload/qh2-rcc154.csv')
file36 = pd.read_csv('workload/qh2-rcc155.csv')
file37 = pd.read_csv('workload/qh2-rcc156.csv')
file38 = pd.read_csv('workload/qh2-rcc157.csv')
file39 = pd.read_csv('workload/qh2-rcc158.csv')
file40 = pd.read_csv('workload/qh2-rcc159.csv')
file41 = pd.read_csv('workload/qh2-rcc170.csv')
file42 = pd.read_csv('workload/qh2-rcc171.csv')
file43 = pd.read_csv('workload/qh2-rcc172.csv')
file44 = pd.read_csv('workload/qh2-rcc173.csv')
file45 = pd.read_csv('workload/qh2-rcc174.csv')
file46 = pd.read_csv('workload/qh2-rcc175.csv')
file47 = pd.read_csv('workload/qh2-rcc176.csv')
file48 = pd.read_csv('workload/qh2-rcc177.csv')
file49 = pd.read_csv('workload/qh2-rcc178.csv')
file50 = pd.read_csv('workload/qh2-rcc179.csv')


files.append(file1)
files.append(file2)
files.append(file3)
files.append(file4)
files.append(file5)
files.append(file6)
files.append(file7)
files.append(file8)
files.append(file9)
files.append(file10)
files.append(file11)
files.append(file12)
files.append(file13)
files.append(file14)
files.append(file15)
files.append(file16)
files.append(file17)
files.append(file18)
files.append(file19)
files.append(file20)
files.append(file21)
files.append(file22)
files.append(file23)
files.append(file24)
files.append(file25)
files.append(file26)
files.append(file27)
files.append(file28)
files.append(file29)
files.append(file30)
files.append(file31)
files.append(file32)
files.append(file33)
files.append(file34)
files.append(file35)
files.append(file36)
files.append(file37)
files.append(file38)
files.append(file39)
files.append(file40)
files.append(file41)
files.append(file42)
files.append(file43)
files.append(file44)
files.append(file45)
files.append(file46)
files.append(file47)
files.append(file48)
files.append(file49)
files.append(file50)




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

class DoubleDeepQNetwork():
    def __init__(self, n_actions, n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 e_greedy_increment=None,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 output_graph=False,
                 double_q=True,
                 sess=None
                 ):
        self.n_actions = n_actions
        print(n_actions)
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.double_q = double_q

        self.__build_net()

        # Get Parameters to be Updated
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # total learning step & Cost Histogram
        self.learn_step_counter = 0
        self.cost_his = []

        # Start Session
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def __build_net(self):
        n_h1 = 1000  # Hidden Layer
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(
            0.1)  # config of layers

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_h1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_h1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_h1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        # ------------------ loss & optimizer ------------------
        with tf.variable_scope('loss'):
            # ('bb',self.q_target)
            # print('cc',self.q_eval)
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            # print('dd',self.loss)
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input

        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_h1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_h1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_h1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)  # Returns the indices of the maximum values along an axis

        if not hasattr(self, 'q'):  # 记录选的 Qmax 值
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q * 0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:
            action = np.random.randint(0, self.n_actions)

        return action

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # replace the old memory with new memory
        transition = np.hstack((s, [a, r], s_))
        print(transition)
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1
       

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
           

        # sample batch memory from all memory   从memory中随机抽取batch_size这么多记忆
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size,
                                            size=self.batch_size)  # Generates a random sample from a given 1-D array
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],  # next observation
                       self.s: batch_memory[:, -self.n_features:]})  # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_action_index = batch_memory[:, self.n_features].astype(
            int)  # Action Index of all batch memory, length equal to batch_index
        reward = batch_memory[:, self.n_features + 1]
        # Update Q Matrix
        if self.double_q:  # 如果是 Double DQN
            max_act4next = np.argmax(q_eval4next, axis=1)  # q_eval 得出的最高奖励动作
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN 选择 q_next 依据 q_eval 选出的动作
        else:  # 如果是 Natural DQN
            selected_q_next = np.max(q_next, axis=1)  # natural DQN

        q_target[batch_index, eval_action_index] = reward + self.gamma * selected_q_next

        # train eval network
        _, self.cost = self.sess.run([self.train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


class Maze():
    def __init__(self):
        dim_1 = sum(server_count) * len(Server_VM[sourceRack][sourceServer])    # 选目的主机
        dim_2 = (T_red - T_initial) / 0.1 + 1  # 选温度       # 需加入类似主机转换函数，确定具体温度值
        self.n_actions = dim_1
        self.n_features = len(Server_VM[sourceRack][sourceServer])      # 相当于State

    # action是选择的源主机号,第vm_index个虚拟机放到第server_index个主机上,vm_id指虚拟机的真实编号
    def move(self, observation, action, vm_index, server_index, vm_id, Server_VM):
        observation_ = observation.copy()
        # 是迁一个VM就变，还是迁好几个才变，参考背包（应该是迁一个就变，变为正常了，再重新开始选）
        observation_[vm_index] = action
        action_Rack, action_Server = choose_server(server_index)
        
        # 执行动作，计算r

        # 迁移之前的值：
        new_cpu_Utilization, new_inlet_Temperature, new_server_Temperature, new_Rack_Power, new_CoP, \
        new_IT_Power, new_Cooling_Power, new_Total_Power, new_Brown_Energy = updateInformation(t, Server_VM, T_sup)

        r1 = []
        r1.append(new_IT_Power)
        r1.append(new_Total_Power / new_Cooling_Power)
        r1.append(new_Brown_Energy)
        r1.append(np.mean(new_inlet_Temperature))
        r1.append(np.mean(new_server_Temperature))
        r1.append(new_Total_Power)

        # r1 = new_TotalCost * new_totalCarbonEmission
        

        # 迁移后的状态
        Server_VM[sourceRack][sourceServer].remove(vm_id)
        if (sourceRack != action_Rack) or (sourceServer != action_Server):
            Server_VM[action_Rack][action_Server].append(vm_id)
        else:
           
            Server_VM[action_Rack][action_Server].insert(vm_index, vm_id)

        
        new_cpu_Utilization, new_inlet_Temperature, new_server_Temperature, new_Rack_Power, new_CoP, \
        new_IT_Power, new_Cooling_Power, new_Total_Power, new_Brown_Energy = updateInformation(t, Server_VM,
                                                                                               T_sup)
        


        r2 = []
        r2.append(new_IT_Power)                                 # 0.15
        r2.append(new_Total_Power / new_Cooling_Power)          # 0.4
        r2.append(new_Brown_Energy)                             # 0.05
        r2.append(np.mean(new_inlet_Temperature))               # 0.1
        r2.append(np.mean(new_server_Temperature))              # 0.1
        r2.append(new_Total_Power)                              # 0.2

        # r2 = new_TotalCost * new_totalCarbonEmission
        

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
            # TotalMigration += 1
        

        # Server_VM = tempServerVM

        return observation_, r, done

    def reset(self):
        
        knapsack = np.array([-1] * self.n_features)     # 各过载主机上有哪些虚拟机
        return knapsack


def run_maze(M, DDQN):
    step = 0
    Server_VM = copy.deepcopy(copyServerVM)
    for episode in range(10):
        observation = M.reset()

        while True:
            step += 1

            action = DDQN.choose_action(observation)
            
           
            vm_index, server_index = findMigrateVM2(action)
            
            vm_id = copyServerVM[sourceRack][sourceServer][vm_index]
       
            while vm_id not in Server_VM[sourceRack][sourceServer]:
                
                action = np.random.randint(0, sum(server_count) * M.n_features)
               
                vm_index, server_index = findMigrateVM2(action)
                
                vm_id = copyServerVM[sourceRack][sourceServer][vm_index]
                

           

            observation_, reward, done = M.move(observation, action, vm_index, server_index, vm_id, Server_VM)
            
            DDQN.store_transition(observation, action, reward, observation_)

           
            if step > 200 and step % 5 == 0:
                DDQN.learn()

            
            observation = observation_

            
            if done:
                
                Server_VM = copy.deepcopy(copyServerVM)
                
                break

    # end of game
    print('Game Over.')


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
   
    SLAV1 = 0
    SLAV1 = SLAV1 + slavServer1(inlet_Temperature, server_Temperature)
    
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
                len1 = len(VM_List)
                availGreen = ArizonaWindPower[t] - Total_Power
                Server_VM, T_sup = RESBFD(sourceRack, sourceServer, Server_VM[sourceRack][sourceServer], t, availGreen, Server_VM, T_sup)
               
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
    dd.append(T_sup)
    ee.append(np.mean(server_Temperature))
    ff.append(np.mean(inlet_Temperature))
    gg.append(SLAV1)