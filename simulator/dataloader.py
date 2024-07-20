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


Rack = [rack for rack in range(10)]   



server_count = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]



server0 = {'name': 'G4', 'CPU': 1860, 'Core': 2, 'Memory': 4048, 'Storage': 100}
server1 = {'name': 'G5', 'CPU': 2660, 'Core': 2, 'Memory': 4048, 'Storage': 100}
server_type = [server0, server1]

# 定义VM数量
vm_count = 300


#定义虚拟机类型
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

# 读取Oregon数据
data3 = pd.read_csv('Dataset/Exp-Oregon-Wind Speed.csv')
OregonWindSpeed = data3['Speed']

# 读取Louisiana数据
data4 = pd.read_csv('Dataset/Exp-Louisiana-Wind Speed.csv')
LouisianaWindSpeed = data4['Speed']

# NE-3000发电机
v_in = 2.5
v_out = 35
v_r = 11
Pr = 3000

# Arizona内单个风叶轮机的产能
ArizonaWindPower = []
for i in range(T):
    print('此时的风速:', ArizonaWindSpeed[i])
    if ArizonaWindSpeed[i] < v_in or ArizonaWindSpeed[i] > v_out:
        ArizonaWindPower.append(0)
    elif v_in < ArizonaWindSpeed[i] < v_r:
        ArizonaWindPower.append(round(Pr * (ArizonaWindSpeed[i] - v_in) / (v_r - v_in), 2) * Wind_Turbine)
    elif v_r < ArizonaWindSpeed[i] < v_out:
        ArizonaWindPower.append(round(Pr, 2) * Wind_Turbine)
    # print('Arizona的风能:', ArizonaWindPower)

# California内单个风叶轮机的产能
CaliforniaWindPower = []
for i in range(T):
    # print('此时的风速:', CaliforniaWindSpeed[i])
    if CaliforniaWindSpeed[i] < v_in or CaliforniaWindSpeed[i] > v_out:
        CaliforniaWindPower.append(0)
    elif v_in < CaliforniaWindSpeed[i] < v_r:
        CaliforniaWindPower.append(round(Pr * (CaliforniaWindSpeed[i] - v_in) / (v_r - v_in), 2) * Wind_Turbine)
    elif v_r < CaliforniaWindSpeed[i] < v_out:
        CaliforniaWindPower.append(round(Pr, 2) * Wind_Turbine)
# print('California的风能:', CaliforniaWindPower)

# Oregon内单个风叶轮机的产能
OregonWindPower = []
for i in range(T):
    # print('此时的风速:', OregonWindSpeed[i])
    if OregonWindSpeed[i] < v_in or OregonWindSpeed[i] > v_out:
        OregonWindPower.append(0)
    elif v_in < OregonWindSpeed[i] < v_r:
        OregonWindPower.append(round(Pr * (OregonWindSpeed[i] - v_in) / (v_r - v_in), 2) * Wind_Turbine)
    elif v_r < OregonWindSpeed[i] < v_out:
        OregonWindPower.append(round(Pr, 2) * Wind_Turbine)
# print('Oregon的风能:', OregonWindPower)

# Louisiana内单个风叶轮机的产能
LouisianaWindPower = []
for i in range(T):
    # print('此时的风速:', LouisianaWindSpeed[i])
    if LouisianaWindSpeed[i] < v_in or LouisianaWindSpeed[i] > v_out:
        LouisianaWindPower.append(0)
    elif v_in < LouisianaWindSpeed[i] < v_r:
        LouisianaWindPower.append(round(Pr * (LouisianaWindSpeed[i] - v_in) / (v_r - v_in), 2) * Wind_Turbine)
    elif v_r < LouisianaWindSpeed[i] < v_out:
        LouisianaWindPower.append(round(Pr, 2) * Wind_Turbine)
# print('Louisiana的风能:', LouisianaWindPower)




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
    # server_class.append(serverClass)
    cpu_Capacity.append(cpuCapacity)
    cpu_Available.append(cpuAvailable)
    cpu_Utilization.append(cpuUtilization)

Server_VM = [[[45, 178, 277, 292], [130, 247, 125, 163, 75, 213], [269, 36], [244, 161, 55, 228, 149], [299, 205, 227]], [[29, 155, 265, 224, 53, 214], [156, 112], [209, 245, 187, 18, 32, 169, 135], [251, 117, 158, 24], [2, 181, 198, 93, 33]], [[252, 193, 260, 182, 268, 199], [42, 62, 56, 256, 13, 190, 67], [173, 287, 188, 266, 88, 241, 83], [273, 154, 271, 71, 194, 89], [102, 37, 11, 253]], [[168, 267, 65, 126, 281, 289, 4, 218, 1, 298], [238, 40, 285, 25, 48, 94], [57, 14], [148, 185], [274, 272, 261, 127, 222, 282, 63]], [[105, 66, 270, 41, 258], [114, 254, 91, 242, 136], [283, 70, 147, 153, 54, 23], [152, 3, 99, 8], [118, 171, 143, 39, 249, 246, 262, 229]], [[236, 115, 79, 96, 28, 279, 176, 85, 16], [87, 121, 108, 104, 197, 9], [122, 206, 107, 72, 61, 35, 184], [232, 288, 20, 210, 264, 77, 142], [140, 106, 124, 225, 78, 159, 239, 12]], [[49, 167, 44, 31], [19, 157, 120, 200, 166, 174, 191, 59, 30], [86, 101, 144, 215, 74, 15, 27], [52, 141, 219, 134, 38], [110, 208, 151, 189, 231]], [[82, 259, 137, 204, 22, 226, 177, 212], [98, 133, 95, 0, 250], [51, 290, 116, 286, 183, 5, 170, 296], [203, 90, 257, 97, 139, 68, 131, 275], [201, 237, 103, 47, 297, 138, 180, 216, 233]], [[243, 284, 294, 278, 123], [], [221, 172, 7, 207, 17, 111, 50, 280, 220, 109, 255, 129, 291], [46, 146, 293, 113, 195], [10, 162, 132, 119, 164, 230, 60, 84, 175, 160]], [[217, 186, 202, 58, 73, 179, 234, 211], [240, 150, 196, 263, 21, 145, 92], [295, 80, 34, 64], [69, 276, 43, 235], [192, 128, 81, 6, 100, 223, 26, 248, 76, 165]]]

print('初始时各主机上的VM分布:\n', Server_VM)   

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

