mport numpy as np
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

# x指机架，y指时刻，z指Server_VM
def calUtilization(x, y, z):
    # print(Server_VM[x])  # 第x个数据中心的主机上有哪些虚拟机
    # print(server_count[x])
    for i in range(server_count[x]):
        # print(Server_VM[x][i])  # 第x个数据中心第i个主机上有哪些虚拟机
        # print('主机的CPU容量:', cpu_Capacity[x][i])  # 第x个数据集中心第i个主机的CPU容量
        temp_VMCPUCapacity = 0
        for j in range(len(z[x][i])):
            # print(vm_cpu[Server_VM[x][i][j]][y])
            temp_VMCPU = vm_cpu[z[x][i][j]][y]
            # print('虚拟机CPU利用率:', temp_VMCPU)  # y时刻，第x个数据中心第i个主机上第j个虚拟机的CPU利用率
            temp_VMNumber = z[x][i][j]
            # print('虚拟机编号:', temp_VMNumber)  # 第x个数据中心第i个主机上第j个虚拟机的编号
            # print('所有虚拟机类型:', vm_type)  # 所有虚拟机的虚拟机类型
            temp_VMType = vm_type[z[x][i][j]]
            # print('虚拟机类型:', temp_VMType)  # 第x个数据中心第i个主机上第j个虚拟机的类型
            temp_VMCapacity = vm[temp_VMType]['CPU']
            # print('虚拟机容量:', temp_VMCapacity)
            temp_VMCPUCapacity = temp_VMCPUCapacity + temp_VMCapacity * temp_VMCPU / 100
        # print('主机%s上VM的容量%s' % (i, temp_VMCPUCapacity))

        # print('第%s个数据中心内，主机%s的CPU利用率%s' % (x, i, round(temp_VMCPUCapacity / cpu_Capacity[x][i], 2)))
        cpu_Available[x][i] = cpu_Capacity[x][i] - temp_VMCPUCapacity  # 待定，有可能对比算法时按剩余空间大小进行VMC
        # print('第%s个数据中心内，主机%s的剩余可用容量%s' % (x, i, cpu_Available[x][i]))
        cpu_Utilization[x][i] = round(temp_VMCPUCapacity / cpu_Capacity[x][i], 2)

    return cpu_Utilization[x]


# 计算当前各主机的能耗 单位 Watt
# x指当前的机架，y指主机编号, z指当前主机的CPU利用率
def serverPower(x, y, z):
    if y < server_count[x] / 2:  # 若当前主机编号在上半区，那么就是G4
        if z == 0:
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


# x指Rack，y指cpu_Utilization
def rackPower(x, y):
    rack_Power = []
    for i in range(len(x)):
        sum = 0
        for j in range(server_count[x[i]]):
            # print('各服务器利用率:', cpu_Utilization[i][j])
            # print('各服务器能耗:', serverPower(i, j, cpu_Utilization[i][j]))
            sum = sum + serverPower(i, j, y[i][j])
        rack_Power.append(sum)      # 为了和后面的RES对应
    # print('各数据中心的能耗:',DC_Power)
    return rack_Power


# x指机架编号，y指服务器编号，z指cpu_Utilization, xx指T_sup
def calInletTemperature(x, y, z, xx):
    # print('当前机架:', x)
    inlet_Temperature = []
    result = 0
    sum = 0
    if z[x][y] != 0:
        for j in range(server_count[x]):
            # print('当前机架上的主机:', j)
            # print('该主机的能耗:', serverPower(x, j, cpu_Utilization[x][j]))
            # print('目标服务器:', y)
            # print('热循环系数:', Rack_HRM[x][y][j])
            sum = sum + serverPower(x, j, z[x][j]) * Rack_HRM[x][y][j]
        # print('预计温度:', sum + T_sup)
        result = round(sum + xx, 2)
    return result


# x指机架编号，y指服务器编号，z指cpu_Utilization, xx指T_sup
def calServerTemperature(x, y, z, xx):
    return round(serverPower(x, y, z[x][y]) * Resistance + calInletTemperature(x, y, z, xx) +
                 T_initial * math.exp(-Resistance * Capacitance), 2)


# x指T_sup
def calCoP(x):
    print('此时CoP的值为:', round(0.0068 * x * x + 0.0008 * x + 0.458, 2))
    return round(0.0068 * x * x + 0.0008 * x + 0.458, 2)


# x指Total能耗，y指可再生能源产量
def calBrownPower(x, y):
    if x <= y:
        return 0
    else:
        return x - y


# x指t，y指Server_VM，z指T_sup
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


# x指机架的CPU利用率，y指机架进气温度，z指主机温度
def findServer(x, y, z):
    serverList = []
    for i in range(len(x)):     # 都用if判断，就可以把符合不同判断条件的主机放进不同的列表中
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


# x指虚拟机编号
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


def choose_server(x):  # x指当前输入进来的动作号
    # indexServer = 0
    # indexDC = 0
    sum = 0
    for i in range(len(server_count)):
        if x < sum + server_count[i]:
            index_Rack = i
            index_Server = x - sum
            break
        sum = sum + server_count[i]
    return index_Rack, index_Server


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
    print('各机架的最大进气温度:', rackInletTemperature)
    return max(rackInletTemperature)


# x指过载主机上的VM列表 -- Server_VM[机架][过载主机], y指时刻
def sortVM(x, y):
    # 给出列表内VM的CPU利用率
    # for i in range(len(x)):
        # print('VM的CPU利用率', vm_cpu[i][y])
    sorted_id = sorted(range(len(x)), key=lambda k: vm_cpu[x[k]][y], reverse=True)
    print('排序后索引:', sorted_id)
    z = []
    for i in range(len(x)):
        z.append(x[sorted_id[i]])
    # print('排序后列表:', z)
    return z


# x指机架，y指主机，z指Server_VM[x][y] -- 即虚拟机列表, xx指t, yy指availGreen, zz指Server_VM, xxx指T_sup
def PABFD(x, y, z, xx, yy, zz, xxx):
    global copyServerVM
    print('按TABFD进行虚拟机迁移')
    print('待迁移虚拟机列表:', z)
    cpu_Utilization, inlet_Temperature, server_Temperature, Rack_Power, CoP, \
    IT_Power, Cooling_Power, Total_Power, Brown_Power = updateInformation(t, zz, xxx)
    tServerPower = 0    # 当前配置所需能耗
    minServerPower = 99999      # 当前最少能耗
    tIncServerPower = 99999     # 增加能耗
    minServerTemperature = 999999          # 当前最低主机温度
    oUtilization = cpu_Utilization
    # zz = z[x][y]    # Server_VM[i][j]
    flag = False
    for k in range(len(z)):
        if flag:
            k = k - 1
        print('当前待迁移VM编号:', z[k])
        print('当前待迁移VM容量:', vm_capacity[z[k]])
        print('当前待迁移VM利用率:', vm_cpu[z[k]][xx])
        for i in range(len(Rack)):
            for j in range(server_count[i]):
                if i == x and j == y:
                    print('重合了')
                    continue
                print('机架%s上的主机%s包含的VM%s:' % (i, j, zz[i][j]))
                print('当前服务器的CPU容量', cpu_Capacity[i][j])
                print('当前服务器的CPU利用率', cpu_Utilization[i][j])
                print('当前服务器的进气温度', inlet_Temperature[i][j])
                print('当前服务器的主机温度', server_Temperature[i][j])
                temp_CPU_Utilization = cpu_Utilization[i][j]
                cpu_Utilization[i][j] = (cpu_Capacity[i][j] * cpu_Utilization[i][j] +
                                   vm_cpu[z[k]][xx] * vm_capacity[z[k]] / 100) / cpu_Capacity[i][j]
                print('放上去以后，该服务器的CPU利用率:', cpu_Utilization[i][j])
                destInletTemperature = calInletTemperature(i, j, cpu_Utilization, xxx)
                print('放上去以后，该服务器的进气温度:', destInletTemperature)
                # destServerTemperature = calServerTemperature(i, j, cpu_Utilization)
                destServerTemperature = calServerTemperature(i, j, cpu_Utilization, xxx) \
                                        - calServerTemperature(x, y, oUtilization, xxx)
                print('放上去以后，该服务器变化的主机温度:', destServerTemperature)
                tServerPower = serverPower(i, j, cpu_Utilization[i][j]) / calCoP(xxx)
                print('新的能耗:', tServerPower)
                oServerPower = serverPower(i, j, temp_CPU_Utilization) / calCoP(xxx)
                print('原能耗:', oServerPower)
                incServerPower = tServerPower - oServerPower
                print('增加的能耗:', incServerPower)
                print('可用的可再生能源为:', yy)
                if cpu_Utilization[i][j] > U_max or destInletTemperature > T_red or destServerTemperature > Host_red:
                    print('目的主机不符合要求')
                    cpu_Utilization[i][j] = temp_CPU_Utilization
                    continue
                else:
                    print('机架%s上目的主机%s符合要求:' % (i, j))
                    if incServerPower < yy:   # 按服务器温度最低迁移
                    # if incServerPower < 50:     # 50是临时的
                        # 按平均温度最低迁移
                        print('满足迁移条件，同时可再生能源充足')
                        print('主机温度:', destServerTemperature)
                        if destServerTemperature < minServerTemperature:
                            print('找到最低主机温度')
                            minServerTemperature = destServerTemperature
                            destRack1 = i
                            destServer1 = j
                            flag = True
                        cpu_Utilization[i][j] = temp_CPU_Utilization
        if flag:
            print('可再生能源充足，已找到合适的目的主机')
            print('目的机架:', destRack1)
            print('目的服务器:', destServer1)
            # 开始迁移
            vm_migrate = z[k]
            print('迁移的VM编号为:', vm_migrate)
            print('目的服务器迁移之前的虚拟机列表:', zz[destRack1][destServer1])
            srcRack1 = x
            print('源主机所在的机架为:', srcRack1)
            srcServer1 = y
            print('源VM所在服务器编号:', srcServer1)
            # print('源VM所在机架内有哪些虚拟机:', Server_VM[srcRack1])
            print('目的服务器迁移之前的虚拟机列表:', zz[srcRack1][srcServer1])
            zz[srcRack1][srcServer1].remove(vm_migrate)
            print('删除之后源VM所在主机上有哪些虚拟机:', zz[srcRack1][srcServer1])
            # 添加到目的主机上
            zz[destRack1][destServer1].append(vm_migrate)
            print('目的主机添加VM后有哪些虚拟机:', zz[destRack1][destServer1])
            print('更新前源服务器的CPU利用率为:', cpu_Utilization[srcRack1][srcServer1])
            print('更新前目的服务器的CPU利用率为:', cpu_Utilization[destRack1][destServer1])
            # 执行更新操作
            cpu_Utilization[srcRack1][srcServer1] = round((cpu_Capacity[srcRack1][srcServer1] *
                                                     cpu_Utilization[srcRack1][srcServer1] - vm_cpu[z[k]][xx]
                                                     * vm_capacity[z[k]] / 100) / cpu_Capacity[srcRack1][srcServer1], 2)
            print('更新后源服务器的CPU利用率为:', cpu_Utilization[srcRack1][srcServer1])
            cpu_Utilization[destRack1][destServer1] = round((cpu_Capacity[destRack1][destServer1] *
                                                      cpu_Utilization[destRack1][destServer1] + vm_cpu[z[k]][xx]
                                                      * vm_capacity[z[k]] / 100) / cpu_Capacity[destRack1][destServer1], 2)
            print('更新后目的服务器的CPU利用率为:', cpu_Utilization[destRack1][destServer1])
            print('看看利用率是否发生了变化:', cpu_Utilization)
            if cpu_Utilization[x][y] <= U_max:
                print('TABFD迁移成功，不过载了')
                break
    if flag is not True:
        print('TABFD迁移失败')
        print('同时调温及负载调度')
        print('贝叶斯优化前的制冷温度:', xxx)
        cpu_Utilization, inlet_Temperature, server_Temperature, Rack_Power, CoP, \
        IT_Power, Cooling_Power, Total_Power, Brown_Power = updateInformation(t, zz, xxx)
        print('过载主机上有哪些虚拟机:', zz[sourceRack][sourceServer])
        M = Maze()
        observation = M.reset()
        sess = tf.Session()
        # 执行迁移操作，看迁移后各主机的利用率
        with tf.variable_scope('Double_DQN'):
            double_DQN = DoubleDeepQNetwork(
                n_actions=M.n_actions, n_features=M.n_features, memory_size=3000,
                e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)
        sess.run(tf.global_variables_initializer())
        tf.reset_default_graph()
        q_double = run_maze(M, double_DQN)
        print('DDQN运行完毕!')
        # ----------------训练好了Q函数（即Q表）---------------- #
        actionList = []
        knapsack = M.reset()
        action = double_DQN.choose_action(knapsack)
        print('训练后选择的动作:', action)
        Server_VM = copy.deepcopy(copyServerVM)
        for iters in itertools.count():
            actionList.append(action)
            T_sup = findTemperature(action)
            vm_index, server_index = findMigrateVM(action)
            print('制冷温度为%s,第%s个虚拟机放到第%s个服务器上' % (T_sup, vm_index, server_index))
            vm_id = copyServerVM[sourceRack][sourceServer][vm_index]
            print('虚拟机实际编号:', vm_id)
            # 如果ServerVM上没有待迁移的虚拟机，那么重新选
            while vm_id not in zz[sourceRack][sourceServer]:
                action = np.random.randint(0, sum(server_count) * M.n_features)
                print('初始选择的动作号:', action)
                # 给出action对应的虚拟机及其动作
                # TT_sup = findTemperature(action)
                vm_index, server_index = findMigrateVM2(action)
                print('第%s个虚拟机放到第%s个服务器上' % (vm_index, server_index))
                vm_id = copyServerVM[sourceRack][sourceServer][vm_index]
                print('虚拟机实际编号:', vm_id)
            print('背包:', knapsack)
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
                    print('目的主机超载!')
                    actionList.pop()
                    break
                else:
                    print('源主机不过载!')
                    action = action_
                    knapsack = knapsack_
                    break
            else:
                action = action_
                knapsack = knapsack_
        print('最终选择的VM:', knapsack)
        print('动作列表:', actionList)
        print('当前制冷温度', xxx)
        cpu_Utilization, inlet_Temperature, server_Temperature, Rack_Power, CoP, \
        IT_Power, Cooling_Power, Total_Power, Brown_Power = updateInformation(t, zz, xxx)

        print('TPABFD执行完毕')
        print('过载主机上有哪些虚拟机:', zz[sourceRack][sourceServer])
        copyServerVM = copy.deepcopy(zz)
        cpu_Utilization, inlet_Temperature, server_Temperature, Rack_Power, CoP, \
        IT_Power, Cooling_Power, Total_Power, Brown_Power = updateInformation(t, zz, xxx)
    return zz, xxx
