import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams['axes.labelsize'] = 16  
plt.rcParams['xtick.labelsize'] = 14  
plt.rcParams['ytick.labelsize'] = 14  

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


TAS= [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 2, 1, 0, 0, 0, 1, 0, 0, 1, 0]

BayesDDQN = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0]

TAWM = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1]

DeepEE = [1, 1, 0, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 5, 0, 0]

TA = [1, 0, 1, 0, 0, 0, 0, 2, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]


plt.plot(x, TAS, label="TAS", linestyle='--', marker='p', markersize=5.5, color='#779043')
plt.plot(x, TA, label="TA", linestyle=':', marker='*', markersize=5.5, color='#7cd6cf')
plt.plot(x, DeepEE, label="DeepEE", linestyle=(0, (3, 3)), marker='^', markersize=5.5, color='#0c84c6')
plt.plot(x, TAWM, label="TAWM", linestyle='--', marker='s', markersize=5.5, color='#f89588')
plt.plot(x, BayesDDQN, label="BayesDDQN", linestyle='-.', marker='o', markersize=5.5, color='#9192ab')


plt.ylabel('Hotspots')
plt.xlabel('Hour')
plt.xlim(0.5,25)
plt.legend()

plt.show()
