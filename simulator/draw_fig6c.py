import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
RES_Utilization=[18.00, 17.81, 19.96, 18.30, 21.27]
labels = ['TAS','TA','DeepEE','TAWM','BayesDDQN']
width = 0.5  
plt.rcParams['axes.labelsize'] = 16  
plt.rcParams['xtick.labelsize'] = 14  
plt.rcParams['ytick.labelsize'] = 14  
fig, ax = plt.subplots()
ax.set_ylabel('Cooling Temperature (°C)')
ax.set_ylim(15, 22)
ax.bar(labels, RES_Utilization, width=width, align='center', color=['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC'])

ax.get_xticklabels()[4].set_fontweight('bold')
ax.get_xticklabels()[4].set_fontstyle('italic')#oblique

for x,y in zip(labels, RES_Utilization):
    
    plt.text(x, y+0.01, "%.2f" % y, ha='center', va='bottom',size=12)


plt.tight_layout()
plt.grid(axis="y")
plt.show()