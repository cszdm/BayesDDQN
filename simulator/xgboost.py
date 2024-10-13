import pandas as pd
import xgboost as xgb
from matplotlib import pyplot
from xgboost import plot_importance
import pylab
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
data = pd.read_csv('workload/qh2-rcc120-vae-energy.csv')

X = data.loc[:10000, ['CPU_Load', 'Fan_speed1', 'Fan_speed2', 'Fan_speed3', 'Fan_speed4', 'RAM_Load', 'No_Of_Running_vms', 'CPU_cores_used']]
y = data.iloc[:10001, -1]   # 目标变量
Features = ['CPU_cores_used', 'No_Of_Running_vms',  'Fan_speed4', 'Fan_speed3', 'Fan_speed2', 'RAM_Load', 'Fan_speed1', 'CPU_Load']
# Sort_Features = [71.0, 173.0, 410.0, 474.0, 495.0, 522.0, 799.0, 1064.0]

# 构建XGBoost回归模型
xg_reg = xgb.XGBRegressor()

# 训练模型
xg_reg.fit(X, y)

# 输出特征重要性
feature_importances = xg_reg.feature_importances_

# 将特征重要性排序并输出
feature_importance = dict(zip(X.columns, feature_importances))
sorted_feature_importance = sorted(feature_importance.items(), key=lambda x: x[1])

print('Feature importance ranking:')
for i, (feature, importance) in enumerate(sorted_feature_importance):
    print('{}. {}: {}'.format(i+1, feature, importance))

# plot_importance(xg_reg)
# pyplot.show()
fasd

plt.figure(figsize=(6, 5))  # 设置画布的尺寸
plt.title('Feature Importance')
# 设置y轴，并设定字号大小
plt.xlabel(u'F score', fontsize=10)
plt.ylabel(u'Features', fontsize=10)
plt.barh(Features, Sort_Features, height=0.2)
plt.xticks(np.arange(0, 1420, 200))


for a, b in zip(Sort_Features, Features):  # 添加数字标签
   print(a, b)
   plt.text(a, b, '%.1f'%float(a))  # a+0.001代表标签位置在柱形图上方0.001处

plt.grid(alpha=1.6)
plt.show()

