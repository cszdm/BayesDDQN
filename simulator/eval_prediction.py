import numpy as np
import pandas as pd
from keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Concatenate, Embedding, GlobalMaxPool1D, Layer, Lambda, Dot, Activation  # Bidirectional  双向LSTM
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from math import sqrt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import concatenate

all_data1 = []
all_data2 = []
time_step = 30

for i in range(time_step, len(full_df_scaled_array1)):
    data_x1 = []
    data_x2 = []
    data_x1.append(full_df_scaled_array1[i-time_step:i, 0:full_df_scaled_array1.shape[1]])
    data_x2.append(full_df_scaled_array2[i - time_step:i, 0:full_df_scaled_array2.shape[1]])
    data_x1 = np.array(data_x1)
    data_x2 = np.array(data_x2)
    prediction1, prediction2 = model.predict([data_x1, data_x2])
    all_data1.append(prediction1)
    all_data2.append(prediction2)
    full_df1.iloc[i, 0] = prediction1
    full_df2.iloc[i, 0] = prediction2
print(all_data1)
print(all_data2)

new_array1 = np.array(all_data1)
new_array1 = new_array1.reshape(-1, 1)
prediction_copies_array1 = np.repeat(new_array1, 6, axis=-1)
y_pred_future_30_days_1 = scaler1.inverse_transform(np.reshape(prediction_copies_array1, (len(new_array1), 6)))[:, 0]
print(y_pred_future_30_days_1)

df_origin1 = df_30_days_future1 = pd.read_csv("workload/qh2-rcc120-MTL LSTM5.csv")
y_origin1 = df_origin1.iloc[-30:, 0]
print(y_origin1)

rmse1 = sqrt(mean_squared_error(y_origin1, y_pred_future_30_days_1))
print("Test RMSE Temperature:%.3f:", rmse1)

new_array2 = np.array(all_data2)
new_array2 = new_array2.reshape(-1, 1)
prediction_copies_array2 = np.repeat(new_array2, 3, axis=-1)
y_pred_future_30_days_2 = scaler2.inverse_transform(np.reshape(prediction_copies_array2, (len(new_array2), 3)))[:, 0]
print(y_pred_future_30_days_2)

df_origin2 = df_30_days_future2 = pd.read_csv("workload/qh2-rcc120-MTL LSTM9.csv")
y_origin2 = df_origin2.iloc[-30:, 0]
print(y_origin2)

# 计算rmse
rmse2 = sqrt(mean_squared_error(y_origin2, y_pred_future_30_days_2))
print("Test RMSE Energy:%.3f:", rmse2)
