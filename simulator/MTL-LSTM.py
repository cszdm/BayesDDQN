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


CPU_Temp_dataset = pd.read_csv('workload/qh2-rcc120-MTL LSTM5-train.csv')
CPU_Energy_dataset = pd.read_csv('workload/qh2-rcc120-MTL LSTM9-train.csv')

CPU_Temp_dataset.columns = ['CPU_Temp_Max', 'CPU_Load', 'Fan_speed1', 'Fan_speed2', 'Fan_speed3', 'Fan_speed4']
print('CPU_Temp_dataset.shape:', CPU_Temp_dataset.shape)

CPU_Energy_dataset.columns = ['Power', 'CPU_Load', 'RAM_Load']
print('CPU_Energy_dataset.shape:', CPU_Energy_dataset.shape)

values1 = CPU_Temp_dataset.values   
values1 = values1.astype('float32')

values2 = CPU_Energy_dataset.values  
values2 = values2.astype('float32')

train1 = values1[:10000, :]
test1 = values1[10000:, :]
print('train1.shape:', train1.shape)
print('test1.shape:', test1.shape)

train2 = values2[:10000, :]
test2 = values2[10000:, :]
print('train2.shape:', train2.shape)
print('test2.shape:', test2.shape)

scaler1 = MinMaxScaler(feature_range=(0, 1))
train_scaled1 = scaler1.fit_transform(train1)
test_scaled1 = scaler1.fit_transform(test1)
print("缩放train:", train_scaled1)
print("缩放test:", test_scaled1)

scaler2 = MinMaxScaler(feature_range=(0, 1))
train_scaled2 = scaler2.fit_transform(train2)
test_scaled2 = scaler2.fit_transform(test2)



def createXY(data, n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(data)):
        dataX.append(data[i-n_past:i, 0:data.shape[1]])
        dataY.append(data[i, 0])
    return np.array(dataX), np.array(dataY)


class Attention(Layer):

    def __init__(self, units=128, **kwargs):
        self.units = units
        super().__init__(**kwargs)

    def __call__(self, inputs):
        """
        Many-to-one attention mechanism for Keras.
        @param inputs: 3D tensor with shape (batch_size, time_steps, input_dim).
        @return: 2D tensor with shape (batch_size, 128)
        @author: felixhao28, philipperemy.
        """
        hidden_states = inputs
        hidden_size = int(hidden_states.shape[2])
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = Dot(axes=[1, 2], name='attention_score')([h_t, score_first_part])
        attention_weights = Activation('softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = Dot(axes=[1, 1], name='context_vector')([hidden_states, attention_weights])
        pre_activation = Concatenate(name='attention_output')([context_vector, h_t])
        attention_vector = Dense(self.units, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector

    def get_config(self):
        return {'units': self.units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


train1X, train1Y = createXY(train_scaled1, 30)
test1X, test1Y = createXY(test_scaled1, 30)

train2X, train2Y = createXY(train_scaled2, 30)
test2X, test2Y = createXY(test_scaled2, 30)

print("train1X Shape-- ", train1X.shape)
print("train1Y Shape-- ", train1Y.shape)
print("test1X Shape-- ", test1X.shape)
print("test1Y Shape-- ", test1Y.shape)

print("train2X Shape-- ", train2X.shape)
print("train2Y Shape-- ", train2Y.shape)
print("test2X Shape-- ", test2X.shape)
print("test2Y Shape-- ", test2Y.shape)

model_input1 = Input(shape=(train1X.shape[1], train1X.shape[2]), name="input_1")
model_input2 = Input(shape=(train2X.shape[1], train2X.shape[2]), name='input_2')

concat = concatenate([model_input1, model_input2], name='concat')

lstm = LSTM(64)(model_input1)
dense_1 = Dense(32, name='dense_1')(lstm)
output_1 = Dense(1, name='output_1')(dense_1)
dense_2 = Dense(32, name='dense_2')(lstm)
output_2 = Dense(1, name='output_2')(dense_2)

model = Model(inputs=[model_input1, model_input2], outputs=[output_1, output_2])

model.compile(loss="mae",
              optimizer="adam",
              loss_weights={'output_1': 1., 'output_2': 1.})

model.summary()
history = model.fit([train1X, train2X], [train1Y, train2Y], epochs=300, batch_size=32)

prediction1, prediction2 = model.predict([test1X, test2X])
# print("prediction\n")
print('pred-test1X:\n', prediction1)
# print('pred-test2X:\n', prediction2)
print("\nPrediction1 Shape-", prediction1.shape)
print("\nPrediction2 Shape-", prediction2.shape)

loss = history.history['loss']
output_1_loss = history.history['output_1_loss']
output_2_loss = history.history['output_2_loss']
plt.plot(loss, label='train_loss')
plt.plot(output_1_loss, label='output_1_loss')
plt.plot(output_2_loss, label='output_2_loss')
plt.xlabel('Epochs')
# plt.ylabel('Loss')
plt.legend()
plt.show()

prediction1_copies_array = np.repeat(prediction1, 6, axis=-1)
print(prediction1_copies_array.shape)
pred1 = scaler1.inverse_transform(np.reshape(prediction1_copies_array, (len(prediction1), 6)))[:, 0]
original1_copies_array = np.repeat(test1Y, 6, axis=-1)
original1 = scaler1.inverse_transform(np.reshape(original1_copies_array, (len(test1Y), 6)))[:, 0]
print("Pred1 Values-- ", pred1)
print("\nOriginal1 Values-- ", original1)
rmse1 = sqrt(mean_squared_error(original1, pred1))
print("Test RMSE1:%.3f:", rmse1)

prediction2_copies_array = np.repeat(prediction2, 3, axis=-1)
print(prediction2_copies_array.shape)
pred2 = scaler2.inverse_transform(np.reshape(prediction2_copies_array, (len(prediction2), 3)))[:, 0]
original2_copies_array = np.repeat(test1Y, 3, axis=-1)
original2 = scaler2.inverse_transform(np.reshape(original2_copies_array, (len(test1Y), 3)))[:, 0]
print("Pred1 Values-- ", pred2)
print("\nOriginal1 Values-- ", original2)
rmse2 = sqrt(mean_squared_error(original2, pred2))
print("Test RMSE2:%.3f:", rmse2)

plt.plot(original1, color='red', label='Origin Temperature')
plt.plot(pred1, color='blue', label='Predicted Temperature')
plt.title('CPU Temperature Prediction')
plt.xlabel('Time')
plt.ylabel('CPU Temperature')
plt.legend()
plt.show()

plt.plot(original2, color='green', label='Origin Energy')
plt.plot(pred2, color='yellow', label='Predicted Energy')
plt.title('Energy Prediction')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend()
plt.show()

df_30_days_past1 = CPU_Temp_dataset.iloc[-30:, :]
print(df_30_days_past1.tail())

df_30_days_future1 = pd.read_csv("workload/qh2-rcc120-MTL LSTM5-test.csv")
print(df_30_days_future1)
print(df_30_days_future1.shape)

df_30_days_past2 = CPU_Energy_dataset.iloc[-30:, :]
print(df_30_days_past2.tail())

df_30_days_future2 = pd.read_csv("workload/qh2-rcc120-MTL LSTM9-test.csv")
print(df_30_days_future2)
print(df_30_days_future2.shape)

df_30_days_future1["CPU_Temp_Max"] = 0
df_30_days_future1 = df_30_days_future1[["CPU_Temp_Max", "CPU_Load", "Fan_speed1", "Fan_speed2", "Fan_speed3", "Fan_speed4"]]
old_scaled_array1 = scaler1.transform(df_30_days_past1)
new_scaled_array1 = scaler1.transform(df_30_days_future1)
new_scaled_df1 = pd.DataFrame(new_scaled_array1)
new_scaled_df1.iloc[:, 0] = np.nan
full_df1 = pd.concat([pd.DataFrame(old_scaled_array1), new_scaled_df1]).reset_index().drop(["index"], axis=1)
print(full_df1.shape)
print(full_df1.tail())
full_df_scaled_array1 = full_df1.values
print(full_df_scaled_array1.shape)

df_30_days_future2["Power"] = 0
df_30_days_future2 = df_30_days_future2[['Power', 'CPU_Load', 'RAM_Load']]
old_scaled_array2 = scaler2.transform(df_30_days_past2)
new_scaled_array2 = scaler2.transform(df_30_days_future2)
new_scaled_df2 = pd.DataFrame(new_scaled_array2)
new_scaled_df2.iloc[:, 0] = np.nan
full_df2 = pd.concat([pd.DataFrame(old_scaled_array2), new_scaled_df2]).reset_index().drop(["index"], axis=1)
print(full_df2.shape)
print(full_df2.tail())
full_df_scaled_array2 = full_df2.values
print(full_df_scaled_array2.shape)

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





