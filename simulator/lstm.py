import numpy as np
import pandas as pd
from keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Concatenate, Embedding, GlobalMaxPool1D, Lambda, Dot, Activation  # Bidirectional  双向LSTM
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from pandas import datetime
from pandas import read_csv
from matplotlib import pyplot
from pandas import DataFrame
from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from keras.layers import Layer
# from attention import Attention

# dataset = pd.read_csv('workload/qh2-rcc192-MTL LSTM 8-train.csv')
dataset = pd.read_csv('workload/qh2-rcc153-MTL LSTM 10-train.csv')

# 手动设置每一列的label
dataset.columns = ['Power', 'CPU_Load', 'RAM_Load']
print(dataset.head())
print(dataset.shape)

values = dataset.values
values = values.astype('float32')
# 划分训练集测试集
train = values[:10000, :]
test = values[10000:, :]
print(train.shape)
print(test.shape)

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.fit_transform(test)
print("缩放train:", train_scaled)
print("缩放test:", test_scaled)


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


trainX, trainY = createXY(train_scaled, 30)
testX, testY = createXY(test_scaled, 30)

print("trainX Shape-- ", trainX.shape)
print("trainY Shape-- ", trainY.shape)
print("testX Shape-- ", testX.shape)
print("testY Shape-- ", testY.shape)


# 构建模型
# def build_model():
#     grid_model = Sequential()
#     grid_model.add(LSTM(50, return_sequences=True, input_shape=(30, 6)))
#     grid_model.add(LSTM(50))
#     grid_model.add(Dropout(0.2))
#     grid_model.add(Dense(1))
#
#     grid_model.compile(loss="mse", optimizer="adam")
#     return grid_model
#
#
# model = build_model()
# grid_search = model.fit(trainX, trainY, batch_size=72, epochs=300, validation_data=(testX, testY), verbose=2, shuffle=False)

# grid_model = KerasRegressor(build_fn=build_model, verbose=1, validation_data=(testX, testY))
#
#
# parameters = {'batch_size': [16, 20],
#               'epochs': [8, 10],
#               'optimizer': ['adam', 'Adadelta']}
#
# grid_search = GridSearchCV(estimator=grid_model, param_grid=parameters, cv=2)
#
# # 将模型拟合到trainX和trainY中
# grid_search = grid_search.fit(trainX, trainY)
# grid_model = Sequential()
# grid_model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
# grid_model.add(Attention(units=32))
# grid_model.add(Dropout(0.2))
# grid_model.add(Dense(1))
# grid_model.compile(loss="mse", optimizer="adam")

model_input = Input(shape=(trainX.shape[1], trainX.shape[2]))
x = LSTM(64, return_sequences=True)(model_input)
x = Attention(32)(x)
x = Dense(1)(x)
model = Model(model_input, x)
model.compile(loss="mae", optimizer="adam")
history = model.fit(trainX, trainY, batch_size=72, epochs=200, validation_data=(trainX, trainY), verbose=2, shuffle=False)
prediction = model.predict(testX)
print("prediction\n", prediction)
print("\nPrediction Shape-", prediction.shape)

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()



prediction_copies_array = np.repeat(prediction, 3, axis=-1)
print(prediction_copies_array.shape)
pred = scaler.inverse_transform(np.reshape(prediction_copies_array, (len(prediction), 3)))[:, 0]
original_copies_array = np.repeat(testY, 3, axis=-1)
original = scaler.inverse_transform(np.reshape(original_copies_array, (len(testY), 3)))[:, 0]
print("Pred Values-- ", pred)
print("\nOriginal Values-- ", original)
# 计算rmse
rmse = sqrt(mean_squared_error(original, pred))
print("Test RMSE:%.3f:", rmse)

plt.plot(original, color='#BEB8DC', label='Origin Energy Consumption')
plt.plot(pred, color='#FA7F6F', label='Predicted Energy Consumption')
plt.title('Energy Consumption Prediction')
plt.xlabel('Time')
plt.ylabel('Energy Consumption (Watt)')
plt.legend()
plt.show()

# mylog1 = open('orig3.txt', mode='a', encoding='utf-8')
# np.set_printoptions(threshold=np.inf)
# for i in range(len(original)):
#     print(original, end=',', file=mylog1)
# mylog1.close()

mylog2 = open('pred6.txt', mode='a', encoding='utf-8')
np.set_printoptions(threshold=np.inf)
for i in range(len(pred)):
    print(pred[i], end=',', file=mylog2)
mylog2.close()

df_30_days_past = dataset.iloc[-30:, :]
print(df_30_days_past.tail())

df_30_days_future = pd.read_csv("workload/qh2-rcc153-MTL LSTM 10-test.csv")
print(df_30_days_future)
print(df_30_days_future.shape)

df_30_days_future["Power"] = 0
df_30_days_future = df_30_days_future[['Power', 'CPU_Load', 'RAM_Load']]
old_scaled_array = scaler.transform(df_30_days_past)
new_scaled_array = scaler.transform(df_30_days_future)
new_scaled_df = pd.DataFrame(new_scaled_array)
new_scaled_df.iloc[:, 0] = np.nan
full_df = pd.concat([pd.DataFrame(old_scaled_array), new_scaled_df]).reset_index().drop(["index"], axis=1)
print(full_df.shape)
print(full_df.tail())
full_df_scaled_array = full_df.values
print(full_df_scaled_array.shape)

all_data = []
time_step = 30
for i in range(time_step, len(full_df_scaled_array)):
    data_x = []
    data_x.append(full_df_scaled_array[i-time_step:i, 0:full_df_scaled_array.shape[1]])
    data_x = np.array(data_x)
    prediction = model.predict(data_x)
    all_data.append(prediction)
    full_df.iloc[i, 0] = prediction
print(all_data)

new_array = np.array(all_data)
new_array = new_array.reshape(-1, 1)
prediction_copies_array = np.repeat(new_array, 3, axis=-1)
y_pred_future_30_days = scaler.inverse_transform(np.reshape(prediction_copies_array, (len(new_array), 3)))[:, 0]
print(y_pred_future_30_days)

df_origin = df_30_days_future = pd.read_csv("workload/qh2-rcc153-MTL LSTM 10.csv")
y_origin = df_origin.iloc[-30:, 0]
print(y_origin)

# 计算rmse
rmse = sqrt(mean_squared_error(y_origin, y_pred_future_30_days))
print("Test RMSE:%.3f:", rmse)







