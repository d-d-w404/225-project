import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv("data.csv")  # 假设文件名为 bitcoin.csv
data = data.iloc[::-1].reset_index(drop=True)

features = ["Open", "High", "Low", "Close", "Volume", "Market Cap"]
data = data[features]

#print(data["Open"])


print(data.head())

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)



print(scaled_data[:5])

time_steps = 7
predict_days=1



def create_sequences(data, time_steps, predict_days):
    X, y = [], []
    for i in range(len(data) - time_steps - predict_days + 1):
        X.append(data[i:i+time_steps, [4,5]])
        y.append(data[i+time_steps:i+time_steps+predict_days, 3])
    return np.array(X), np.array(y)















X, y = create_sequences(scaled_data, time_steps,predict_days)

# 查看形状
print("输入数据形状:", X.shape)
print("目标数据形状:", y.shape)

split_index = int(0.9 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print("训练集形状:", X_train.shape, y_train.shape)
print("测试集形状:", X_test.shape, y_test.shape)

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))


model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))


model.add(Dense(units=predict_days))  # 输出第11天的收盘价


model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

predicted = model.predict(X_test)






filled_scaled_data = np.zeros((len(predicted), scaled_data.shape[1]))


filled_scaled_data[:, 3] = predicted.flatten()


unscaled_data = scaler.inverse_transform(filled_scaled_data)


predicted_unscaled = unscaled_data[:, 3]
print("反归一化后的 Close 数据:")
print(predicted_unscaled)



zeros = np.zeros((len(y_test), data.shape[1]))


zeros[:, 3] = y_test.flatten()


y_test_unscaled = scaler.inverse_transform(zeros)[:, 3]



print("反归一化后的 Close 列数据:")
print(y_test_unscaled)




look1=predict_days-1

plt.figure(figsize=(12, 6))



plt.plot(predicted_unscaled, label="Predicted Price", color="red")
plt.plot(y_test_unscaled, label="Real Price", color="green")

plt.title(f"Bitcoin Prediction ({predict_days} Days)")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.legend()
plt.show()




mse = mean_squared_error(y_test_unscaled.flatten(), predicted_unscaled.flatten())
mae = mean_absolute_error(y_test_unscaled.flatten(), predicted_unscaled.flatten())
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")


from sklearn.metrics import r2_score

r2 = r2_score(y_test_unscaled.flatten(), predicted_unscaled.flatten())
print(f"R-squared (R²): {r2:.4f}")
