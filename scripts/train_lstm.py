import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Load data
X_train, X_test = np.load("data/X_train.npy"), np.load("data/X_test.npy")
y_train, y_test = np.load("data/y_train.npy"), np.load("data/y_test.npy")

# Check for NaN values and remove them
print(f"🔹 NaN in X_train: {np.isnan(X_train).sum()}") 
print(f"🔹 NaN in y_train: {np.isnan(y_train).sum()}") 
print(f"🔹 NaN in X_test: {np.isnan(X_test).sum()}")  
print(f"🔹 NaN in y_test: {np.isnan(y_test).sum()}")  

# Fill NaN values with zero or mean
X_train = np.nan_to_num(X_train, nan=0.0)
y_train = np.nan_to_num(y_train, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)
y_test = np.nan_to_num(y_test, nan=0.0)

# Reshape for LSTM (ensure correct input format)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build LSTM Model with `tanh` activation instead of `relu`
model = Sequential([
    LSTM(50, activation="tanh", return_sequences=True, input_shape=(1, X_train.shape[2])),
    LSTM(50, activation="tanh"),
    Dense(y_train.shape[1])
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Save Model in `.keras` format
model.save("models/lstm_mocap_model.keras")

# Evaluate Performance
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ LSTM Model RMSE: {rmse:.4f}")
