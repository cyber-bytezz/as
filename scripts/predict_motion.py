import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import os

# Load data
X_test = np.load("data/X_test.npy")

# Load models
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgboost_mocap_model.json")

lstm_model = tf.keras.models.load_model("models/lstm_mocap_model.keras")

# Ensure output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Predict using XGBoost
dtest = xgb.DMatrix(X_test)
xgb_predictions = xgb_model.predict(dtest)

# Predict using LSTM
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
lstm_predictions = lstm_model.predict(X_test_reshaped)

# Save predictions
df_xgb_predictions = pd.DataFrame(xgb_predictions, columns=[f"L{i}" for i in range(xgb_predictions.shape[1])])
df_lstm_predictions = pd.DataFrame(lstm_predictions, columns=[f"L{i}" for i in range(lstm_predictions.shape[1])])

df_xgb_predictions.to_excel(f"{output_dir}/xgb_predictions.xlsx", index=False)
df_lstm_predictions.to_excel(f"{output_dir}/lstm_predictions.xlsx", index=False)

print("✅ Predictions saved successfully in 'output/' folder!")
