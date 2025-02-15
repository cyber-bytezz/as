import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# File Paths
actual_file = "input/c3d_data.xlsx"  # Change if needed
predicted_file_xgb = "output/xgb_predictions.xlsx"
predicted_file_lstm = "output/lstm_predictions.xlsx"

# Load Data
df_actual = pd.read_excel(actual_file)
df_xgb = pd.read_excel(predicted_file_xgb)
df_lstm = pd.read_excel(predicted_file_lstm)

# Ensure Actual Data Matches Frame Order
df_actual.sort_values(by="Frame", inplace=True)

# Extract Motion Marker Columns
actual_markers = [col for col in df_actual.columns if col.startswith("L")]  # Left-side markers
predicted_markers = df_xgb.columns.tolist()  # XGB predicted markers

print(f"🔹 Actual Data Columns: {list(df_actual.columns)}")
print(f"🔹 Predicted Data Columns: {list(df_xgb.columns)}")

if not actual_markers or not predicted_markers:
    raise ValueError("❌ Error: Missing left-side markers in actual or predicted data!")

print(f"🟢 Found Left-Side Markers in Actual Data: {actual_markers}")
print(f"🟢 Found Left-Side Markers in Predicted Data: {predicted_markers}")

# Fill Missing Values (if any)
df_actual.fillna(method="ffill", inplace=True)
df_xgb.fillna(method="ffill", inplace=True)
df_lstm.fillna(method="ffill", inplace=True)

# 🔹 Align Data Lengths (Fix Mismatch Issue)
min_length = min(len(df_actual), len(df_xgb), len(df_lstm))
df_actual = df_actual.iloc[:min_length]
df_xgb = df_xgb.iloc[:min_length]
df_lstm = df_lstm.iloc[:min_length]

# Compute Errors (Mean Absolute Error & Root Mean Squared Error)
mae_xgb = mean_absolute_error(df_actual[actual_markers], df_xgb[predicted_markers])
rmse_xgb = np.sqrt(mean_squared_error(df_actual[actual_markers], df_xgb[predicted_markers]))

mae_lstm = mean_absolute_error(df_actual[actual_markers], df_lstm[predicted_markers])
rmse_lstm = np.sqrt(mean_squared_error(df_actual[actual_markers], df_lstm[predicted_markers]))

print(f"📊 **XGBoost Model Metrics**")
print(f"✅ MAE: {mae_xgb:.4f}")
print(f"✅ RMSE: {rmse_xgb:.4f}")

print(f"📊 **LSTM Model Metrics**")
print(f"✅ MAE: {mae_lstm:.4f}")
print(f"✅ RMSE: {rmse_lstm:.4f}")

# Save Results
results_df = pd.DataFrame({
    "Model": ["XGBoost", "LSTM"],
    "MAE": [mae_xgb, mae_lstm],
    "RMSE": [rmse_xgb, rmse_lstm]
})

results_df.to_csv("output/error_analysis.csv", index=False)
print("✅ Error analysis saved to `output/error_analysis.csv` 🎯")
