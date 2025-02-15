import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load actual and predicted motion data
df_actual = pd.read_excel("input/c3d_data.xlsx")
df_predicted = pd.read_excel("data/motion_predictions.xlsx")

# Trim actual data to match predicted data length
df_actual = df_actual.iloc[:df_predicted.shape[0]]

# Select markers for visualization (Example: Left Back Waist - LBWT)
marker_actual = ["LBWT_X", "LBWT_Y", "LBWT_Z"]
marker_predicted = df_predicted.columns[:3]  # Assuming 1st 3 cols correspond to LBWT

# Create time-series plot
plt.figure(figsize=(12, 6))
for i in range(3):
    plt.plot(df_actual["Frame"], df_actual[marker_actual[i]], label=f"Actual {marker_actual[i]}", linestyle="dashed")
    plt.plot(df_actual["Frame"], df_predicted[marker_predicted[i]], label=f"Predicted {marker_actual[i]}")

plt.xlabel("Frame")
plt.ylabel("Position Value")
plt.title("Actual vs Predicted Motion for LBWT (Left Back Waist)")
plt.legend()
plt.show()
