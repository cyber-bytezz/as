import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load extracted Excel data
file_path = "input/c3d_data.xlsx"
df = pd.read_excel(file_path)

# Check for missing values and fill them
df = df.interpolate(method='linear')

# Select markers
right_markers = [col for col in df.columns if col.startswith("R")]
left_markers = [col for col in df.columns if col.startswith("L")]

# Ensure columns exist
if not right_markers or not left_markers:
    raise ValueError("Error: No right-side or left-side markers found in dataset!")

X = df[right_markers]  # Features: Right-side markers
y = df[left_markers]   # Targets: Left-side markers

# Normalize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Save scalers
joblib.dump(scaler_X, "data/scaler_X.pkl")
joblib.dump(scaler_y, "data/scaler_y.pkl")

# Split into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Save datasets
np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)

print(f"✅ Data is ready for training! {X_train.shape} features, {y_train.shape} targets")
