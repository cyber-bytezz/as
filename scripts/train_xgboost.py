import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import mean_squared_error

# Load training data
X_train, X_test = np.load("data/X_train.npy"), np.load("data/X_test.npy")
y_train, y_test = np.load("data/y_train.npy"), np.load("data/y_test.npy")

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=10)
xgb_model.fit(X_train, y_train)

# Save model
xgb_model.save_model("models/xgboost_mocap_model.json")

# Evaluate
y_pred = xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ XGBoost Model RMSE: {rmse:.4f}")
