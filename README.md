
# **🏃 Motion Prediction Using Machine Learning and Deep Learning (XGBoost & LSTM)**

## **Project Overview**
In this project, we focus on predicting human motion using **machine learning** (XGBoost) and **deep learning** (LSTM). The goal is to **predict and analyze motion data** (such as body parts' positions in 3D space), based on motion capture data collected from **C3D files**.

### **What This Project Does:**
- **Data Extraction**: Process and extract **motion capture data** from **C3D files**.
- **Feature Engineering**: Extract key motion features and prepare the data for model training.
- **Model Training**: Train two different types of models:
    1. **XGBoost (Extreme Gradient Boosting)** for handling tabular data.
    2. **LSTM (Long Short-Term Memory)**, a type of Recurrent Neural Network (RNN), for learning temporal sequences.
- **Prediction**: Generate **motion predictions** using the trained models.
- **Error Analysis**: Compare the actual motion data with the predicted results to assess model performance.
- **Visualization**: Visualize and interpret the results through error plots and actual vs predicted comparisons.

---

## **🔑 Key Components**
1. **Data Preprocessing**: Organize raw motion capture data for machine learning.
2. **Model Training**: Train both **XGBoost** and **LSTM** to predict the motion data.
3. **Prediction and Evaluation**: Predict the motion of body parts and evaluate model performance.
4. **Error Analysis & Visualization**: Analyze errors, visualize the predictions, and create reports.

---

## **📁 Project Folder Structure**
```
new/
│── data/               # 📂 Stores processed and raw data
│   ├── X_train.npy      # Training data (features)
│   ├── y_train.npy      # Training data (labels)
│   ├── X_test.npy       # Testing data (features)
│   ├── y_test.npy       # Testing data (labels)
│   ├── scaler_X.pkl     # Data normalizer (features)
│   ├── scaler_y.pkl     # Data normalizer (labels)
│
│── input/              # 📂 Stores input files (raw C3D and Excel data)
│   ├── 1.c3d           # Raw C3D motion capture data
│   ├── c3d_data.xlsx   # Extracted data from C3D file in Excel format
│
│── models/             # 📂 Stores trained machine learning models
│   ├── lstm_mocap_model.keras     # Trained LSTM model
│   ├── xgboost_mocap_model.json   # Trained XGBoost model
│
│── output/             # 📂 Stores results like predictions, error analysis
│   ├── xgb_predictions.xlsx      # Predictions made by XGBoost
│   ├── lstm_predictions.xlsx     # Predictions made by LSTM
│   ├── error_analysis.csv        # MAE, RMSE values for model comparison
│   ├── motion_comparison.png     # Plot showing actual vs predicted motion
│
│── scripts/            # 📂 Python scripts for all stages
│   ├── process_mocap.py        # Processes C3D motion data into a structured format
│   ├── train_xgboost.py        # Trains XGBoost model
│   ├── train_lstm.py           # Trains LSTM model
│   ├── predict_motion.py       # Uses trained models to generate predictions
│   ├── analyze_errors.py       # Compares actual vs predicted data and calculates errors
│   ├── visualize_motion.py     # Visualizes results with plots
│
│── venv/               # 📂 Virtual environment (Python dependencies)
│── README.md           # 📖 Project documentation
```

---

## **💻 How the Project Works (In Detail)**

### **Step 1: Preprocessing Motion Data**

📌 **Script**: `scripts/process_mocap.py`  
📌 **Input**: Raw **C3D** motion data and corresponding **Excel files**.  
📌 **Output**: Preprocessed **training** and **testing** datasets:  
- `X_train.npy`: Features (e.g., motion data points)  
- `y_train.npy`: Labels (e.g., the corresponding positions)  
- `X_test.npy`: Testing features  
- `y_test.npy`: Testing labels  

#### **What Happens?**
1. **C3D Data Extraction**: The raw **C3D file** contains 3D motion data of body parts (markers). The script extracts **X, Y, Z positions** of these markers.
2. **Data Normalization**: The extracted data is normalized using **scalers** (`scaler_X.pkl` and `scaler_y.pkl`) to scale down large values and improve the model's efficiency.
3. **Data Splitting**: The dataset is split into **training** (80%) and **testing** (20%) sets for model evaluation.

---

### **Step 2: Training Machine Learning Models**

📌 **Scripts**:  
1. **XGBoost Model**: `scripts/train_xgboost.py`  
2. **LSTM Model**: `scripts/train_lstm.py`  

📌 **Input**: `data/X_train.npy`, `data/y_train.npy`  
📌 **Output**:  
- Trained **XGBoost Model** saved as `models/xgboost_mocap_model.json`  
- Trained **LSTM Model** saved as `models/lstm_mocap_model.keras`  

#### **What Happens?**
1. **XGBoost Model**:
   - **XGBoost** is an ensemble technique that uses decision trees to make predictions. It is especially good at handling **tabular data**.
   - The model is trained on the **training dataset**, learning the relationship between the **features** (X positions, Y, Z) and **labels** (actual motion data).
   
2. **LSTM Model**:
   - **LSTM** is a type of **Recurrent Neural Network (RNN)** designed to handle sequential data. This model is ideal for **time-series predictions**, as motion data has inherent sequential patterns.
   - LSTM is trained to predict the **next positions of markers** in the sequence.

---

### **Step 3: Generating Predictions**

📌 **Script**: `scripts/predict_motion.py`  
📌 **Input**: Trained models (`models/xgboost_mocap_model.json`, `models/lstm_mocap_model.keras`)  
📌 **Output**: Predictions:  
- `output/xgb_predictions.xlsx`  
- `output/lstm_predictions.xlsx`  

#### **What Happens?**
1. **XGBoost Prediction**: Uses the trained **XGBoost model** to make predictions on the **testing set** (`X_test.npy`), generating predicted motion data.
2. **LSTM Prediction**: Uses the trained **LSTM model** to generate predictions based on the same testing data.
3. **Saving Predictions**: The predictions are saved to Excel files for further analysis and visualization.

---

### **Step 4: Analyzing Errors**

📌 **Script**: `scripts/analyze_errors.py`  
📌 **Input**: `output/xgb_predictions.xlsx`, `output/lstm_predictions.xlsx`  
📌 **Output**:  
- `output/error_analysis.csv`  

#### **What Happens?**
1. **Comparing Actual vs Predicted**: The script compares **actual motion data** (`y_test.npy`) with the **predicted motion data** from both models (XGBoost and LSTM).
2. **Calculating Errors**: The **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** are calculated to assess model accuracy.
3. **Saving Error Analysis**: The error metrics are saved in a CSV file for reporting and review.

---

### **Step 5: Visualizing Predictions**

📌 **Script**: `scripts/visualize_motion.py`  
📌 **Input**: `output/xgb_predictions.xlsx`, `output/lstm_predictions.xlsx`  
📌 **Output**:  
- `output/motion_comparison.png` (Visualization)  

#### **What Happens?**
1. **Plotting Actual vs Predicted**: The script generates visual plots comparing **actual motion** vs **predicted motion** for key body parts, such as the **left back waist (LBWT)** markers.
2. **Plotting Error Trends**: It shows how well the models are predicting motion for each axis (X, Y, Z).
3. **Output**: A **graphical plot** is generated, helping the team visualize the model's performance.

---

## **📊 Results & Observations**

### **Model Performance**
- **XGBoost** and **LSTM** both provided reasonable predictions, but each has **room for improvement**.
- **XGBoost** performed better on structured features, while **LSTM** was better at capturing sequential motion patterns.
- **MAE** and **RMSE** metrics show that **LSTM and XGBoost are both effective** but can still be fine-tuned.

---

## **📝 Conclusion & Future Work**
### **What We’ve Done**:
- We successfully **extracted and processed motion capture data**, trained **XGBoost** and **LSTM** models, and used them to predict human motion.
- The project provides a full **end-to-end pipeline**, from data processing to model training, prediction, error analysis, and visualization.

### **Next Steps for Improvement**:
- **Improve Model Accuracy** by training with more data and using advanced models like **GRU** or **Transformer**.
- **Optimize Hyperparameters** to reduce errors and improve the RMSE score.
- **Deploy Model for Real-Time Use**, to predict human motion in real-time systems.

---

## **🚀 How to Run the Project**
1. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```
   
2. **Preprocess Data**:
   ```
   python scripts/process_mocap.py
   ```

3. **Train Models**:
   ```
   python scripts/train_xgboost.py
   python scripts/train_lstm.py
   ```

4. **Generate Predictions**:
   ```
   python scripts/predict_motion.py
   ```

5. **Analyze Errors**:
   ```
   python scripts/analyze_errors.py
   ```

6. **Visualize Predictions**:
   ```
   python scripts/visualize_motion.py
   ```

