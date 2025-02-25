# AI_Timing_Prediction
import os
import re
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function to extract features from RTL files
def extract_features(rtl_code):
    gate_count = len(re.findall(r'\b(AND|OR|NAND|NOR|XOR|XNOR|NOT)\b', rtl_code, re.IGNORECASE))
    fan_in = len(re.findall(r'input', rtl_code, re.IGNORECASE))
    fan_out = len(re.findall(r'output', rtl_code, re.IGNORECASE))
    return [gate_count, fan_in, fan_out]

# Load dataset from RTL files
def load_dataset(rtl_folder, synthesis_report):
    data = []
    labels = []
    for file in os.listdir(rtl_folder):
        if file.endswith(".v") or file.endswith(".sv"):
            with open(os.path.join(rtl_folder, file), 'r') as f:
                rtl_code = f.read()
                features = extract_features(rtl_code)
                data.append(features)
                labels.append(synthesis_report.get(file, 0))  # Assuming synthesis_report has file-depth mappings
    return np.array(data), np.array(labels)

# Sample dataset (Replace with actual RTL files and synthesis data)
synthesis_report = {'module1.v': 5, 'module2.v': 7, 'module3.v': 4}
X, y = load_dataset("rtl_files", synthesis_report)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ML model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")

# Prediction function
def predict_logic_depth(rtl_code):
    features = np.array(extract_features(rtl_code)).reshape(1, -1)
    return model.predict(features)[0]

# Example usage
test_rtl_code = """
module example (input a, b, output y);
  wire n1;
  AND g1 (n1, a, b);
  NOT g2 (y, n1);
endmodule
"""
print(f"Predicted Logic Depth: {predict_logic_depth(test_rtl_code)}")
