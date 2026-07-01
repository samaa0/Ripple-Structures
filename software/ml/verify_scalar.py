# verify_scaler.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from ripple_paths import model_path

# ---------------------------------------------------------------------------------------
# Step 1: Load the saved scaler
# ---------------------------------------------------------------------------------------
try:
    scaler = joblib.load(model_path('feature_scaler.pkl'))
    print("Scaler loaded successfully.\n")
except FileNotFoundError:
    print("Error: The scaler file 'feature_scaler.pkl' was not found.")
    exit(1)

# ---------------------------------------------------------------------------------------
# Step 2: Load the raw training features
# ---------------------------------------------------------------------------------------
try:
    train_features_raw = pd.read_csv('train_features.csv')
    print("Raw training features loaded successfully.\n")
except FileNotFoundError:
    print("Error: The file 'train_features_raw.csv' was not found.")
    exit(1)

# ---------------------------------------------------------------------------------------
# Step 3: Apply the loaded scaler to the raw features
# ---------------------------------------------------------------------------------------
# Separate the features and labels
feature_columns = [col for col in train_features_raw.columns if col not in ['label', 'sequence']]
X_raw = train_features_raw[feature_columns]
y_train = train_features_raw['label']

# Apply the scaler to the raw features
X_scaled = scaler.transform(X_raw)
print("Scaler applied to raw training features.\n")

# ---------------------------------------------------------------------------------------
# Step 4: Load the normalized training features
# ---------------------------------------------------------------------------------------
try:
    train_features_normalized = pd.read_csv('train_features_normalized.csv')
    print("Normalized training features loaded successfully.\n")
except FileNotFoundError:
    print("Error: The file 'train_features_normalized.csv' was not found.")
    exit(1)

X_normalized = train_features_normalized[feature_columns]

# ---------------------------------------------------------------------------------------
# Step 5: Compare the scaled features with the normalized features
# ---------------------------------------------------------------------------------------
# Calculate the absolute differences between the two sets of features
differences = np.abs(X_scaled - X_normalized.values)
max_difference = np.max(differences)
mean_difference = np.mean(differences)

print(f"Maximum difference between scaled features and normalized features: {max_difference:.6f}")
print(f"Mean difference between scaled features and normalized features: {mean_difference:.6f}\n")

# Set a tolerance level for floating-point comparisons
tolerance = 1e-6

if max_difference < tolerance:
    print("Verification success: Scaled features match the normalized features within the acceptable tolerance.")
    print("The saved scaler is correct and usable.\n")
else:
    print("Verification failure: There is a significant discrepancy between scaled features and normalized features.")
    print("Please check the scaler and the feature normalization process.\n")

# ---------------------------------------------------------------------------------------
# Step 6: Check the mean and standard deviation of the scaled features
# ---------------------------------------------------------------------------------------
mean_scaled = np.mean(X_scaled, axis=0)
std_scaled = np.std(X_scaled, axis=0)

print("Mean of scaled features (should be close to 0):")
print(mean_scaled)
print("\nStandard deviation of scaled features (should be close to 1):")
print(std_scaled)
