# test_model.py

# Import necessary libraries
import numpy as np
import pandas as pd
import joblib  # For loading scikit-learn models
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
from ripple_paths import model_path

# Step 1: Load Test Data

# Load the merged test predictions, features, and labels
df_merged_test_preds = pd.read_csv('merged_test_predictions.csv')
df_merged_test_features = pd.read_csv('merged_test_features.csv')
test_labels_df = pd.read_csv('test_labels.csv')

# Prepare data for testing
X_test_preds = df_merged_test_preds.filter(regex='class_.*').values
X_test_features = df_merged_test_features.drop(columns=['sequence', 'label']).values
y_test = test_labels_df['label'].values.astype(int)

# Combine predictions and features for testing
X_test_combined = np.hstack([X_test_features, X_test_preds])

# Step 2: Load the Trained Scikit-Learn Model

# Load the model using joblib
model = joblib.load(model_path('best_meta_learner_model.pkl'))

# Step 3: Evaluate the Model on the Test Set

# Make predictions
y_test_pred = model.predict(X_test_combined)

# Step 4: Calculate Evaluation Metrics

macro_f1 = f1_score(y_test, y_test_pred, average='macro')
accuracy = accuracy_score(y_test, y_test_pred)
kappa = cohen_kappa_score(y_test, y_test_pred)

print(f"Macro F1-Score: {macro_f1}")
print(f"Accuracy: {accuracy}")
print(f"Cohen's Kappa: {kappa}")
