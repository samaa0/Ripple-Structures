import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, cohen_kappa_score
import joblib  # For loading the scaler
from ripple_paths import model_path
# Load the scaler
scaler = joblib.load(model_path('feature_scaler.pkl'))

# Load the trained LightGBM model
model = lgb.Booster(model_file=str(model_path('best_lgbm_model_rank_1.txt')))

# Load the training data
train_data = pd.read_csv('test_features.csv')  # Replace with the path to your training data

# Separate features and labels
feature_columns = [col for col in train_data.columns if col not in ['label', 'sequence']]
X_train = train_data[feature_columns]
y_train = train_data['label']

# Apply the scaler to the features
X_train_scaled = scaler.transform(X_train)

# Make predictions using the loaded model
y_pred_probs = model.predict(X_train_scaled)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# Compute the confusion matrix and performance metrics
conf_mat = confusion_matrix(y_train, y_pred_labels)
macro_f1 = f1_score(y_train, y_pred_labels, average='macro')
accuracy = accuracy_score(y_train, y_pred_labels) * 100
kappa = cohen_kappa_score(y_train, y_pred_labels)

# Print the confusion matrix and performance metrics
print("Confusion Matrix:")
print(conf_mat)
print("\nPerformance Metrics on Training Set:")
print(f"Macro F1-Score: {macro_f1:.4f}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Cohen's Kappa: {kappa:.4f}")
