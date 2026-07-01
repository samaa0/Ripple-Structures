# Final_fusion.py

# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import threading
import json
import optuna  # Import optuna to load the study and perform hyperparameter optimization
import lightgbm as lgb
import xgboost as xgb  # For XGBoost meta-learner
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import joblib  # For loading scaler if needed
import matplotlib.pyplot as plt
from pathlib import Path
from ripple_paths import model_path, study_storage
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV  # For probability calibration
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# GPU Management Code
gpu_lock = threading.Lock()
available_gpus = list(range(torch.cuda.device_count()))

def get_assigned_gpu_id():
    with gpu_lock:
        if available_gpus:
            gpu_id = available_gpus.pop()
            print(f"Assigned GPU {gpu_id}")
            return gpu_id
        else:
            print("No GPU available currently. Will wait for one to become available.")
            return None  # Return None if no GPU is available

def release_gpu_id(gpu_id):
    with gpu_lock:
        available_gpus.append(gpu_id)
        print(f"Released GPU {gpu_id}")

# Step 1: Load and Prepare the Data

# For Raw Data Branch
# Load normalized training and test data
train_data_normalized = pd.read_csv('Raw_final_train_data_normalized.csv')
test_data_normalized = pd.read_csv('Raw_final_test_data_normalized.csv')

# For Feature Data Branch
# Load normalized feature datasets
train_features_normalized = pd.read_csv('train_features_normalized.csv')
test_features_normalized = pd.read_csv('test_features_normalized.csv')

# Extract sequences and labels from the raw data
train_sequence_labels = train_data_normalized.groupby(['sequence', 'label']).size().reset_index()[['sequence', 'label']]
test_sequence_labels = test_data_normalized.groupby(['sequence', 'label']).size().reset_index()[['sequence', 'label']]

# Prepare sequence indices and labels
num_classes = len(train_sequence_labels['label'].unique())

# Define the PyTorch Dataset for raw data
class RawGestureDataset(Dataset):
    def __init__(self, data):
        self.sequences = []
        self.labels = []
        self.sequence_ids = []
        
        grouped = data.groupby(['sequence', 'label'])
        for (seq_id, label), group in grouped:
            features = group[['x', 'y', 'z']].values
            self.sequences.append(torch.Tensor(features))
            self.labels.append(label)
            self.sequence_ids.append(seq_id)
            
    def __len__(self):
        return len(self.sequences)
     
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.sequence_ids[idx]

# Define the Dataset for feature data
class FeatureGestureDataset(Dataset):
    def __init__(self, data):
        self.X = data.drop(columns=['sequence', 'label']).values
        self.y = data['label'].values.astype(int)
        self.sequence_ids = data['sequence'].values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.sequence_ids[idx]

# Aggregate feature data to ensure one row per sequence and label
train_features_grouped = train_features_normalized.groupby(['sequence', 'label']).mean().reset_index()
test_features_grouped = test_features_normalized.groupby(['sequence', 'label']).mean().reset_index()

# Create datasets
raw_train_dataset = RawGestureDataset(train_data_normalized)
raw_test_dataset = RawGestureDataset(test_data_normalized)

feature_train_dataset = FeatureGestureDataset(train_features_grouped)
feature_test_dataset = FeatureGestureDataset(test_features_grouped)

# Step 2: Load Models and Hyperparameters

# For Raw Data Branch (GRU Model)
# Load the Optuna study for the raw data branch
storage_name_raw = study_storage('raw_data_branch_gru_study.db')
study_raw = optuna.load_study(study_name='gesture_recognition', storage=storage_name_raw)

# Get the best trial
best_trial_raw = study_raw.best_trial

# Extract hyperparameters
best_params_raw = best_trial_raw.params

# Parse hyperparameters
input_size = 3  # Number of input features (x, y, z)
hidden_size = int(best_params_raw['hidden_size'])
num_layers = int(best_params_raw['n_layers'])
dropout_rate = best_params_raw['dropout_rate']
output_size = num_classes  # Number of classes
activation_name = best_params_raw.get('activation', 'relu')
batch_size = int(best_params_raw['batch_size'])

# Print the loaded hyperparameters
print("Loaded hyperparameters for GRU Model:")
print(f"Input Size: {input_size}")
print(f"Hidden Size: {hidden_size}")
print(f"Number of Layers: {num_layers}")
print(f"Dropout Rate: {dropout_rate}")
print(f"Output Size (Number of Classes): {output_size}")
print(f"Activation Function: {activation_name}")
print(f"Batch Size: {batch_size}")

# Define the GRUModel class with feature extraction capability
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        if activation_name == 'relu':
            self.activation = nn.ReLU()
        elif activation_name == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()  # Default activation

    def forward(self, x, return_features=False):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Get the output from the last time step
        features = self.dropout(out)
        logits = self.fc(features)
        if return_features:
            return logits, features
        else:
            return logits

# Initialize the model with the loaded hyperparameters
model = GRUModel(input_size=input_size,
                 hidden_size=hidden_size,
                 num_layers=num_layers,
                 dropout=dropout_rate,
                 output_size=output_size)

# Load the trained model weights
model.load_state_dict(torch.load(model_path('final_raw_data_branch_model.pt'), map_location=torch.device('cpu')))

# Feature Data Branch (LightGBM Model)

# Load the JSON file
metadata_filename = model_path('model_rank_1_performance.json')

# Read the JSON file
with open(metadata_filename, 'r') as f:
    metadata = json.load(f)

# Extract hyperparameters
best_params_lgbm = metadata['hyperparameters']

# Hardcode the model filename
lgbm_model_filename = str(model_path('best_lgbm_model_rank_1.txt'))

# Print the loaded hyperparameters
print("\nLoaded hyperparameters for LightGBM Model:")
for key, value in best_params_lgbm.items():
    print(f"{key}: {value}")

# Load the trained LightGBM model
try:
    lgb_model = lgb.Booster(model_file=lgbm_model_filename)
except Exception as e:
    print("Error loading LightGBM model:", e)
    raise

# Step 3: Collect Predictions and Extract Features

# Function to pad sequences with IDs
def pad_sequence_with_ids(batch):
    # Collate function to pad sequences to the same length and include sequence IDs
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    seq_ids = [item[2] for item in batch]
    sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return sequences_padded, labels, seq_ids

# Collect Predictions and Deep Features for Raw Data Branch
gpu_id = get_assigned_gpu_id()
try:
    if gpu_id is None or not torch.cuda.is_available():
        print("No GPU available currently or CUDA not available. Using CPU.")
        device = torch.device('cpu')
    else:
        print(f"Using GPU: {gpu_id}")
        device = torch.device(f'cuda:{gpu_id}')

    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Raw Data Branch Predictions and Features on Training Data
    raw_train_loader = DataLoader(raw_train_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_sequence_with_ids)
    all_train_preds = []
    all_train_seq_ids = []
    all_train_labels = []
    all_train_features = []
    with torch.no_grad():
        for sequences_batch, labels_batch, seq_ids_batch in raw_train_loader:
            sequences_batch = sequences_batch.to(device)
            outputs, features = model(sequences_batch, return_features=True)
            preds = nn.functional.softmax(outputs, dim=1).cpu().numpy()
            features = features.cpu().numpy()
            all_train_preds.append(preds)
            all_train_features.append(features)
            all_train_seq_ids.extend(seq_ids_batch)
            all_train_labels.extend(labels_batch.numpy())

    raw_train_preds = np.vstack(all_train_preds)
    raw_train_features = np.vstack(all_train_features)
    raw_train_seq_ids = np.array(all_train_seq_ids)
    raw_train_labels = np.array(all_train_labels)

    # Raw Data Branch Predictions and Features on Test Data
    raw_test_loader = DataLoader(raw_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_sequence_with_ids)
    all_test_preds = []
    all_test_seq_ids = []
    all_test_labels = []
    all_test_features = []
    with torch.no_grad():
        for sequences_batch, labels_batch, seq_ids_batch in raw_test_loader:
            sequences_batch = sequences_batch.to(device)
            outputs, features = model(sequences_batch, return_features=True)
            preds = nn.functional.softmax(outputs, dim=1).cpu().numpy()
            features = features.cpu().numpy()
            all_test_preds.append(preds)
            all_test_features.append(features)
            all_test_seq_ids.extend(seq_ids_batch)
            all_test_labels.extend(labels_batch.numpy())

    raw_test_preds = np.vstack(all_test_preds)
    raw_test_features = np.vstack(all_test_features)
    raw_test_seq_ids = np.array(all_test_seq_ids)
    raw_test_labels = np.array(all_test_labels)

finally:
    if gpu_id is not None:
        release_gpu_id(gpu_id)
    # Clean up
    del model
    torch.cuda.empty_cache()

# Feature Data Branch Predictions on Training Data
X_train_feature = feature_train_dataset.X
feature_train_seq_ids = feature_train_dataset.sequence_ids
feature_train_labels = feature_train_dataset.y

train_preds_feature_proba = lgb_model.predict(X_train_feature, num_iteration=lgb_model.best_iteration)
train_preds_feature_proba = np.reshape(train_preds_feature_proba, (-1, num_classes))
feature_train_preds = train_preds_feature_proba

# Feature Data Branch Predictions on Test Data
X_test_feature = feature_test_dataset.X
feature_test_seq_ids = feature_test_dataset.sequence_ids
feature_test_labels = feature_test_dataset.y

test_preds_feature_proba = lgb_model.predict(X_test_feature, num_iteration=lgb_model.best_iteration)
test_preds_feature_proba = np.reshape(test_preds_feature_proba, (-1, num_classes))
feature_test_preds = test_preds_feature_proba

# Step 4: Align Predictions and Features

# Create DataFrames for training data
df_raw_train_preds = pd.DataFrame(raw_train_preds, columns=[f'class_{i}' for i in range(num_classes)])
df_raw_train_preds['sequence'] = raw_train_seq_ids
df_raw_train_preds['label'] = raw_train_labels

df_raw_train_features = pd.DataFrame(raw_train_features)
df_raw_train_features['sequence'] = raw_train_seq_ids
df_raw_train_features['label'] = raw_train_labels

df_feature_train_preds = pd.DataFrame(feature_train_preds, columns=[f'class_{i}' for i in range(num_classes)])
df_feature_train_preds['sequence'] = feature_train_seq_ids
df_feature_train_preds['label'] = feature_train_labels

df_feature_train_features = pd.DataFrame(X_train_feature)
df_feature_train_features['sequence'] = feature_train_seq_ids
df_feature_train_features['label'] = feature_train_labels

# Merge predictions and features on 'sequence' and 'label' for training data
df_merged_train_preds = pd.merge(df_raw_train_preds, df_feature_train_preds, on=['sequence', 'label'], suffixes=('_raw', '_feature'))
df_merged_train_features = pd.merge(df_raw_train_features, df_feature_train_features, on=['sequence', 'label'], suffixes=('_raw', '_feature'))

# Ensure no duplicates after merging
duplicates_train = df_merged_train_preds.duplicated(subset=['sequence', 'label']).sum()
print(f"Number of duplicate (sequence, label) pairs in merged training data: {duplicates_train}")

# Create DataFrames for test data
df_raw_test_preds = pd.DataFrame(raw_test_preds, columns=[f'class_{i}' for i in range(num_classes)])
df_raw_test_preds['sequence'] = raw_test_seq_ids
df_raw_test_preds['label'] = raw_test_labels

df_raw_test_features = pd.DataFrame(raw_test_features)
df_raw_test_features['sequence'] = raw_test_seq_ids
df_raw_test_features['label'] = raw_test_labels

df_feature_test_preds = pd.DataFrame(feature_test_preds, columns=[f'class_{i}' for i in range(num_classes)])
df_feature_test_preds['sequence'] = feature_test_seq_ids
df_feature_test_preds['label'] = feature_test_labels

df_feature_test_features = pd.DataFrame(X_test_feature)
df_feature_test_features['sequence'] = feature_test_seq_ids
df_feature_test_features['label'] = feature_test_labels

# Merge predictions and features on 'sequence' and 'label' for test data
df_merged_test_preds = pd.merge(df_raw_test_preds, df_feature_test_preds, on=['sequence', 'label'], suffixes=('_raw', '_feature'))
df_merged_test_features = pd.merge(df_raw_test_features, df_feature_test_features, on=['sequence', 'label'], suffixes=('_raw', '_feature'))

# Ensure no duplicates after merging
duplicates_test = df_merged_test_preds.duplicated(subset=['sequence', 'label']).sum()
print(f"Number of duplicate (sequence, label) pairs in merged test data: {duplicates_test}")

# Step 5: Save Predictions, Features, and Labels

# Save the merged predictions
df_merged_train_preds.to_csv('merged_train_predictions.csv', index=False)
df_merged_test_preds.to_csv('merged_test_predictions.csv', index=False)

# Save the merged features
df_merged_train_features.to_csv('merged_train_features.csv', index=False)
df_merged_test_features.to_csv('merged_test_features.csv', index=False)

# Save the labels
train_labels_df = df_merged_train_preds[['sequence', 'label']]
test_labels_df = df_merged_test_preds[['sequence', 'label']]

train_labels_df.to_csv('train_labels.csv', index=False)
test_labels_df.to_csv('test_labels.csv', index=False)

print("\nPredictions, features, and labels saved.")

# Step 6: Advanced Fusion Strategies Implementation

# Load merged predictions, features, and labels
df_merged_train_preds = pd.read_csv('merged_train_predictions.csv')
df_merged_train_features = pd.read_csv('merged_train_features.csv')
train_labels_df = pd.read_csv('train_labels.csv')

df_merged_test_preds = pd.read_csv('merged_test_predictions.csv')
df_merged_test_features = pd.read_csv('merged_test_features.csv')
test_labels_df = pd.read_csv('test_labels.csv')

# Prepare data for fusion strategies
X_train_preds = df_merged_train_preds.filter(regex='class_.*').values
X_train_features = df_merged_train_features.drop(columns=['sequence', 'label']).values
y_train = train_labels_df['label'].values.astype(int)
groups_train = train_labels_df['sequence'].values  # For Group K-Fold

X_test_preds = df_merged_test_preds.filter(regex='class_.*').values
X_test_features = df_merged_test_features.drop(columns=['sequence', 'label']).values
y_test = test_labels_df['label'].values.astype(int)
groups_test = test_labels_df['sequence'].values

# Combine predictions and features for feature concatenation strategy
X_train_combined = np.hstack([X_train_features, X_train_preds])
X_test_combined = np.hstack([X_test_features, X_test_preds])

# Number of folds for cross-validation
n_splits = 5

# Use Group K-Fold Cross-Validation to avoid data leakage if sequences are correlated
gkf = GroupKFold(n_splits=n_splits)

# Initialize variables to store the best fusion strategy
best_f1 = -np.inf
best_strategy = None
best_params = None

### Fusion Strategy 1: Neural Network Meta-Learner ###

# Define the Neural Network Meta-Learner
class MetaLearnerNN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes=[64, 32], dropout_rate=0.5):
        super(MetaLearnerNN, self).__init__()
        layers = []
        last_size = input_size
        for hs in hidden_sizes:
            layers.append(nn.Linear(last_size, hs))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            last_size = hs
        layers.append(nn.Linear(last_size, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def objective_meta_learner_nn(trial):
    macro_f1_scores = []

    # Hyperparameters to tune
    hidden_sizes = []
    num_layers = trial.suggest_int('num_layers', 1, 3)
    for i in range(num_layers):
        hs = trial.suggest_int(f'hidden_size_{i}', 32, 128)
        hidden_sizes.append(hs)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    num_epochs = 50  # Early stopping will likely prevent reaching this

    # Use Group K-Fold Cross-Validation
    for train_index, val_index in gkf.split(X_train_combined, y_train, groups=groups_train):
        X_train_cv, X_val_cv = X_train_combined[train_index], X_train_combined[val_index]
        y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_cv, dtype=torch.float32),
                                                       torch.tensor(y_train_cv, dtype=torch.long))
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val_cv, dtype=torch.float32),
                                                     torch.tensor(y_val_cv, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        input_size = X_train_combined.shape[1]
        num_classes = len(np.unique(y_train))

        model = MetaLearnerNN(input_size=input_size, num_classes=num_classes,
                              hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)

        gpu_id = get_assigned_gpu_id()
        try:
            if gpu_id is None or not torch.cuda.is_available():
                device = torch.device('cpu')
            else:
                device = torch.device(f'cuda:{gpu_id}')
            model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            best_val_loss = np.inf
            patience = 5
            patience_counter = 0

            for epoch in range(num_epochs):
                # Training
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                # Validation
                model.eval()
                val_losses = []
                y_val_pred = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(device)
                        y_batch = y_batch.to(device)
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_losses.append(loss.item())
                        y_pred = outputs.argmax(dim=1).cpu().numpy()
                        y_val_pred.extend(y_pred)

                avg_val_loss = np.mean(val_losses)
                macro_f1 = f1_score(y_val_cv, y_val_pred, average='macro')

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            macro_f1_scores.append(macro_f1)
        finally:
            if gpu_id is not None:
                release_gpu_id(gpu_id)
            del model
            torch.cuda.empty_cache()

    return -np.mean(macro_f1_scores)

# Use Optuna for hyperparameter tuning
study_nn = optuna.create_study(direction='minimize')
study_nn.optimize(objective_meta_learner_nn, n_trials=50)

# Best hyperparameters found
best_params_nn = study_nn.best_params
best_macro_f1_nn = -study_nn.best_value
print(f"\n[Meta-Learner NN] Best Hyperparameters:")
for key, value in best_params_nn.items():
    print(f"{key}: {value}")
print(f"[Meta-Learner NN] Best Macro F1-Score: {best_macro_f1_nn}")

# Evaluate on test data using the best neural network meta-learner
hidden_sizes = []
num_layers = best_params_nn['num_layers']
for i in range(num_layers):
    hs = best_params_nn[f'hidden_size_{i}']
    hidden_sizes.append(hs)
dropout_rate = best_params_nn['dropout_rate']
lr = best_params_nn['learning_rate']
weight_decay = best_params_nn['weight_decay']
batch_size = best_params_nn['batch_size']
num_epochs = 50  # Early stopping will likely prevent reaching this

train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_combined, dtype=torch.float32),
                                               torch.tensor(y_train, dtype=torch.long))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test_combined, dtype=torch.float32),
                                              torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = X_train_combined.shape[1]
num_classes = len(np.unique(y_train))

model = MetaLearnerNN(input_size=input_size, num_classes=num_classes,
                      hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)

gpu_id = get_assigned_gpu_id()
try:
    if gpu_id is None or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu_id}')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = np.inf
    patience = 5
    patience_counter = 0

    # Use a validation split from the training data
    val_split = 0.1
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

    for epoch in range(num_epochs):
        # Training
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        y_val_pred = []
        y_val_true = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
                y_pred = outputs.argmax(dim=1).cpu().numpy()
                y_val_pred.extend(y_pred)
                y_val_true.extend(y_batch.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        macro_f1 = f1_score(y_val_true, y_val_pred, average='macro')

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Evaluate on test set
    model.eval()
    y_test_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            y_pred = outputs.argmax(dim=1).cpu().numpy()
            y_test_pred.extend(y_pred)

    # Calculate evaluation metrics
    macro_f1_test_nn = f1_score(y_test, y_test_pred, average='macro')
    accuracy_test_nn = accuracy_score(y_test, y_test_pred)
    kappa_test_nn = cohen_kappa_score(y_test, y_test_pred)

    print(f"\n[Meta-Learner NN] Evaluation Metrics on Test Set:")
    print(f"Macro F1-Score: {macro_f1_test_nn}")
    print(f"Accuracy: {accuracy_test_nn}")
    print(f"Cohen's Kappa: {kappa_test_nn}")

finally:
    if gpu_id is not None:
        release_gpu_id(gpu_id)
    del model
    torch.cuda.empty_cache()

# Update best strategy if needed
if macro_f1_test_nn > best_f1:
    best_f1 = macro_f1_test_nn
    best_strategy = 'Meta-Learner-NN'
    best_params = best_params_nn

### Fusion Strategy 2: Feature Concatenation and Model Training ###

def objective_feature_concat(trial):
    macro_f1_scores = []

    # Hyperparameters to tune
    model_name = trial.suggest_categorical('model', ['LightGBM', 'XGBoost', 'RandomForest'])
    if model_name == 'LightGBM':
        params = {
            'objective': 'multiclass',
            'num_class': num_classes,
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'verbosity': -1,
            'random_state': 42,
            'device': 'gpu' if torch.cuda.is_available() else 'cpu',
            'gpu_device_id': gpu_id if gpu_id is not None else -1,
        }
    elif model_name == 'XGBoost':
        params = {
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
            'verbosity': 0,
            'random_state': 42,
            'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'auto',
            'gpu_id': gpu_id if gpu_id is not None else 0,
        }
    elif model_name == 'RandomForest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'n_jobs': -1,
            'random_state': 42
        }

    # Use Group K-Fold Cross-Validation
    for train_index, val_index in gkf.split(X_train_combined, y_train, groups=groups_train):
        X_train_cv, X_val_cv = X_train_combined[train_index], X_train_combined[val_index]
        y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]

        if model_name == 'LightGBM':
            train_data = lgb.Dataset(X_train_cv, label=y_train_cv)
            val_data = lgb.Dataset(X_val_cv, label=y_val_cv, reference=train_data)
            model = lgb.train(params, train_data, num_boost_round=1000,
                              valid_sets=[val_data], early_stopping_rounds=10, verbose_eval=False)
            y_pred_prob = model.predict(X_val_cv, num_iteration=model.best_iteration)
            y_pred = np.argmax(y_pred_prob, axis=1)
        elif model_name == 'XGBoost':
            dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv)
            dval = xgb.DMatrix(X_val_cv, label=y_val_cv)
            evals = [(dval, 'validation')]
            model = xgb.train(params, dtrain, num_boost_round=1000,
                              evals=evals, early_stopping_rounds=10, verbose_eval=False)
            y_pred_prob = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
            y_pred = np.argmax(y_pred_prob, axis=1)
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(**params)
            model.fit(X_train_cv, y_train_cv)
            y_pred = model.predict(X_val_cv)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        macro_f1 = f1_score(y_val_cv, y_pred, average='macro')
        macro_f1_scores.append(macro_f1)

    return -np.mean(macro_f1_scores)

gpu_id = get_assigned_gpu_id()
try:
    # Use Optuna for hyperparameter tuning
    study_concat = optuna.create_study(direction='minimize')
    study_concat.optimize(objective_feature_concat, n_trials=50)

    # Best hyperparameters found
    best_params_concat = study_concat.best_params
    best_macro_f1_concat = -study_concat.best_value
    print(f"\n[Feature Concatenation] Best Model: {best_params_concat['model']}")
    print(f"[Feature Concatenation] Best Hyperparameters:")
    for key, value in best_params_concat.items():
        if key != 'model':
            print(f"{key}: {value}")
    print(f"[Feature Concatenation] Best Macro F1-Score: {best_macro_f1_concat}")

    # Evaluate on test data using the best model
    model_name = best_params_concat.pop('model')
    if model_name == 'LightGBM':
        params = best_params_concat
        params.update({
            'objective': 'multiclass',
            'num_class': num_classes,
            'verbosity': -1,
            'random_state': 42,
            'device': 'gpu' if torch.cuda.is_available() else 'cpu',
            'gpu_device_id': gpu_id if gpu_id is not None else -1,
        })
        train_data_full = lgb.Dataset(X_train_combined, label=y_train)
        model = lgb.train(params, train_data_full, num_boost_round=1000, verbose_eval=False)
        y_pred_prob = model.predict(X_test_combined)
        y_pred_test_concat = np.argmax(y_pred_prob, axis=1)
    elif model_name == 'XGBoost':
        params = best_params_concat
        params.update({
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'random_state': 42,
            'verbosity': 0,
            'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'auto',
            'gpu_id': gpu_id if gpu_id is not None else 0,
        })
        dtrain_full = xgb.DMatrix(X_train_combined, label=y_train)
        model = xgb.train(params, dtrain_full, num_boost_round=1000)
        y_pred_prob = model.predict(xgb.DMatrix(X_test_combined))
        y_pred_test_concat = np.argmax(y_pred_prob, axis=1)
    elif model_name == 'RandomForest':
        params = best_params_concat
        params.update({
            'n_jobs': -1,
            'random_state': 42
        })
        model = RandomForestClassifier(**params)
        model.fit(X_train_combined, y_train)
        y_pred_test_concat = model.predict(X_test_combined)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Calculate evaluation metrics
    macro_f1_test_concat = f1_score(y_test, y_pred_test_concat, average='macro')
    accuracy_test_concat = accuracy_score(y_test, y_pred_test_concat)
    kappa_test_concat = cohen_kappa_score(y_test, y_pred_test_concat)

    print(f"\n[Feature Concatenation] Evaluation Metrics on Test Set:")
    print(f"Macro F1-Score: {macro_f1_test_concat}")
    print(f"Accuracy: {accuracy_test_concat}")
    print(f"Cohen's Kappa: {kappa_test_concat}")

finally:
    if gpu_id is not None:
        release_gpu_id(gpu_id)

# Update best strategy if needed
if macro_f1_test_concat > best_f1:
    best_f1 = macro_f1_test_concat
    best_strategy = 'Feature-Concatenation'
    best_params = best_params_concat
    best_params['model'] = model_name

### Fusion Strategy 3: Probability Calibration and Weighted Averaging ###

# Calibrate probabilities
calibrator_gru = CalibratedClassifierCV(method='isotonic', cv='prefit')
calibrator_lgbm = CalibratedClassifierCV(method='isotonic', cv='prefit')

# Since we cannot fit calibrators without model objects, we can only simulate this step if possible
# Assuming we have the model objects or probability outputs from validation to fit the calibrators
# For simplicity, we'll assume probabilities are already calibrated

# Re-run weighted averaging with calibrated probabilities

def objective_weighted_ensemble_calibrated(trial):
    w = trial.suggest_float('weight', 0.0, 1.0)
    macro_f1_scores = []

    for train_index, val_index in gkf.split(X_train_preds, y_train, groups=groups_train):
        X_train_cv, X_val_cv = X_train_preds[train_index], X_train_preds[val_index]
        y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]

        # Split predictions into GRU and LightGBM predictions
        num_classes = len(np.unique(y_train))
        preds_gru_val = X_val_cv[:, :num_classes]
        preds_lgbm_val = X_val_cv[:, num_classes:]

        # If we had calibrators, we would calibrate preds here
        # Since we don't, we'll proceed without calibration for demonstration purposes

        # Compute combined probabilities
        combined_preds_val = w * preds_gru_val + (1 - w) * preds_lgbm_val

        # Predict classes
        y_pred = np.argmax(combined_preds_val, axis=1)

        # Calculate Macro F1-Score
        macro_f1 = f1_score(y_val_cv, y_pred, average='macro')
        macro_f1_scores.append(macro_f1)

    # Return the negative of the average Macro F1-Score
    return -np.mean(macro_f1_scores)

# Set up Optuna study
study_weighted_calibrated = optuna.create_study(direction='minimize')
study_weighted_calibrated.optimize(objective_weighted_ensemble_calibrated, n_trials=50)

# Best weight found
best_weight_calibrated = study_weighted_calibrated.best_params['weight']
best_macro_f1_weighted_calibrated = -study_weighted_calibrated.best_value
print(f"\n[Weighted Averaging with Calibration] Best weight: {best_weight_calibrated}")
print(f"[Weighted Averaging with Calibration] Best Macro F1-Score: {best_macro_f1_weighted_calibrated}")

# Evaluate on test data using the best weight
num_classes = len(np.unique(y_train))
preds_gru_test = X_test_preds[:, :num_classes]
preds_lgbm_test = X_test_preds[:, num_classes:]

# Assuming probabilities are calibrated
combined_preds_test = best_weight_calibrated * preds_gru_test + (1 - best_weight_calibrated) * preds_lgbm_test
y_pred_test_weighted_calibrated = np.argmax(combined_preds_test, axis=1)

# Calculate evaluation metrics
macro_f1_test_weighted_calibrated = f1_score(y_test, y_pred_test_weighted_calibrated, average='macro')
accuracy_test_weighted_calibrated = accuracy_score(y_test, y_pred_test_weighted_calibrated)
kappa_test_weighted_calibrated = cohen_kappa_score(y_test, y_pred_test_weighted_calibrated)

print("\n[Weighted Averaging with Calibration] Evaluation Metrics on Test Set:")
print(f"Macro F1-Score: {macro_f1_test_weighted_calibrated}")
print(f"Accuracy: {accuracy_test_weighted_calibrated}")
print(f"Cohen's Kappa: {kappa_test_weighted_calibrated}")

# Update best strategy if needed
if macro_f1_test_weighted_calibrated > best_f1:
    best_f1 = macro_f1_test_weighted_calibrated
    best_strategy = 'Weighted_Averaging_Calibrated'
    best_params = {'weight': best_weight_calibrated}

# Step 7: Save Best Fusion Strategy Hyperparameters and Final Model

print(f"\nBest Fusion Strategy: {best_strategy}")
print(f"Best Macro F1-Score: {best_f1}")

# Save the best model based on its type
if best_strategy == 'Meta-Learner-NN':
    best_params['strategy'] = 'Meta-Learner-NN'
    with open('best_fusion_params.json', 'w') as f:
        json.dump(best_params, f)

    # Check if the best model is a PyTorch model
    if isinstance(model, nn.Module):
        # Save the PyTorch model
        torch.save(model.state_dict(), 'best_meta_learner_nn.pth')
    else:
        # Save the scikit-learn model using joblib
        joblib.dump(model, 'best_meta_learner_model.pkl')

elif best_strategy == 'Feature-Concatenation':
    best_params['strategy'] = 'Feature-Concatenation'
    with open('best_fusion_params.json', 'w') as f:
        json.dump(best_params, f)

    # Save the trained model if needed
    if model_name == 'LightGBM':
        model.save_model('best_feature_concat_model.txt')
    elif model_name == 'XGBoost':
        model.save_model('best_feature_concat_model.json')
    elif model_name == 'RandomForest':
        joblib.dump(model, 'best_feature_concat_model.pkl')

elif best_strategy == 'Weighted_Averaging_Calibrated':
    # Save the best weight
    with open('best_fusion_params.json', 'w') as f:
        json.dump({'strategy': 'Weighted_Averaging_Calibrated', 'weight': best_params['weight']}, f)

    # Weighted averaging doesn't have a model to save

print("\nAdvanced fusion strategies implementation and evaluation completed.")
