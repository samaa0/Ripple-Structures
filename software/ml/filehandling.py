# filehandling.py

import pandas as pd
import numpy as np
import random
import torch
import joblib  # For saving the scaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ripple_paths import MODEL_DIR, model_path, study_storage

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

# Detect if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data
try:
    data = pd.read_csv('1409_resampled_raw_all.csv')
    print("Data loaded successfully.")
    print(f"Initial data shape: {data.shape}\n")
except FileNotFoundError:
    print("Error: The data file was not found.")
    exit(1)

# Check for required columns
required_columns = {'sequence', 'timestamp', 'x', 'y', 'z', 'label'}
missing_cols = required_columns - set(data.columns)
if missing_cols:
    print(f"Error: Missing required columns: {missing_cols}")
    exit(1)

# Ensure 'label' column is integer type
data['label'] = data['label'].astype(int)

# Unique labels
unique_labels = data['label'].unique()
print(f"Unique labels in the data: {unique_labels}")
print(f"Number of unique labels: {len(unique_labels)}\n")

# Sequence count per label BEFORE cleaning
sequence_counts = data.groupby('label')['sequence'].nunique()
print("Sequence count per label BEFORE cleaning:")
print(sequence_counts, "\n")

# Calculate sequence lengths by grouping on both 'label' and 'sequence'
sequence_lengths = data.groupby(['label', 'sequence'])['timestamp'].count().reset_index()
sequence_lengths.columns = ['label', 'sequence', 'seq_length']

# Merge sequence lengths back into data
data = data.merge(sequence_lengths, on=['label', 'sequence'], how='left')

# Expected sequence length
expected_sequence_length = 501

# Identify valid sequences (those with acceptable lengths)
acceptable_lengths = [500, 501]
valid_sequences = sequence_lengths[
    sequence_lengths['seq_length'].isin(acceptable_lengths)
][['label', 'sequence']]
print(f"Number of sequences with acceptable lengths {acceptable_lengths}: {len(valid_sequences)}\n")

# Filter data to include only valid sequences
data_filtered = data.merge(valid_sequences, on=['label', 'sequence'], how='inner', suffixes=('', '_y'))

# Drop redundant columns if any
columns_to_drop = ['seq_length_y']
for col in columns_to_drop:
    if col in data_filtered.columns:
        data_filtered.drop(columns=[col], inplace=True)

# Final sequence count per label after including acceptable lengths
final_sequence_counts = data_filtered.groupby('label')['sequence'].nunique()
print("Final sequence count per label after including acceptable lengths:")
print(final_sequence_counts, "\n")

# Final unique labels
final_labels = data_filtered['label'].unique()
print(f"Final unique labels in the data after including acceptable lengths: {final_labels}")
print(f"Number of labels after cleaning: {len(final_labels)}\n")

print(f"Using device: {device}")

# ---------------------------------------------------------------------------------------
# Data Splitting (Training (70%), Validation (15%), Test (15%))
# ---------------------------------------------------------------------------------------

print("Starting Data Splitting...\n")

# Get unique sequences from the cleaned data
unique_sequences = data_filtered[['label', 'sequence']].drop_duplicates()

# Dataset Partitioning: Training (70%), Validation (15%), Test (15%)
train_sequences, temp_sequences = train_test_split(
    unique_sequences,
    test_size=0.3,  # 30% will be split into validation and test sets
    stratify=unique_sequences['label'],
    random_state=random_seed
)

validation_sequences, test_sequences = train_test_split(
    temp_sequences,
    test_size=0.5,  # Split the 30% temp_sequences equally into validation and test sets
    stratify=temp_sequences['label'],
    random_state=random_seed
)

print(f"Number of training sequences: {len(train_sequences)}")
print(f"Number of validation sequences: {len(validation_sequences)}")
print(f"Number of test sequences: {len(test_sequences)}\n")

# Merge sequences back to get the actual data splits
train_data = data_filtered.merge(train_sequences, on=['label', 'sequence'], how='inner')
validation_data = data_filtered.merge(validation_sequences, on=['label', 'sequence'], how='inner')
test_data = data_filtered.merge(test_sequences, on=['label', 'sequence'], how='inner')

print(f"Training data shape: {train_data.shape}")
print(f"Validation data shape: {validation_data.shape}")
print(f"Test data shape: {test_data.shape}\n")

print("Sequence counts per label in Training set:")
train_sequence_counts = train_data.groupby('label')['sequence'].nunique()
print(train_sequence_counts)
print("\n")

print("Sequence counts per label in Validation set:")
validation_sequence_counts = validation_data.groupby('label')['sequence'].nunique()
print(validation_sequence_counts)
print("\n")

print("Sequence counts per label in Test set:")
test_sequence_counts = test_data.groupby('label')['sequence'].nunique()
print(test_sequence_counts)
print("\n")

# ---------------------------------------------------------------------------------------
# b. Outlier Detection and Removal (Label-wise) training set only
# ---------------------------------------------------------------------------------------

print("Starting Outlier Detection and Removal on Training Set (Label-wise)...\n")

# Define the features
feature_cols = ['x', 'y', 'z']

# Initialize an empty list to collect outlier sequences
outlier_sequences = []

# Loop over each label in the training set
labels = train_data['label'].unique()
for label in labels:
    print(f"Processing label: {label}")
    # Filter data for the current label
    label_data = train_data[train_data['label'] == label]

    # Compute mean and std for the current label in the training set
    label_mean = label_data[feature_cols].mean()
    label_std = label_data[feature_cols].std()

    # Calculate Z-scores for the label data
    label_data_zscored = label_data.copy()
    for col in feature_cols:
        label_data_zscored[col + '_zscore'] = (label_data_zscored[col] - label_mean[col]) / label_std[col]

    # Identify outlier sequences within the label
    outlier_mask = (label_data_zscored[[col + '_zscore' for col in feature_cols]].abs() > 3).any(axis=1)
    label_outlier_sequences = label_data_zscored.loc[outlier_mask, ['sequence']].drop_duplicates()

    # Add label information
    label_outlier_sequences['label'] = label

    # Append to the list of outlier sequences
    outlier_sequences.append(label_outlier_sequences)

# Combine outlier sequences from all labels
if outlier_sequences:
    outlier_sequences_df = pd.concat(outlier_sequences, ignore_index=True)
else:
    outlier_sequences_df = pd.DataFrame(columns=['sequence', 'label'])

print("Sequences identified as outliers in the training set:")
print(outlier_sequences_df)
print(f"\nNumber of sequences identified as outliers: {len(outlier_sequences_df)}\n")

# Proceed to remove the outlier sequences from train_data
if not outlier_sequences_df.empty:
    train_data_cleaned = train_data.merge(
        outlier_sequences_df,
        on=['label', 'sequence'],
        how='outer',
        indicator=True
    )
    train_data_cleaned = train_data_cleaned[train_data_cleaned['_merge'] == 'left_only']
    train_data_cleaned.drop(columns=['_merge'], inplace=True)
else:
    train_data_cleaned = train_data.copy()
    print("No outlier sequences were found in the training data.\n")

# Discard Z-score columns
zscore_cols = [col + '_zscore' for col in feature_cols]
train_data_cleaned = train_data_cleaned.drop(columns=[col for col in zscore_cols if col in train_data_cleaned.columns])

# Print data statistics AFTER outlier detection
print("Data statistics AFTER outlier detection (Training Set):")
data_mean_after = train_data_cleaned[feature_cols].mean()
data_std_after = train_data_cleaned[feature_cols].std()
print("Mean of x, y, z:")
print(data_mean_after)
print("\nStandard deviation of x, y, z:")
print(data_std_after)
print("\n")

# Final sequence count per label after outlier removal
final_sequence_counts_after = train_data_cleaned.groupby('label')['sequence'].nunique()
print("Final sequence count per label in Training Set AFTER outlier detection:")
print(final_sequence_counts_after, "\n")

# ---------------------------------------------------------------------------------------
# Calculate and Print Outlier Counts and Percentage Removed in Training Set
# ---------------------------------------------------------------------------------------

# Count the number of outlier sequences per label
outlier_counts = outlier_sequences_df.groupby('label')['sequence'].nunique()
print("Number of outlier sequences per label in Training Set:")
print(outlier_counts)

# Calculate the percentage of sequences removed per label
sequence_counts_before = train_data.groupby('label')['sequence'].nunique()
sequence_counts_after = train_data_cleaned.groupby('label')['sequence'].nunique()
sequences_removed = sequence_counts_before - sequence_counts_after
percentage_removed = (sequences_removed / sequence_counts_before) * 100

print("\nPercentage of sequences removed per label in Training Set:")
print(percentage_removed)
print("\n")

# Update the training data reference
train_data = train_data_cleaned

# ---------------------------------------------------------------------------------------
# 2. Feature Preparation
# Branch 1: Raw Data Branch
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------------------
# Analyze the distribution of sequences across gesture labels in the training set
train_sequence_distribution = train_data.groupby('label')['sequence'].nunique().reset_index()
train_sequence_distribution.columns = ['label', 'sequence_count']

print("Sequence distribution in the Training Set:")
print(train_sequence_distribution)
print("\n")

# Augmentation Functions
def add_noise(data, noise_level=0.001):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def scale_data(data, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return data * scale

def jitter_data(data, jitter_level=0.01):
    jitter = np.random.normal(0, jitter_level, data.shape)
    return data + jitter

def time_warp(data, warp_range=(0.9, 1.1)):
    # Simple time warp implementation
    original_length = data.shape[0]
    warp_factor = np.random.uniform(warp_range[0], warp_range[1])
    warped_length = int(original_length * warp_factor)
    warped_data = np.interp(
        np.linspace(0, original_length, warped_length),
        np.arange(original_length),
        data
    )
    # Resize back to original length if necessary
    if len(warped_data) != original_length:
        warped_data = np.interp(
            np.linspace(0, len(warped_data), original_length),
            np.arange(len(warped_data)),
            warped_data
        )
    return warped_data

# Set the target number of sequences per label
target_sequence_count = 400  # Desired number of sequences per label

# Create an empty DataFrame to store the balanced training data
train_data_balanced = pd.DataFrame()

# List to store augmented data
augmented_data_list = []

# Iterate over each label
for index, row in train_sequence_distribution.iterrows():
    label = row['label']
    sequence_count = row['sequence_count']
    
    # Get all data for the current label
    label_data = train_data[train_data['label'] == label]
    label_sequences = label_data['sequence'].unique()
    
    if sequence_count > target_sequence_count:
        # Downsample to target_sequence_count sequences
        print(f"Downsampling label {label}: reducing from {sequence_count} to {target_sequence_count} sequences.")
        selected_sequences = np.random.choice(label_sequences, size=target_sequence_count, replace=False)
        label_data_downsampled = label_data[label_data['sequence'].isin(selected_sequences)]
        downsampled_sequence_count = label_data_downsampled['sequence'].nunique()
        print(f"After downsampling, label {label} has {downsampled_sequence_count} sequences.\n")
        train_data_balanced = pd.concat([train_data_balanced, label_data_downsampled], ignore_index=True)
    elif sequence_count < target_sequence_count:
        # Keep existing sequences
        train_data_balanced = pd.concat([train_data_balanced, label_data], ignore_index=True)
        
        sequences_to_generate = target_sequence_count - sequence_count
        print(f"Augmenting label {label}: {sequences_to_generate} new sequences needed.")
        
        # For each sequence to generate
        for i in range(sequences_to_generate):
            # Randomly select a sequence to augment
            seq = np.random.choice(label_sequences)
            seq_data = label_data[label_data['sequence'] == seq].copy()
            
            # Apply a random augmentation technique
            augmentation_choice = np.random.choice(['noise', 'scale', 'jitter', 'time_warp'])
            if augmentation_choice == 'noise':
                seq_data[['x', 'y', 'z']] = add_noise(seq_data[['x', 'y', 'z']].values)
            elif augmentation_choice == 'scale':
                seq_data[['x', 'y', 'z']] = scale_data(seq_data[['x', 'y', 'z']].values)
            elif augmentation_choice == 'jitter':
                seq_data[['x', 'y', 'z']] = jitter_data(seq_data[['x', 'y', 'z']].values)
            elif augmentation_choice == 'time_warp':
                # Apply time warping to each axis separately
                for axis in ['x', 'y', 'z']:
                    warped_series = time_warp(seq_data[axis].values)
                    seq_data[axis] = warped_series

            # Assign a new sequence ID
            new_sequence_id = train_data['sequence'].max() + 1 + len(augmented_data_list)
            seq_data['sequence'] = new_sequence_id

            # Append augmented data to the list
            augmented_data_list.append(seq_data)
    else:
        # If sequence_count == target_sequence_count
        print(f"Label {label} already has {target_sequence_count} sequences. No action taken.\n")
        train_data_balanced = pd.concat([train_data_balanced, label_data], ignore_index=True)

# Concatenate augmented data if any
if augmented_data_list:
    augmented_data = pd.concat(augmented_data_list, ignore_index=True)
    # Combine augmented data with the balanced training data
    train_data_balanced = pd.concat([train_data_balanced, augmented_data], ignore_index=True)
else:
    print("No augmentation was necessary.")

# Reset index after balancing
train_data_balanced.reset_index(drop=True, inplace=True)

# Analyze the distribution after balancing
balanced_sequence_distribution = train_data_balanced.groupby('label')['sequence'].nunique().reset_index()
balanced_sequence_distribution.columns = ['label', 'sequence_count']

print("\nSequence distribution in the Balanced Training Set:")
print(balanced_sequence_distribution)
print("\n")

# Verify sequence counts per label in train_data_balanced before saving
balanced_sequence_counts = train_data_balanced.groupby('label')['sequence'].nunique()
print("Sequence counts per label in train_data_balanced before saving:")
print(balanced_sequence_counts)
print("\nTotal sequences in train_data_balanced:", train_data_balanced['sequence'].nunique())
print("Total records in train_data_balanced:", train_data_balanced.shape[0])
print("\n")

# For example, when saving the augmented training data:
# Specify the output file path
augmented_data_filepath = 'augmented_raw_data_branch_hyperparameter_optimization_train_data.csv'

# Save the balanced training data
train_data_balanced.to_csv(augmented_data_filepath, index=False)
print(f"Balanced training data has been saved to '{augmented_data_filepath}'.\n")

# ---------------------------------------------------------------------------------------
# Sequence-Wise Normalization
# ---------------------------------------------------------------------------------------

print("Starting Sequence-Wise Normalization...\n")

# Define the feature columns
feature_cols = ['x', 'y', 'z']

# Function to perform sequence-wise normalization
def sequence_wise_normalization(df, feature_cols):
    # Copy the dataframe to avoid modifying the original data
    df_normalized = df.copy()
    # Group by 'sequence' and 'label' to ensure correct grouping in case sequences are reused across labels
    df_normalized[feature_cols] = df_normalized.groupby(['label', 'sequence'])[feature_cols].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    return df_normalized

# Apply sequence-wise normalization to the training set
train_data_normalized = sequence_wise_normalization(train_data_balanced, feature_cols)
print("Sequence-wise normalization on training data completed.\n")

# Apply sequence-wise normalization to the validation set
validation_data_normalized = sequence_wise_normalization(validation_data, feature_cols)
print("Sequence-wise normalization on validation data completed.\n")

# Apply sequence-wise normalization to the test data
test_data_normalized = sequence_wise_normalization(test_data, feature_cols)
print("Sequence-wise normalization on test data completed.\n")

# Optionally, save the normalized datasets
train_data_normalized.to_csv('train_raw_data_branch_hyperparameter_optimization_normalized_data.csv', index=False)
validation_data_normalized.to_csv('validation_raw_data_branch_hyperparameter_optimization_normalized_data.csv', index=False)
test_data_normalized.to_csv('test_raw_data_branch_hyperparameter_optimization_normalized_data.csv', index=False)
print("Normalized datasets have been saved.\n")

# Verify sequence counts in the normalized training data
normalized_train_sequence_counts = train_data_normalized.groupby('label')['sequence'].nunique()
print("Sequence counts per label in train_data_normalized:")
print(normalized_train_sequence_counts)
print("\nTotal sequences in train_data_normalized:", train_data_normalized['sequence'].nunique())
print("Total records in train_data_normalized:", train_data_normalized.shape[0])

# ---------------------------------------------------------------------------------------
# Branch 2: Feature Data Branch
# ---------------------------------------------------------------------------------------

# Function to compute features for each sequence
def extract_features(df):
    feature_list = []

    # Group the data by 'label' and 'sequence'
    grouped = df.groupby(['label', 'sequence'])

    for (label, sequence), group in grouped:
        # Ensure the data is sorted by timestamp
        group = group.sort_values('timestamp').reset_index(drop=True)

        # Extract position data
        x = group['x'].values
        y = group['y'].values
        z = group['z'].values
        timestamps = group['timestamp'].values

        # Calculate time differences
        dt = np.diff(timestamps)
        dt = np.where(dt == 0, 0.01, dt)  # Replace zero deltas with 0.01s to avoid division by zero

        # Calculate velocities
        vx = np.diff(x) / dt
        vy = np.diff(y) / dt
        vz = np.diff(z) / dt
        v = np.sqrt(vx**2 + vy**2 + vz**2)

        # Calculate accelerations
        ax = np.diff(vx) / dt[:-1]
        ay = np.diff(vy) / dt[:-1]
        az = np.diff(vz) / dt[:-1]
        a = np.sqrt(ax**2 + ay**2 + az**2)

        # Calculate jerks
        jerk_x = np.diff(ax) / dt[:-2]
        jerk_y = np.diff(ay) / dt[:-2]
        jerk_z = np.diff(az) / dt[:-2]
        jerk = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)

        # Calculate displacements
        displacement_x = x[-1] - x[0]
        displacement_y = y[-1] - y[0]
        displacement_z = z[-1] - z[0]

        # Calculate energies
        energy_x = np.sum(vx**2)
        energy_y = np.sum(vy**2)
        energy_z = np.sum(vz**2)

        # Calculate RMS values
        rms_velocity = np.sqrt(np.mean(v**2)) if len(v) > 0 else 0
        rms_acceleration = np.sqrt(np.mean(a**2)) if len(a) > 0 else 0

        # Frequency domain features using FFT
        fft_x = np.fft.fft(x - np.mean(x))
        fft_y = np.fft.fft(y - np.mean(y))
        fft_z = np.fft.fft(z - np.mean(z))

        freqs = np.fft.fftfreq(len(x), d=0.01)

        # Get dominant frequencies (excluding zero frequency)
        idx_x = np.argsort(np.abs(fft_x))[::-1]
        idx_y = np.argsort(np.abs(fft_y))[::-1]
        idx_z = np.argsort(np.abs(fft_z))[::-1]

        dominant_freq_x = freqs[idx_x[1]] if len(idx_x) > 1 else 0  # Skip the zero frequency component
        dominant_freq_y = freqs[idx_y[1]] if len(idx_y) > 1 else 0
        dominant_freq_z = freqs[idx_z[1]] if len(idx_z) > 1 else 0

        # Compile features into a dictionary
        features = {
            'label': label,
            'sequence': sequence,
            'Mean_Velocity': np.mean(v) if len(v) > 0 else 0,
            'Peak_Velocity': np.max(v) if len(v) > 0 else 0,
            'RMS_Velocity': rms_velocity,
            'Mean_Acceleration': np.mean(a) if len(a) > 0 else 0,
            'Peak_Acceleration': np.max(a) if len(a) > 0 else 0,
            'RMS_Acceleration': rms_acceleration,
            'Displacement_X': displacement_x,
            'Displacement_Y': displacement_y,
            'Displacement_Z': displacement_z,
            'Jerk_X': np.mean(jerk_x) if len(jerk_x) > 0 else 0,
            'Jerk_Y': np.mean(jerk_y) if len(jerk_y) > 0 else 0,
            'Jerk_Z': np.mean(jerk_z) if len(jerk_z) > 0 else 0,
            'Energy_X': energy_x,
            'Energy_Y': energy_y,
            'Energy_Z': energy_z,
            'FFT_Dominant_Freq_X': abs(dominant_freq_x),
            'FFT_Dominant_Freq_Y': abs(dominant_freq_y),
            'FFT_Dominant_Freq_Z': abs(dominant_freq_z)
        }

        feature_list.append(features)

    # Create a DataFrame from the list of feature dictionaries
    feature_df = pd.DataFrame(feature_list)
    return feature_df

# Perform feature extraction on the training set
print("Extracting features from the Training Set...\n")
train_features = extract_features(train_data)
print("Feature extraction for Training Set completed.\n")

# Perform feature extraction on the validation set
print("Extracting features from the Validation Set...\n")
validation_features = extract_features(validation_data)
print("Feature extraction for Validation Set completed.\n")

# Perform feature extraction on the test set
print("Extracting features from the Test Set...\n")
test_features = extract_features(test_data)
print("Feature extraction for Test Set completed.\n")

# Save the extracted features to CSV files
train_features.to_csv('train_features.csv', index=False)
validation_features.to_csv('validation_features.csv', index=False)
test_features.to_csv('test_features.csv', index=False)

print("Feature extraction completed. Feature datasets have been saved.\n")

# ---------------------------------------------------------------------------------------
# Feature Normalization
# - Combine training and validation sets
# - Fit scaler on the combined dataset
# - Apply scaling to all sets
# - Save scaler and normalized data
# ---------------------------------------------------------------------------------------
# Combine training and validation features
combined_features = pd.concat([train_features, validation_features], ignore_index=True)
print("Combined training and validation features.\n")

# Separate features and labels
feature_columns = [col for col in combined_features.columns if col not in ['label', 'sequence']]

X_combined = combined_features[feature_columns]
y_combined = combined_features['label']

X_test = test_features[feature_columns]
y_test = test_features['label']

# Fit the scaler on the combined training and validation features
scaler = StandardScaler()
scaler.fit(X_combined)

# Save the scaler
MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(scaler, model_path('feature_scaler.pkl'))
print(f"Feature scaler has been saved as '{model_path('feature_scaler.pkl')}'.\n")

# Apply the scaler to the training, validation, and test features
X_train_scaled = scaler.transform(train_features[feature_columns])
X_validation_scaled = scaler.transform(validation_features[feature_columns])
X_test_scaled = scaler.transform(X_test)

# Create DataFrames with normalized features and labels
train_features_normalized = pd.DataFrame(X_train_scaled, columns=feature_columns)
train_features_normalized['label'] = train_features['label']
train_features_normalized['sequence'] = train_features['sequence']

validation_features_normalized = pd.DataFrame(X_validation_scaled, columns=feature_columns)
validation_features_normalized['label'] = validation_features['label']
validation_features_normalized['sequence'] = validation_features['sequence']

test_features_normalized = pd.DataFrame(X_test_scaled, columns=feature_columns)
test_features_normalized['label'] = test_features['label']
test_features_normalized['sequence'] = test_features['sequence']

# Save the normalized feature datasets to CSV files
train_features_normalized.to_csv('train_features_normalized.csv', index=False)
validation_features_normalized.to_csv('validation_features_normalized.csv', index=False)
test_features_normalized.to_csv('test_features_normalized.csv', index=False)

print("Normalized feature datasets have been saved.\n")

# ---------------------------------------------------------------------------------------
# Model Training and Hyperparameter Optimization
# Raw data branch
# ---------------------------------------------------------------------------------------

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, cohen_kappa_score
import optuna
import os
import threading
import json

# Load normalized datasets
train_data_normalized = pd.read_csv('train_raw_data_branch_hyperparameter_optimization_normalized_data.csv')
validation_data_normalized = pd.read_csv('validation_raw_data_branch_hyperparameter_optimization_normalized_data.csv')
test_data_normalized = pd.read_csv('test_raw_data_branch_hyperparameter_optimization_normalized_data.csv')

# Create directories if they don't exist
MODEL_DIR.mkdir(exist_ok=True)
if not os.path.exists('saved_raw_data_branch_models'):
    os.makedirs('saved_raw_data_branch_models')

# Define the Dataset
class GestureDataset(Dataset):
    def __init__(self, data):
        self.sequences = []
        self.labels = []
        grouped = data.groupby(['sequence', 'label'])
        for (seq, label), group in grouped:
            features = group[['x', 'y', 'z']].values
            self.sequences.append(torch.Tensor(features))
            self.labels.append(label)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Create Dataset objects
train_dataset = GestureDataset(train_data_normalized)
validation_dataset = GestureDataset(validation_data_normalized)

import threading

# Initialize a lock for GPU assignment
gpu_lock = threading.Lock()
available_gpus = list(range(torch.cuda.device_count()))  # Detect available GPUs

def get_assigned_gpu_id():
    with gpu_lock:
        if available_gpus:
            gpu_id = available_gpus.pop()
            print(f"Assigned GPU {gpu_id}")
            return gpu_id
        else:
            print("No GPU available currently. Will wait for one to become available.")
            return None

def release_gpu_id(gpu_id):
    with gpu_lock:
        available_gpus.append(gpu_id)
        print(f"Released GPU {gpu_id}")

# Best models tracking
best_models_lock = threading.Lock()
best_models = []  # List of tuples (val_loss, model_filename)
top_k = 3

# Logging setup
log_lock = threading.Lock()
log_filename = 'Raw_Data_Branch_hyperparameter_optimization_trial_logs.csv'
if not os.path.exists(log_filename):
    with open(log_filename, 'w') as f:
        f.write('trial_number,model_type,hyperparameters,val_loss,val_f1,val_kappa\n')

# Model definitions with regularization techniques applied
import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout_rate, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # LSTM layer with dropout applied between layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True,
                            dropout=dropout_rate if n_layers > 1 else 0.0, bidirectional=bidirectional)

        # Layer normalization after LSTM
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.layer_norm = nn.LayerNorm(lstm_output_size)

        # Dropout layer after LayerNorm
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(lstm_output_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        h_0 = torch.zeros(self.n_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.n_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))
        # Apply layer normalization
        out = self.layer_norm(out)
        # Apply dropout
        out = self.dropout(out)
        # Get the output from the last time step
        out = out[:, -1, :]  # (batch_size, lstm_output_size)
        out = self.fc(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout_rate, bidirectional=False):
        super(GRUModel, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # GRU layer with dropout applied between layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers=n_layers, batch_first=True,
                          dropout=dropout_rate if n_layers > 1 else 0.0, bidirectional=bidirectional)

        # Layer normalization after GRU
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        self.layer_norm = nn.LayerNorm(gru_output_size)

        # Dropout layer after LayerNorm
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(gru_output_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        h_0 = torch.zeros(self.n_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)

        out, _ = self.gru(x, h_0)
        # Apply layer normalization
        out = self.layer_norm(out)
        # Apply dropout
        out = self.dropout(out)
        # Get the output from the last time step
        out = out[:, -1, :]  # (batch_size, gru_output_size)
        out = self.fc(out)
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, n_heads, num_classes, num_layers, dim_feedforward, dropout_rate, activation):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)

        # Transformer encoder layer with specified dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation=activation
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization after Transformer encoder
        self.layer_norm = nn.LayerNorm(d_model)

        # Dropout layer after LayerNorm
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        x = self.input_proj(x)  # (batch_size, seq_length, d_model)
        x = x.transpose(0, 1)  # Transformer expects input of shape (seq_length, batch_size, d_model)

        out = self.transformer_encoder(x)
        # Apply layer normalization
        out = self.layer_norm(out)
        # Apply dropout
        out = self.dropout(out)

        # Global average pooling over sequence length
        out = out.mean(dim=0)  # (batch_size, d_model)
        out = self.fc(out)
        return out

# Define the objective function for Optuna
def objective(trial):
    # Get an available GPU
    gpu_id = get_assigned_gpu_id()

    try:
        if gpu_id is None or not torch.cuda.is_available():
            print("No GPU available currently. Using CPU.")
            device = torch.device('cpu')
        else:
            print(f"Using GPU: {gpu_id}")
            device = torch.device(f'cuda:{gpu_id}')

        # Hyperparameter suggestions
        model_type = trial.suggest_categorical('model_type', ['LSTM', 'BLSTM', 'GRU', 'Transformer'])

        # Common hyperparameters
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)  # Adjusted range for L2 regularization
        dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.5, step=0.1)   # Adjusted range for dropout
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])

        # Early stopping parameters
        patience = 10
        min_delta = 0.001

        # Learning rate scheduler parameters
        scheduler_patience = 5
        scheduler_factor = 0.1
        scheduler_min_lr = 1e-6

        # Model-specific hyperparameters
        if model_type in ['LSTM', 'BLSTM', 'GRU']:
            n_layers = trial.suggest_int('n_layers', 1, 3)
            hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
            if model_type == 'BLSTM':
                bidirectional = True
            elif model_type == 'GRU':
                bidirectional = trial.suggest_categorical('bidirectional', [True, False])
            else:
                bidirectional = False
        elif model_type == 'Transformer':
            num_layers = trial.suggest_int('num_layers', 1, 4)
            d_model = trial.suggest_categorical('d_model', [64, 128, 256])
            possible_n_heads = [n for n in [2, 4, 8, 16] if d_model % n == 0 and n <= d_model]
            n_heads = trial.suggest_categorical('n_heads', possible_n_heads)
            dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256, 512])
            activation = trial.suggest_categorical('activation', ['relu', 'gelu'])
            activation = activation.lower()
        else:
            pass  # Should not reach here

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        # Define the model
        num_classes = train_data_normalized['label'].nunique()
        input_size = 3  # x, y, z

        if model_type == 'LSTM':
            model = LSTMModel(input_size, hidden_size, num_classes, n_layers, dropout_rate, bidirectional=False).to(device)
        elif model_type == 'BLSTM':
            model = LSTMModel(input_size, hidden_size, num_classes, n_layers, dropout_rate, bidirectional=True).to(device)
        elif model_type == 'GRU':
            model = GRUModel(input_size, hidden_size, num_classes, n_layers, dropout_rate, bidirectional).to(device)
        elif model_type == 'Transformer':
            model = TransformerModel(input_size, d_model, n_heads, num_classes, num_layers,
                                     dim_feedforward, dropout_rate, activation).to(device)
        else:
            pass  # Should not reach here

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            pass

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr
        )

        # Training loop with early stopping
        num_epochs = 50  # Adjusted number of epochs
        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            train_losses = []
            for sequences, labels in train_loader:
                sequences = sequences.to(device)
                labels = labels.to(device).long()
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            model.eval()
            val_losses = []
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences = sequences.to(device)
                    labels = labels.to(device).long()
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    val_losses.append(loss.item())
                    preds = outputs.argmax(dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())

            val_loss = sum(val_losses) / len(val_losses)
            val_f1 = f1_score(val_targets, val_preds, average='macro')
            val_kappa = cohen_kappa_score(val_targets, val_preds)
            print(f"Trial {trial.number}, Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Kappa: {val_kappa:.4f}")

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Check for improvement
            if best_val_loss - val_loss > min_delta:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Save the best model
        model_filename = os.path.join('saved_raw_data_branch_models', f"best_raw_data_branch_model_trial_{trial.number}.pt")
        torch.save(best_model_state, model_filename)

        # Update best_models list
        with best_models_lock:
            best_models.append((best_val_loss, model_filename))
            # Sort by val_loss
            best_models.sort(key=lambda x: x[0])
            # Keep only top k models
            if len(best_models) > top_k:
                # Remove the worst models
                for val_loss_, model_file in best_models[top_k:]:
                    if os.path.exists(model_file):
                        os.remove(model_file)
                del best_models[top_k:]

        # Log the trial performance
        hyperparameters = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'dropout_rate': dropout_rate,
            'optimizer': optimizer_name,
        }

        if model_type in ['LSTM', 'BLSTM']:
            hyperparameters.update({
                'n_layers': n_layers,
                'hidden_size': hidden_size,
                'bidirectional': bidirectional
            })
        elif model_type == 'GRU':
            hyperparameters.update({
                'n_layers': n_layers,
                'hidden_size': hidden_size,
                'bidirectional': bidirectional
            })
        elif model_type == 'Transformer':
            hyperparameters.update({
                'num_layers': num_layers,
                'd_model': d_model,
                'n_heads': n_heads,
                'dim_feedforward': dim_feedforward,
                'activation': activation
            })

        with log_lock:
            with open(log_filename, 'a') as f:
                f.write(f"{trial.number},{model_type},{json.dumps(hyperparameters)},{best_val_loss},{val_f1},{val_kappa}\n")

        # Log the best_val_loss
        trial.set_user_attr('best_val_loss', best_val_loss)

        return best_val_loss
    finally:
        # Release GPU
        release_gpu_id(gpu_id)

# Set up Optuna study storage
storage_name = study_storage('raw_data_branch_gru_study.db')
study = optuna.create_study(direction='minimize', study_name='gesture_recognition', storage=storage_name, load_if_exists=True)

# Optimize
study.optimize(objective, n_trials=50)

# ---------------------------------------------------------------------------------------
# final model training and evaluation
# Raw data branch

# ---------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import random
import torch

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

# Load the data
try:
    data = pd.read_csv('1409_resampled_raw_all.csv')
    print("Data loaded successfully.")
    print(f"Initial data shape: {data.shape}\n")
except FileNotFoundError:
    print("Error: The data file was not found.")
    exit(1)

# Check for required columns
required_columns = {'sequence', 'timestamp', 'x', 'y', 'z', 'label'}
missing_cols = required_columns - set(data.columns)
if missing_cols:
    print(f"Error: Missing required columns: {missing_cols}")
    exit(1)

# Ensure 'label' column is integer type
data['label'] = data['label'].astype(int)

# Check for missing values
if data.isnull().values.any():
    print("Missing values found in the data. Proceeding to remove sequences with missing values.")
    sequences_with_null = data[data.isnull().any(axis=1)]['sequence'].unique()
    data = data[~data['sequence'].isin(sequences_with_null)]
    print(f"Removed sequences with missing values. Data shape after removal: {data.shape}")
else:
    print("No missing values found in the data.\n")

# Verify sequence lengths
# First, calculate sequence lengths
sequence_lengths = data.groupby(['label', 'sequence'])['timestamp'].count().reset_index()
sequence_lengths.columns = ['label', 'sequence', 'seq_length']

# Expected sequence length
expected_sequence_length = 501

# Identify valid sequences (those with the expected sequence length)
valid_sequences = sequence_lengths[sequence_lengths['seq_length'] == expected_sequence_length][['label', 'sequence']]
print(f"Number of sequences with expected length ({expected_sequence_length}): {len(valid_sequences)}\n")

# Filter data to include only valid sequences
data_filtered = data.merge(valid_sequences, on=['label', 'sequence'], how='inner')

# Print data shape after filtering
print(f"Data shape after filtering sequences with expected length: {data_filtered.shape}\n")

# Final sequence count per label
final_sequence_counts = data_filtered.groupby('label')['sequence'].nunique()
print("Sequence count per label after filtering:")
print(final_sequence_counts, "\n")

# Reset index (optional)
data_filtered.reset_index(drop=True, inplace=True)

# Step 2: Split the Data
# Split the dataset into 85% training set and 15% test set, keeping entire sequences intact

from sklearn.model_selection import train_test_split

print("Starting Data Splitting...\n")

# Get unique sequences from the cleaned data
unique_sequences = data_filtered[['label', 'sequence']].drop_duplicates()

# Split into training and test sets
train_sequences, test_sequences = train_test_split(
    unique_sequences,
    test_size=0.15,  # 15% for test set
    stratify=unique_sequences['label'],
    random_state=random_seed
)

print(f"Number of training sequences: {len(train_sequences)}")
print(f"Number of test sequences: {len(test_sequences)}\n")

# Merge sequences back to get the actual data splits
train_data = data_filtered.merge(train_sequences, on=['label', 'sequence'], how='inner')
test_data = data_filtered.merge(test_sequences, on=['label', 'sequence'], how='inner')

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}\n")

# Verify sequence counts per label in the splits
print("Sequence counts per label in Training set:")
train_sequence_counts = train_data.groupby('label')['sequence'].nunique()
print(train_sequence_counts)
print("\nSequence counts per label in Test set:")
test_sequence_counts = test_data.groupby('label')['sequence'].nunique()
print(test_sequence_counts)

# Save the split datasets
train_data.to_csv('Raw_final_train_data.csv', index=False)
test_data.to_csv('Raw_final_test_data.csv', index=False)
print("\nTraining and test datasets have been saved as 'Raw_final_train_data.csv' and 'Raw_final_test_data.csv'.\n")

# ---------------------------------------------------------------------------------------
# Raw Data Branch
# Step 2: Reapply Data Preprocessing to the Training Set
# - Perform Outlier Detection and Removal
# - Apply Data Augmentation
# - Ensure that augmentation is applied only to the training data (the combined 85%)
# - Do not augment the test set

# ---------------------------------------------------------------------------------------
# Step 2: Reapply Data Preprocessing to the Training Set
# ---------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import random
import torch

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

# ---------------------------------------------------------------------------------------
# a. Data Loading and Initial Cleaning
# ---------------------------------------------------------------------------------------

# Load the training and test data splits
try:
    train_data = pd.read_csv('Raw_final_train_data.csv')
    test_data = pd.read_csv('Raw_final_test_data.csv')
    print("Training and test data loaded successfully.")
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}\n")
except FileNotFoundError:
    print("Error: The training or test data file was not found.")
    exit(1)

# Ensure 'label' column is integer type
train_data['label'] = train_data['label'].astype(int)
test_data['label'] = test_data['label'].astype(int)

# Check for missing values in training data
if train_data.isnull().values.any():
    print("Missing values found in the training data. Proceeding to remove sequences with missing values.")
    sequences_with_null = train_data[train_data.isnull().any(axis=1)]['sequence'].unique()
    train_data = train_data[~train_data['sequence'].isin(sequences_with_null)]
    print(f"Removed sequences with missing values from training data. New shape: {train_data.shape}")
else:
    print("No missing values found in the training data.\n")

# Check for missing values in test data
if test_data.isnull().values.any():
    print("Missing values found in the test data. Proceeding to remove sequences with missing values.")
    sequences_with_null = test_data[test_data.isnull().any(axis=1)]['sequence'].unique()
    test_data = test_data[~test_data['sequence'].isin(sequences_with_null)]
    print(f"Removed sequences with missing values from test data. New shape: {test_data.shape}")
else:
    print("No missing values found in the test data.\n")

# Verify sequence lengths in training data
sequence_lengths_train = train_data.groupby(['label', 'sequence'])['timestamp'].count().reset_index()
sequence_lengths_train.columns = ['label', 'sequence', 'seq_length']

# Expected sequence length
expected_sequence_length = 501

# Identify valid sequences (those with the expected sequence length)
valid_sequences_train = sequence_lengths_train[
    sequence_lengths_train['seq_length'] == expected_sequence_length
][['label', 'sequence']]
print(f"Number of sequences with expected length ({expected_sequence_length}) in training data: {len(valid_sequences_train)}\n")

# Filter training data to include only valid sequences
train_data_filtered = train_data.merge(valid_sequences_train, on=['label', 'sequence'], how='inner')

# Print data shape after filtering
print(f"Training data shape after filtering sequences with expected length: {train_data_filtered.shape}\n")

# Final sequence count per label in training data
final_sequence_counts_train = train_data_filtered.groupby('label')['sequence'].nunique()
print("Sequence count per label in Training set after filtering:")
print(final_sequence_counts_train)
print("\n")

# Reset index (optional)
train_data_filtered.reset_index(drop=True, inplace=True)

# ---------------------------------------------------------------------------------------
# b. Outlier Detection and Removal (Label-wise) on Training Set
# ---------------------------------------------------------------------------------------

print("Starting Outlier Detection and Removal on Training Set (Label-wise)...\n")

# Define the features
feature_cols = ['x', 'y', 'z']

# Initialize an empty list to collect outlier sequences
outlier_sequences = []

# Loop over each label in the training set
labels = train_data_filtered['label'].unique()
for label in labels:
    print(f"Processing label: {label}")
    # Filter data for the current label
    label_data = train_data_filtered[train_data_filtered['label'] == label]

    # Compute mean and std for the current label in the training set
    label_mean = label_data[feature_cols].mean()
    label_std = label_data[feature_cols].std()

    # Calculate Z-scores for the label data
    label_data_zscored = label_data.copy()
    for col in feature_cols:
        label_data_zscored[col + '_zscore'] = (label_data_zscored[col] - label_mean[col]) / label_std[col]

    # Identify outlier sequences within the label
    outlier_mask = (label_data_zscored[[col + '_zscore' for col in feature_cols]].abs() > 3).any(axis=1)
    label_outlier_sequences = label_data_zscored.loc[outlier_mask, ['sequence']].drop_duplicates()

    # Add label information
    label_outlier_sequences['label'] = label

    # Append to the list of outlier sequences
    outlier_sequences.append(label_outlier_sequences)

# Combine outlier sequences from all labels
if outlier_sequences:
    outlier_sequences_df = pd.concat(outlier_sequences, ignore_index=True)
else:
    outlier_sequences_df = pd.DataFrame(columns=['sequence', 'label'])

print("Sequences identified as outliers in the training set:")
print(outlier_sequences_df)
print(f"\nNumber of sequences identified as outliers: {len(outlier_sequences_df)}\n")

# Proceed to remove the outlier sequences from train_data_filtered
if not outlier_sequences_df.empty:
    train_data_cleaned = train_data_filtered.merge(
        outlier_sequences_df,
        on=['label', 'sequence'],
        how='outer',
        indicator=True
    )
    train_data_cleaned = train_data_cleaned[train_data_cleaned['_merge'] == 'left_only']
    train_data_cleaned.drop(columns=['_merge'], inplace=True)
else:
    train_data_cleaned = train_data_filtered.copy()
    print("No outlier sequences were found in the training data.\n")

# Discard Z-score columns
zscore_cols = [col + '_zscore' for col in feature_cols]
train_data_cleaned.drop(columns=[col for col in zscore_cols if col in train_data_cleaned.columns], inplace=True)

# Print data statistics AFTER outlier detection
print("Data statistics AFTER outlier detection (Training Set):")
data_mean_after = train_data_cleaned[feature_cols].mean()
data_std_after = train_data_cleaned[feature_cols].std()
print("Mean of x, y, z:")
print(data_mean_after)
print("\nStandard deviation of x, y, z:")
print(data_std_after)
print("\n")

# Final sequence count per label after outlier removal
final_sequence_counts_after = train_data_cleaned.groupby('label')['sequence'].nunique()
print("Final sequence count per label in Training Set AFTER outlier detection:")
print(final_sequence_counts_after, "\n")

# ---------------------------------------------------------------------------------------
# c. Data Augmentation (Training Set Only) with Fixed Target Sequence Count
# ---------------------------------------------------------------------------------------

print("Starting Data Augmentation on Training Set...\n")

# Analyze the distribution of sequences across gesture labels in the training set
train_sequence_distribution = train_data_cleaned.groupby('label')['sequence'].nunique().reset_index()
train_sequence_distribution.columns = ['label', 'sequence_count']

print("Sequence distribution in the Training Set after outlier removal:")
print(train_sequence_distribution)
print("\n")

# Set the target number of sequences per label
# We'll set it to the median value of the sequence counts
target_sequence_count = 500
print(f"Target sequence count per label: {target_sequence_count}\n")

# Augmentation Functions
def add_noise(data, noise_level=0.01):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def scale_data(data, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return data * scale

def jitter_data(data, jitter_level=0.01):
    jitter = np.random.normal(0, jitter_level, data.shape)
    return data + jitter

def time_warp(data, warp_range=(0.9, 1.1)):
    # Simple time warp implementation
    original_length = data.shape[0]
    warp_factor = np.random.uniform(warp_range[0], warp_range[1])
    warped_length = int(original_length * warp_factor)
    if warped_length < 2:
        warped_length = 2  # Ensure at least two points
    warped_data = np.interp(
        np.linspace(0, original_length - 1, warped_length),
        np.arange(original_length),
        data
    )
    # Resize back to original length if necessary
    if len(warped_data) != original_length:
        warped_data = np.interp(
            np.linspace(0, len(warped_data) - 1, original_length),
            np.arange(len(warped_data)),
            warped_data
        )
    return warped_data

# Create an empty DataFrame to store the balanced training data
train_data_balanced = pd.DataFrame()

# List to store augmented data
augmented_data_list = []

# Initialize sequence ID counter for new sequences
max_sequence_id = train_data_cleaned['sequence'].max()

# Set random seed for reproducibility
np.random.seed(random_seed)

# Iterate over each label
for index, row in train_sequence_distribution.iterrows():
    label = row['label']
    sequence_count = row['sequence_count']

    # Get all data for the current label
    label_data = train_data_cleaned[train_data_cleaned['label'] == label]
    label_sequences = label_data['sequence'].unique()

    if sequence_count > target_sequence_count:
        # Downsample to target_sequence_count sequences
        print(f"Label {label} has more sequences than target ({target_sequence_count}). Downsampling...")
        # Randomly select sequences to keep
        selected_sequences = np.random.choice(label_sequences, size=target_sequence_count, replace=False)
        label_data_downsampled = label_data[label_data['sequence'].isin(selected_sequences)]
        train_data_balanced = pd.concat([train_data_balanced, label_data_downsampled], ignore_index=True)
    elif sequence_count < target_sequence_count:
        # Keep existing sequences
        train_data_balanced = pd.concat([train_data_balanced, label_data], ignore_index=True)

        sequences_to_generate = target_sequence_count - sequence_count
        print(f"Augmenting label {label}: {sequences_to_generate} new sequences needed.")

        # For each sequence to generate
        for i in range(sequences_to_generate):
            # Randomly select a sequence to augment
            seq = np.random.choice(label_sequences)
            seq_data = label_data[label_data['sequence'] == seq].copy()

            # Apply a random augmentation technique
            augmentation_choice = np.random.choice(['noise', 'scale', 'jitter', 'time_warp'])
            if augmentation_choice == 'noise':
                seq_data[['x', 'y', 'z']] = add_noise(seq_data[['x', 'y', 'z']].values)
            elif augmentation_choice == 'scale':
                seq_data[['x', 'y', 'z']] = scale_data(seq_data[['x', 'y', 'z']].values)
            elif augmentation_choice == 'jitter':
                seq_data[['x', 'y', 'z']] = jitter_data(seq_data[['x', 'y', 'z']].values)
            elif augmentation_choice == 'time_warp':
                # Apply time warping to each axis separately
                for axis in ['x', 'y', 'z']:
                    warped_series = time_warp(seq_data[axis].values)
                    seq_data[axis] = warped_series

            # Assign a new sequence ID
            max_sequence_id += 1
            seq_data['sequence'] = max_sequence_id

            # Append augmented data to the list
            augmented_data_list.append(seq_data)
    else:
        # If sequence_count == target_sequence_count
        print(f"Label {label} already has {target_sequence_count} sequences. No augmentation needed.\n")
        train_data_balanced = pd.concat([train_data_balanced, label_data], ignore_index=True)

# Concatenate augmented data if any
if augmented_data_list:
    augmented_data = pd.concat(augmented_data_list, ignore_index=True)
    # Combine augmented data with the balanced training data
    train_data_balanced = pd.concat([train_data_balanced, augmented_data], ignore_index=True)
    print("Augmented data has been added to the training set.\n")
else:
    print("No augmentation was necessary.\n")

# Reset index after balancing
train_data_balanced.reset_index(drop=True, inplace=True)

# Analyze the distribution after balancing
balanced_sequence_distribution = train_data_balanced.groupby('label')['sequence'].nunique().reset_index()
balanced_sequence_distribution.columns = ['label', 'sequence_count']

print("Sequence distribution in the Balanced Training Set after augmentation and downsampling:")
print(balanced_sequence_distribution)
print("\n")

# Verify sequence counts per label in train_data_balanced before saving
balanced_sequence_counts = train_data_balanced.groupby('label')['sequence'].nunique()
print("Sequence counts per label in train_data_balanced before saving:")
print(balanced_sequence_counts)
print("\nTotal sequences in train_data_balanced:", train_data_balanced['sequence'].nunique())
print("Total records in train_data_balanced:", train_data_balanced.shape[0])
print("\n")

# Optionally, save the balanced training data
balanced_data_filepath = 'Raw_final_train_data_balanced.csv'
train_data_balanced.to_csv(balanced_data_filepath, index=False)
print(f"Balanced training data has been saved to '{balanced_data_filepath}'.\n")

# ---------------------------------------------------------------------------------------
# d. Sequence-Wise Normalization (Training and Test Sets)
# ---------------------------------------------------------------------------------------

print("Starting Sequence-Wise Normalization...\n")

# Function to perform sequence-wise normalization
def sequence_wise_normalization(df, feature_cols):
    # Copy the dataframe to avoid modifying the original data
    df_normalized = df.copy()
    # Group by 'sequence' and 'label' to ensure correct grouping
    df_normalized[feature_cols] = df_normalized.groupby(['label', 'sequence'])[feature_cols].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    return df_normalized

# Apply sequence-wise normalization to the training set
train_data_normalized = sequence_wise_normalization(train_data_balanced, feature_cols)
print("Sequence-wise normalization on training data completed.\n")

# Apply sequence-wise normalization to the test set
test_data_normalized = sequence_wise_normalization(test_data, feature_cols)
print("Sequence-wise normalization on test data completed.\n")

# Optionally, save the normalized datasets
train_data_normalized.to_csv('Raw_final_train_data_normalized.csv', index=False)
test_data_normalized.to_csv('Raw_final_test_data_normalized.csv', index=False)
print("Normalized training and test datasets have been saved.\n")

# Verify sequence counts in the normalized training data
normalized_train_sequence_counts = train_data_normalized.groupby('label')['sequence'].nunique()
print("Sequence counts per label in train_data_normalized:")
print(normalized_train_sequence_counts)
print("\nTotal sequences in train_data_normalized:", train_data_normalized['sequence'].nunique())
print("Total records in train_data_normalized:", train_data_normalized.shape[0])

# Verify sequence counts in the normalized test data
normalized_test_sequence_counts = test_data_normalized.groupby('label')['sequence'].nunique()
print("\nSequence counts per label in test_data_normalized:")
print(normalized_test_sequence_counts)
print("\nTotal sequences in test_data_normalized:", test_data_normalized['sequence'].nunique())
print("Total records in test_data_normalized:", test_data_normalized.shape[0])

# ---------------------------------------------------------------------------------------
# Data Preprocessing Completed
# Next Steps:
# - Proceed to retrain the model using the best hyperparameters
# - Evaluate the model on the test set
# ---------------------------------------------------------------------------------------
import optuna

# Set the storage path and study name
storage_name = study_storage('raw_data_branch_gru_study.db')
study_name = 'gesture_recognition'

# Load the Optuna study
study = optuna.load_study(study_name=study_name, storage=storage_name)


# Retrieve the best trial
best_trial = study.best_trial

# Print the best trial's parameters
print('Number of finished trials: {}'.format(len(study.trials)))
print("Best trial:")
print("  Trial number: {}".format(best_trial.number))
print("  Value: {:.5f}".format(best_trial.value))
print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))

    import torch
import torch.nn as nn
import torch.optim as optim

# Assuming you have already defined your model classes: LSTMModel, GRUModel, TransformerModel
# Also, define the number of classes based on your dataset
num_classes = 14
  # Replace with the actual number of gesture classes

# Retrieve the best hyperparameters
best_params = best_trial.params

# Extract hyperparameters
model_type = best_params.get('model_type', 'LSTM')  # Default to 'LSTM' if not specified
input_size = 3  # Number of input features (x, y, z)
hidden_size = int(best_params['hidden_size'])
num_layers = int(best_params['n_layers'])
dropout_rate = best_params['dropout_rate']
learning_rate = best_params['learning_rate']
batch_size = int(best_params['batch_size'])
weight_decay = best_params['weight_decay']
optimizer_name = best_params['optimizer']
activation_name = best_params.get('activation', 'relu')

# Initialize the model based on the model type
if model_type == 'LSTM':
    # Define your LSTM model class accordingly
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.ReLU() if activation_name == 'relu' else nn.Tanh()

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]  # Get the output from the last time step
            out = self.dropout(out)
            out = self.fc(out)
            return out

    model = LSTMModel(input_size=input_size,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      dropout=dropout_rate,
                      output_size=num_classes)

elif model_type == 'GRU':
    # Define your GRU model class accordingly
    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
            super(GRUModel, self).__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.ReLU() if activation_name == 'relu' else nn.Tanh()

        def forward(self, x):
            out, _ = self.gru(x)
            out = out[:, -1, :]  # Get the output from the last time step
            out = self.dropout(out)
            out = self.fc(out)
            return out

    model = GRUModel(input_size=input_size,
                     hidden_size=hidden_size,
                     num_layers=num_layers,
                     dropout=dropout_rate,
                     output_size=num_classes)
elif model_type == 'Transformer':
    # Define your Transformer model class accordingly
    class TransformerModel(nn.Module):
        def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, dropout, output_size):
            super(TransformerModel, self).__init__()
            self.input_fc = nn.Linear(input_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                       nhead=nhead,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=dropout,
                                                       activation=activation_name)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, output_size)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = self.input_fc(x)
            x = x.permute(1, 0, 2)  # Transformer expects input of shape [seq_len, batch_size, d_model]
            out = self.transformer_encoder(x)
            out = out.mean(dim=0)  # Aggregate over the sequence length
            out = self.dropout(out)
            out = self.fc(out)
            return out

    d_model = int(best_params['d_model'])
    nhead = int(best_params['n_heads'])
    dim_feedforward = int(best_params['dim_feedforward'])

    model = TransformerModel(input_size=input_size,
                             d_model=d_model,
                             nhead=nhead,
                             num_layers=num_layers,
                             dim_feedforward=dim_feedforward,
                             dropout=dropout_rate,
                             output_size=num_classes)
else:
    raise ValueError(f"Unknown model type: {model_type}")

# Move the model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set up the optimizer
if optimizer_name == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer_name == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
else:
    raise ValueError(f"Unknown optimizer: {optimizer_name}")

# Optionally, set up the learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode='min',
                                                 factor=0.1,
                                                 patience=5,
                                                 min_lr=1e-6)

# ---------------------------------------------------------------------------------------
# raw data branch final model training
# ---------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import threading
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd

# Initialize a lock for GPU assignment
gpu_lock = threading.Lock()

# Automatically detect available GPUs
available_gpus = list(range(torch.cuda.device_count()))

def get_assigned_gpu_id():
    with gpu_lock:
        if available_gpus:
            gpu_id = available_gpus.pop()
            print(f"Assigned GPU {gpu_id}")
            return gpu_id
        else:
            print("No GPU available currently. Will wait for one to become available.")
            # Optionally, implement waiting logic here
            return None  # For simplicity, return None if no GPU is available

def release_gpu_id(gpu_id):
    with gpu_lock:
        available_gpus.append(gpu_id)
        print(f"Released GPU {gpu_id}")

# Load normalized datasets
train_data_normalized = pd.read_csv('Raw_final_train_data_normalized.csv')
test_data_normalized = pd.read_csv('Raw_final_test_data_normalized.csv')

# Define the Dataset class
class GestureDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.sequences = []
        self.labels = []
        grouped = data.groupby(['sequence', 'label'])
        for (seq, label), group in grouped:
            features = group[['x', 'y', 'z']].values
            self.sequences.append(torch.Tensor(features))
            self.labels.append(label)
            
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Create Dataset objects
train_dataset = GestureDataset(train_data_normalized)
test_dataset = GestureDataset(test_data_normalized)

# Create DataLoaders
batch_size = int(best_params['batch_size'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Get an available GPU
gpu_id = get_assigned_gpu_id()

try:
    if gpu_id is None or not torch.cuda.is_available():
        print("No GPU available currently. Using CPU.")
        device = torch.device('cpu')
    else:
        print(f"Using GPU: {gpu_id}")
        device = torch.device(f'cuda:{gpu_id}')

    # Move the model to device
    model.to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define early stopping parameters
    patience = 10
    min_delta = 0.001

    # Learning rate scheduler parameters
    scheduler_patience = 5
    scheduler_factor = 0.1
    scheduler_min_lr = 1e-6

    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    training_losses = []
    validation_losses = []
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_losses = []
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        avg_train_loss = np.mean(epoch_train_losses)
        training_losses.append(avg_train_loss)

        # Validation phase (we can use a portion of the training data as validation if needed)
        # For this example, we'll skip validation during training and focus on test evaluation after training

        # Early stopping logic can be adjusted accordingly
        # For demonstration, we'll monitor the training loss
        # You may split a validation set from the training data if desired

        # Check for improvement
        if best_val_loss - avg_train_loss > min_delta:
            best_val_loss = avg_train_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Learning rate scheduling
        scheduler.step(avg_train_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        # Early stopping condition
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # Save the best model
    MODEL_DIR.mkdir(exist_ok=True)
    model_filename = model_path("final_raw_data_branch_model.pt")
    torch.save(best_model_state, model_filename)
    print(f"Best model saved to {model_filename}")

finally:
    # Ensure the GPU is released even if an error occurs
    if gpu_id is not None:
        release_gpu_id(gpu_id)

    # Load the best model
model.load_state_dict(torch.load(model_filename))
model.to(device)
model.eval()  # Set the model to evaluation mode

all_preds = []
all_targets = []

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        labels = labels.to(device).long()
        outputs = model(sequences)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

# Calculate metrics
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, confusion_matrix

macro_f1 = f1_score(all_targets, all_preds, average='macro')
kappa = cohen_kappa_score(all_targets, all_preds)
accuracy = accuracy_score(all_targets, all_preds)
conf_mat = confusion_matrix(all_targets, all_preds)

print(f"Test Set Performance:")
print(f"Macro F1-Score: {macro_f1:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_mat)
