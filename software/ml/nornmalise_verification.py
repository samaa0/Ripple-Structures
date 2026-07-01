# verification.py

import pandas as pd
import numpy as np

# Define the paths to your normalized data files
train_data_path = 'Raw_final_train_data_normalized.csv'
# validation_data_path = 'validation_data_normalized.csv'
test_data_path = 'Raw_final_test_data_normalized.csv'

# Define the feature columns
feature_cols = ['x', 'y', 'z']

# Function to verify normalization
def verify_sequence_normalization(data_path, dataset_name):
    print(f"\nVerifying normalization for {dataset_name} dataset...\n")

    # Read the normalized data
    data_normalized = pd.read_csv(data_path)

    # Group by 'label' and 'sequence'
    grouped = data_normalized.groupby(['label', 'sequence'])[feature_cols]

    # Calculate mean and standard deviation for each sequence
    means = grouped.mean()
    stds = grouped.std()

    # Tolerances for checking
    mean_tolerance = 1e-6
    std_tolerance = 1e-6

    # Check if means are close to zero
    mean_deviation = means.abs().max(axis=1) > mean_tolerance
    sequences_with_mean_deviation = means[mean_deviation]

    # Check if standard deviations are close to one
    std_deviation = (stds - 1).abs().max(axis=1) > std_tolerance
    sequences_with_std_deviation = stds[std_deviation]

    # Report results
    if sequences_with_mean_deviation.empty and sequences_with_std_deviation.empty:
        print(f"All sequences in {dataset_name} are normalized correctly.")
    else:
        if not sequences_with_mean_deviation.empty:
            print(f"Sequences in {dataset_name} with mean deviation beyond tolerance ({mean_tolerance}):")
            print(sequences_with_mean_deviation)
            print("\n")

        if not sequences_with_std_deviation.empty:
            print(f"Sequences in {dataset_name} with standard deviation deviation beyond tolerance ({std_tolerance}):")
            print(sequences_with_std_deviation)
            print("\n")

# Verify normalization for each dataset
verify_sequence_normalization(train_data_path, 'Training')
# verify_sequence_normalization(validation_data_path, 'Validation')
verify_sequence_normalization(test_data_path, 'Test')

