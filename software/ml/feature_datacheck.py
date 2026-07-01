import pandas as pd
import numpy as np

# Step 1: Load your datasets
# Replace the file paths with the actual paths to your CSV files
train_features_normalized = pd.read_csv('train_features_normalized.csv')
validation_features_normalized = pd.read_csv('validation_features_normalized.csv')
test_features_normalized = pd.read_csv('test_features_normalized.csv')

# Combine training and validation data
train_val_data = pd.concat([train_features_normalized, validation_features_normalized], ignore_index=True)

# Step 2: Check data types of all columns in the training data
print("Data types in training data:")
print(train_val_data.dtypes)

# Step 3: Check if 'label' column is integer or object (string)
label_dtype = train_val_data['label'].dtype
print(f"\nLabel column data type in training data: {label_dtype}")

# Check data types in the test data
print("\nData types in test data:")
print(test_features_normalized.dtypes)

test_label_dtype = test_features_normalized['label'].dtype
print(f"\nLabel column data type in test data: {test_label_dtype}")

# Step 4: If labels are strings, convert them to integers
if train_val_data['label'].dtype == 'object':
    print("\nConverting training labels from string to integer.")
    train_val_data['label'] = train_val_data['label'].astype(int)

if test_features_normalized['label'].dtype == 'object':
    print("\nConverting test labels from string to integer.")
    test_features_normalized['label'] = test_features_normalized['label'].astype(int)

# Verify conversion
print(f"\nNew label column data type in training data: {train_val_data['label'].dtype}")
print(f"New label column data type in test data: {test_features_normalized['label'].dtype}")

# Step 5: Check unique labels and their counts in the training data
print("\nUnique labels in training data:")
print(sorted(train_val_data['label'].unique()))

print("\nClass distribution in training data:")
train_label_counts = train_val_data['label'].value_counts().sort_index()
print(train_label_counts)

# Step 6: Check unique labels and their counts in the test data
print("\nUnique labels in test data:")
print(sorted(test_features_normalized['label'].unique()))

print("\nClass distribution in test data:")
test_label_counts = test_features_normalized['label'].value_counts().sort_index()
print(test_label_counts)

# Step 7: Check for missing values in features and labels in the training data
print("\nChecking for missing values in training data:")
missing_train = train_val_data.isnull().sum()
print(missing_train[missing_train > 0])

# Step 8: Check for missing values in features and labels in the test data
print("\nChecking for missing values in test data:")
missing_test = test_features_normalized.isnull().sum()
print(missing_test[missing_test > 0])

# Step 9: Ensure feature consistency between training and test datasets
train_features = set(train_val_data.columns) - {'label', 'sequence'}
test_features = set(test_features_normalized.columns) - {'label', 'sequence'}

print("\nFeatures only in training data:")
print(sorted(train_features - test_features))

print("\nFeatures only in test data:")
print(sorted(test_features - train_features))

# Step 10: Verify that feature distributions are similar across datasets
common_features = train_features & test_features

print("\nComparing feature statistics between training and test datasets:")
for feature in sorted(common_features):
    train_stats = train_val_data[feature].describe()
    test_stats = test_features_normalized[feature].describe()
    print(f"\nFeature '{feature}':")
    print(f"  Training Data - min: {train_stats['min']:.4f}, max: {train_stats['max']:.4f}, mean: {train_stats['mean']:.4f}, std: {train_stats['std']:.4f}")
    print(f"  Test Data     - min: {test_stats['min']:.4f}, max: {test_stats['max']:.4f}, mean: {test_stats['mean']:.4f}, std: {test_stats['std']:.4f}")

# Step 11: Check for constant features in the training data
print("\nChecking for features with zero variance in training data:")
constant_features = []
for feature in common_features:
    if train_val_data[feature].nunique() == 1:
        constant_features.append(feature)
if constant_features:
    print("Constant features in training data:", constant_features)
else:
    print("No constant features found in training data.")

# Step 12: Check for features with extremely low variance
print("\nChecking for features with low variance in training data:")
low_variance_features = []
for feature in common_features:
    variance = train_val_data[feature].var()
    if variance < 1e-5:
        low_variance_features.append((feature, variance))
if low_variance_features:
    print("Features with low variance (feature, variance):")
    for feature, variance in low_variance_features:
        print(f"  {feature}: {variance:.6f}")
else:
    print("No features with low variance found in training data.")

# Step 13: Check correlation among features (optional)
# Calculating pairwise correlations can be computationally intensive for large datasets
print("\nCalculating pairwise correlations among features...")
correlation_matrix = train_val_data[sorted(common_features)].corr()

# Identify pairs of features with high correlation
threshold = 0.95  # Define a threshold for high correlation
print(f"\nPairs of features with correlation higher than {threshold}:")
correlated_pairs = []
corr_matrix_abs = correlation_matrix.abs()
for i in range(len(corr_matrix_abs.columns)):
    for j in range(i+1, len(corr_matrix_abs.columns)):
        feature_i = corr_matrix_abs.columns[i]
        feature_j = corr_matrix_abs.columns[j]
        correlation = corr_matrix_abs.iloc[i, j]
        if correlation > threshold:
            correlated_pairs.append((feature_i, feature_j, correlation))
if correlated_pairs:
    for feature_i, feature_j, correlation in correlated_pairs:
        print(f"  {feature_i} and {feature_j}: correlation = {correlation:.4f}")
else:
    print("No highly correlated feature pairs found.")
