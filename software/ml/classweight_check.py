import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Define the class counts from your training data
class_counts = {
    0: 425,
    1: 407,
    2: 390,
    3: 399,
    4: 382,
    5: 423,
    6: 405,
    7: 424,
    8: 424,
    9: 600,
    10: 405,
    11: 386,
    12: 389,
    13: 397
}

# Create arrays of labels based on counts
labels = []
for cls, count in class_counts.items():
    labels.extend([cls] * count)
labels = np.array(labels)

# Define the mapping from labels to class names
label_mapping = {
    0: 'Hand Waving While Walking',
    1: 'Hand Waving While Sitting',
    2: 'Hand Waving While Standing',
    3: 'Hand Stationary While Sitting',
    4: 'Hand Stationary While Walking',
    5: 'Hand Stationary While Standing',
    6: 'Hand Stationary While Sleeping',
    7: 'Hand Rising While Standing',
    8: 'Hand Rising While Sitting',
    9: 'Standing Up',
    10: 'Sitting Down',
    11: 'Lying Down',
    12: 'Waking Up',
    13: 'Drop'
}

# Compute class weights using sklearn's compute_class_weight
classes = np.unique(labels)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}

# Print out the class weights with class names
print("Class Weights:")
for cls in classes:
    weight = class_weights_dict[cls]
    class_name = label_mapping[cls]
    print(f"Class {cls} ({class_name}): Weight {weight:.4f}")