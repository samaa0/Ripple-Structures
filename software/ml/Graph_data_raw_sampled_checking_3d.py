import pandas as pd
import plotly.express as px
import os

label_check = 13
sequence_check = 2730  # Specify the exact sequence number you want to plot

# Load the data
try:
    data = pd.read_csv('augmented_train_data.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: The data file was not found.")
    exit(1)

# Ensure 'label' and 'sequence' columns are integer types
data['label'] = data['label'].astype(int)
data['sequence'] = data['sequence'].astype(int)

# Convert sequence_check to int
sequence_check = int(sequence_check)

# Get the list of unique labels
labels = sorted(data['label'].unique())
print(f"Unique labels in the data: {labels}\n")

# Create an output directory for the interactive plots
output_dir = 'interactive_3d_plots'
os.makedirs(output_dir, exist_ok=True)

# Check if the specified label exists
if label_check in labels:
    # Filter data for the specified label
    label_data = data[data['label'] == label_check]
    # Get the list of sequences for this label and convert to Python int
    sequences = sorted(label_data['sequence'].unique().tolist())
    print(f"Available sequences for label {label_check}: {sequences}\n")

    # Debugging prints
    print(f"Type of sequences elements: {type(sequences[0])}")
    print(f"Sequence check: {sequence_check}, Type: {type(sequence_check)}\n")

    # Define sequences_label_0 for label 0
    if label_check == 0:
        sequences_label_0 = sequences  # sequences already contains sequences for label 0
        print("Sequences available for label 0:")
        print(sequences_label_0)

    # Check if the specified sequence exists in this label
    if sequence_check in sequences:
        print(f"Generating interactive 3D plot for label {label_check}, sequence {sequence_check}")
        # Filter data for the specified sequence
        seq_data = label_data[label_data['sequence'] == sequence_check]

        # Create an interactive 3D scatter plot using Plotly
        fig = px.scatter_3d(
            seq_data,
            x='x',
            y='y',
            z='z',
            color='timestamp',  # Color by timestamp to show progression
            title=f'Label {label_check} - Sequence {sequence_check}',
            labels={'x': 'X-axis', 'y': 'Y-axis', 'z': 'Z-axis', 'timestamp': 'Timestamp'},
        )

        # Update marker size
        fig.update_traces(marker=dict(size=3))

        # Display the plot interactively
        fig.show()

        print(f"Interactive 3D plot displayed for label {label_check}, sequence {sequence_check}")
    else:
        print(f"Sequence {sequence_check} not found for label {label_check}.")
else:
    print(f"Label {label_check} not found in the data.")

# Get all labels that contain sequence 100
labels_with_sequence_100 = data[data['sequence'] == sequence_check]['label'].unique()

if len(labels_with_sequence_100) > 0:
    print(f"Sequence {sequence_check} exists under labels: {labels_with_sequence_100}")
else:
    print(f"Sequence {sequence_check} does not exist in the dataset.")

# To avoid NameError, ensure sequences_label_0 is defined even if label_check is not 0
# You can initialize sequences_label_0 as follows:
if label_check != 0:
    sequences_label_0 = data[data['label'] == 0]['sequence'].unique().tolist()
