import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

# Ensure plots are displayed inline (useful if running in a notebook)
# %matplotlib inline

# Load the data
try:
    data = pd.read_csv('1409_resampled_raw_all.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: The data file was not found.")
    exit(1)

# Ensure 'label' and 'sequence' columns are integer types
data['label'] = data['label'].astype(int)
data['sequence'] = data['sequence'].astype(int)

# Get the list of unique labels
labels = sorted(data['label'].unique())
print(f"Unique labels in the data: {labels}\n")

# Create an output directory for the plots
output_dir = 'sequence_3d_plots'
os.makedirs(output_dir, exist_ok=True)

# Generate 3D plots for the first 10 sequences of each label
for label in labels:
    # Filter data for the current label
    label_data = data[data['label'] == label]
    
    # Get the first 10 sequence numbers for this label
    sequences = sorted(label_data['sequence'].unique())[:10]
    print(f"Generating 3D plots for label {label}, sequences: {sequences}")
    
    for seq in sequences:
        # Filter data for the current sequence
        seq_data = label_data[label_data['sequence'] == seq]
        
        # Create a new 3D figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot x, y, z in 3D space
        ax.plot(seq_data['x'], seq_data['y'], seq_data['z'], marker='o', markersize=2, label=f'Sequence {seq}')
        
        # Set labels
        ax.set_title(f'Label {label} - Sequence {seq}')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        
        # Optional: Set equal aspect ratio for all axes
        ax.set_box_aspect([1,1,1])
        
        # Add a legend
        ax.legend()
        
        # Save the plot to a file
        plot_filename = f'label_{label}_sequence_{seq}_3d.png'
        plt.savefig(os.path.join(output_dir, plot_filename), dpi=300)
        plt.close()
        
        print(f"3D plot saved: {plot_filename}")
