# Test.py

import socket
import re
import time
from threading import Thread, Lock
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import optuna  # For loading the Optuna study
import os
import warnings
from ripple_paths import model_path, study_storage

warnings.filterwarnings("ignore")

# Set up UDP listener
UDP_IP_SEND = "127.0.0.1"
UDP_PORT_SEND = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP_SEND, UDP_PORT_SEND))

# Regular expression pattern to extract x, y, z
pattern = re.compile(r'^(\d+),([\d\.-]+),([\d\.-]+),([\d\.-]+),[\d\.-]+$')

print(f"Listening for UDP data on {UDP_IP_SEND}:{UDP_PORT_SEND}...\n")

# Buffer to store incoming data
data_buffer = deque()  # Stores tuples of (timestamp, x, y, z)
buffer_lock = Lock()

def udp_listener():
    """Function to listen for UDP data and store it in a buffer."""
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            line = data.decode('utf-8').strip()
            match = pattern.match(line)
            if not match:
                continue
            x = float(match.group(2))
            y = float(match.group(3))
            z = float(match.group(4))
            timestamp = time.time()

            # Append the data to the buffer
            with buffer_lock:
                data_buffer.append((timestamp, x, y, z))

                # Remove old data beyond 10 seconds
                while data_buffer and data_buffer[0][0] < timestamp - 10:
                    data_buffer.popleft()

        except KeyboardInterrupt:
            print("Interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

# ---------------------------------------------------------------------------------------
# Raw Data Model Setup
# ---------------------------------------------------------------------------------------

# Load the Optuna study to get the best hyperparameters
# Update the storage_name_raw to point to your actual Optuna database file
storage_name_raw = study_storage('raw_data_branch_gru_study.db')
study_name = 'gesture_recognition'

try:
    study_raw = optuna.load_study(study_name=study_name, storage=storage_name_raw)
    best_trial_raw = study_raw.best_trial
    best_params_raw = best_trial_raw.params

    # Extract hyperparameters
    input_size = 3  # Number of input features (x, y, z)
    hidden_size = int(best_params_raw['hidden_size'])
    num_layers = int(best_params_raw['n_layers'])
    dropout_rate = best_params_raw['dropout_rate']
    activation_name = best_params_raw.get('activation', 'relu')

    print("\nLoaded hyperparameters for GRU Model:")
    print(f"Input Size: {input_size}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Number of Layers: {num_layers}")
    print(f"Dropout Rate: {dropout_rate}")
    print(f"Activation Function: {activation_name}")

except Exception as e:
    print(f"Error loading Optuna study: {e}")
    print("Using default hyperparameters.")
    # Default hyperparameters (adjust as needed)
    input_size = 3
    hidden_size = 64
    num_layers = 2
    dropout_rate = 0.5
    activation_name = 'relu'

# Output size (number of gesture classes)
output_size = 14  # Ensure this matches your actual number of classes

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

# Initialize and load the trained model
model = GRUModel(input_size=input_size,
                 hidden_size=hidden_size,
                 num_layers=num_layers,
                 dropout=dropout_rate,
                 output_size=output_size)

# Path to the saved model weights
raw_model_path = model_path('final_raw_data_branch_model.pt')

if os.path.exists(raw_model_path):
    model.load_state_dict(torch.load(raw_model_path, map_location=torch.device('cpu')))
    print(f"Raw data model weights loaded from {raw_model_path}")
else:
    print(f"Model weights file {raw_model_path} not found. Please ensure the model file exists.")
    exit(1)

model.eval()  # Set the model to evaluation mode

# ---------------------------------------------------------------------------------------
# Gesture Labels (Ensure Order Matches Training Labels)
# ---------------------------------------------------------------------------------------

# gesture_labels = [
#     'Hand_waving_while_walking',
#     'Hand_waving_while_sitting',
#     'Hand_waving_while_standing',
#     'Hand_stationary_while_sitting',
#     'Hand_stationary_while_walking',
#     'Hand_stationary_while_standing',
#     'Hand_stationary_while_sleeping',
#     'Hand_rising_while_standing',
#     'Hand_rising_while_sitting',
#     'Standing_up',
#     'Sitting_down',
#     'Lying_down',
#     'Waking_up',
#     'Drop'
# ]

gesture_labels = [
    'walking',
    'Hand_waving',
    'Hand_waving',
    'Hand_stationary',
    'Hand_stationary',
    'Hand_stationary',
    'Hand_stationary',
    'Hand_rising',
    'Hand_rising',
    'Standing_up',
    'Sitting_down',
    'Lying_down',
    'Waking_up',
    'Drop'
]

# ---------------------------------------------------------------------------------------
# Data Processor Function
# ---------------------------------------------------------------------------------------

def data_processor():
    """Function to process the data buffer and perform prediction using the raw data model."""
    prediction_history = deque(maxlen=4)  # Stores the last 5 predictions for smoothing
    while True:
        time.sleep(0.3)  # Process data every 0.3 seconds
        current_time = time.time()

        # Get the latest 5 seconds of data
        with buffer_lock:
            recent_data = [item for item in data_buffer if item[0] >= current_time - 5.0]

        if len(recent_data) >= 2:
            ### Raw Data Branch Prediction ###
            # Prepare data for raw data model (resampling and normalization)
            timestamps = np.array([item[0] for item in recent_data])
            x_vals = np.array([item[1] for item in recent_data])
            y_vals = np.array([item[2] for item in recent_data])
            z_vals = np.array([item[3] for item in recent_data])

            # Shift timestamps to start from 0
            start_time = timestamps[0]
            times = timestamps - start_time  # Now times[0] == 0

            # Resample to uniform timestamps (e.g., every 0.01 seconds)
            fs = 100  # Sampling frequency (adjust as needed)
            duration = 5.0  # Duration in seconds
            new_time_index = np.linspace(0, duration, int(duration * fs))

            # Use interpolation to align the x, y, z values with the new uniform timestamps
            x_resampled = np.interp(new_time_index, times, x_vals)
            y_resampled = np.interp(new_time_index, times, y_vals)
            z_resampled = np.interp(new_time_index, times, z_vals)

            # Apply sequence-wise normalization
            mu_x = x_resampled.mean()
            sigma_x = x_resampled.std() if x_resampled.std() != 0 else 1
            mu_y = y_resampled.mean()
            sigma_y = y_resampled.std() if y_resampled.std() != 0 else 1
            mu_z = z_resampled.mean()
            sigma_z = z_resampled.std() if z_resampled.std() != 0 else 1

            # Normalize the data
            x_normalized = (x_resampled - mu_x) / sigma_x
            y_normalized = (y_resampled - mu_y) / sigma_y
            z_normalized = (z_resampled - mu_z) / sigma_z

            # Stack x_normalized, y_normalized, z_normalized into data shape (sequence_length, 3)
            sequence_data = np.stack((x_normalized, y_normalized, z_normalized), axis=1)  # Shape: (sequence_length, 3)

            # Convert to torch.Tensor
            input_tensor = torch.Tensor(sequence_data).unsqueeze(0)  # Add batch dimension, shape: (1, sequence_length, 3)

            # Feed into the model to get predictions
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)

            # Store the prediction for smoothing
            prediction_history.append(predicted_class.item())

            # Apply majority vote
            most_common_prediction = max(set(prediction_history), key=prediction_history.count)
            predicted_label_raw = gesture_labels[most_common_prediction]

            # Set a confidence threshold (adjust as needed)
            confidence_threshold = 0.97  # 60%

            # After obtaining probabilities
            max_prob, predicted_class = torch.max(probabilities, dim=1)
            max_prob = max_prob.item()
            predicted_class = predicted_class.item()

            if max_prob >= confidence_threshold:
                predicted_label_raw = gesture_labels[predicted_class]
                print(f"Predicted gesture (Raw Data Model): {predicted_label_raw} (Confidence: {max_prob:.2f})\n")
            else:
                print(f"Prediction confidence too low ({max_prob:.2f}). No gesture predicted.\n")

        else:
            print("Not enough data to process.")

# ---------------------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------------------

# Run the UDP listener and data processor in separate threads
listener_thread = Thread(target=udp_listener)
processor_thread = Thread(target=data_processor)

listener_thread.start()
processor_thread.start()

listener_thread.join()
processor_thread.join()