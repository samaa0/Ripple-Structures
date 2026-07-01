import socket
import re
import time
from threading import Thread, Lock
from collections import deque
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import torch
import torch.nn as nn
import optuna
import os
import warnings
from ripple_paths import model_path, study_storage

warnings.filterwarnings("ignore")

# Check if MPS is available (for Macs with Apple Silicon)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS backend.")
else:
    device = torch.device("cpu")
    print("Using CPU backend.")

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

# Shared variables for predictions and features
prediction_gru = None
prediction_lgbm = None
features_gru = None
features_lgbm = None
prediction_lock = Lock()

# ------------------- UDP Listener -------------------

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

# ------------------- LightGBM Feature Extraction -------------------

def feature_extraction():
    """Function to extract features from the last 5 seconds of data and make predictions using LightGBM."""
    prediction_interval = 0.3

    # Load the scaler used during training
    try:
        scaler = joblib.load(model_path('feature_scaler.pkl'))
    except FileNotFoundError:
        print("Scaler file not found. Make sure 'feature_scaler.pkl' exists.")
        return

    # Load LightGBM model
    try:
        lgb_model = lgb.Booster(model_file=str(model_path('best_lgbm_model_rank_1.txt')))
    except FileNotFoundError:
        print("LightGBM model file not found. Make sure 'best_lgbm_model_rank_1.txt' exists.")
        return

    # Load the label mapping if necessary
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

    # Initialize prediction history deque for smoothing
    prediction_history = deque(maxlen=4)  # Adjust the length as needed

    # Set a confidence threshold
    confidence_threshold = 0.1  # Adjust the threshold as needed

    while True:
        time.sleep(prediction_interval)
        with buffer_lock:
            data_list = list(data_buffer)
            if len(data_list) < 2:
                # Not enough data to process
                continue

            timestamps, xs, ys, zs = zip(*data_list)
            df = pd.DataFrame({'Timestamp': timestamps, 'X': xs, 'Y': ys, 'Z': zs})

        features = compute_features(df)
        if features is None:
            continue

        features_scaled = scaler.transform(features)

        # Predict with LightGBM model
        prediction = lgb_model.predict(features_scaled, num_iteration=lgb_model.best_iteration)

        if prediction.ndim > 1 and prediction.shape[1] > 1:
            # Multi-class classification
            predicted_label = np.argmax(prediction, axis=1)[0]
            probability = prediction[0][predicted_label]
        else:
            # Binary classification or regression
            predicted_label = int(round(prediction[0]))
            probability = prediction[0]

        # Store the prediction for smoothing
        prediction_history.append(predicted_label)

        # Apply majority vote for smoothing
        most_common_prediction = max(set(prediction_history), key=prediction_history.count)
        predicted_gesture = label_mapping.get(most_common_prediction, 'Unknown')

        with prediction_lock:
            global prediction_lgbm, features_lgbm
            # Store the features and prediction
            prediction_lgbm = most_common_prediction
            # Save the features scaled (used for prediction)
            features_lgbm = features_scaled

def compute_features(df):
    """Compute the required features from the raw data."""
    try:
        time_diffs = np.diff(df['Timestamp'])
        if len(time_diffs) == 0:
            # Not enough data points to compute features
            return None
        sampling_rate = 1 / np.mean(time_diffs)

        # Calculate velocities
        df['Velocity_X'] = df['X'].diff() * sampling_rate
        df['Velocity_Y'] = df['Y'].diff() * sampling_rate
        df['Velocity_Z'] = df['Z'].diff() * sampling_rate

        # Calculate accelerations
        df['Acceleration_X'] = df['Velocity_X'].diff() * sampling_rate
        df['Acceleration_Y'] = df['Velocity_Y'].diff() * sampling_rate
        df['Acceleration_Z'] = df['Velocity_Z'].diff() * sampling_rate

        # Calculate jerks
        df['Jerk_X'] = df['Acceleration_X'].diff() * sampling_rate
        df['Jerk_Y'] = df['Acceleration_Y'].diff() * sampling_rate
        df['Jerk_Z'] = df['Acceleration_Z'].diff() * sampling_rate

        # Compute velocity magnitude
        v = np.sqrt(df['Velocity_X']**2 + df['Velocity_Y']**2 + df['Velocity_Z']**2).dropna()

        # Compute acceleration magnitude
        a = np.sqrt(df['Acceleration_X']**2 + df['Acceleration_Y']**2 + df['Acceleration_Z']**2).dropna()

        # Compute jerk magnitude
        jerk = np.sqrt(df['Jerk_X']**2 + df['Jerk_Y']**2 + df['Jerk_Z']**2).dropna()

        # Displacements
        displacement_x = df['X'].iloc[-1] - df['X'].iloc[0]
        displacement_y = df['Y'].iloc[-1] - df['Y'].iloc[0]
        displacement_z = df['Z'].iloc[-1] - df['Z'].iloc[0]

        # Energies
        energy_x = np.sum(df['Velocity_X'].dropna() ** 2)
        energy_y = np.sum(df['Velocity_Y'].dropna() ** 2)
        energy_z = np.sum(df['Velocity_Z'].dropna() ** 2)

        # RMS values
        rms_velocity = np.sqrt(np.mean(v**2)) if len(v) > 0 else 0
        rms_acceleration = np.sqrt(np.mean(a**2)) if len(a) > 0 else 0

        # Mean and peak values
        mean_velocity = v.mean() if len(v) > 0 else 0
        peak_velocity = v.max() if len(v) > 0 else 0
        mean_acceleration = a.mean() if len(a) > 0 else 0
        peak_acceleration = a.max() if len(a) > 0 else 0

        # Mean jerk components
        mean_jerk_x = df['Jerk_X'].mean() if not df['Jerk_X'].isna().all() else 0
        mean_jerk_y = df['Jerk_Y'].mean() if not df['Jerk_Y'].isna().all() else 0
        mean_jerk_z = df['Jerk_Z'].mean() if not df['Jerk_Z'].isna().all() else 0

        # FFT dominant frequencies
        fft_freq_x = get_dominant_frequency(df['X'].dropna(), sampling_rate)
        fft_freq_y = get_dominant_frequency(df['Y'].dropna(), sampling_rate)
        fft_freq_z = get_dominant_frequency(df['Z'].dropna(), sampling_rate)

        feature_dict = {
            'Mean_Velocity': [mean_velocity],
            'Peak_Velocity': [peak_velocity],
            'RMS_Velocity': [rms_velocity],
            'Mean_Acceleration': [mean_acceleration],
            'Peak_Acceleration': [peak_acceleration],
            'RMS_Acceleration': [rms_acceleration],
            'Displacement_X': [displacement_x],
            'Displacement_Y': [displacement_y],
            'Displacement_Z': [displacement_z],
            'Jerk_X': [mean_jerk_x],
            'Jerk_Y': [mean_jerk_y],
            'Jerk_Z': [mean_jerk_z],
            'Energy_X': [energy_x],
            'Energy_Y': [energy_y],
            'Energy_Z': [energy_z],
            'FFT_Dominant_Freq_X': [abs(fft_freq_x)],
            'FFT_Dominant_Freq_Y': [abs(fft_freq_y)],
            'FFT_Dominant_Freq_Z': [abs(fft_freq_z)],
        }

        # Ensure the feature columns are in the same order as during training
        feature_names = [
            'Mean_Velocity',
            'Peak_Velocity',
            'RMS_Velocity',
            'Mean_Acceleration',
            'Peak_Acceleration',
            'RMS_Acceleration',
            'Displacement_X',
            'Displacement_Y',
            'Displacement_Z',
            'Jerk_X',
            'Jerk_Y',
            'Jerk_Z',
            'Energy_X',
            'Energy_Y',
            'Energy_Z',
            'FFT_Dominant_Freq_X',
            'FFT_Dominant_Freq_Y',
            'FFT_Dominant_Freq_Z'
        ]

        feature_df = pd.DataFrame(feature_dict)
        feature_df = feature_df[feature_names]

        return feature_df
    except Exception as e:
        # Error computing features
        return None

def get_dominant_frequency(signal, sampling_rate):
    """Compute the dominant frequency of a signal using FFT."""
    try:
        n = len(signal)
        freqs = np.fft.fftfreq(n, d=1/sampling_rate)
        fft_values = np.fft.fft(signal)
        fft_values = np.abs(fft_values[:n//2])
        freqs = freqs[:n//2]
        dominant_freq = freqs[np.argmax(fft_values)]
        return dominant_freq
    except Exception as e:
        # Error computing dominant frequency
        return 0

# ------------------- GRU Model Setup -------------------

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
    model.load_state_dict(torch.load(raw_model_path, map_location=device))
    model.to(device)
    print(f"Raw data model weights loaded from {raw_model_path}")
else:
    print(f"Model weights file {raw_model_path} not found. Please ensure the model file exists.")
    exit(1)

model.eval()  # Set the model to evaluation mode

# Gesture Labels (Ensure Order Matches Training Labels)
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

# ------------------- GRU Data Processor Function -------------------

def data_processor_gru():
    """Function to process the data buffer and perform prediction using the raw data model."""
    prediction_history = deque(maxlen=4)  # Stores the last 4 predictions for smoothing
    while True:
        time.sleep(0.3)  # Process data every 0.3 seconds
        current_time = time.time()

        # Get the latest 5 seconds of data
        with buffer_lock:
            recent_data = [item for item in data_buffer if item[0] >= current_time - 5.0]

        if len(recent_data) >= 2:
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
            duration = times[-1]  # Duration in seconds
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
            input_tensor = torch.Tensor(sequence_data).unsqueeze(0).to(device)  # Add batch dimension, shape: (1, sequence_length, 3)

            # Feed into the model to get predictions
            with torch.no_grad():
                output, features = model(input_tensor, return_features=True)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)

            # Store the prediction for smoothing
            prediction_history.append(predicted_class.item())

            # Apply majority vote
            most_common_prediction = max(set(prediction_history), key=prediction_history.count)
            predicted_label_raw = gesture_labels[most_common_prediction]

            # Set a confidence threshold (adjust as needed)
            confidence_threshold = 0.1  # 60%

            # After obtaining probabilities
            max_prob, predicted_class = torch.max(probabilities, dim=1)
            max_prob = max_prob.item()
            predicted_class = predicted_class.item()

            with prediction_lock:
                global prediction_gru, features_gru
                # Store the features and prediction
                prediction_gru = most_common_prediction
                # Save the features (deep features from the model)
                features_gru = features.cpu().numpy()

        else:
            # Not enough data to process
            with prediction_lock:
                prediction_gru = None
                features_gru = None

# ------------------- Processing and Printing Function -------------------

def process_and_print_predictions():
    """Function to merge features from both models and print predictions and processed data."""
    sequence_counter = 0  # Sequence ID counter
    while True:
        time.sleep(0.3)
        with prediction_lock:
            pred_gru = prediction_gru
            pred_lgbm = prediction_lgbm
            feats_gru = features_gru
            feats_lgbm = features_lgbm

        # Check if both features are available
        if feats_gru is not None and feats_lgbm is not None:
            # Make sure feats_gru and feats_lgbm are numpy arrays of shape (1, num_features)
            if len(feats_gru.shape) == 2:
                feats_gru = feats_gru  # shape (1, num_features)
            else:
                feats_gru = feats_gru.reshape(1, -1)  # Ensure shape is (1, num_features)

            if len(feats_lgbm.shape) == 2:
                feats_lgbm = feats_lgbm  # shape (1, num_features)
            else:
                feats_lgbm = feats_lgbm.reshape(1, -1)  # Ensure shape is (1, num_features)

            # Get number of features for headers
            num_feats_gru = feats_gru.shape[1]
            num_feats_lgbm = feats_lgbm.shape[1]

            # Concatenate features
            merged_features = np.concatenate([feats_gru, feats_lgbm], axis=1)  # Shape: (1, total_features)

            # Create headers
            headers = ['{}_raw'.format(i) for i in range(num_feats_gru)] + \
                      ['sequence', 'label'] + \
                      ['{}_feature'.format(i) for i in range(num_feats_lgbm)]

            # Create DataFrame
            df_merged = pd.DataFrame(merged_features, columns=headers[:-2])

            # Add sequence and label (assuming we don't have ground-truth label, set to predicted label)
            df_merged['sequence'] = sequence_counter
            df_merged['label'] = 'N/A'  # Or None or predicted label if available

            # Print out the processed data along with the predictions
            print("\nProcessed Data:")
            print(df_merged)
            print(f"Predictions -> GRU: {pred_gru}, LightGBM: {pred_lgbm}")

            sequence_counter += 1
        else:
            # For debugging, if None, set to 'None'
            pred_gru_str = str(pred_gru) if pred_gru is not None else 'None'
            pred_lgbm_str = str(pred_lgbm) if pred_lgbm is not None else 'None'
            print(f"GRU: {pred_gru_str} | LightGBM: {pred_lgbm_str}")

# ------------------- Main Execution -------------------

if __name__ == '__main__':
    # Start the UDP listener in a separate thread
    listener_thread = Thread(target=udp_listener, daemon=True)
    listener_thread.start()

    # Start the feature extraction and prediction using LightGBM
    lgbm_thread = Thread(target=feature_extraction, daemon=True)
    lgbm_thread.start()

    # Start the GRU data processor
    gru_thread = Thread(target=data_processor_gru, daemon=True)
    gru_thread.start()

    # Start the processing and printing function
    process_thread = Thread(target=process_and_print_predictions, daemon=True)
    process_thread.start()

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
