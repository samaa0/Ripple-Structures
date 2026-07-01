import socket
import re
import time
from threading import Thread, Lock
from collections import deque
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler  # or use the scaler you used during training
import json
import joblib
from ripple_paths import model_path

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

                # Remove old data beyond 5 seconds
                while data_buffer and data_buffer[0][0] < timestamp - 5:
                    data_buffer.popleft()

        except KeyboardInterrupt:
            print("Interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

def feature_extraction():
    """Function to extract features from the last 5 seconds of data."""
    prediction_interval = 0.3

    # Load the scaler used during training
    try:
        scaler = joblib.load(model_path('feature_scaler.pkl'))
    except FileNotFoundError:
        print("Scaler file not found. Make sure 'feature_scaler.pkl' exists.")
        return

    # Load the hyperparameters and model
    try:
        with open(model_path('model_rank_1_performance_MLP.json'), 'r') as f:
            performance = json.load(f)
            hyperparameters = performance['hyperparameters']
    except FileNotFoundError:
        print("Hyperparameter file not found. Make sure 'model_rank_1_performance_MLP.json' exists.")
        return

    hidden_layer_sizes = hyperparameters['hidden_layer_sizes']
    activation = hyperparameters['activation']
    dropout_rate = hyperparameters['dropout_rate']

    if isinstance(hidden_layer_sizes, str):
        hidden_layer_sizes = eval(hidden_layer_sizes)

    if activation == 'relu':
        activation_fn = nn.ReLU()
    elif activation == 'tanh':
        activation_fn = nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

    input_size = 18  # Adjust based on your feature count

    # Set num_classes explicitly to 14
    num_classes = 14

    class MLP(nn.Module):
        def __init__(self, input_size, hidden_layers, num_classes, activation_fn, dropout_rate):
            super(MLP, self).__init__()
            layers = []
            prev_size = input_size
            for hidden_size in hidden_layers:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(activation_fn)
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(dropout_rate))
                prev_size = hidden_size
            layers.append(nn.Linear(prev_size, num_classes))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    # Initialize the model
    model = MLP(
        input_size=input_size,
        hidden_layers=hidden_layer_sizes,
        num_classes=num_classes,
        activation_fn=activation_fn,
        dropout_rate=dropout_rate
    )

    # Determine the device and move the model to that device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load the model state dict with map_location
    try:
        model.load_state_dict(torch.load('best_mlp_model_rank_1.pth', map_location=device))
    except FileNotFoundError:
        print("Model file not found. Make sure 'best_mlp_model_rank_1.pth' exists.")
        return

    model.eval()

    while True:
        time.sleep(prediction_interval)
        with buffer_lock:
            data_list = list(data_buffer)
            if len(data_list) < 1:
                continue

            timestamps, xs, ys, zs = zip(*data_list)
            df = pd.DataFrame({'Timestamp': timestamps, 'X': xs, 'Y': ys, 'Z': zs})

            features = compute_features(df)
            if features is None:
                continue

            features_scaled = scaler.transform(features)
            X_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)

            with torch.no_grad():
                output = model(X_tensor)
                probabilities = nn.functional.softmax(output, dim=1)
                predicted_label = torch.argmax(probabilities, dim=1).item()
                probability = probabilities[0, predicted_label].item()
                print(f"Predicted Label: {predicted_label}, Probability: {probability:.4f}")

def compute_features(df):
    """Compute the required features from the raw data."""
    try:
        time_diffs = np.diff(df['Timestamp'])
        if len(time_diffs) == 0:
            print("Not enough data points to compute features.")
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
        return pd.DataFrame(feature_dict)
    except Exception as e:
        print(f"Error computing features: {e}")
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
        print(f"Error computing dominant frequency: {e}")
        return 0

# Start the UDP listener in a separate thread
listener_thread = Thread(target=udp_listener, daemon=True)
listener_thread.start()

# Start the feature extraction and prediction
feature_extraction()
