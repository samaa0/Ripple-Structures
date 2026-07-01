import socket
import re
import time
from threading import Thread, Lock
from collections import deque
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import json
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
                print("Not enough data to process.")
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

        if probability >= confidence_threshold:
            # High confidence prediction
            print(f"Predicted Gesture: {predicted_gesture} (Confidence: {probability:.2f})\n")
        else:
            # Low confidence - no gesture predicted
            print(f"Prediction confidence too low ({probability:.2f}). No gesture predicted.\n")

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

if __name__ == '__main__':
    # Start the UDP listener in a separate thread
    listener_thread = Thread(target=udp_listener, daemon=True)
    listener_thread.start()

    # Start the feature extraction and prediction using LightGBM
    feature_extraction()
