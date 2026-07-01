import socket
import numpy as np
import time
from collections import deque
from scipy.interpolate import interp1d
from threading import Thread, Lock
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import threading
from collections import Counter
import requests
import json
import os
from pathlib import Path


MODEL_DIR = Path(os.getenv("RIPPLE_PREDICTION_MODEL_DIR", Path(__file__).resolve().parent / "models")).expanduser()
AUTOMATION_API_ENDPOINT = os.getenv("RIPPLE_AUTOMATION_API_ENDPOINT")
AUTOMATION_USERNAME = os.getenv("RIPPLE_AUTOMATION_USERNAME", "User1")
AUTOMATION_DEVICE_IDS = [
    int(device_id.strip())
    for device_id in os.getenv("RIPPLE_AUTOMATION_DEVICE_IDS", "1,2,3").split(",")
    if device_id.strip()
]
AUTOMATION_STATUS = os.getenv("RIPPLE_AUTOMATION_STATUS", "Off")


def configured_model_path(env_name, default_filename):
    configured_path = Path(os.getenv(env_name, default_filename)).expanduser()
    if configured_path.is_absolute():
        return configured_path
    return MODEL_DIR / configured_path


model1 = load_model(configured_model_path("RIPPLE_PREDICTION_MODEL_1", "CNN_feature_fold1.h5"))
model2 = load_model(configured_model_path("RIPPLE_PREDICTION_MODEL_2", "CNN_feature_fold2.h5"))
model3 = load_model(configured_model_path("RIPPLE_PREDICTION_MODEL_3", "CNN_feature_fold3.h5"))
model4 = load_model(configured_model_path("RIPPLE_PREDICTION_MODEL_4", "CNN_feature_fold4.h5"))
model5 = load_model(configured_model_path("RIPPLE_PREDICTION_MODEL_5", "CNN_feature_fold5.h5"))
# Initialize a queue to hold the raw data
data_stream = deque()




def udp_listener():
    start_time = time.time()
    # Set up the UDP socket
    UDP_IP = os.getenv("RIPPLE_PREDICTION_HOST", "0.0.0.0")
    UDP_PORT = int(os.getenv("RIPPLE_PREDICTION_PORT", "5005"))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    while True:
        current_time=time.time()
        data, addr = sock.recvfrom(1024)
        data_str = data.decode('utf-8')
        x, y, z = map(float, data_str.split(','))
        time_e = current_time - start_time

        # print(f'received:{np.array([time_e, x, y, z])}')
        # if current_time-start_time>=2:
        #     raw_data_queue.clear()
        #     start_time = current_time
        data_stream.append([current_time, x, y, z])
        if current_time - start_time >=2 :
            sock.close()
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((UDP_IP, UDP_PORT))
            start_time = current_time


def send_put_request(api_endpoint, username, device_id, status):
    # Construct the JSON payload
    payload = {
        "Username": username,
        "DeviceID": device_id,
        "Status": status
    }

    # Send the PUT request
    response = requests.put(api_endpoint, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        print('Request was successful.')
        # Perform actions with the response if necessary
        # data = response.json()
        # print(data)
    else:
        print('Request failed with status code:', response.status_code)
        # Print the error message if needed
        # print(response.text)

    return response

def resample_data_from_array(data_deque, num_samples=150):
    # Convert the deque to a NumPy array
    data_array = np.array(data_deque)

    # Rest of your function...
    if len(data_array) < 2:
        print("Not enough data to resample.")
        return None

    # Separate the timestamps and the sensor data
    timestamps = data_array[:, 0]
    sensor_data = data_array[:, 1:]

    # Normalize time to start at 0 and end at 2 seconds
    normalized_time = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0]) * 2

    # Define the new time vector for resampling
    resampled_time = np.linspace(0, 2, num_samples)

    # Interpolate the data for each axis
    resampled_data = np.empty((num_samples, 3))
    for i in range(3):  # x, y, z coordinates
        interpolator = interp1d(normalized_time, sensor_data[:, i], kind='linear', fill_value='extrapolate')
        resampled_data[:, i] = interpolator(resampled_time)

    # Take absolute value of the z-axis
    resampled_data[:, 2] = np.abs(resampled_data[:, 2])

    return resampled_data


def global_feature_extraction(x_data):
    features_list = []


    sequence = x_data
    standard_deviation = np.std(sequence, axis=0)
    total_distance = np.sum(np.sqrt(np.sum(np.diff(sequence, axis=0) ** 2, axis=1)))
    start_point = sequence[0]
    end_point = sequence[-1]
    dist_start_end = np.linalg.norm(end_point - start_point)
    xy_correlation = np.corrcoef(sequence[:, 0], sequence[:, 1])[0, 1]
    xz_correlation = np.corrcoef(sequence[:, 0], sequence[:, 2])[0, 1]
    yz_correlation = np.corrcoef(sequence[:, 1], sequence[:, 2])[0, 1]

    # Ensure that all elements are numpy arrays of the same length (1 in case of scalars).
    standard_deviation = np.atleast_1d(standard_deviation)
    total_distance = np.atleast_1d(total_distance)
    dist_start_end = np.atleast_1d(dist_start_end)
    xy_correlation = np.atleast_1d(xy_correlation)
    xz_correlation = np.atleast_1d(xz_correlation)
    yz_correlation = np.atleast_1d(yz_correlation)

    # Combine all features into a single array for this sequence.
    sequence_features = np.concatenate(
        [standard_deviation, total_distance, dist_start_end, xy_correlation, xz_correlation, yz_correlation]
    )

    features_list.append(sequence_features)

    # Convert the list of feature arrays into a 2D numpy array.
    feature_vector = np.vstack(features_list)

    return feature_vector

# Assuming scaler and model are defined elsewhere and are not thread-safe
scaler_lock = threading.Lock()
model_lock = threading.Lock()

def process_window_data(window_data):
    # Convert window_data to a numpy array
    window_data_array = np.array(window_data)

    # Resample the data
    resampled_data = resample_data_from_array(window_data_array, 150)

    # Normalize the data with thread-safe access
    with scaler_lock:
        scaler.fit(resampled_data)
        normalized_data = scaler.transform(resampled_data)

    # Extract features
    feature_vector = global_feature_extraction(normalized_data)

    # Prepare the input data
    Input_data = np.expand_dims(normalized_data, axis=0)

    # Make prediction with thread-safe access
    # with model_lock:
    #     prediction = model.predict([Input_data, feature_vector])
    prediction1 = model1.predict([Input_data, feature_vector])
    prediction2 = model2.predict([Input_data, feature_vector])
    prediction3 = model3.predict([Input_data, feature_vector])
    prediction4 = model4.predict([Input_data, feature_vector])
    prediction5 = model5.predict([Input_data, feature_vector])

    # Use np.argmax to find the index of the maximum value in predictions
    predicted_classes = [
        np.argmax(prediction1, axis=-1),
        np.argmax(prediction2, axis=-1),
        np.argmax(prediction3, axis=-1),
        np.argmax(prediction4, axis=-1),
        np.argmax(prediction5, axis=-1)
    ]
    # print(predicted_classes)
    # Flatten the list if the predictions are multidimensional
    predicted_classes = [item for sublist in predicted_classes for item in sublist]

    # Count the occurrences of each predicted class
    counts = Counter(predicted_classes)

    # Find the most common class (mode)
    most_common_prediction = counts.most_common(1)[0][0]
    pred_action = most_common_prediction
    # if(pred_action== 0):
    #     print(f'predicted action : clap')
    if (pred_action == 1):
        print(f'predicted action : wave')
    if (pred_action == 2):
        print(f'predicted action : walk')
    if (pred_action == 3):
        print(f'predicted action : idle')

    # print(type(most_common_prediction))
    # # Output the most common prediction
    # print(f"The most frequent prediction is: {most_common_prediction}")
    if pred_action == 1:
        if not AUTOMATION_API_ENDPOINT:
            print("Automation API endpoint is not configured; skipping device update.")
            return
        for device_id in AUTOMATION_DEVICE_IDS:
            send_put_request(
                api_endpoint=AUTOMATION_API_ENDPOINT,
                username=AUTOMATION_USERNAME,
                device_id=device_id,
                status=AUTOMATION_STATUS,
            )



def threaded_process_window_data(window_data):
    Thread(target=process_window_data, args=(window_data,)).start()

# Main loop for processing and resampling data
WINDOW_DURATION = 2.0  # Duration of the data window in seconds
PREDICTION_STEP = 2  # Time step for each prediction
last_prediction_time = time.time() - PREDICTION_STEP  # Initialize to make an immediate prediction

s_time = time.time()
# Create a thread to run the udp_listener function
listener_thread = Thread(target=udp_listener)
listener_thread.daemon = True  # Set as a daemon so it will be killed when the main thread exits
listener_thread.start()
# Initialize the StandardScaler
scaler = StandardScaler()

counter = 0
window_size=2.0
step=0.5
start_time = time.time()
last_check_time = time.time()


while True:
    current_time = time.time()
    # Calculate the start of the current window
    window_start = current_time - window_size

    # Check if it's time to process the window
    if current_time >= last_check_time + step:
        last_check_time += step

        # Snapshot the current state of the deque for thread safety
        with threading.Lock():  # Locking to ensure thread-safe access to deque
            # Remove items that are too old for the new window
            while data_stream and data_stream[0][0] < window_start:
                data_stream.popleft()

            # Take a snapshot for processing
            window_data_snapshot = list(data_stream)

        # Extract the data for the current window and normalize time
        window_data = [(element[0] - window_start, element[1], element[2], element[3]) for element in
                       window_data_snapshot if window_start <= element[0] <= current_time]
        # print(window_data)
        # Process the data for the current window
        # This is where you do something meaningful with the window_data
        threaded_process_window_data(window_data)

        print(f"Processed window at {time.strftime('%X')} with {len(window_data)} data points.")

    time.sleep(0.001)  # Sleep for 1 ms to avoid busy waiting
