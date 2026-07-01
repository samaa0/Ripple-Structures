import threading
import re
import numpy as np
from scipy.optimize import least_squares
from collections import deque
from vispy import scene
from vispy import app
from vispy.visuals.transforms import STTransform
from vispy.scene.visuals import Line, Text
from filterpy.kalman import KalmanFilter
from sklearn.linear_model import RANSACRegressor
from joblib import Parallel, delayed
import socket
import time
import itertools
import requests
import json
import os


def env_int(name, default):
    return int(os.getenv(name, default))


def env_float(name, default):
    return float(os.getenv(name, default))

# Setup UDP server
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((os.getenv("RIPPLE_UDP_HOST", "0.0.0.0"), env_int("RIPPLE_UDP_PORT", 20002)))

# Create a new UDP socket for transmitting data to Unity
sock_unity = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
UNITY_IP = os.getenv("RIPPLE_UNITY_HOST", "127.0.0.1")
UNITY_PORT = env_int("RIPPLE_UNITY_PORT", 5065)

sock_dataCollect = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
DC_IP = os.getenv("RIPPLE_DATA_HOST", "127.0.0.1")
DC_PORT = env_int("RIPPLE_DATA_PORT", 5005)
# Define the length of the moving average window
WINDOW_SIZE = env_int("RIPPLE_WINDOW_SIZE", 80)
# A larger window size will result in smoother output, but it might miss out on detecting rapid
# changes in the data.

second_average = env_int("RIPPLE_SECOND_AVERAGE", 400)
# A larger value will smooth out the output further, but may also lead to slower response times
# to changes in the data.

# Define the standard deviation threshold
STD_DEV_THRESHOLD = env_float("RIPPLE_STD_DEV_THRESHOLD", 500)
# A higher threshold will result in fewer anomalies being detected.

# Define the length of the standard deviation window
STD_DEV_WINDOW_SIZE = env_int("RIPPLE_STD_DEV_WINDOW_SIZE", 1)
# A larger window size may smooth out the output, but it could also delay the detection of
# rapid changes.

#Kalman Filter
transition_covariance_value = env_float("RIPPLE_TRANSITION_COVARIANCE", 100000)
# system noise, Higher values imply more uncertainty about the state evolution.
observation_covariance_value = env_float("RIPPLE_OBSERVATION_COVARIANCE", 10000)
# measurement noise, Higher values indicate more uncertainty about the observations

# Define the alpha value for the low-pass filter
alpha = env_float("RIPPLE_LOW_PASS_ALPHA", 0.03)
# A higher value (closer to 1) gives more weight to recent data points, making the output respond more
# quickly to changes.

# Define the least median square
sample_size_value = env_int("RIPPLE_RANSAC_SAMPLE_SIZE", 500)
max_iterations_value = env_int("RIPPLE_RANSAC_MAX_ITERATIONS", 500)

# Define the max number of points to display per RX
MAX_POINTS_PER_RX = env_int("RIPPLE_MAX_POINTS_PER_RX", 5)

# Store RX data
rx_data = {}

# Store scatter plot data
scatter_data = {
    'positions': {},  # To store positions of all RX in a dictionary
    'colors': {}      # To store colors corresponding to each RX position in a dictionary
}



# Function to update RX positions in scatter_data
def update_rx_positions(rx_id, new_position, new_color):
    # We will use a dictionary with rx_id as keys to track positions for each RX
    if rx_id not in scatter_data['positions']:
        scatter_data['positions'][rx_id] = deque(maxlen=MAX_POINTS_PER_RX)
        scatter_data['colors'][rx_id] = deque(maxlen=MAX_POINTS_PER_RX)

    # Append new position and color, automatically discarding the oldest if over limit
    scatter_data['positions'][rx_id].append(new_position)
    scatter_data['colors'][rx_id].append(new_color)


# Function to generate distinct colors
def get_distinct_colors():
    # Define some base colors
    base_colors = [
        (1.0, 0.0, 0.0),  # red
        (0.0, 1.0, 0.0),  # green
        (0.0, 0.0, 1.0),  # blue
        (1.0, 1.0, 0.0),  # yellow
        (1.0, 0.0, 1.0),  # magenta
        (0.0, 1.0, 1.0),  # cyan
    ]
    # If more colors are needed, use itertools to create combinations
    for color in itertools.cycle(base_colors):
        yield color

color_gen = get_distinct_colors()

# Initialize the color map for RX IDs
rx_color_map = {}

# Function to get color for RX ID
def get_rx_color(rx_id):
    if rx_id not in rx_color_map:
        rx_color_map[rx_id] = next(color_gen)
    return rx_color_map[rx_id]


def send_to_url(url, data):
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.RequestException as e:
        print(f"HTTP Request failed: {e}")

# Error function with weights
def weighted_error(p, distances, coordinates):
    weights = 1 #/ np.array(distances)  # inverse distance weighting
    return weights * np.array([calc_distance(p, coord) - dist for dist, coord in zip(distances, coordinates)])

# Moving average function
def moving_average(a, n=0):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# Distance function
def calc_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


# Error function
def error(p, distances, coordinates):
    return [calc_distance(p, coord) - dist for dist, coord in zip(distances, coordinates)]


# Read data from UDP
def read_from_udp(sock):
    while True:
        data, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
        line = data.decode('utf-8').strip()

        # Adjust regex to match the new message format: [RX_ID]TX[TX_ID] (x.x, y.y, z.z) d.ddm
        match = re.match(r'\[(\d{4})\] \(([\d.]+), ([\d.]+), ([\d.]+)\) ([\d.]+)m', line)

        if match:
            rx_id = int(match.group(1)[:2])  # Extract RX ID
            tx_num = int(match.group(1)[2:])  # Extract TX ID
            coord = np.array([float(match.group(2)), float(match.group(3)), float(match.group(4))])
            dist = float(match.group(5))
            return rx_id, tx_num, coord, dist

        return None, None, None, None

def ransac_least_median_squares(coordinates, distances, initial_guess, sample_size,
                                max_iterations=max_iterations_value):
    best_median = float('inf')
    best_model = None

    # Convert coordinates and distances to numpy arrays
    coordinates = np.array(coordinates)
    distances = np.array(distances)

    # Create a RANSAC regressor
    ransac = RANSACRegressor(min_samples=sample_size, max_trials=max_iterations, random_state=42)

    # Define a function to be run in parallel for each iteration
    def run_iteration(_):
        try:
            # Fit the model to the data using RANSAC
            ransac.fit(coordinates, distances)

            # Get the inlier mask that denotes if a point is an inlier
            inlier_mask = ransac.inlier_mask_

            # Use only the inliers to fit the model
            result = least_squares(error, initial_guess, args=(distances[inlier_mask], coordinates[inlier_mask]))

            if result.success:
                model = result.x

                # Calculate the residuals for all data
                residuals = np.abs(np.array(error(model, distances, coordinates)))

                # Calculate the median of the residuals
                median = np.median(residuals)

                return median, model

        except ValueError as e:
            if 'All `max_trials` iterations were skipped because each randomly chosen sub-sample failed the passing criteria.' in str(
                    e):
                print("Skipping this iteration due to the error:", str(e))
                return float('inf'), None
            else:
                raise e

    # Run the iterations in parallel
    results = Parallel(n_jobs=-1)(delayed(run_iteration)(i) for i in range(max_iterations))

    # Find the best model
    for median, model in results:
        if median < best_median:
            best_median = median
            best_model = model

    return best_model

# Initializing vispy scene
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
scatter = scene.visuals.Markers(parent=view.scene)
scatter.set_data(np.array([[0, 0, 0]]), face_color='pink')  # initialize with a single point
view.camera = 'turntable'
view.camera.flip = (0, 0, 0)

x_label = Text('X', parent=view.scene, color='red')
y_label = Text('Y', parent=view.scene, color='green')
z_label = Text('Z', parent=view.scene, color='blue')

# Define the length of each axis
axis_length = 3

# Define the axes
axes = [Line(pos=np.array(((0, 0, 0), [axis_length if i == j else 0 for i in range(3)])),
             color=[1 if i == j else 0 for i in range(3)],
             width=2,
             method='gl',
             connect='segments',
             parent=view.scene) for j in range(3)]

# Position the labels
x_label.transform = STTransform(translate=(1050, 0, 0))
y_label.transform = STTransform(translate=(0, 1050, 0))
z_label.transform = STTransform(translate=(0, 0, 1050))

# Define the length of each axis
axis_length = 1000

# Define the axes
axes = [Line(pos=np.array(((0, 0, 0), [axis_length if i==j else 0 for i in range(3)])),
             color=[1 if i==j else 0 for i in range(3)],
             width=2,
             method='gl',
             connect='segments',
             parent=view.scene) for j in range(3)]

# Define the speed text
speed_text = Text('Speed: 0.00 m/sec', parent=view.scene, color='white')
speed_text.transform = STTransform(translate=(850, 950, 0))  # Adjust these values to position the text

# Create a new Text visual for displaying the speed data
speed_text = scene.visuals.Text('', color='white', parent=canvas.scene, pos=(canvas.size[0] - 10, 10), anchor_x='right')



# Initialize previous time value
prev_time = None

# Initialize speeds deque for moving average calculation and last speed update time
# speeds = deque(maxlen=second_average)  # store speeds for moving average calculation
last_speed_update = time.time() # store the last time the speed was updated

# Store TX data
tx_data = {}
coordinates = []  # store for plot

# Initialize previous filtered value
prev_filtered_rx_position = np.array([0, 0, 0])

# Initialize previous time value
prev_position = None
prev_time = None


def initialize_rx_data(rx_id, initial_position=None):
    """Initialize data structure for a new RX."""
    if initial_position is None:
        initial_position = np.zeros(3)

    kf = KalmanFilter(dim_x=3, dim_z=3)
    kf.x = initial_position  # state mean, set to initial position
    kf.F = np.eye(3)  # state transition matrix
    kf.P *= 1000.0  # covariance matrix
    kf.H = np.eye(3)  # observation matrix
    kf.Q = transition_covariance_value * np.eye(3)  # process noise
    kf.R = observation_covariance_value * np.eye(3)  # measurement noise

    rx_data[rx_id] = {
        'tx_data': {},
        'kf': kf,
        'prev_filtered_rx_position': initial_position,
        'prev_time': time.time(),
        'rx_positions': deque(maxlen=WINDOW_SIZE),
        'final_rx_positions': deque(maxlen=WINDOW_SIZE),
        'speed': 0.0
    }
    return kf

def send_to_udp(ip, port, data):
    try:
        if not isinstance(data, (list, dict)):
            raise ValueError("Data must be a list or dictionary to be JSON serialized.")


        message = json.dumps(data)

        sock_dataCollect.sendto(message.encode(), (ip, port))
    except Exception as e:
        print(f"UDP sending failed: {e}")

def update_visuals():
    flat_positions = [pos for pos_list in scatter_data['positions'].values() for pos in pos_list]
    flat_colors = [color for color_list in scatter_data['colors'].values() for color in color_list]

    if flat_positions:
        scatter.set_data(np.array(flat_positions), face_color=np.array(flat_colors), size=5)

    canvas.update()
    app.process_events()

def read_data():
    global rx_data, prev_time, last_speed_update, sock_unity, UNITY_IP, UNITY_PORT
    speed = 0.0

    while True:
        rx_id, tx_num, coord, dist = read_from_udp(sock)
        if rx_id is not None:
            color = get_rx_color(rx_id)
            if rx_id not in rx_data:
                kf = initialize_rx_data(rx_id, coord)
            else:
                kf = rx_data[rx_id]['kf']



            if tx_num not in rx_data[rx_id]['tx_data']:
                rx_data[rx_id]['tx_data'][tx_num] = {'coord': coord, 'distances': deque(maxlen=WINDOW_SIZE)}

            rx_data[rx_id]['tx_data'][tx_num]['distances'].append(dist)


            valid_tx_data = [data for data in rx_data[rx_id]['tx_data'].values() if len(data['distances']) > 0]


            if len(valid_tx_data) >= 3:
                initial_guess = rx_data[rx_id]['prev_filtered_rx_position']
                distances = [np.mean(data['distances']) for data in valid_tx_data]
                coordinates = [data['coord'] for data in valid_tx_data]


                result = least_squares(weighted_error, initial_guess, args=(distances, coordinates))
                model = result.x if result.success else None

                if model is not None:
                    rx_position = model
                    rx_data[rx_id]['rx_positions'].append(rx_position)


                    kf = rx_data[rx_id]['kf']
                    kf.predict()
                    kf.update(rx_position)
                    state_mean = kf.x
                    state_covariance = kf.P


                    filtered_rx_position = alpha * state_mean + (1 - alpha) * rx_data[rx_id]['prev_filtered_rx_position']
                    rx_data[rx_id]['prev_filtered_rx_position'] = filtered_rx_position
                    rx_data[rx_id]['final_rx_positions'].append(filtered_rx_position)




                    if len(rx_data[rx_id]['final_rx_positions']) == WINDOW_SIZE:
                        final_rx_position_ma = np.mean(rx_data[rx_id]['final_rx_positions'], axis=0)


                        current_time = time.time()
                        if rx_data[rx_id]['prev_time']:
                            time_elapsed = current_time - rx_data[rx_id]['prev_time']
                            if time_elapsed > 0:
                                distance = np.linalg.norm(final_rx_position_ma - rx_data[rx_id]['prev_filtered_rx_position'])
                                rx_data[rx_id]['speed'] = distance / time_elapsed


                        rx_data[rx_id]['prev_time'] = current_time


                        final_speed = rx_data[rx_id]['speed']
                        print(f'RX[{rx_id}] Position: {final_rx_position_ma}, Speed: {final_speed} m/s')


                        message = f'{rx_id},{final_rx_position_ma[0]},{final_rx_position_ma[1]},{abs(final_rx_position_ma[2])},{final_speed}'
                        sock_unity.sendto(message.encode(), (UNITY_IP, UNITY_PORT))
                        data = f'{final_rx_position_ma[0]},{final_rx_position_ma[1]},{final_rx_position_ma[2]}'
                        sock_dataCollect.sendto(data.encode(), (DC_IP, DC_PORT))
                        update_rx_positions(rx_id, final_rx_position_ma.tolist(), color)

                        # Prepare the data to send; packing one integer and three floats
                        # data_to_send = struct.pack('ifff', rx_id, final_rx_position_ma[0], final_rx_position_ma[1],
                        #                            final_rx_position_ma[2])


                        # Example: forward final positions to another UDP service.
                        # SERVER_IP = "example.local"
                        # SERVER_PORT = 20002
                        # data_to_send = [rx_id, final_rx_position_ma[0], final_rx_position_ma[1],
                        #                 final_rx_position_ma[2]]
                        #
                        # send_to_udp(SERVER_IP, SERVER_PORT, data_to_send)




                        update_visuals()

                        for data in valid_tx_data:
                            data['distances'].clear()


        if time.time() - last_speed_update >= 1:
            speed_text.text = "{:.2f} m/s".format(speed)
            last_speed_update = time.time()

        app.process_events()
        update_visuals()

data_thread = threading.Thread(target=read_data)
data_thread.start()

app.run()
