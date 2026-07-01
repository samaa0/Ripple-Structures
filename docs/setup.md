# Setup

## Firmware

Copy the example configs before compiling:

```sh
cp firmware/transmitter/config.example.h firmware/transmitter/config.h
cp firmware/receiver/config.example.h firmware/receiver/config.h
```

Set each transmitter to a unique `TX_TO_UPLOAD` ID. In the receiver config, set WiFi credentials, `SERVER_IP`, UDP port, receiver ID, transmitter count, anchor coordinates, and distance offsets.

## Positioning Server

```sh
cd software/positioning_server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python positioning_server.py
```

Common environment variables:

- `RIPPLE_UDP_HOST`, `RIPPLE_UDP_PORT`
- `RIPPLE_UNITY_HOST`, `RIPPLE_UNITY_PORT`
- `RIPPLE_DATA_HOST`, `RIPPLE_DATA_PORT`
- `RIPPLE_WINDOW_SIZE`, `RIPPLE_SECOND_AVERAGE`
- `RIPPLE_STD_DEV_THRESHOLD`, `RIPPLE_STD_DEV_WINDOW_SIZE`
- `RIPPLE_TRANSITION_COVARIANCE`, `RIPPLE_OBSERVATION_COVARIANCE`
- `RIPPLE_LOW_PASS_ALPHA`
- `RIPPLE_RANSAC_SAMPLE_SIZE`, `RIPPLE_RANSAC_MAX_ITERATIONS`

## ML Scripts

```sh
cd software/ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model artifacts are read from `software/ml/models/` by default. Set `RIPPLE_ML_MODEL_DIR` or `RIPPLE_ML_DATA_DIR` to use a different local model/data directory.

Raw CSV datasets are not included in the public repository. Add private datasets locally under `software/ml/data/` or pass repo-relative paths when adapting the scripts.

## Unity

Open `unity/ripple-visualizer/` using Unity `2022.3.10f1`. The private scanned room scene is excluded. Create a neutral scene, add a receiver object, and attach `UdpReceiver`, `RXCollisionHandler`, and the helper scripts needed for your workflow.

## Hardware

PCB files are in `hardware/pcb/`; case files are in `hardware/case/`. Check board dimensions, connector orientation, and enclosure fit before fabrication.
