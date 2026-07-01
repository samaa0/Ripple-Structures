# Positioning Server

This folder contains the real-time UDP positioning and plotting code.

## Files

- `positioning_server.py` receives receiver measurements, estimates position, plots the stream, and forwards coordinates.
- `prediction_handler.py` is an optional gesture/action handler for a separate Keras model workflow.
- `config.example.yaml` lists the public defaults mirrored by environment variables.

## Run

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python positioning_server.py
```

## Configuration

The server keeps the original defaults but reads public configuration from environment variables:

- `RIPPLE_UDP_HOST`, `RIPPLE_UDP_PORT`
- `RIPPLE_UNITY_HOST`, `RIPPLE_UNITY_PORT`
- `RIPPLE_DATA_HOST`, `RIPPLE_DATA_PORT`
- `RIPPLE_WINDOW_SIZE`, `RIPPLE_SECOND_AVERAGE`
- `RIPPLE_STD_DEV_THRESHOLD`, `RIPPLE_STD_DEV_WINDOW_SIZE`
- `RIPPLE_TRANSITION_COVARIANCE`, `RIPPLE_OBSERVATION_COVARIANCE`
- `RIPPLE_LOW_PASS_ALPHA`
- `RIPPLE_RANSAC_SAMPLE_SIZE`, `RIPPLE_RANSAC_MAX_ITERATIONS`

`prediction_handler.py` optionally calls an automation API. No private endpoint is committed; set `RIPPLE_AUTOMATION_API_ENDPOINT` and related `RIPPLE_AUTOMATION_*` variables locally if needed.

Optional Keras fold models for `prediction_handler.py` are not included in this release. Place them in `software/positioning_server/models/` or set `RIPPLE_PREDICTION_MODEL_DIR`.
