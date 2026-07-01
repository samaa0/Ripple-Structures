# Firmware

This folder contains the Arduino sketches for the DWM3000 UWB transmitter and receiver.

## Transmitter

```sh
cp transmitter/config.example.h transmitter/config.h
```

Set `TX_TO_UPLOAD` to a unique DWM3000 transmitter ID before flashing `transmitter.ino`.

## Receiver

```sh
cp receiver/config.example.h receiver/config.h
```

Set WiFi credentials, server IP/port, receiver ID, transmitter count, DWM3000 anchor coordinates, and distance offsets before flashing `receiver.ino`.

`config.h` is intentionally ignored by Git so private credentials and calibration values stay local.
