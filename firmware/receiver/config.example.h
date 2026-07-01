#pragma once

// Copy this file to config.h before compiling.
// Keep config.h private: it contains WiFi credentials and deployment geometry.
#define WIFI_SSID "YOUR_WIFI_SSID"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
#define SERVER_IP "YOUR_SERVER_IP"
#define SERVER_UDP_PORT 20002

// Receiver ID and number of anchors/transmitters used in the deployment.
#define RX_TO_UPLOAD 1
#define NUMBER_OF_TRANSMITTERS 3

// Anchor/transmitter coordinates in meters, matching NUMBER_OF_TRANSMITTERS.
// Replace these sample values with your measured deployment coordinates.
#define TX_COORDINATES { \
  { 0.0, 0.0, 0.0 }, \
  { 1.0, 0.0, 0.0 }, \
  { 0.0, 1.0, 1.0 } \
}

// Per-transmitter distance calibration offsets in meters. The firmware supports
// up to 50 transmitter IDs, so the table intentionally contains 50 entries.
#define DISTANCE_OFFSETS { \
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 \
}
