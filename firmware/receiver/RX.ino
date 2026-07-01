#include "dw3000.h"
#include <cmath>
#include <math.h>
#include <vector>
#include <algorithm>
#include "dw3000_config_options.h"
#include <WiFi.h>
#include <WiFiUdp.h>
#include "config.h"
extern dwt_txconfig_t txconfig_options_ch9;
//
// #define PIN_RST 25
// #define PIN_IRQ 26
// #define PIN_SS 5
//
#define PIN_RST 0
#define PIN_IRQ 1
#define PIN_SS 7

#define RNG_DELAY_MS 0
#define TX_ANT_DLY 16385
#define RX_ANT_DLY 16385
#define ALL_MSG_COMMON_LEN 10
#define ALL_MSG_SN_IDX 2
#define RESP_MSG_POLL_RX_TS_IDX 10
#define RESP_MSG_RESP_TX_TS_IDX 14
#define RESP_MSG_TS_LEN 4
#define POLL_TX_TO_RESP_RX_DLY_UUS 240
#define RESP_RX_TIMEOUT_UUS 400

/* Default communication configuration. We use default non-STS DW mode. */
static dwt_config_t config = {
  5,                /* Channel number. */
  DWT_PLEN_128,     /* Preamble length. Used in TX only. */
  DWT_PAC8,         /* Preamble acquisition chunk size. Used in RX only. */
  9,                /* TX preamble code. Used in TX only. */
  9,                /* RX preamble code. Used in RX only. */
  1,                /* 0 to use standard 8 symbol SFD, 1 to use non-standard 8 symbol, 2 for non-standard 16 symbol SFD and 3 for 4z 8 symbol SDF type */
  DWT_BR_6M8,       /* Data rate. */
  DWT_PHRMODE_STD,  /* PHY header mode. */
  DWT_PHRRATE_STD,  /* PHY header rate. */
  (129 + 8 - 8),    /* SFD timeout (preamble length + 1 + SFD length - PAC size). Used in RX only. */
  DWT_STS_MODE_OFF, /* STS disabled */
  DWT_STS_LEN_64,   /* STS length see allowed values in Enum dwt_sts_lengths_e */
  DWT_PDOA_M0       /* PDOA mode off */
};


struct Point3D {
  double x, y, z;
};


static uint8_t tx_poll_msg[] = { 0x41, 0x88, 0, 0xCA, 0xDE, 'T', 'X', '0', '1', 0xE0, 0, 0 };
static uint8_t rx_resp_msg[] = { 0x41, 0x88, 0, 0xCA, 0xDE, 'V', 'E', 'W', 'A', 0xE1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
char tx[50][4] = { { 'T', 'X', '0', '1' }, { 'T', 'X', '0', '2' }, { 'T', 'X', '0', '3' }, { 'T', 'X', '0', '4' }, { 'T', 'X', '0', '5' }, { 'T', 'X', '0', '6' }, { 'T', 'X', '0', '7' }, { 'T', 'X', '0', '8' }, { 'T', 'X', '0', '9' }, { 'T', 'X', '1', '0' }, { 'T', 'X', '1', '1' }, { 'T', 'X', '1', '2' }, { 'T', 'X', '1', '3' }, { 'T', 'X', '1', '4' }, { 'T', 'X', '1', '5' }, { 'T', 'X', '1', '6' }, { 'T', 'X', '1', '7' }, { 'T', 'X', '1', '8' }, { 'T', 'X', '1', '9' }, { 'T', 'X', '2', '0' }, { 'T', 'X', '2', '1' }, { 'T', 'X', '2', '2' }, { 'T', 'X', '2', '3' }, { 'T', 'X', '2', '4' }, { 'T', 'X', '2', '5' }, { 'T', 'X', '2', '6' }, { 'T', 'X', '2', '7' }, { 'T', 'X', '2', '8' }, { 'T', 'X', '2', '9' }, { 'T', 'X', '3', '0' }, { 'T', 'X', '3', '1' }, { 'T', 'X', '3', '2' }, { 'T', 'X', '3', '3' }, { 'T', 'X', '3', '4' }, { 'T', 'X', '3', '5' }, { 'T', 'X', '3', '6' }, { 'T', 'X', '3', '7' }, { 'T', 'X', '3', '8' }, { 'T', 'X', '3', '9' }, { 'T', 'X', '4', '0' }, { 'T', 'X', '4', '1' }, { 'T', 'X', '4', '2' }, { 'T', 'X', '4', '3' }, { 'T', 'X', '4', '4' }, { 'T', 'X', '4', '5' }, { 'T', 'X', '4', '6' }, { 'T', 'X', '4', '7' }, { 'T', 'X', '4', '8' }, { 'T', 'X', '4', '9' }, { 'T', 'X', '5', '0' } };

//coordinate of each TX
Point3D tx_coordinate[NUMBER_OF_TRANSMITTERS] = TX_COORDINATES;

// this is a function change TX to corresponding RX number ["this"]["this"][""][""]
void change_tx() {
  int i;
  char str[3];
  sprintf(str, "%02d", RX_TO_UPLOAD);
  for (i = 0; i < 50; ++i) {
    tx[i][0] = str[0];
    tx[i][1] = str[1];
  }
}

const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;
// UDP settings
const char* remoteIp = SERVER_IP;
unsigned int remoteUdpPort = SERVER_UDP_PORT;
WiFiUDP udp;

static uint8_t frame_seq_nb = 0;
static uint8_t rx_buffer[20];
static uint32_t status_reg = 0;
static double tof;
static double distance;
static double distance1;
static double distance2;
static double distance3;
extern dwt_txconfig_t txconfig_options;
int current_tx = 1;

void setup() {
  UART_init();

  spiBegin(PIN_IRQ, PIN_RST);
  spiSelect(PIN_SS);

  delay(2);  // Time needed for DW3000 to start up (transition from INIT_RC to IDLE_RC, or could wait for SPIRDY event)

  while (!dwt_checkidlerc())  // Need to make sure DW IC is in IDLE_RC before proceeding
  {
    UART_puts("IDLE FAILED\r\n");
    while (1)
      ;
  }

  if (dwt_initialise(DWT_DW_INIT) == DWT_ERROR) {
    UART_puts("INIT FAILED\r\n");
    while (1)
      ;
  }

  //  // Enabling LEDs here for debug so that for each TX the D1 LED will flash on DW3000 red eval-shield boards.
  dwt_setleds(DWT_LEDS_ENABLE | DWT_LEDS_INIT_BLINK);

  /* Configure DW IC. See NOTE 6 below. */
  if (dwt_configure(&config))  // if the dwt_configure returns DWT_ERROR either the PLL or RX calibration has failed the host should reset the device
  {
    UART_puts("CONFIG FAILED\r\n");
    while (1)
      ;
  }

  /* Configure the TX spectrum parameters (power, PG delay and PG count) */
  dwt_configuretxrf(&txconfig_options);

  //if (dwt_configure(&config_options)) // if the dwt_configure returns DWT_ERROR either the PLL or RX calibration has failed the host should reset the device
  //{
  //    UART_puts("CONFIG FAILED\r\n");
  //    while (1)
  //      ;
  //}
  //
  ///* Configure the TX spectrum parameters (power, PG delay and PG count) */
  //dwt_configuretxrf(&txconfig_options_ch9);

  /* Apply default antenna delay value. See NOTE 2 below. */
  dwt_setrxantennadelay(RX_ANT_DLY);
  dwt_settxantennadelay(TX_ANT_DLY);

  /* Set expected response's delay and timeout. See NOTE 1 and 5 below.
     As this example only handles one incoming frame with always the same delay and timeout, those values can be set here once for all. */
  dwt_setrxaftertxdelay(POLL_TX_TO_RESP_RX_DLY_UUS);
  dwt_setrxtimeout(RESP_RX_TIMEOUT_UUS);

  /* Next can enable TX/RX states output on GPIOs 5 and 6 to help debug, and also TX/RX LEDs
     Note, in real low power applications the LEDs should not be used. */
  //  dwt_setlnapamode(DWT_LNA_ENABLE | DWT_PA_ENABLE);

  Serial.begin(115200);

  Serial.println("Range RX");
  Serial.println("Setup over........");



  //change TX
  change_tx();


  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }

  Serial.println("Connected to WiFi");

  // Start UDP communication
  udp.begin(remoteUdpPort);

}

void loop() {
  //  Serial.println("Send_to_TX: " + String(Send_to_TX));
  /* Write frame data to DW IC and prepare transmission. See NOTE 7 below. */
  tx_poll_msg[ALL_MSG_SN_IDX] = frame_seq_nb;
  dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_TXFRS_BIT_MASK);
  dwt_writetxdata(sizeof(tx_poll_msg), tx_poll_msg, 0); /* Zero offset in TX buffer. */
  dwt_writetxfctrl(sizeof(tx_poll_msg), 0, 1);          /* Zero offset in TX buffer, ranging. */

  /* Start transmission, indicating that a response is expected so that reception is enabled automatically after the frame is sent and the delay
     set by dwt_setrxaftertxdelay() has elapsed. */

  dwt_starttx(DWT_START_TX_IMMEDIATE | DWT_RESPONSE_EXPECTED);

  /* We assume that the transmission is achieved correctly, poll for reception of a frame or error/timeout. See NOTE 8 below. */
  // waiting response
  //
  //    Serial.print("tx_poll_msg: ");
  //    for (int i = 5; i < 9; i++) {
  //      Serial.print(tx_poll_msg[i], HEX);
  //      Serial.print(" ");
  //    }
  //    Serial.println();
  //    // print rx_buffer
  //    Serial.print("rx_buffer: ");
  //    for (int i = 5; i < 9; i++) {
  //      Serial.print(rx_buffer[i], HEX);
  //      Serial.print(" ");
  //    }
  //
  //    Serial.println();

  //  uint32_t start_time = millis();
  while (!((status_reg = dwt_read32bitreg(SYS_STATUS_ID)) & (SYS_STATUS_RXFCG_BIT_MASK | SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR))) {
  };

  /* Increment frame sequence number after transmission of the poll message (modulo 256). */
  frame_seq_nb++;






  // uploading tx[][] to tx_poll_msg[] ["this"]["this"]["this"]["this"]
  memcpy(tx_poll_msg + 5, tx[current_tx ], 4 * sizeof(uint8_t));

  // increase current_tx to next tx
  current_tx = (current_tx + 1) % NUMBER_OF_TRANSMITTERS;


  //  Serial.print("current_TX1: ");
  //  Serial.print(current_tx);
  //  Serial.print(" ");
  //  Serial.print("tx_poll_msg1: ");
  //  for (int i = 5; i < 9; i++) {
  //    Serial.print(tx_poll_msg[i], HEX);
  //  }
  //  Serial.print(" ");
  //  Serial.print("rx_resp_msg1: ");
  //  for (int i = 5; i < 9; i++) {
  //    Serial.print(rx_resp_msg[i], HEX);
  //  }
  //
  //  Serial.print(" \n");


  if (status_reg & SYS_STATUS_RXFCG_BIT_MASK) {
    //    Serial.printf("status: %d\n", (status_reg & SYS_STATUS_RXFCG_BIT_MASK) > 0);
    uint32_t frame_len;

    //    Serial.print("current_TX after: ");
    //    Serial.print(current_tx);
    //    Serial.print(" ");
    //    Serial.print("tx_poll_msg after: ");
    //    for (int i = 5; i < 9; i++) {
    //      Serial.print(tx_poll_msg[i], HEX);
    //    }
    //    Serial.print(" ");
    //    Serial.print("rx_resp_msg after: ");
    //    for (int i = 5; i < 9; i++) {
    //      Serial.print(rx_resp_msg[i], HEX);
    //    }
    //    Serial.println();
    /* Clear good RX frame event in the DW IC status register. */
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RXFCG_BIT_MASK);

    /* A frame has been received, read it into the local buffer. */
    frame_len = dwt_read32bitreg(RX_FINFO_ID) & RXFLEN_MASK;
    if (frame_len <= sizeof(rx_buffer)) {
      //      Serial.printf("frame_len: %d\n", (frame_len <= sizeof(rx_buffer)) > 0);
      dwt_readrxdata(rx_buffer, frame_len, 0);

      /* Check that the frame is the expected response from the companion "SS TWR responder" example.
         As the sequence number field of the frame is not relevant, it is cleared to simplify the validation of the frame. */
      rx_buffer[ALL_MSG_SN_IDX] = 0;

      //      int tens = tx_poll_msg[7] - '0';
      //      int ones = tx_poll_msg[8] - '0';
      //      int number = tens * 10 + ones;
      //      number--;
      //      tens = number / 10;
      //      ones = number % 10;
      //      tx_poll_msg[7] = '0' + tens;
      //      tx_poll_msg[8] = '0' + ones;


      //      tx_poll_msg[8]--;
      //      if (tx_poll_msg[8] < '0') {
      //        tx_poll_msg[8] = '9';
      //        tx_poll_msg[7]--;
      //      }


      //updating rx_resp_msg for later checking with TX_resp_meg
      for (int i = 0; i < NUMBER_OF_TRANSMITTERS; i++) {
        if (memcmp(rx_buffer + 5, tx[i], 4) == 0) {
          memcpy(rx_resp_msg + 5, tx[i], 4 * sizeof(uint8_t));
        }
      }

      //      Serial.println();
      //      Serial.print("tx_poll_msg after 3: ");
      //      for (int i = 5; i < 9; i++) {
      //        Serial.print(tx_poll_msg[i], HEX);
      //      }
      //      Serial.print(" ");
      //      Serial.print("rx_resp_msg after 3: ");
      //      for (int i = 5; i < 9; i++) {
      //        Serial.print(rx_resp_msg[i], HEX);
      //      }


      //            Serial.println();
      //            Serial.print("rx_buffer2: ");
      //            for (int i = 5; i < 9; i++) {
      //              Serial.print(rx_buffer[i], HEX);
      //            }
      //      Serial.print(" ");
      //      Serial.print("current_TX2: ");
      //      Serial.print(current_tx);
      //
      //      Serial.print(" ");
      //      Serial.print("tx_poll_msg2: ");
      //      for (int i = 5; i < 9; i++) {
      //        Serial.print(tx_poll_msg[i], HEX);
      //      }
      //            // print tx_resp_msg buffer
      //            Serial.print(" ");
      //            Serial.print("rx_resp_msg2: ");
      //            for (int i = 5; i < 9; i++) {
      //              Serial.print(rx_resp_msg[i], HEX);
      //            }
      //            Serial.println();
      //
      //


      // check TX_resp_msg with rx_resp_msg
      if (memcmp(rx_buffer, rx_resp_msg, ALL_MSG_COMMON_LEN) == 0) {
        //        Serial.println("in");
        //        for (int i = 5; i < 9; i++) {
        //          Serial.printf("%c", rx_buffer[i]);
        //        }
        //        Serial.println();

        uint32_t poll_tx_ts, resp_rx_ts, poll_rx_ts, resp_tx_ts;
        int32_t rtd_init, rtd_resp;
        float clockOffsetRatio;

        /* Retrieve poll transmission and response reception timestamps. See NOTE 9 below. */
        poll_tx_ts = dwt_readtxtimestamplo32();
        resp_rx_ts = dwt_readrxtimestamplo32();

        /* Read carrier integrator value and calculate clock offset ratio. See NOTE 11 below. */
        clockOffsetRatio = ((float)dwt_readclockoffset()) / (uint32_t)(1 << 26);

        /* Get timestamps embedded in response message. */
        /*This line extracts the timestamp at which the responder received the 'Poll' message.
          The RESP_MSG_POLL_RX_TS_IDX constant is an index into the rx_buffer array where the 'Poll' reception timestamp is located. */
        resp_msg_get_ts(&rx_buffer[RESP_MSG_POLL_RX_TS_IDX], &poll_rx_ts);
        /*This line extracts the timestamp at which the responder transmitted the 'Response' message.
          The RESP_MSG_RESP_TX_TS_IDX constant is an index into the rx_buffer array where the 'Response' transmission timestamp is located. */
        resp_msg_get_ts(&rx_buffer[RESP_MSG_RESP_TX_TS_IDX], &resp_tx_ts);

        /* Compute time of flight and distance, using clock offset ratio to correct for differing local and remote clock rates */
        rtd_init = resp_rx_ts - poll_tx_ts;
        rtd_resp = resp_tx_ts - poll_rx_ts;

        tof = ((rtd_init - rtd_resp * (1 - clockOffsetRatio)) / 2.0) * DWT_TIME_UNITS;
        distance = tof * SPEED_OF_LIGHT;





        if (distance < 0) {
          goto restart;
        }




        double distances[NUMBER_OF_TRANSMITTERS] = { 0.0 };  // Initialize all distances to zero
        double offsets[50] = DISTANCE_OFFSETS;

        // Check with TX_resp_meg and identify which TX it is from
        for (int i = 0; i < NUMBER_OF_TRANSMITTERS; i++) {
//                    Serial.printf("%c%c%c%c, %c%c%c%c\n", rx_buffer[5], rx_buffer[6], rx_buffer[7], rx_buffer[8]
          //                        , tx[i][0], tx[i][1], tx[i][2], tx[i][3]);
          if (memcmp(rx_buffer + 5, tx[i], 4) == 0) {
            distances[i] = distance + offsets[i];
            if (i + 1 < 10) {
                            Serial.print( "[" + String(tx[0][0]) + String(tx[0][1]) + String(0) + String(i + 1) + "] (" + String(tx_coordinate[i].x) + ", " + String(tx_coordinate[i].y) + ", " + String(tx_coordinate[i].z) + ") " + String(distances[i]) + "m\n");
              udp.beginPacket(remoteIp, remoteUdpPort);
              udp.print("[" + String(tx[0][0]) + String(tx[0][1]) + String(0) + String(i + 1) + "] (" + String(tx_coordinate[i].x) + ", " + String(tx_coordinate[i].y) + ", " + String(tx_coordinate[i].z) + ") " + String(distances[i]) + "m");
              udp.endPacket();
            } else {
                            Serial.print( "[" + String(tx[0][0]) + String(tx[0][1]) + String(i + 1) + "] (" + String(tx_coordinate[i].x) + ", " + String(tx_coordinate[i].y) + ", " + String(tx_coordinate[i].z) + ") " + String(distances[i]) + "m\n");
              udp.beginPacket(remoteIp, remoteUdpPort);
              udp.print("[" + String(tx[0][0]) + String(tx[0][1]) + String(i + 1) + "] (" + String(tx_coordinate[i].x) + ", " + String(tx_coordinate[i].y) + ", " + String(tx_coordinate[i].z) + ") " + String(distances[i]) + "m");
              udp.endPacket();
            }
          }
        }
      }
    }
  } else {
restart:
    /* Clear RX error/timeout events in the DW IC status register. */
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR);
  }

  /* Execute a delay between ranging exchanges. */
  Sleep(RNG_DELAY_MS);
}

/*****************************************************************************************************************************************************
   NOTES:

   1. The single-sided two-way ranging scheme implemented here has to be considered carefully as the accuracy of the distance measured is highly
      sensitive to the clock offset error between the devices and the length of the response delay between frames. To achieve the best possible
      accuracy, this response delay must be kept as low as possible. In order to do so, 6.8 Mbps data rate is used in this example and the response
      delay between frames is defined as low as possible. The user is referred to User Manual for more details about the single-sided two-way ranging
      process.  NB:SEE ALSO NOTE 11.

      Initiator: |Poll TX| ..... |Resp RX|
      Responder: |Poll RX| ..... |Resp TX|
                     ^|P RMARKER|                    - time of Poll TX/RX
                                     ^|R RMARKER|    - time of Resp TX/RX

                         <--TDLY->                   - POLL_TX_TO_RESP_RX_DLY_UUS (RDLY-RLEN)
                                 <-RLEN->            - RESP_RX_TIMEOUT_UUS   (length of response frame)
                      <----RDLY------>               - POLL_RX_TO_RESP_TX_DLY_UUS (depends on how quickly responder can turn around and reply)


   2. The sum of the values is the TX to RX antenna delay, this should be experimentally determined by a calibration process. Here we use a hard coded
      value (expected to be a little low so a positive error will be seen on the resultant distance estimate). For a real production application, each
      device should have its own antenna delay properly calibrated to get good precision when performing range measurements.
   3. The frames used here are Decawave specific ranging frames, complying with the IEEE 802.15.4 standard data frame encoding. The frames are the
      following:
       - a poll message sent by the initiator to trigger the ranging exchange.
       - a response message sent by the responder to complete the exchange and provide all information needed by the initiator to compute the
         time-of-flight (distance) estimate.
      The first 10 bytes of those frame are common and are composed of the following fields:
       - byte 0/1: frame control (0x8841 to indicate a data frame using 16-bit addressing).
       - byte 2: sequence number, incremented for each new frame.
       - byte 3/4: PAN ID (0xDECA).
       - byte 5/6: destination address, see NOTE 4 below.
       - byte 7/8: source address, see NOTE 4 below.
       - byte 9: function code (specific values to indicate which message it is in the ranging process).
      The remaining bytes are specific to each message as follows:
      Poll message:
       - no more data
      Response message:
       - byte 10 -> 13: poll message reception timestamp.
       - byte 14 -> 17: response message transmission timestamp.
      All messages end with a 2-byte checksum automatically set by DW IC.
   4. Source and destination addresses are hard coded constants in this example to keep it simple but for a real product every device should have a
      unique ID. Here, 16-bit addressing is used to keep the messages as short as possible but, in an actual application, this should be done only
      after an exchange of specific messages used to define those short addresses for each device participating to the ranging exchange.
   5. This timeout is for complete reception of a frame, i.e. timeout duration must take into account the length of the expected frame. Here the value
      is arbitrary but chosen large enough to make sure that there is enough time to receive the complete response frame sent by the responder at the
      6.8M data rate used (around 200 µs).
   6. In a real application, for optimum performance within regulatory limits, it may be necessary to set TX pulse bandwidth and TX power, (using
      the dwt_configuretxrf API call) to per device calibrated values saved in the target system or the DW IC OTP memory.
   7. dwt_writetxdata() takes the full size of the message as a parameter but only copies (size - 2) bytes as the check-sum at the end of the frame is
      automatically appended by the DW IC. This means that our variable could be two bytes shorter without losing any data (but the sizeof would not
      work anymore then as we would still have to indicate the full length of the frame to dwt_writetxdata()).
   8. We use polled mode of operation here to keep the example as simple as possible but all status events can be used to generate interrupts. Please
      refer to DW IC User Manual for more details on "interrupts". It is also to be noted that STATUS register is 5 bytes long but, as the event we
      use are all in the first bytes of the register, we can use the simple dwt_read32bitreg() API call to access it instead of reading the whole 5
      bytes.
   9. The high order byte of each 40-bit time-stamps is discarded here. This is acceptable as, on each device, those time-stamps are not separated by
      more than 2**32 device time units (which is around 67 ms) which means that the calculation of the round-trip delays can be handled by a 32-bit
      subtraction.
   10. The user is referred to DecaRanging ARM application (distributed with EVK1000 product) for additional practical example of usage, and to the
       DW IC API Guide for more details on the DW IC driver functions.
   11. The use of the clock offset value to correct the TOF calculation, significantly improves the result of the SS-TWR where the remote
       responder unit's clock is a number of PPM offset from the local initiator unit's clock.
       As stated in NOTE 2 a fixed offset in range will be seen unless the antenna delay is calibrated and set correctly.
   12. In this example, the DW IC is put into IDLE state after calling dwt_initialise(). This means that a fast SPI rate of up to 20 MHz can be used
       thereafter.
   13. Desired configuration by user may be different to the current programmed configuration. dwt_configure is called to set desired
       configuration.*/

//#include "dw3000.h"
//#include <cmath>
//#include <math.h>
//#include <vector>
//#include <algorithm>
//#include "dw3000_config_options.h"
//#include <WiFi.h>
//#include <WiFiUdp.h>
//
//extern dwt_txconfig_t txconfig_options_ch9;
//
////#define PIN_RST 25
////#define PIN_IRQ 26
////#define PIN_SS 5
//
//#define PIN_RST 0
//#define PIN_IRQ 1
//#define PIN_SS 7
//
//#define RNG_DELAY_MS 0
//#define TX_ANT_DLY 16385
//#define RX_ANT_DLY 16385
//#define ALL_MSG_COMMON_LEN 10
//#define ALL_MSG_SN_IDX 2
//#define RESP_MSG_POLL_RX_TS_IDX 10
//#define RESP_MSG_RESP_TX_TS_IDX 14
//#define RESP_MSG_TS_LEN 4
//#define POLL_TX_TO_RESP_RX_DLY_UUS 240
//#define RESP_RX_TIMEOUT_UUS 400
//
//#define NUMBER_OF_TRANSMITTERS 4 //!!!
//#define rx_to_upload 1 //!!!
//
//#define OFFSET1 0.29
//#define OFFSET2 0.31
//#define OFFSET3 0.3
//#define OFFSET4 0.31
//#define OFFSET5 0.23
//#define OFFSET6 0.31
//#define OFFSET7 0.23
//#define OFFSET8 0.0
//#define OFFSET9 0.0
//#define OFFSET10 0.0
//#define OFFSET11 0.25
//#define OFFSET12 0.25
//#define OFFSET13 0.25
//#define OFFSET14 0.25
//#define OFFSET15 0.25
//#define OFFSET16 0.25
//#define OFFSET17 0.25
//#define OFFSET18 0.25
//#define OFFSET19 0.25
//#define OFFSET20 0.25
//#define OFFSET21 0.25
//#define OFFSET22 0.25
//#define OFFSET23 0.25
//#define OFFSET24 0.25
//#define OFFSET25 0.25
//#define OFFSET26 0.25
//#define OFFSET27 0.25
//#define OFFSET28 0.25
//#define OFFSET29 0.25
//#define OFFSET30 0.25
//#define OFFSET31 0.25
//#define OFFSET32 0.25
//#define OFFSET33 0.25
//#define OFFSET34 0.25
//#define OFFSET35 0.25
//#define OFFSET36 0.25
//#define OFFSET37 0.25
//#define OFFSET38 0.25
//#define OFFSET39 0.25
//#define OFFSET40 0.25
//#define OFFSET41 0.25
//#define OFFSET42 0.25
//#define OFFSET43 0.25
//#define OFFSET44 0.25
//#define OFFSET45 0.25
//#define OFFSET46 0.25
//#define OFFSET47 0.25
//#define OFFSET48 0.25
//#define OFFSET49 0.25
//#define OFFSET50 0.25
//
//      Initiator: |Poll TX| ..... |Resp RX|
//      Responder: |Poll RX| ..... |Resp TX|
//                     ^|P RMARKER|                    - time of Poll TX/RX
//                                     ^|R RMARKER|    - time of Resp TX/RX
//
//                         <--TDLY->                   - POLL_TX_TO_RESP_RX_DLY_UUS (RDLY-RLEN)
//                                 <-RLEN->            - RESP_RX_TIMEOUT_UUS   (length of response frame)
//                      <----RDLY------>               - POLL_RX_TO_RESP_TX_DLY_UUS (depends on how quickly responder can turn around and reply)
//
//
//   2. The sum of the values is the TX to RX antenna delay, this should be experimentally determined by a calibration process. Here we use a hard coded
//      value (expected to be a little low so a positive error will be seen on the resultant distance estimate). For a real production application, each
//      device should have its own antenna delay properly calibrated to get good precision when performing range measurements.
//   3. The frames used here are Decawave specific ranging frames, complying with the IEEE 802.15.4 standard data frame encoding. The frames are the
//      following:
//       - a poll message sent by the initiator to trigger the ranging exchange.
//       - a response message sent by the responder to complete the exchange and provide all information needed by the initiator to compute the
//         time-of-flight (distance) estimate.
//      The first 10 bytes of those frame are common and are composed of the following fields:
//       - byte 0/1: frame control (0x8841 to indicate a data frame using 16-bit addressing).
//       - byte 2: sequence number, incremented for each new frame.
//       - byte 3/4: PAN ID (0xDECA).
//       - byte 5/6: destination address, see NOTE 4 below.
//       - byte 7/8: source address, see NOTE 4 below.
//       - byte 9: function code (specific values to indicate which message it is in the ranging process).
//      The remaining bytes are specific to each message as follows:
//      Poll message:
//       - no more data
//      Response message:
//       - byte 10 -> 13: poll message reception timestamp.
//       - byte 14 -> 17: response message transmission timestamp.
//      All messages end with a 2-byte checksum automatically set by DW IC.
//   4. Source and destination addresses are hard coded constants in this example to keep it simple but for a real product every device should have a
//      unique ID. Here, 16-bit addressing is used to keep the messages as short as possible but, in an actual application, this should be done only
//      after an exchange of specific messages used to define those short addresses for each device participating to the ranging exchange.
//   5. This timeout is for complete reception of a frame, i.e. timeout duration must take into account the length of the expected frame. Here the value
//      is arbitrary but chosen large enough to make sure that there is enough time to receive the complete response frame sent by the responder at the
//      6.8M data rate used (around 200 µs).
//   6. In a real application, for optimum performance within regulatory limits, it may be necessary to set TX pulse bandwidth and TX power, (using
//      the dwt_configuretxrf API call) to per device calibrated values saved in the target system or the DW IC OTP memory.
//   7. dwt_writetxdata() takes the full size of the message as a parameter but only copies (size - 2) bytes as the check-sum at the end of the frame is
//      automatically appended by the DW IC. This means that our variable could be two bytes shorter without losing any data (but the sizeof would not
//      work anymore then as we would still have to indicate the full length of the frame to dwt_writetxdata()).
//   8. We use polled mode of operation here to keep the example as simple as possible but all status events can be used to generate interrupts. Please
//      refer to DW IC User Manual for more details on "interrupts". It is also to be noted that STATUS register is 5 bytes long but, as the event we
//      use are all in the first bytes of the register, we can use the simple dwt_read32bitreg() API call to access it instead of reading the whole 5
//      bytes.
//   9. The high order byte of each 40-bit time-stamps is discarded here. This is acceptable as, on each device, those time-stamps are not separated by
//      more than 2**32 device time units (which is around 67 ms) which means that the calculation of the round-trip delays can be handled by a 32-bit
//      subtraction.
//   10. The user is referred to DecaRanging ARM application (distributed with EVK1000 product) for additional practical example of usage, and to the
//       DW IC API Guide for more details on the DW IC driver functions.
//   11. The use of the clock offset value to correct the TOF calculation, significantly improves the result of the SS-TWR where the remote
//       responder unit's clock is a number of PPM offset from the local initiator unit's clock.
//       As stated in NOTE 2 a fixed offset in range will be seen unless the antenna delay is calibrated and set correctly.
//   12. In this example, the DW IC is put into IDLE state after calling dwt_initialise(). This means that a fast SPI rate of up to 20 MHz can be used
//       thereafter.
//   13. Desired configuration by user may be different to the current programmed configuration. dwt_configure is called to set desired
//       configuration.*/
