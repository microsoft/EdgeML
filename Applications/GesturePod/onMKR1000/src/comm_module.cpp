/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 * 
 */

#include "comm_module.h"

// BLE Definitions
// Serial2 pin and pad definitions (in Arduino files Variant.h & Variant.cpp)
// Pin description number for PIO_SERCOM on D1
#define PIN_SERIAL2_RX       (1ul)
// Pin description number for PIO_SERCOM on D0
#define PIN_SERIAL2_TX       (0ul)
// SERCOM pad 0 TX
#define PAD_SERIAL2_TX       (UART_TX_PAD_0)
// SERCOM pad 1 RX
#define PAD_SERIAL2_RX       (SERCOM_RX_PAD_1)

// Instantiate the extra Serial classes
Uart Serial2(&sercom3, PIN_SERIAL2_RX, PIN_SERIAL2_TX, PAD_SERIAL2_RX, PAD_SERIAL2_TX);

// Interrupt handler for SERCOM3
void SERCOM3_Handler(){
  Serial2.IrqHandler();
}
// End of BLE declarations

int BLE::init(){
    // Assign pins 0 & 1 SERCOM functionality
    pinPeripheral(0, PIO_SERCOM);   
    pinPeripheral(1, PIO_SERCOM);
    /* Do not change this for HM-10
     * For HC-05 use  HC_05_BAUD_RATE
     * Baud rates defined in config.h
    */
    Serial2.begin(HM_10_BAUD_RATE);
    // Setup() for BLE
    if(bleSetup()){ 
    // If successful  
        return 0;
    } else {
    // If not successful
        return 1;
    }
}

int BLE::write(const char* x){
    return (Serial2.print(x));
}

int BLE::writeln(const char* x){
    return (Serial2.println(x));
}

int BLE::bleSetup(){
#ifdef __DEBUG_BLE__
    bool wake = 0;
    Serial.println("Initial Setup : Please ensure device is not connected");
    Serial.println("Putting device to sleep");
    Serial2.println("AT+SLEEP");
    delay(1000);
    while(Serial2.available()){
        char c = Serial2.read();
        Serial.print(c);
    }
    //  Wake up Device:
    unsigned long startTime = millis();
    while(wake==0){
        Serial.println("Pinging HM-10 with \"AT\" ");
        Serial2.println("AT");
        Serial.println("Response: ");
        while(Serial2.available()){
            char c = Serial2.read();
            if((c == 'O') || (c =='+')){
                wake = 1;
                delay(10);
            }
            Serial.print(c);
        }
    }
    unsigned long endTime = millis();
    Serial.println("Time to respond to AT (in ms):" + String((endTime - startTime))); 
    // Get Address:
    Serial.println("Pinging HM-10 with \"AT+ADDR\" ");
    Serial2.println("AT+ADDR");
    Serial.print("Response: ");
    delay(50);
    while(Serial2.available()){
        char c = Serial2.read();
        Serial.print(c);
    }
#endif //__DEBUG_BLE__
    delay(500);
    return 1;
}