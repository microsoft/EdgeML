/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * MKR1000 has ability to add extra hardware serial ports
 * This sketch has the following serial interfaces:
 *   Serial2 - Extra serial port on D0, D1 (Sercom 3)
 *   Good explanation of sercom funcationality here: 
 *   https://learn.adafruit.com/using-atsamd21-sercom-to-add-more-spi-i2c-serial-ports/muxing-it-up
 *
 * Declaration for communication modules and BLE
 */

#ifndef __COMMUNICATION__
#define __COMMUNICATION__

// include Arduino.h before wiring_private.h
#include <Arduino.h> 
// wiring private is for BLE                           
#include <wiring_private.h>
#include <Arduino.h>
#include "./../config.h"

// Base class for all communication modules
class CommunicationModule{
public:
    virtual int init();
    virtual int write(const char* x);
    virtual int writeln(const char* x);
};

// BLE Class declaration. 
// BLE class is Concrete class derived from CommunicationModule
class BLE: public CommunicationModule{
private:
    int bleSetup();
public:
    int init();
    int write(const char* x);
    int writeln(const char* x);
};

#endif //__COMMUNICATION__