/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 * 
 * Connections to MPU6050 using I2C protocol.
 */

/*
 * Parts of this code are borrowed from Jeff Rowberg's I2Cdev device library.
 * Below are the terms of license for I2Cdev device library:
 */

/*
 * I2Cdev device library code is placed under the MIT license
 * Copyright (c) 2012 Jeff Rowberg
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "config.h"
#include "src/utils.h"
#include "src/lib/I2CDev.h"
#include "src/lib/MPU6050_6Axis_MotionApps20.h"
// include Arduino.h before wiring_private.h
#include <Arduino.h>   

/* Arduino Wire library is required if I2Cdev
 * I2CDEV_ARDUINO_WIRE implementation is used
 * in I2Cdev.h
 */
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
    #include "Wire.h"
#endif

MPU6050 mpu1;
uint16_t fifoPacketSize;
uint8_t fifoBuffer[64];
uint16_t currBucketBufferSize = 0;
void csvPrint(VectorInt16 *a, char sep=',');


void setup() {
	/* join I2C bus (I2Cdev library doesn't do this automatically) */
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
		Wire.begin();
		/* 400kHz I2C clock. Comment this line if having
		 * compilation difficulties
		 */
		Wire.setClock(400000);
#elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
		Fastwire::setup(400, true);
#endif
	Serial.begin(BAUD_RATE);
	delay(2000);
	Serial.println("Initialize MPU..");
	mpu1.initialize();
	uint8_t devStatus = mpu1.dmpInitialize();
	if (devStatus == 0){
		Serial.print("Enabling DMP...");
		mpu1.setDMPEnabled(true);
		Serial.println("Done.");
	} else {
		Serial.print("DMP Initialization failed with code ");
        Serial.println(devStatus);
        while(true){
            delay(200);
            Serial.print("DMP Initialization failed with code ");
            Serial.println(devStatus);
        }
    }
	fifoPacketSize = mpu1.dmpGetFIFOPacketSize();
	pinMode(LED_PIN, OUTPUT);
}


void loop() {
	uint16_t fifoCount = mpu1.getFIFOCount();
	uint8_t mpuIntStatus = mpu1.getIntStatus();
	// check for overflow (this should never happen)
	if ((mpuIntStatus & 0x10) || fifoCount == 1024){
		// Buffer time 
        // 350ms @66hz  (empirical) i.e. 13ms between samples
        // 195ms @100Hz (empirical) i.e. 8ms between samples
        // 73ms  @200Hz (empirical) i.e. 3ms between samples
		Serial.print(millis());
		Serial.println(": FIFO Overflow!");
		mpu1.resetFIFO();
	// otherwise, check for DMP data ready (this should happen frequently)
	} else if ((mpuIntStatus & 0x02)){
		while(fifoCount < fifoPacketSize) fifoCount = mpu1.getFIFOCount();
		while(fifoCount >= fifoPacketSize) {
			mpu1.getFIFOBytes(fifoBuffer, fifoPacketSize);
			VectorInt16 acc__, gyr__;
			mpu1.dmpGetAccel(&acc__, fifoBuffer);
			mpu1.dmpGetGyro(&gyr__, fifoBuffer);
			//VectorInt16 from jrowberg to generic Vector3D
			Serial.print(millis());
			Serial.print(",");
			csvPrint(&acc__);
			csvPrint(&gyr__);
			Serial.println();
			fifoCount = mpu1.getFIFOCount();
		}
	}
}


void csvPrint(VectorInt16 *a, char sep){
	Serial.print(a->x);
	Serial.print(sep);
	Serial.print(a->y);
	Serial.print(sep);
	Serial.print(a->z);
	Serial.print(sep);
}