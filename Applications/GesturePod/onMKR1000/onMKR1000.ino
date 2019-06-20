/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 * 
 * ============================
 * Connections to BLE
 * ============================
 * MKR1000   --------->    HM10
 * VCC                     VCC
 * GND                     GND
 * 0                       RX
 * 1                       TX
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
#include "src/protoNN.h"
#include "src/featurizer.h"
#include "src/utils.h"
#include "src/lib/I2CDev.h"
#include "src/lib/MPU6050_6Axis_MotionApps20.h"
#include "src/comm_module.h"
// include Arduino.h before wiring_private.h
#include <Arduino.h>   

/* Arduino Wire library is required if I2Cdev
 * I2CDEV_ARDUINO_WIRE implementation is used
 * in I2Cdev.h
 */
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
    #include "Wire.h"
#endif

// To debug BLE
// #define __DEBUG_BLE__

FIFOCircularQ<float, 400> normAX, normAY, normAZ; 
FIFOCircularQ<float, 400> normGX, normGY, normGZ;
/* Featurizer class constructor
 * Set of hand crafted features for gesture recognition with IMU data - EdgeML
 */
Featurizer featurizer(BUCKET_WIDTH, &normAX, &normAY, &normAZ, 
                      &normGX, &normGY, &normGZ);
/* ProtoNN class constructor. 
 * ProtoNN is a Multi-class classification algorithm - EdgeML
 */
ProtoNNF predictor1;
// Voting class constructor takes as input the (index of max no of labels + 1)
Vote Vote1(10);
int VOTE_RESULT;
int COUNT_AFTER_RESET;
// BLE class constructor for communicating gestures through BLE
BLE BLE_module;
/* 
 * Used for min-max normalization.
 * These values may have to be changed depending on MPU
 * Values defined in config.h
 */
Vector3D<int16_t> minAcc(MIN_ACC, MIN_ACC, MIN_ACC);
Vector3D<int16_t> maxAcc(MAX_ACC, MAX_ACC, MAX_ACC);
Vector3D<int16_t> minGyr(MIN_GYR_X, MIN_GYR_Y, MIN_GYR_Z);
Vector3D<int16_t> maxGyr(MAX_GYR_X, MAX_GYR_Y, MAX_GYR_Z);
/*
 * For MPU6050 and DMP usage
 */
MPU6050 mpu1;
uint16_t fifoPacketSize;
uint8_t fifoBuffer[64];
uint16_t currBucketBufferSize = 0;
uint16_t numNewReadings = 0;
/*
 * Post gesture processing
 */
const int L = 6;
int scores[L];
unsigned long LAST_SENT_TIME = 0;
unsigned long samplesAfterReset = 0;
// Gestures are mapped to classes - Do not change ordering!
const char *GESTURE_TO_COMMUNICATE[10] = {"", "", "", "double_tap", 
                                          "right_twist", "left_twist", "",
                                          "twirl", "", "double_swipe"};

void setup() {
    bool initSuccess = true;
    /* join I2C bus (I2Cdev library doesn't do this automatically)*/
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
    if (predictor1.getErrorCode()){
        Serial.print("ProtoNNF initialization failed with code ");
        Serial.println(predictor1.getErrorCode());
        initSuccess = false;
    }
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
    digitalWrite(LED_PIN, HIGH);
    // Setup() for BLE
    if(!BLE_module.init())
        Serial.println("BLE Setup Complete. Please connect");
    else{
        /* If BLE setup fails, you cannot receive information through 
         * BLE. However, predictions will still be displayed on console.
         */
        Serial.println("Failed to Setup BLE");
        Serial.println("Use \"#define __DEBUG_BLE__\" to debug");
    }
    COUNT_AFTER_RESET = 0;
    VOTE_RESULT = 1;
}

void loop() {
    uint16_t fifoCount = mpu1.getFIFOCount();
    uint8_t mpuIntStatus = mpu1.getIntStatus();
    Vector3D<float> normAcc, normGyr;
    // check for overflow (this should never happen)
    if ((mpuIntStatus & 0x10) || fifoCount == 1024){
        // Buffer time 
        // 350ms @66hz  (empirical) i.e. 13ms between samples
        // 195ms @100Hz (empirical) i.e. 8ms between samples
        // 73ms  @200Hz (empirical) i.e. 3ms between samples
        Serial.print(millis());
        Serial.println(": FIFO Overflow!");
        mpu1.resetFIFO();
        samplesAfterReset = 0;
        numNewReadings = 0;
    // otherwise, check for DMP data ready (this should happen frequently)
    } else if ((mpuIntStatus & 0x02)){
        while(fifoCount < fifoPacketSize) fifoCount = mpu1.getFIFOCount();
        while(fifoCount >= fifoPacketSize) {
            mpu1.getFIFOBytes(fifoBuffer, fifoPacketSize);
            VectorInt16 acc__, gyr__;
            mpu1.dmpGetAccel(&acc__, fifoBuffer);
            mpu1.dmpGetGyro(&gyr__, fifoBuffer);
            //VectorInt16 from jrowberg to generic Vector3D
            Vector3D<int16_t> acc(acc__.x, acc__.y, acc__.z);
            Vector3D<int16_t> gyr(gyr__.x, gyr__.y, gyr__.z);
            minMaxNormalize(&acc, &minAcc, &maxAcc, &normAcc);
            minMaxNormalize(&gyr, &minGyr, &maxGyr, &normGyr);
            normAX.forceAdd(normAcc.x); normGX.forceAdd(normGyr.x);
            normAY.forceAdd(normAcc.y); normGY.forceAdd(normGyr.y);
            normAZ.forceAdd(normAcc.z); normGZ.forceAdd(normGyr.z);
            fifoCount = mpu1.getFIFOCount();
            numNewReadings += 1;
            samplesAfterReset += 1;
        }
    }
     
    /* format of feature vector:[indexPosEdge, countPosEdge, countNegEdge,
     * indexNegEdge, ax(20buckets), ay(20buckets), az(20buckets),
     * gx(20buckets), gy(20buckets), gz(20buckets)]
     */
    int featureVector[FEATURE_LENGTH]={0}; 
    float featureVectorF[FEATURE_LENGTH]={0}; 
    // For every STRIDE number of readings, do a prediction
    // In case of FIFO overflow, wait for 250 samples, before predicting
    if(samplesAfterReset < 250){
        if(numNewReadings == STRIDE) numNewReadings = 0;
    } else if(numNewReadings == STRIDE){
        samplesAfterReset = 401;

        int featureStatus = featurizer.featurize(featureVector); 
        // Since predictor expects a float type. 
        // But feature computation with floats is expensive.
        for(int i = 0; i < FEATURE_LENGTH; i++){
            featureVectorF[i] = featureVector[i];
        }
        int result = predictor1.predict(featureVectorF, 
                                        FEATURE_LENGTH, 
                                        scores);
        
        // Printing of Scores to Console
        Serial.print("Result: "); Serial.print(result);
        Serial.print(",Score:") ; Serial.print(scores[result]);Serial.print(",");

        // Voting to get rid of stray gestures
        Vote1.forcePush(result);
        VOTE_RESULT = Vote1.result();
        Serial.print(" Vote Result: "); Serial.println(VOTE_RESULT);

        /* Send a new gesture every 2 seconds. This is to ensure phone
         * text to speech(TTS) conveys something meaningful.
         */
        if (millis() - LAST_SENT_TIME >= 2000){
            if((VOTE_RESULT == 3)||(VOTE_RESULT == 4)||(VOTE_RESULT == 5)||(VOTE_RESULT == 7)||(VOTE_RESULT == 9)){
                digitalWrite(LED_PIN, HIGH);
                Serial.print("Communicating gesture: ");
                Serial.println(GESTURE_TO_COMMUNICATE[VOTE_RESULT]);
                BLE_module.writeln(GESTURE_TO_COMMUNICATE[VOTE_RESULT]);
                LAST_SENT_TIME = millis();
                digitalWrite(LED_PIN, LOW);
            }
        }
        numNewReadings = 0;
    }
}
