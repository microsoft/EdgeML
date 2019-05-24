// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "config.h"
#include "predict.h"
#include "model.h"

using namespace model;

void setup() {
	Serial.begin(115200);
	return;
}

void loop() {

#ifdef ACCURACY
	accuracy();
#endif

#ifdef PREDICTIONTIME
	predictionTime();
#endif

	return;
}

void accuracy() {

	welcome();

	while (true) {
		unsigned long startTime = micros();

		int classID = predict();

		unsigned long elapsedTime = micros() - startTime;

		Serial.println(classID);
		Serial.println(elapsedTime);
	}

	return;
}

void predictionTime() {
	unsigned long totalTime = 0;
	int iterations = 0;

	while(true) {
		Serial.print(iterations + 1);
		Serial.print(": ");

		unsigned long startTime = micros();

		int classID = predict();

		unsigned long elapsedTime = micros() - startTime;
		
		if ((classID + 1) == Y)
			Serial.println("Correct prediction");
		else
			Serial.println("WARNING: Incorrect prediction");

		totalTime += elapsedTime;
		iterations++;

		if (iterations % 100 == 0) {
			Serial.println("\n------------------------");
			Serial.println("Average prediction time:");
			Serial.println((float)totalTime / iterations);
			Serial.println("------------------------\n");
		}
	}

	return;
}

MYINT getIntFeature(MYINT i) {
#ifdef ACCURACY
	char buff[13];
	while (!Serial.available())
		;
	Serial.readBytes(buff, 13);
	double f = (float)(atof(buff));
#endif

#ifdef PREDICTIONTIME
	double f = ((float) pgm_read_float_near(&X[i]));
#endif

	double f_int = ldexp(f, -scaleOfX);
	return (MYINT)(f_int);
}

MYINT getIntFeatureOld(MYINT i) {
#ifdef ACCURACY
#ifdef INT16
	char buff[10];
	while (!Serial.available())
		;
	Serial.readBytes(buff, 10);
	return (MYINT)(atol(buff));
#endif
#ifdef INT32
	char buff[14];
	while (!Serial.available())
		;
	Serial.readBytes(buff, 14);
	return (MYINT)(atoll(buff));
#endif
#endif

#ifdef PREDICTIONTIME
#ifdef INT16
	return ((MYINT) pgm_read_word_near(&X[i]));
#endif
#ifdef INT32
	return ((MYINT) pgm_read_dword_near(&X[i]));
#endif
#endif
}

float getFloatFeature(MYINT i) {
#ifdef ACCURACY
	char buff[13];
	while (!Serial.available())
		;
	Serial.readBytes(buff, 13);
	return (float)(atof(buff));
#endif

#ifdef PREDICTIONTIME
	return ((float) pgm_read_float_near(&X[i]));
#endif
}

// Setup serial communication with the computer
void welcome() {

	// Wait for input
	while (!Serial.available())
		;

	// Match input with the message
	const char *syncMsg = "fixed";
	int len = strlen(syncMsg);
	byte index = 0;
	do {
		char ch = Serial.read();
		if (ch == syncMsg[index])
			index += 1;
	} while (index < len);

	Serial.println("point");

	return;
}
