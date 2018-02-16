#include "predict.h"

void setup() {
	Serial.begin(115200);
	return;
}

void loop() {

	welcome();

	while (true)
		predict(Serial);

	return;
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
