/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Helper methods used in various parts.
 */
#include "utils.h"

/**
 * Busy waits until a input from serial is received.
 * Once a character is available, reads and returns it. 
 */
char waitForSerialInput(){
	while(!(Serial.available() > 0));
	return Serial.read();
	
}