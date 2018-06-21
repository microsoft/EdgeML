// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stdio.h>
#include "lstm.h"

int main(){
	printf("Running tests.\n");
	#ifdef __TEST_LSTM__
	printf("LSTM error Code: %u\n", runLSTMTests());
	#endif
	printf("Done.\n");
}