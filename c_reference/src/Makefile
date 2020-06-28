# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

include ../config.mk

INCLUDE_DIR=../include
CMSISDSP_DIR=../CMSIS/DSP/Include
CMSISCORE_DIR=../CMSIS/Core/Include
IFLAGS = -I $(INCLUDE_DIR)
CMSISIFLAGS = -I $(CMSISCORE_DIR) -I $(CMSISDSP_DIR)

all: utils.o fastgrnn.o classifier.o rnnpool.o quantized_utils.o quantized_fastgrnn.o quantized_rnnpool.o quantized_mbconv.o

utils.o: utils.c
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

fastgrnn.o: fastgrnn.c
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

classifier.o: classifier.c 
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

rnnpool.o: rnnpool.c
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

quantized_utils.o: quantized_utils.c
	$(CC) -o $@ $(CMSISIFLAGS) $(IFLAGS) $(CFLAGS) -c $^

quantized_fastgrnn.o: quantized_fastgrnn.c
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

quantized_rnnpool.o: quantized_rnnpool.c
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

quantized_mbconv.o: quantized_mbconv.c
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

.PHONY: clean cleanest

clean: 
	rm -f *.o *.gch

cleanest: clean
	rm *~
