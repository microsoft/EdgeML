# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

include ../config.mk

INCLUDE_DIR=../include
IFLAGS=-I $(INCLUDE_DIR)

all: dscnn.o conv1d.o utils.o fastgrnn.o classifier.o rnnpool.o quantized_utils.o quantized_fastgrnn.o quantized_rnnpool.o quantized_mbconv.o rnn_bricked.o

dscnn.o : dscnn.c
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

conv1d.o : conv1d.c
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

utils.o: utils.c
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

fastgrnn.o: fastgrnn.c
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

classifier.o: classifier.c 
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

rnnpool.o: rnnpool.c
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

rnn_bricked.o: rnn_bricked.c
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

quantized_utils.o: quantized_utils.c
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

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
