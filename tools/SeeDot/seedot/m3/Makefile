# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

LDFLAGS=-lm -ldl
CC=gcc
CFLAGS=-g -fPIC -O3 -w

INCLUDE_DIR=./
IFLAGS=-I $(INCLUDE_DIR)

all: predictor

predictor:	main.c predict.o quantized_utils.o quantized_mbconv.o
	$(CC) -o $@ $^ $(CFLAGS) $(IFLAGS) -lm

predict.o:	predict.c
	$(CC) -o $@ $(CFLAGS) $(IFLAGS) -c $^

quantized_utils.o:	quantized_utils.c
	$(CC) -o $@ $(CFLAGS) $(IFLAGS) -c $^

quantized_mbconv.o:	quantized_mbconv.c
	$(CC) -o $@ $(CFLAGS) $(IFLAGS) -c $^

clean:
	rm *.o
	rm predictor

run: predictor
	./predictor
