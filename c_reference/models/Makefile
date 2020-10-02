# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

include ../config.mk

INCLUDE_DIR=../include
IFLAGS = -I $(INCLUDE_DIR)

all: quantized_face_detection.o quantized_face_detection_fast.o

quantized_face_detection.o: quantized_face_detection.c
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

quantized_face_detection_fast.o: quantized_face_detection_fast.c
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

.PHONY: clean cleanest

clean: 
	rm -f *.o *.gch

cleanest: clean
	rm *~
