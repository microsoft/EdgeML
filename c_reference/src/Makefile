# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

include ../config.mk

INCLUDE_DIR=../include
IFLAGS = -I $(INCLUDE_DIR)

all: utils.o fastgrnn.o classifier.o rnnpool.o

utils.o: utils.c
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

fastgrnn.o: fastgrnn.c
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

classifier.o: classifier.c 
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

rnnpool.o: rnnpool.c
	$(CC) -o $@ $(IFLAGS) $(CFLAGS) -c $^

.PHONY: clean cleanest

clean: 
	rm -f *.o *.gch

cleanest: clean
	rm *~
