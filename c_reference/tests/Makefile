# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

include ../config.mk

INCLUDE_DIR=../include
SRC_DIR=../src
IFLAGS = -I $(INCLUDE_DIR)

all: test_fastgrnn_lr test_rnnpool

FASTGRNN_DIR=fastgrnn
test_fastgrnn_lr: $(FASTGRNN_DIR)/test_fastgrnn_lr.c $(SRC_DIR)/utils.o $(SRC_DIR)/fastgrnn.o $(SRC_DIR)/classifier.o
	$(CC) -o $@ $^ $(IFLAGS) $(CFLAGS) -lm

RNNPOOL_DIR=rnnpool
test_rnnpool: $(RNNPOOL_DIR)/test_rnnpool.c  $(SRC_DIR)/utils.o $(SRC_DIR)/fastgrnn.o $(SRC_DIR)/rnnpool.o
	$(CC) -o $@ $^ $(IFLAGS) $(CFLAGS) -lm


.PHONY: clean cleanest

clean: 
	rm -f *.o *.gch test_fastgrnn_lr test_rnnpool

cleanest: clean
	rm *~
