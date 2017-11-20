# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

include ../../../config.mk

SOURCE_DIR=../../../src

COMMON_DIR=$(SOURCE_DIR)/common
PROTONN_DIR=$(SOURCE_DIR)/ProtoNN
IFLAGS = -I ../../../eigen -I$(MKL_ROOT)/include \
	 -I$(COMMON_DIR) -I$(PROTONN_DIR)

all: ../../../ProtoNNTrainDriver.o

../../../ProtoNNTrainDriver.o: ProtoNNTrainDriver.cpp 
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

.PHONY: clean cleanest

clean:
	rm -f ../../../ProtoNNTrainDriver.o

cleanest: clean	
	rm *~
