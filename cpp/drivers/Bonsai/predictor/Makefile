# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

include ../../../config.mk

SOURCE_DIR=../../../src

COMMON_DIR=$(SOURCE_DIR)/common
PROTONN_DIR=$(SOURCE_DIR)/Bonsai
IFLAGS = -I ../../../eigen -I$(MKL_ROOT)/include \
	 -I$(COMMON_DIR) -I$(PROTONN_DIR)

all: ../../../BonsaiPredictDriver.o

../../../BonsaiPredictDriver.o: BonsaiPredictDriver.cpp 
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

.PHONY: clean cleanest

clean:
	rm -f ../../../BonsaiPredictDriver.o

cleanest: clean	
	rm *~
