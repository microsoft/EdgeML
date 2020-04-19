# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

include config.mk

INCLUDE_DIR=include
TEST_DIR=tests

IFLAGS = -I $(INCLUDE_DIR)

all: test_fastgrnn_lr

test_fastgrnn_lr: $(TEST_DIR)/test_fastgrnn_lr.c
	$(CC) -o $(TEST_DIR)/$@ $(IFLAGS) $(CFLAGS) $^ -lm

.PHONY: clean cleanest

clean: 
	rm -f *.o *.gch

cleanest: clean
	rm *~
