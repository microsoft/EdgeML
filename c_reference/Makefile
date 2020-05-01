# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

include config.mk

SRC_DIR=src
TEST_DIR=tests

all:  
	$(MAKE) -C $(SRC_DIR)
	$(MAKE) -C $(TEST_DIR)

.PHONY: clean cleanest

clean: 
	rm -f *.o *.gch
	$(MAKE) -C $(SRC_DIR) clean
	$(MAKE) -C $(TEST_DIR) clean

cleanest: clean
	rm *~
	$(MAKE) -C $(SRC_DIR) cleanest
	$(MAKE) -C $(TEST_DIR) cleanest
