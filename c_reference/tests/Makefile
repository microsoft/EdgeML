# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

include ../config.mk

INCLUDE_DIR=../include
MODEL_DIR=../models
SRC_DIR=../src
IFLAGS = -I $(INCLUDE_DIR) -I $(MODEL_DIR)

all: test_fastgrnn_lr test_rnnpool test_quantized_utils test_quantized_fastgrnn test_quantized_rnnpool test_quantized_mbconv test_quantized_face_detection test_quantized_face_detection_fast

FASTGRNN_DIR=fastgrnn
test_fastgrnn_lr: $(FASTGRNN_DIR)/test_fastgrnn_lr.c $(SRC_DIR)/utils.o $(SRC_DIR)/fastgrnn.o $(SRC_DIR)/classifier.o
	$(CC) -o $@ $^ $(IFLAGS) $(CFLAGS) -lm
test_quantized_fastgrnn: $(FASTGRNN_DIR)/test_quantized_fastgrnn.c $(SRC_DIR)/quantized_utils.o $(SRC_DIR)/quantized_fastgrnn.o
	$(CC) -o $@ $^ $(IFLAGS) $(CFLAGS) -Wno-unused-variable -lm

RNNPOOL_DIR=rnnpool
test_rnnpool: $(RNNPOOL_DIR)/test_rnnpool.c  $(SRC_DIR)/utils.o $(SRC_DIR)/fastgrnn.o $(SRC_DIR)/rnnpool.o
	$(CC) -o $@ $^ $(IFLAGS) $(CFLAGS) -lm
test_quantized_rnnpool: $(RNNPOOL_DIR)/test_quantized_rnnpool.c $(SRC_DIR)/quantized_utils.o $(SRC_DIR)/quantized_fastgrnn.o $(SRC_DIR)/quantized_rnnpool.o
	$(CC) -o $@ $^ $(IFLAGS) $(CFLAGS) -Wno-unused-result -lm

UTILS_DIR=utils
test_quantized_utils: $(UTILS_DIR)/test_quantized_utils.c $(SRC_DIR)/quantized_utils.o
	$(CC) -o $@ $^ $(IFLAGS) $(CFLAGS) -lm

MBCONV_DIR=mbconv
test_quantized_mbconv: $(MBCONV_DIR)/test_quantized_mbconv.c $(SRC_DIR)/quantized_utils.o $(SRC_DIR)/quantized_mbconv.o
	$(CC) -o $@ $^ $(IFLAGS) $(CFLAGS) -Wno-unused-result -lm

FACE_DETECTION_DIR=face_detection
test_quantized_face_detection: $(FACE_DETECTION_DIR)/test_quantized_face_detection.c $(SRC_DIR)/quantized_utils.o $(SRC_DIR)/quantized_fastgrnn.o $(SRC_DIR)/quantized_rnnpool.o $(SRC_DIR)/quantized_mbconv.o $(MODEL_DIR)/quantized_face_detection.o
	$(CC) -o $@ $^ $(IFLAGS) $(CFLAGS) -Wno-unused-result -lm
test_quantized_face_detection_fast: $(FACE_DETECTION_DIR)/test_quantized_face_detection_fast.c $(SRC_DIR)/quantized_utils.o $(SRC_DIR)/quantized_fastgrnn.o $(SRC_DIR)/quantized_rnnpool.o $(SRC_DIR)/quantized_mbconv.o $(MODEL_DIR)/quantized_face_detection_fast.o
	$(CC) -o $@ $^ $(IFLAGS) $(CFLAGS) -Wno-unused-result -lm

.PHONY: clean cleanest

clean: 
	rm -f *.o *.gch test_fastgrnn_lr test_rnnpool test_quantized_utils test_quantized_fastgrnn test_quantized_rnnpool test_quantized_mbconv test_quantized_face_detection test_quantized_face_detection_fast

cleanest: clean
	rm *~
