# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


include config.mk

SOURCE_DIR=src
DRIVER_DIR=drivers

COMMON_DIR=$(SOURCE_DIR)/common
PROTONN_DIR=$(SOURCE_DIR)/ProtoNN
BONSAI_DIR=$(SOURCE_DIR)/Bonsai

IFLAGS = -I eigen/ -I$(MKL_ROOT)/include \
	 -I$(COMMON_DIR) -I$(PROTONN_DIR) -I$(BONSAI_DIR)

all: ProtoNNTrain ProtoNNPredict BonsaiTrain BonsaiPredict Bonsai #ProtoNNIngestTest BonsaiIngestTest 

libcommon.so: $(COMMON_INCLUDES)
	$(MAKE) -C $(SOURCE_DIR)/common

libProtoNN.so: $(PROTONN_INCLUDES)
	$(MAKE) -C $(SOURCE_DIR)/ProtoNN

libBonsai.so: $(BONSAI_INCLUDES)
	$(MAKE) -C $(SOURCE_DIR)/Bonsai

ProtoNNTrainDriver.o:
	$(MAKE) -C $(DRIVER_DIR)/ProtoNN/trainer

ProtoNNPredictDriver.o:
	$(MAKE) -C $(DRIVER_DIR)/ProtoNN/predictor

BonsaiLocalDriver.o:
	$(MAKE) -C $(DRIVER_DIR)/Bonsai/local

BonsaiTrainDriver.o:
	$(MAKE) -C $(DRIVER_DIR)/Bonsai/trainer

BonsaiPredictDriver.o:
	$(MAKE) -C $(DRIVER_DIR)/Bonsai/predictor

#ProtoNNIngestTest.o BonsaiIngestTest.o:

ProtoNNTrain: ProtoNNTrainDriver.o libcommon.so libProtoNN.so
	$(CC) -o $@ $^ $(CFLAGS) $(MKL_PAR_LDFLAGS) $(CILK_LDFLAGS)

ProtoNNPredict: ProtoNNPredictDriver.o libcommon.so libProtoNN.so
	$(CC) -o $@ $^ $(CFLAGS) $(MKL_PAR_LDFLAGS) $(CILK_LDFLAGS)

#ProtoNNIngestTest: ProtoNNIngestTest.o libcommon.so libProtoNN.so
#	$(CC) -o $@ $^ $(CFLAGS) $(MKL_PAR_LDFLAGS) $(CILK_LDFLAGS)

Bonsai: BonsaiLocalDriver.o libcommon.so libBonsai.so
	$(CC) -o $@ $^ $(CFLAGS) $(MKL_SEQ_LDFLAGS) $(CILK_LDFLAGS)

BonsaiTrain: BonsaiTrainDriver.o libcommon.so libBonsai.so
	$(CC) -o $@ $^ $(CFLAGS) $(MKL_SEQ_LDFLAGS) $(CILK_LDFLAGS)

BonsaiPredict: BonsaiPredictDriver.o libcommon.so libBonsai.so
	$(CC) -o $@ $^ $(CFLAGS) $(MKL_SEQ_LDFLAGS) $(CILK_LDFLAGS)

#BonsaiIngestTest: BonsaiIngestTest.o libcommon.so libBonsai.so
#	$(CC) -o $@ $^ $(CFLAGS) $(MKL_PAR_LDFLAGS) $(CILK_LDFLAGS)


.PHONY: clean cleanest

clean: 
	rm -f *.o
	$(MAKE) -C $(SOURCE_DIR)/common clean
	$(MAKE) -C $(SOURCE_DIR)/ProtoNN clean
	$(MAKE) -C $(SOURCE_DIR)/Bonsai clean
	$(MAKE) -C $(DRIVER_DIR)/ProtoNN/trainer clean
	$(MAKE) -C $(DRIVER_DIR)/ProtoNN/predictor clean
	$(MAKE) -C $(DRIVER_DIR)/Bonsai/trainer clean
	$(MAKE) -C $(DRIVER_DIR)/Bonsai/predictor clean

cleanest: clean
	rm -f ProtoNN ProtoNNPredict ProtoNNIngestTest BonsaiIngestTest Bonsai
	$(MAKE) -C $(SOURCE_DIR)/common cleanest
	$(MAKE) -C $(SOURCE_DIR)/ProtoNN cleanest
	$(MAKE) -C $(SOURCE_DIR)/Bonsai cleanest
	$(MAKE) -C $(DRIVER_DIR)/ProtoNN/trainer cleanest
	$(MAKE) -C $(DRIVER_DIR)/ProtoNN/predictor cleanest
	$(MAKE) -C $(DRIVER_DIR)/Bonsai/trainer cleanest
	$(MAKE) -C $(DRIVER_DIR)/Bonsai/predictor cleanest
