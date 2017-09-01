# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


include config.mk

COMMON_INCLUDES=$(SOURCE_DIR)/common
PROTONN_INCLUDES=$(SOURCE_DIR)/ProtoNN
BONSAI_INCLUDES=$(SOURCE_DIR)/Bonsai

IFLAGS=-I eigen/ -I$(MKL_ROOT)/include \
	 -I$(COMMON_INCLUDES) -I$(PROTONN_INCLUDES) -I$(BONSAI_INCLUDES)

all: ProtoNN Bonsai #ProtoNNIngestTest BonsaiIngestTest 

libcommon.so: $(COMMON_INCLUDES)
	$(MAKE) -C $(SOURCE_DIR)/common

libProtoNN.so: $(PROTONN_INCLUDES)
	$(MAKE) -C $(SOURCE_DIR)/ProtoNN

libBonsai.so: $(BONSAI_INCLUDES)
	$(MAKE) -C $(SOURCE_DIR)/Bonsai

ProtoNNLocalDriver.o: ProtoNNLocalDriver.cpp $(PROTONN_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

ProtoNNIngestTest.o: ProtoNNIngestTest.cpp $(PROTONN_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

BonsaiLocalDriver.o:BonsaiLocalDriver.cpp $(BONSAI_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

BonsaiIngestTest.o:BonsaiIngestTest.cpp $(BONSAI_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<


ProtoNN: ProtoNNLocalDriver.o libcommon.so libProtoNN.so
	$(CC) -o $@ $^ $(CFLAGS) $(MKL_PAR_LDFLAGS) $(CILK_LDFLAGS)

#ProtoNNIngestTest: ProtoNNIngestTest.o libcommon.so libProtoNN.so
#	$(CC) -o $@ $^ $(CFLAGS) $(MKL_PAR_LDFLAGS) $(CILK_LDFLAGS)

Bonsai: BonsaiLocalDriver.o libcommon.so libBonsai.so
	$(CC) -o $@ $^ $(CFLAGS) $(MKL_SEQ_LDFLAGS) $(CILK_LDFLAGS)

#BonsaiIngestTest: BonsaiIngestTest.o libcommon.so libBonsai.so
#	$(CC) -o $@ $^ $(CFLAGS) $(MKL_PAR_LDFLAGS) $(CILK_LDFLAGS)


.PHONY: clean cleanest

clean: 
	rm -f *.o
	$(MAKE) -C $(SOURCE_DIR)/common clean
	$(MAKE) -C $(SOURCE_DIR)/ProtoNN clean
	$(MAKE) -C $(SOURCE_DIR)/Bonsai clean

cleanest: clean
	rm -f ProtoNN ProtoNNIngestTest BonsaiIngestTest Bonsai
	$(MAKE) -C $(SOURCE_DIR)/common cleanest
	$(MAKE) -C $(SOURCE_DIR)/ProtoNN cleanest
	$(MAKE) -C $(SOURCE_DIR)/Bonsai cleanest
