# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

include ../../config.mk

COMMON_INCLUDE_DIR=../common

IFLAGS= -I ../../eigen/ -I $(COMMON_INCLUDE_DIR) -I$(MKL_ROOT)/include 

PARAMETER_SPARSITY_FLAGS = -DSPARSE_LABEL_PROTONN #-DSPARSE_Z_PROTONN #-DSPARSE_B_PROTONN #-DSPARSE_W_PROTONN
PROTONN_FLAGS = -DL2 $(PARAMETER_SPARSITY_FLAGS)

CFLAGS += $(PROTONN_FLAGS)

PROTONN_INCLUDES = ProtoNN.h ProtoNNFunctions.h \
		   $(COMMON_INCLUDE_DIR)
PROTONN_OBJS = ProtoNNModel.o ProtoNNHyperParams.o ProtoNNParams.o \
               ProtoNNTrainer.o ProtoNNPredictor.o ProtoNNFunctions.o cluster.o

PROTONN_LIB = ../../libProtoNN.so

all: $(PROTONN_LIB)

../../libProtoNN.so: $(PROTONN_OBJS)
	$(CC) -o $@ -shared -fPIC $^ -lc 

ProtoNNModel.o: ProtoNNModel.cpp $(PROTONN_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

ProtoNNHyperParams.o: ProtoNNHyperParams.cpp $(PROTONN_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

ProtoNNParams.o: ProtoNNParams.cpp $(PROTONN_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

ProtoNNTrainer.o: ProtoNNTrainer.cpp $(PROTONN_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

ProtoNNPredictor.o: ProtoNNPredictor.cpp $(PROTONN_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

ProtoNNFunctions.o: ProtoNNFunctions.cpp $(PROTONN_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

cluster.o: cluster.cpp $(PROTONN_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<


.PHONY: clean cleanest

clean: 
	rm -f *.o

cleanest: clean
	rm *~
	rm $(PROTONN_LIB)
