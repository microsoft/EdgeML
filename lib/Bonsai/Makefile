# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

include ../../config.mk

COMMON_INCLUDE_DIR=../common/

PARAMETER_SPARSITY_FLAGS = -DSPARSE_LABEL_BONSAI #-DSPARSE_Z_BONSAI #-DSPARSE_V_BONSAI -DSPARSE_THETA_BONSAI -DSPARSE_W_BONSAI
BONSAI_FLAGS = $(PARAMETER_SPARSITY_FLAGS)
CFLAGS += $(BONSAI_FLAGS)

IFLAGS= -I ../../eigen/ -I $(COMMON_INCLUDE_DIR) -I $(MKL_ROOT)/include 

BONSAI_INCLUDES = Bonsai.h BonsaiFunctions.h \
                  $(COMMON_INCLUDE_DIR)
BONSAI_OBJS = BonsaiModel.o BonsaiHyperParams.o BonsaiParams.o \
		BonsaiTrainer.o BonsaiPredictor.o BonsaiFunctions.o

BONSAI_LIB = ../../libBonsai.so

all: $(BONSAI_LIB)

../../libBonsai.so: $(BONSAI_OBJS)
	$(CC) -o $@ -shared -fPIC $^ -lc 

BonsaiModel.o: BonsaiModel.cpp $(BONSAI_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

BonsaiHyperParams.o: BonsaiHyperParams.cpp $(BONSAI_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

BonsaiParams.o: BonsaiParams.cpp $(BONSAI_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

BonsaiTrainer.o: BonsaiTrainer.cpp $(BONSAI_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

BonsaiPredictor.o: BonsaiPredictor.cpp $(BONSAI_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

BonsaiFunctions.o: BonsaiFunctions.cpp $(BONSAI_INCLUDES) 
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

.PHONY: clean cleanest

clean: 
	rm -f *.o

cleanest: clean
	rm *~
	rm $(BONSAI_LIB)
