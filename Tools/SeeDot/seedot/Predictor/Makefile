# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

CC=g++
CFLAGS= -Wall -p -g -fPIC -O3 -std=c++11

PREDICTOR_INCLUDES = bonsai_float_model.h \
					datatypes.h \
					library.h predictors.h \
					profile.h \
					protonn_float_model.h \
					seedot_fixed_model.h
				  
PREDICTOR_OBJS = bonsai_float.o library.o \
				main.o profile.o \
				protonn_float.o \
				seedot_fixed.o

all: Predictor

Predictor: bonsai_float.o library.o main.o profile.o protonn_float.o seedot_fixed.o
	$(CC) -o $@ $^ $(CFLAGS)

bonsai_float.o: bonsai_float.cpp $(PREDICTOR_INCLUDES)
	$(CC) -c -o $@ $(CFLAGS) $<

library.o: library.cpp $(PREDICTOR_INCLUDES)
	$(CC) -c -o $@ $(CFLAGS) $<

main.o: main.cpp $(PREDICTOR_INCLUDES)
	$(CC) -c -o $@ $(CFLAGS) $<

profile.o: profile.cpp $(PREDICTOR_INCLUDES)
	$(CC) -c -o $@ $(CFLAGS) $<

protonn_float.o: protonn_float.cpp $(PREDICTOR_INCLUDES) 
	$(CC) -c -o $@ $(CFLAGS) $<

seedot_fixed.o: seedot_fixed.cpp $(PREDICTOR_INCLUDES) 
	$(CC) -c -o $@ $(CFLAGS) $<

clean: 
	rm -f *.o
	rm -f Predictor
