# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

CC=g++
CFLAGS= -Wall -p -g -fPIC -O3 -std=c++11 -pthread

PREDICTOR_INCLUDES = datatypes.h \
					library_fixed.h \
					library_float.h \
					model_fixed.h \
					model_float.h \
					predictors.h \
					profile.h \
					vars_fixed.h \
					vars_float.h \
				  
PREDICTOR_OBJS = debug.o \
				library_fixed.o \
				library_float.o \
				main.o \
				profile.o \
				seedot_fixed.o \
				seedot_float.o \

all: Predictor

Predictor: $(PREDICTOR_OBJS)
	$(CC) -o $@ $^ $(CFLAGS)

debug.o: debug.cpp $(PREDICTOR_INCLUDES)
	$(CC) -c -o $@ $(CFLAGS) $<

library_fixed.o: library_fixed.cpp $(PREDICTOR_INCLUDES)
	$(CC) -c -o $@ $(CFLAGS) $<

library_float.o: library_float.cpp $(PREDICTOR_INCLUDES)
	$(CC) -c -o $@ $(CFLAGS) $<

main.o: main.cpp $(PREDICTOR_INCLUDES)
	$(CC) -c -o $@ $(CFLAGS) $<

profile.o: profile.cpp $(PREDICTOR_INCLUDES)
	$(CC) -c -o $@ $(CFLAGS) $<

seedot_fixed.o: seedot_fixed.cpp $(PREDICTOR_INCLUDES)
	$(CC) -c -o $@ $(CFLAGS) $<

seedot_float.o: seedot_float.cpp $(PREDICTOR_INCLUDES)
	$(CC) -c -o $@ $(CFLAGS) $<

clean: 
	rm -f *.o
	rm -f Predictor
