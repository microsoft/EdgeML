# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

CC=gcc
CFLAGS=-Wall -g -std=c99
TESTFLAGS= 
INCLUDEARGS=-I./includes

testlstm: TESTFLAGS= -D__TEST_LSTM__ 
testlstm: testlstm.c helpermath.o lstm.o lstmparams.o
	$(CC) $(CFLAGS) $(TESTFLAGS) $(INCLUDEARGS) $^ -o $@ -lm
	rm -rvf *.o *.out

lstm.o: lstm.c includes/lstm.h includes/helpermath.h includes/paramstruct.h
	$(CC) $(CFLAGS) -c $(INCLUDEARGS) $(TESTFLAGS) $< -o $@

lstmparams.o: lstmparams.c includes/paramstruct.h
	$(CC) $(CFLAGS) -c $(INCLUDEARGS) $(TESTFLAGS) $< -o $@	

helpermath.o: helpermath.c includes/helpermath.h
	$(CC) $(CFLAGS) -c $(INCLUDEARGS) $< -o $@ -lm

clean:
	rm -rvf *.o *.out testlstmandprotonn testlstmprotonn testmfcc \
	testpipeline testcircularq testreuse testinterface testpi3 testwav \
	testlstm
