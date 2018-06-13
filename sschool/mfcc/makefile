# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

CC=gcc
LD=ld
CFLAGS=-Wall -g -std=c99
TESTFLAGS= 
INCLUDEARGS=-I./includes -I./muFFT

testmfcc: testmfcc.c mfcc.o
	$(CC) $(CFLAGS) $(INCLUDEARGS) $^ -lm -o $@
	rm -rvf cpu.o kernel.o fft.o mfcc.o

mfcc.o: cpu.o kernel.o fft.o mfcc_.o
	$(LD) -r $^ -o $@
	rm -rvf cpu.o kernel.o fft.o mfcc_.o

mfcc_.o: mfcc.c includes/mfcc.h muFFT/fft.h
	$(CC) $(CFLAGS) -c $(INCLUDEARGS) $< -lm -o $@

cpu.o: muFFT/cpu.c muFFT/fft_internal.h
	$(CC) $(CFLAGS) -c $(INCLUDEARGS) $< -lm -o $@ 

kernel.o: muFFT/kernel.c muFFT/fft_internal.h
	$(CC) $(CFLAGS) -c $(INCLUDEARGS) $< -lm -o $@

fft.o: muFFT/fft.c muFFT/fft.h muFFT/fft_internal.h
	$(CC) $(CFLAGS) -c $(INCLUDEARGS) $< -lm -o $@

clean:
	rm -rvf *.o *.out testmfcc
