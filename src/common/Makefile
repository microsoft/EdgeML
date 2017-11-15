# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

include ../../config.mk

IFLAGS= -I ../../eigen/ -I$(MKL_ROOT)/include


COMMON_INCLUDES = logger.h timer.h \
		  blas_routines.h par_utils.h \
		  mmaped.h utils.h \
		  goldfoil.h Data.h \
		  metrics.h

COMMON_OBJS = logger.o timer.o blas_routines.o  par_utils.o mmaped.o utils.o goldfoil.o Data.o metrics.o

COMMON_LIB = ../../libcommon.so


all: $(COMMON_LIB)

../../libcommon.so: $(COMMON_OBJS)
	$(CC) -o $@ -shared -fPIC $^ -lc 

Data.o:Data.cpp $(COMMON_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

mmaped.o: mmaped.cpp $(COMMON_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

logger.o: logger.cpp $(COMMON_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $< 

timer.o: timer.cpp $(COMMON_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

blas_routines.o: blas_routines.cpp $(COMMON_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

par_utils.o: par_utils.cpp $(COMMON_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

utils.o: utils.cpp  $(COMMON_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

goldfoil.o: goldfoil.cpp $(COMMON_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<

metrics.o: metrics.cpp $(COMMON_INCLUDES)
	$(CC) -c -o $@ $(IFLAGS) $(CFLAGS) $<


.PHONY: clean cleanest

clean: 
	rm -f *.o

cleanest: clean
	rm *~
	rm $(COMMON_LIB)
