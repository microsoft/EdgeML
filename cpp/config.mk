# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

DEBUGGING_FLAGS = #-DLIGHT_LOGGER #-DLOGGER #-DTIMER -DCONCISE #-DSTDERR_ONSCREEN #-DLIGHT_LOGGER -DVERBOSE #-DDUMP #-DVERIFY
CONFIG_FLAGS = -DSINGLE #-DXML -DZERO_BASED_IO 

MKL_EIGEN_FLAGS = -DEIGEN_USE_BLAS -DMKL_ILP64

LDFLAGS= -lm -ldl

MKL_ROOT=/opt/intel/mkl

MKL_COMMON_LDFLAGS=-L $(MKL_ROOT)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_core
MKL_SEQ_LDFLAGS = $(MKL_COMMON_LDFLAGS) -lmkl_sequential
MKL_PAR_LDFLAGS = $(MKL_COMMON_LDFLAGS) -lmkl_gnu_thread -lgomp -lpthread
MKL_PAR_STATIC_LDFLAGS = -Wl,--start-group /opt/intel/mkl/lib/intel64/libmkl_intel_ilp64.a /opt/intel/mkl/lib/intel64/libmkl_gnu_thread.a /opt/intel/mkl/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl

CILK_LDFLAGS = -lcilkrts
CILK_FLAGS = -fcilkplus -DCILK

CC=g++-5

CFLAGS= -p -g -fPIC -O3 -std=c++11 -DLINUX $(DEBUGGING_FLAGS) $(CONFIG_FLAGS) $(MKL_EIGEN_FLAGS) $(CILK_FLAGS)
