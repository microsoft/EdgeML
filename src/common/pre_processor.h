// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __PRE_PROCESSOR_H__
#define __PRE_PROCESSOR_H__

#include "mkl.h"

////////////////////////////////////////////////////////
// DO NOT REORDER THIS: must occur before Eigen includes
// Defines Eigen's index type to MKL_INT
//
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE MKL_INT
////////////////////////////////////////////////////////

#include "goldfoil.h"
#include "timer.h"
#include <cfloat>
#include <vector>
#include <cmath>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdlib>
#include <queue>
#include <random>
#ifdef _MSC_VER 
#include <direct.h> 
#else
#include <sys/stat.h> //Check for Windows
#endif
using namespace Eigen;
//additional dependent includes at the end of the program


#ifndef COMPILER_GCC
//#pragma comment(lib, "../../../../Libraries/MKL/Win/Microsoft.MachineLearning.MklImports.lib")
#endif 

//input and output are assumed to be of the same format (single/double)
#ifdef DOUBLE
typedef double      FP_TYPE;
typedef FP_TYPE     LABEL_TYPE;
#define FP_TYPE_MIN DBL_MIN
#define FP_TYPE_MAX DBL_MAX
#define gemm        cblas_dgemm
#define gemv        cblas_dgemv
#define cscmv       mkl_dcscmv
#define cscmm       mkl_dcscmm
#define csrmm       mkl_dcsrmm
#define omatcopy    mkl_domatcopy
#define amax        cblas_idamax
#define dot         cblas_ddot
#define imin        cblas_idamin
#define axpy        cblas_daxpy
#define vMul		vdMul
#define vTanh		vdTanh
#define scal		cblas_dscal
#define vSqr		vdSqr
#define vDiv 		vdDiv
#endif

#ifdef SINGLE
typedef float       FP_TYPE;
typedef float       LABEL_TYPE;
#define FP_TYPE_MIN FLT_MIN
#define FP_TYPE_MAX FLT_MAX
#define gemm        cblas_sgemm
#define gemv        cblas_sgemv
#define cscmv       mkl_scscmv
#define cscmm       mkl_scscmm
#define csrmm       mkl_scsrmm
#define omatcopy    mkl_somatcopy
#define amax        cblas_isamax
#define dot         cblas_sdot
#define imin        cblas_isamin
#define axpy        cblas_saxpy
#define vMul		vsMul
#define vTanh		vsTanh
#define scal		cblas_sscal
#define vSqr 		vsSqr
#define vDiv		vsDiv
#endif


//number of features and datapoints are assumed to 
//be of the same format (unsigned long long or now)
typedef MKL_UINT dataCount_t;
typedef MKL_UINT labelCount_t;
typedef MKL_UINT featureCount_t;
typedef MKL_INT sparseIndex_t;
#define MIN_DEN 1e-8L

//typedef unsigned long long ULL;
//typedef unsigned long size_t;

#ifdef CILK
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/reducer_opadd.h>
#define pfor cilk_for
#else
#define pfor for
#define cilk_spawn
#define cilk_sync
#endif

#ifdef CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#endif

#ifdef ROWMAJOR
#define MatrixXuf Matrix<FP_TYPE,Dynamic,Dynamic,RowMajor>
#define SparseMatrixuf SparseMatrix<FP_TYPE,RowMajor,sparseIndex_t>
#else
#define MatrixXuf Matrix<FP_TYPE,Dynamic,Dynamic,ColMajor>
#define SparseMatrixuf SparseMatrix<FP_TYPE,ColMajor,sparseIndex_t>
#endif

#define MatrixXufINT MatrixXuf
#define VectorXf Matrix<FP_TYPE,Dynamic,1>
#define Trip Triplet<FP_TYPE,sparseIndex_t>


// Logger Included here because it needs MatrixXuf and SparseMatrixXuf defs
#include "logger.h"

static IOFormat eigen_tsv(FullPrecision, DontAlignCols, "\t", "\n", "", "", "", "");

#endif

