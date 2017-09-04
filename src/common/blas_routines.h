// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __BLAS_ROUTINES_H__
#define __BLAS_ROUTINES_H__

#include "pre_processor.h"

namespace EdgeML
{
  void mm(MatrixXuf& out,
    const MatrixXuf& in1,
    const CBLAS_TRANSPOSE t1,
    const MatrixXuf& in2,
    const CBLAS_TRANSPOSE t2,
    const FP_TYPE alpha,
    const FP_TYPE beta,
    Eigen::Index in1ColsBegin = -1,
    Eigen::Index in1ColsEnd = -1);

  void mm(MatrixXuf& out,
    const SparseMatrixuf& in1,
    const CBLAS_TRANSPOSE t1,
    const MatrixXuf& in2,
    const CBLAS_TRANSPOSE t2,
    const FP_TYPE alpha,
    const FP_TYPE beta,
    Eigen::Index in1ColsBegin = -1,
    Eigen::Index in1ColsEnd = -1);

  void mm(MatrixXuf& out,
    const MatrixXuf& in1,
    const CBLAS_TRANSPOSE t1,
    const SparseMatrixuf& in2,
    const CBLAS_TRANSPOSE t2,
    const FP_TYPE alpha,
    const FP_TYPE beta,
    Eigen::Index in2ColsBegin = -1,
    Eigen::Index in2ColsEnd = -1);

  Eigen::Index getnnzs(const SparseMatrixuf& A);

  FP_TYPE maxAbsVal(const MatrixXuf& A);
  FP_TYPE maxAbsVal(const SparseMatrixuf& A);
};

#endif
