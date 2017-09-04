// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "blas_routines.h"

using namespace EdgeML;

// OUT = alpha*t1(in1)*t2(in2) + beta*out
void EdgeML::mm(
  MatrixXuf& out,
  const MatrixXuf& in1,
  const CBLAS_TRANSPOSE t1,
  const MatrixXuf& in2,
  const CBLAS_TRANSPOSE t2,
  const FP_TYPE alpha,
  const FP_TYPE beta,
  Eigen::Index in1ColsBegin,
  Eigen::Index in1ColsEnd)
{
  static Logger local_logger("dense_dense_mm");
  Timer timer("dense_dense_mm");
  LOG_DIAGNOSTIC(in1);
  LOG_DIAGNOSTIC(in2);
  assert(sizeof(MKL_UINT) == sizeof(Eigen::Index));

  if (in1ColsBegin == -1) assert(in1ColsEnd == -1);
  if (in1ColsEnd == -1) assert(in1ColsBegin == -1);

  Eigen::Index in1Cols = ((in1ColsBegin == -1) ? in1.cols() : (in1ColsEnd - in1ColsBegin));

  assert((out.IsRowMajor == in1.IsRowMajor) && (out.IsRowMajor == in2.IsRowMajor));
  assert(out.rows() == ((t1 == CblasTrans) ? in1Cols : in1.rows()));
  assert(out.cols() == ((t2 == CblasTrans) ? in2.rows() : in2.cols()));
  assert(((t1 == CblasTrans) ? in1.rows() : in1Cols)
    == ((t2 == CblasTrans) ? in2.cols() : in2.rows()));

  std::string inputCharacteristics
    = "computing input characteristics: in1.rows = " + std::to_string(in1.rows())
    + ", in1.cols = " + std::to_string(in1.cols()) + ", in2.cols = " + std::to_string(in2.cols());
  timer.nextTime(inputCharacteristics);
#ifdef CUDA
  // FILL IN CUDA's gemm here.
#endif
#ifdef EIGEN_USE_BLAS
  gemm(out.IsRowMajor ? CblasRowMajor : CblasColMajor, t1, t2,
    out.rows(), out.cols(), t1 == CblasTrans ? in1.rows() : in1Cols,
    alpha,
    (in1ColsBegin == -1)
    ? in1.data()
    : (in1.IsRowMajor
      ? (in1.data() + (MKL_UINT)in1ColsBegin)
      : (in1.data() + (MKL_UINT)in1ColsBegin*(MKL_UINT)in1.rows())),
    in1.IsRowMajor
    ? in1Cols
    : in1.rows(),
    in2.data(),
    in2.IsRowMajor
    ? in2.cols()
    : in2.rows(),
    beta,
    out.data(), out.IsRowMajor ? out.cols() : out.rows());
#endif
  LOG_DIAGNOSTIC(out);
}


void EdgeML::mm(
  Matrix<FP_TYPE, Dynamic, Dynamic, ColMajor>& out,
  const SparseMatrixuf& in1,
  const CBLAS_TRANSPOSE t1,
  const MatrixXuf& in2,
  const CBLAS_TRANSPOSE t2,
  const FP_TYPE alpha,
  const FP_TYPE beta,
  Eigen::Index in1ColsBegin,
  Eigen::Index in1ColsEnd)
{
  static Logger local_logger("dense_mm");
  LOG_DIAGNOSTIC(in1);
  LOG_DIAGNOSTIC(in2);

  // TODO: Add transpose flag for output
  Timer timer("sp_dn_mm");

  // MKL assumes row-major dense matrix for both out and input2 in calls ?cscmm and ?csrmm
  assert(sizeof(MKL_INT) == sizeof(Eigen::Index));
  assert(in1ColsBegin == -1 && in1ColsEnd == -1);

#ifdef LINUX
#pragma GCC diagnostic ignored "-Wenum-compare"  // suppresses single warning
#endif
  //in1.IsRowMajor checks if in1 is in csr
  assert(in1.IsRowMajor == in2.IsRowMajor);
  assert(out.rows() == ((t1 == CblasTrans) ? in1.cols() : in1.rows()));
  assert(out.cols() == ((t2 == CblasTrans) ? in2.rows() : in2.cols()));
  assert(((t1 == CblasTrans) ? in1.rows() : in1.cols())
    == ((t2 == CblasTrans) ? in2.cols() : in2.rows()));

  std::string inputCharacteristics =
    "computing input characteristics: in1.rows = " + std::to_string(in1.rows())
    + ", in1.cols = " + std::to_string(in1.cols()) + ", nnzs(in1) = "
    + std::to_string(getnnzs(in1)) + ", in2.cols = " + std::to_string(in2.cols());
  timer.nextTime(inputCharacteristics);
#ifdef CUDA
  // Fill in CUDA csrmm here
#endif
#ifdef EIGEN_USE_BLAS
  MKL_INT ldIn2 = ((t2 == CblasNoTrans) ? in2.cols() : in2.rows());
  MKL_INT ldOut = out.cols();
  MKL_INT m = in1.rows();
  MKL_INT n = out.cols();
  MKL_INT k = in1.cols();

  FP_TYPE *in2Transpose = new FP_TYPE[in2.rows()*in2.cols()];
  omatcopy(in2.IsRowMajor ? 'R' : 'C', 't',
    in2.rows(), in2.cols(),
    1.0,
    in2.data(), in2.IsRowMajor ? in2.cols() : in2.rows(),
    in2Transpose, in2.IsRowMajor ? in2.rows() : in2.cols());
  timer.nextTime("transposing the dense input matrix");

  Matrix<FP_TYPE, Dynamic, Dynamic, RowMajor> out_(out.rows(), out.cols());
  timer.nextTime("creating a dense output matrix that stores the rowmajor version");

  omatcopy('C', 't',
    out.rows(), out.cols(),
    1.0,
    out.data(), out.rows(),
    out_.data(), out_.cols());
  timer.nextTime("converting the dense output matrix from colmajor to rowmajor");

  const char matdescra[6] = { 'G', 'X', 'X', 'C', 'X', 'X' }; // 'X' means unused

  // If the sparse matrix is not transposed, and the sparse matrix is in csc format, ...
  // cscmm is quite slow. Instead, we convert the sparse matrix to csr format (using eigen's call), ...
  // and then call csrmm
  if (t1 == CblasTrans) {
    char transa = 't';
    assert(in1.IsRowMajor == false);

    cscmm(&transa,
      &m, &n, &k,
      &alpha,
      matdescra,
      in1.valuePtr(), in1.innerIndexPtr(),
      in1.outerIndexPtr(), in1.outerIndexPtr() + 1,
      (in2.IsRowMajor ^ (t2 == CblasTrans)) ? in2.data() : in2Transpose, &ldIn2,
      &beta,
      out_.data(), &ldOut);
    timer.nextTime("cscmm");
  }
  else {
    char transa = 'n';
    // Irrespective of what in1 originally was, this will create an sp that IS row major
    SparseMatrix<FP_TYPE, RowMajor, sparseIndex_t> sp(in1);
    timer.nextTime("creating a rowmajor in1");

    csrmm(&transa,
      &m, &n, &k,
      &alpha,
      matdescra,
      sp.valuePtr(), sp.innerIndexPtr(), sp.outerIndexPtr(), sp.outerIndexPtr() + 1,
      (in2.IsRowMajor ^ (t2 == CblasTrans)) ? in2.data() : in2Transpose, &ldIn2,
      &beta,
      out_.data(), &ldOut);
    timer.nextTime("csrmm");
  }

  omatcopy('R', 't',
    out_.rows(), out_.cols(),
    1.0,
    out_.data(),
    out_.cols(),
    out.data(),
    out.rows());
  timer.nextTime("converting the computed output matrix from rowmajor to columnmajor");

  delete[] in2Transpose;
#endif
  LOG_DIAGNOSTIC(out);
}


void mm(
  Map<Matrix<FP_TYPE, Dynamic, Dynamic, RowMajor>>& out,
  const SparseMatrixuf& in1,
  const CBLAS_TRANSPOSE t1,
  const MatrixXuf& in2,
  const CBLAS_TRANSPOSE t2,
  const FP_TYPE alpha,
  const FP_TYPE beta,
  Eigen::Index in1ColsBegin,
  Eigen::Index in1ColsEnd)
{
  static Logger local_logger("dense_mm");
  LOG_DIAGNOSTIC(in1);
  LOG_DIAGNOSTIC(in2);

  // TODO: Add transpose flag for output
  Timer timer("sp_dn_mm");

  // MKL assumes row-major dense matrix for both out and input2 in calls ?cscmm and ?csrmm
  assert(sizeof(MKL_INT) == sizeof(Eigen::Index));
  assert(in1ColsBegin == -1 && in1ColsEnd == -1);

#ifdef LINUX
#pragma GCC diagnostic ignored "-Wenum-compare"  // suppresses single warning
#endif  
  //in1.IsRowMajor checks if in1 is in csr
  assert(in1.IsRowMajor == in2.IsRowMajor);
  assert(out.rows() == ((t1 == CblasTrans) ? in1.cols() : in1.rows()));
  assert(out.cols() == ((t2 == CblasTrans) ? in2.rows() : in2.cols()));
  assert(((t1 == CblasTrans) ? in1.rows() : in1.cols())
    == ((t2 == CblasTrans) ? in2.cols() : in2.rows()));

  std::string input_characteristics
    = "computing input characteristics: in1.rows = " + std::to_string(in1.rows())
    + ", in1.cols = " + std::to_string(in1.cols()) + ", nnzs(in1) = "
    + std::to_string(getnnzs(in1)) + ",, in2.cols = " + std::to_string(in2.cols());
  timer.nextTime(input_characteristics);
#ifdef CUDA
  // Fill in CUDA csrmm here
#endif
#ifdef EIGEN_USE_BLAS
  MKL_INT ldIn2 = ((t2 == CblasNoTrans) ? in2.cols() : in2.rows());
  MKL_INT ldOut = out.cols();
  MKL_INT m = in1.rows();
  MKL_INT n = out.cols();
  MKL_INT k = in1.cols();

  FP_TYPE *in2Transpose = new FP_TYPE[in2.rows()*in2.cols()];
  omatcopy(in2.IsRowMajor ? 'R' : 'C', 't',
    in2.rows(), in2.cols(),
    1.0,
    in2.data(), in2.IsRowMajor ? in2.cols() : in2.rows(),
    in2Transpose, in2.IsRowMajor ? in2.rows() : in2.cols());
  timer.nextTime("transposing the dense input matrix");

  const char matdescra[6] = { 'G', 'X', 'X', 'C', 'X', 'X' }; // 'X' means unused

  // If the sparse matrix is not transposed, and the sparse matrix is in csc format, ...
  // cscmm is quite slow. Instead, we convert the sparse matrix to csr format (using eigen's call), ...
  // and then call csrmm
  if (t1 == CblasTrans) {
    char transa = 't';
    assert(in1.IsRowMajor == false);

    cscmm(&transa,
      &m, &n, &k,
      &alpha,
      matdescra,
      in1.valuePtr(), in1.innerIndexPtr(),
      in1.outerIndexPtr(), in1.outerIndexPtr() + 1,
      (in2.IsRowMajor ^ (t2 == CblasTrans)) ? in2.data() : in2Transpose, &ldIn2,
      &beta,
      out.data(), &ldOut);

    assert(t1 == CblasTrans && t2 == CblasTrans);
    //out = in1.transpose() * in2.transpose();
    timer.nextTime("cscmm");
  }
  else {
    char transa = 'n';
    // Irrespective of what in1 originally was, this will create an sp that IS row major
    SparseMatrix<FP_TYPE, RowMajor, sparseIndex_t> sp(in1);
    timer.nextTime("creating a rowmajor in1");

    csrmm(&transa,
      &m, &n, &k,
      &alpha,
      matdescra,
      sp.valuePtr(), sp.innerIndexPtr(), sp.outerIndexPtr(), sp.outerIndexPtr() + 1,
      (in2.IsRowMajor ^ (t2 == CblasTrans)) ? in2.data() : in2Transpose, &ldIn2,
      &beta,
      out.data(), &ldOut);
    timer.nextTime("csrmm");
  }

  delete[] in2Transpose;
#endif
  // Calling LOG_DIAGNOSTIC(out) creates a conversion from Map<Matrix> to Matrix which brings a huge overhead with it!
  // LOG_DIAGNOSTIC(out); 
}


// out = alpha*t1(in1)*t2(in2) + beta*out
// t2 is in sparse format
void EdgeML::mm(
  MatrixXuf& out,
  const MatrixXuf& in1,
  const CBLAS_TRANSPOSE t1,
  const SparseMatrixuf& in2,
  const CBLAS_TRANSPOSE t2,
  const FP_TYPE alpha,
  const FP_TYPE beta,
  Eigen::Index in2ColsBegin,
  Eigen::Index in2ColsEnd)
{
  // TODO: Rewrite this call to use space just the allocated for @out, 
  // without creating another array for @outTranspose.
  static Logger local_logger("dense_mm");
  LOG_DIAGNOSTIC(in1);
  LOG_DIAGNOSTIC(in2);
  Timer timer("dn_sp_mm");

  assert(sizeof(MKL_INT) == sizeof(Eigen::Index));
  std::string input_characteristics
    = "computing input characteristics: in1.rows = " + std::to_string(in1.rows())
    + ", in1.cols = " + std::to_string(in1.cols()) + ", nnzs(in2) = " +
    std::to_string(getnnzs(in2)) + ", in2.cols = " + std::to_string(in2.cols());
  timer.nextTime(input_characteristics);

  if (!out.IsRowMajor) {
    Map<Matrix<FP_TYPE, Dynamic, Dynamic, RowMajor>> outMap(out.data(), out.cols(), out.rows());
    mm(outMap,
      in2, (t2 == CblasTrans) ? CblasNoTrans : CblasTrans,
      in1, (t1 == CblasTrans) ? CblasNoTrans : CblasTrans,
      alpha, beta,
      in2ColsBegin, in2ColsEnd);
    timer.nextTime("ret from sp_dn_mm");
  }
  else {
    MatrixXuf outTranspose(out.cols(), out.rows());
    assert(in2.outerIndexPtr()[0] == 0);

    omatcopy(out.IsRowMajor ? 'R' : 'C', 't',
      out.rows(), out.cols(),
      1.0,
      out.data(),
      out.IsRowMajor ? out.cols() : out.rows(),
      outTranspose.data(),
      out.IsRowMajor ? out.rows() : out.cols());
    timer.nextTime("outTranspose() = out.transpose()");

    mm(outTranspose,
      in2, (t2 == CblasTrans) ? CblasNoTrans : CblasTrans,
      in1, (t1 == CblasTrans) ? CblasNoTrans : CblasTrans,
      alpha, beta,
      in2ColsBegin, in2ColsEnd);
    timer.nextTime("ret from sp_dn_mm");

    omatcopy(outTranspose.IsRowMajor ? 'R' : 'C', 't',
      outTranspose.rows(), outTranspose.cols(),
      1.0,
      outTranspose.data(),
      outTranspose.IsRowMajor ? outTranspose.cols() : outTranspose.rows(),
      out.data(),
      outTranspose.IsRowMajor ? outTranspose.rows() : outTranspose.cols());

    timer.nextTime("out = outTranspose.transpose()");
  }

  LOG_DIAGNOSTIC(out);
}


Eigen::Index EdgeML::getnnzs(const SparseMatrixuf& A)
{
#ifdef ROWMAJOR
  sparseIndex_t nnz = A.outerIndexPtr()[A.rows()] - A.outerIndexPtr()[0];
#else
  sparseIndex_t nnz = A.outerIndexPtr()[A.cols()] - A.outerIndexPtr()[0];
#endif
  return nnz;
}


FP_TYPE EdgeML::maxAbsVal(const MatrixXuf& A)
{
  return *(A.data() + amax(A.rows() * A.cols(), A.data(), 1));
}


FP_TYPE EdgeML::maxAbsVal(const SparseMatrixuf& A)
{
  return *(A.valuePtr() + amax(getnnzs(A), A.valuePtr(), 1));
}
