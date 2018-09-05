// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __UTILS_H__
#define __UTILS_H__

#include "pre_processor.h"

namespace EdgeML
{
  struct sparseMatrixMetaData
  {
    featureCount_t nRows;
    dataCount_t nCols;
    Eigen::Index nnzs;

    sparseMatrixMetaData();
    sparseMatrixMetaData(
      const featureCount_t& nRows_,
      const dataCount_t& nCols_,
      const Eigen::Index& nnzs_);

    static size_t structStat();
    size_t exportToBuffer(char *const buffer);
    size_t importFromBuffer(const char *const buffer);
    size_t importSparseMatrixStat();
  };

  struct denseMatrixMetaData
  {
    featureCount_t nRows;
    dataCount_t nCols;

    denseMatrixMetaData();
    denseMatrixMetaData(
      const featureCount_t& nRows_,
      const dataCount_t& nCols_);

    static size_t structStat();
    size_t exportToBuffer(char *const buffer);
    size_t importFromBuffer(const char *const buffer);
  };



  //safe division routine. Checks for underflow before dividing.
  void checkDenominator(const FP_TYPE &denominator);
  FP_TYPE safeDiv(const FP_TYPE &num, const FP_TYPE &den);

  FP_TYPE computeModelSizeInkB(
    const FP_TYPE& lambdaW,
    const FP_TYPE& lambdaZ,
    const FP_TYPE& lambda_B,
    const MatrixXuf& W,
    const MatrixXuf& Z,
    const MatrixXuf& B);

  // sequentialQuickSelect is in place, which means that the data matrix that is passed will be corrupted after the function returns.  
  FP_TYPE sequentialQuickSelect(FP_TYPE* data, size_t count, size_t order);

  inline void typeMismatchAssign(MatrixXuf& dst, const MatrixXuf& src) { dst = src; }
  inline void typeMismatchAssign(MatrixXuf& dst, const SparseMatrixuf& src) { dst = MatrixXuf(src); }
  inline void typeMismatchAssign(SparseMatrixuf& dst, const MatrixXuf& src) { dst = src.sparseView(); }
  inline void typeMismatchAssign(SparseMatrixuf& dst, const SparseMatrixuf& src) { dst = src; }

  void randPick(const MatrixXuf& source, MatrixXuf& target, dataCount_t seed = 42);
  void randPick(const SparseMatrixuf& source, SparseMatrixuf& target, dataCount_t seed = 42);

  inline double rand_fraction()
  {
    const double normalizer = (double)(((uint64_t)RAND_MAX + 1) * ((uint64_t)RAND_MAX + 1));
    return ((double)rand() + (double)rand()*(((double)RAND_MAX + 1.0))) / normalizer;
  }

  size_t sparseExportStat(const SparseMatrixuf& mat);
  size_t sparseExportStat(const MatrixXuf& mat);
  size_t denseExportStat(const MatrixXuf& mat);
  size_t denseExportStat(const SparseMatrixuf& mat);

  size_t exportDenseMatrix(const MatrixXuf& mat, const size_t& bufferSize, char *const buffer);
  size_t exportDenseMatrix(const SparseMatrixuf& mat, const size_t& bufferSize, char *const buffer);
  size_t exportSparseMatrix(const SparseMatrixuf& mat, const size_t& bufferSize, char *const buffer);
  size_t exportSparseMatrix(const MatrixXuf& mat, const size_t& bufferSize, char *const buffer);

  size_t importSparseMatrix(SparseMatrixuf& mat, const char *const buffer);
  size_t importDenseMatrix(MatrixXuf& mat, const size_t& bufferSize, const char *const buffer);

  void writeMatrixInASCII(const MatrixXuf& mat, const std::string& outDir, const std::string& fileName);
  void writeSparseMatrixInASCII(const SparseMatrixuf& mat, const std::string& outDir, const std::string& fileName);
}
#endif
