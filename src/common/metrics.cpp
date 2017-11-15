// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "metrics.h"

using namespace EdgeML;

EdgeML::ResultStruct::ResultStruct()
  :
  problemType(undefinedProblem),
  accuracy(0),
  precision1(0),
  precision3(0),
  precision5(0)
{}

void EdgeML::ResultStruct::scaleAndAdd(ResultStruct& results, FP_TYPE scale)
{
  if ((problemType == undefinedProblem) && (results.problemType != undefinedProblem))
    problemType = results.problemType;
  assert(problemType == results.problemType);

  accuracy += scale * results.accuracy;
  precision1 += scale * results.precision1;
  precision3 += scale * results.precision3;
  precision5 += scale * results.precision5;
}

void EdgeML::ResultStruct::scale(FP_TYPE scale)
{
  accuracy *= scale;
  precision1 *= scale;
  precision3 *= scale;
  precision5 *= scale;
}


// computes accuracy for binary/multiclass datasets, and prec1, prec3, prec5 for multilabel datasets
EdgeML::ResultStruct EdgeML::evaluate(
  const MatrixXuf& Yscores, 
  const SparseMatrixuf& Y,
  const ProblemFormat problemType)
{
  assert(Yscores.cols() == Y.cols());
  assert(Yscores.rows() == Y.rows());
  MatrixXuf Ytrue(Y);
  MatrixXuf Ypred = Yscores;

  FP_TYPE acc = 0;
  FP_TYPE prec1 = 0;
  FP_TYPE prec3 = 0;
  FP_TYPE prec5 = 0;
  
  EdgeML::ResultStruct res;
  res.problemType = problemType;

  if (problemType == EdgeML::ProblemFormat::binary || problemType == EdgeML::ProblemFormat::multiclass) {
    dataCount_t Ytrue_, Ypred_;

    for (Eigen::Index i = 0; i < Ytrue.cols(); ++i) {
      Ytrue.col(i).maxCoeff(&Ytrue_);
      Ypred.col(i).maxCoeff(&Ypred_);

      if (Ytrue_ == Ypred_)
        acc += 1;
    }

    assert(Y.cols() != 0);
    res.accuracy = acc = safeDiv(acc, (FP_TYPE)Y.cols());
  }
  else if (problemType == EdgeML::ProblemFormat::multilabel) {
    const labelCount_t k = 5;

    assert(k * Ypred.cols() < 3e9);
    std::vector<labelCount_t> topInd(k * Ypred.cols());
    pfor(Eigen::Index i = 0; i < Ytrue.cols(); ++i) {
      for (Eigen::Index j = 0; j < Ytrue.rows(); ++j) {
        FP_TYPE val = Ypred(j, i);
        if (j >= k && (val < Ypred(topInd[i*k + (k - 1)], i)))
          continue;
        size_t top = std::min(j, (Eigen::Index)k - 1);
        while (top > 0 && (Ypred(topInd[i*k + (top - 1)], i) < val)) {
          topInd[i*k + (top)] = topInd[i*k + (top - 1)];
          top--;
        }
        topInd[i*k + top] = j;
      }
    }

    assert(k >= 5);
    for (Eigen::Index i = 0; i < Ytrue.cols(); ++i) {
      for (labelCount_t j = 0; j < 1; ++j) {
        if (Ytrue(topInd[i*k + j], i) == 1)
          prec1++;
      }
      for (labelCount_t j = 0; j < 3; ++j) {
        if (Ytrue(topInd[i*k + j], i) == 1)
          prec3++;
      }
      for (labelCount_t j = 0; j < 5; ++j) {
        if (Ytrue(topInd[i*k + j], i) == 1)
          prec5++;
      }
    }

    dataCount_t totalCount = Y.cols();
    assert(totalCount != 0);
    res.precision1 = (prec1 /= (FP_TYPE)totalCount);
    res.precision3 = (prec3 /= ((FP_TYPE)totalCount)*3);
    res.precision5 = (prec5 /= ((FP_TYPE)totalCount)*5);
  }

  return res;
}


void EdgeML::getTopKScoresBatch(
  const MatrixXuf& Yscores,
  MatrixXuf& topKindices,
  MatrixXuf& topKscores,
  int k)
{
  dataCount_t totalCount = Yscores.cols();
  dataCount_t totalLabels = Yscores.rows();

  if (k < 1)
    k = 5;

  if (totalLabels < k)
    k = totalLabels;

  topKindices = MatrixXuf::Zero(k, totalCount);
  topKscores = MatrixXuf::Zero(k, totalCount);

  pfor(Eigen::Index i = 0; i < Yscores.cols(); ++i) {
    for (Eigen::Index j = 0; j < Yscores.rows(); ++j) {
      FP_TYPE val = Yscores(j, i);
      if (j >= k && (val < Yscores(topKindices(k-1, i), i)))
        continue;
      size_t top = std::min(j, (Eigen::Index)k - 1);
      while (top > 0 && (Yscores(topKindices(top-1, i), i) < val)) {
        topKindices(top, i) = topKindices(top - 1, i);
        top--;
      }
      topKindices(top, i) = j;
    }
  }
  
  pfor(Eigen::Index i = 0; i < topKindices.cols(); ++i)
    for (Eigen::Index j = 0; j < topKindices.rows(); ++j)
      topKscores(j, i) = Yscores(topKindices(j, i), i);
}


