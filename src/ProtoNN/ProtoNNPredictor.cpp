// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "blas_routines.h"
#include "Data.h"
#include "ProtoNNFunctions.h"


using namespace EdgeML;
using namespace EdgeML::ProtoNN;

ProtoNNPredictor::ProtoNNPredictor(
  const size_t numBytes,
  const char *const fromModel)
  : model(numBytes, fromModel)
{
  // Set to 0 and use in scoring function 
  WX = MatrixXuf::Zero(model.hyperParams.d, 1);
  WXColSum = MatrixXuf::Zero(1, 1);
  D = MatrixXuf::Zero(1, model.hyperParams.m);

  BColSum = MatrixXuf::Zero(1, model.hyperParams.m);
  BAccumulator = MatrixXuf::Constant(1, model.hyperParams.d, 1.0);
  B_B = model.params.B.cwiseProduct(model.params.B);
  mm(BColSum, BAccumulator, CblasNoTrans, B_B, CblasNoTrans, 1.0, 0.0L);

  gammaSq = model.hyperParams.gamma * model.hyperParams.gamma;
  gammaSqRow = MatrixXuf::Constant(1, model.hyperParams.m, -gammaSq);
  gammaSqCol = MatrixXuf::Constant(WX.cols(), 1, -gammaSq);

  data = new FP_TYPE[model.hyperParams.D];

#ifdef SPARSE_Z
  ZRows = model.params.Z.rows();
  ZCols = model.params.Z.cols();
  alpha = 1.0;
  beta = 0.0;
#endif
}

ProtoNNPredictor::~ProtoNNPredictor()
{
  delete[] data;
}

FP_TYPE ProtoNNPredictor::testDenseDataPoint(
  const FP_TYPE *const values,
  const labelCount_t *const labels,
  const labelCount_t& num_labels,
  const ProblemFormat& problemType)
{
  // create a local matrix to translate the ingested data point into internal format
  MatrixXuf Xtest = MatrixXuf::Zero(model.hyperParams.D, 1); // We want a column vector
  MatrixXuf Ytest = MatrixXuf::Zero(model.hyperParams.l, 1); // We want a column vector

  FP_TYPE* features = Xtest.data();
  FP_TYPE* currentLabels = Ytest.data();

  memcpy(features, values, sizeof(FP_TYPE)*model.hyperParams.D);
  for (labelCount_t id = 0; id < num_labels; id = id + 1) {
    assert(labels[id] < model.hyperParams.l);
    currentLabels[labels[id]] = 1; // TODO: Figure something elegant instead of casting
  }

  // compute matrix to be passed to accuracy function 
  SparseMatrixuf Yval = Ytest.sparseView();
  MatrixXuf WX = MatrixXuf::Zero(model.params.W.rows(), Xtest.cols());
  mm(WX, model.params.W, CblasNoTrans, Xtest, CblasNoTrans, 1.0, 0.0L);

  return accuracy(model.params.Z, Yval,
    gaussianKernel(model.params.B, WX, model.hyperParams.gamma),
    problemType);
}

void ProtoNNPredictor::RBF()
{
  //    WX.array().square().colwise().sum();  
  WXColSum(0, 0) = dot(model.hyperParams.d, WX.data(), 1, WX.data(), 1);

  mm(D,
    WX, CblasTrans,
    model.params.B, CblasNoTrans,
    (FP_TYPE)2.0*gammaSq, 0.0L);
  mm(D, gammaSqCol, CblasNoTrans, BColSum, CblasNoTrans, 1.0, 1.0);
  mm(D, WXColSum, CblasTrans, gammaSqRow, CblasNoTrans, 1.0, 1.0);

  parallelExp(D);
}

void ProtoNNPredictor::scoreDenseDataPoint(
  FP_TYPE* scores,
  const FP_TYPE *const values)
{
  //  mm(WX, model.params.W, CblasNoTrans, Xtest, CblasNoTrans, 1.0, 0.0L);
  gemv(CblasColMajor, CblasNoTrans,
    model.params.W.rows(), model.params.W.cols(),
    1.0, model.params.W.data(), model.params.W.rows(),
    values, 1, 0.0, WX.data(), 1);

  //  MatrixXuf D = gaussianKernel(model.params.B, WX, model.hyperParams.gamma);
  RBF();

  //  mm(scoresMat, model.params.Z, CblasNoTrans, D, CblasTrans, 1.0, 0.0L);
#ifdef SPARSE_Z
  cscmv(&transa,
    &ZRows, &ZCols,
    &alpha, matdescra, model.params.Z.valuePtr(),
    model.params.Z.innerIndexPtr(),
    model.params.Z.outerIndexPtr(),
    model.nnnparams.Z.outerIndexPtr() + 1,
    D.data(), &beta, scores);
#else
  gemv(CblasColMajor, CblasNoTrans,
    model.params.Z.rows(), model.params.Z.cols(),
    1.0, model.params.Z.data(), model.params.Z.rows(),
    D.data(), 1, 0.0, scores, 1);
#endif

  // DO WE NEED TO NORMALIZE SCORES?
}

void ProtoNNPredictor::scoreSparseDataPoint(
  FP_TYPE* scores,
  const FP_TYPE *const values,
  const featureCount_t *indices,
  const featureCount_t numIndices)
{
  memset(data, 0, sizeof(FP_TYPE)*model.hyperParams.D);

  pfor(featureCount_t i = 0; i < numIndices; ++i) {
    assert(indices[i] < model.hyperParams.D);
    data[indices[i]] = values[i];
  }

  gemv(CblasColMajor, CblasNoTrans,
    model.params.W.rows(), model.params.W.cols(),
    1.0, model.params.W.data(), model.params.W.rows(),
    data, 1, 0.0, WX.data(), 1);

  //  MatrixXuf D = gaussianKernel(model.params.B, WX, model.hyperParams.gamma);
  RBF();

  //  mm(scoresMat, model.params.Z, CblasNoTrans, D, CblasTrans, 1.0, 0.0L);
#ifdef SPARSE_Z
  cscmv(&transa,
    &ZRows, &ZCols,
    &alpha, matdescra, model.params.Z.valuePtr(),
    model.params.Z.innerIndexPtr(),
    model.params.Z.outerIndexPtr(),
    model.params.Z.outerIndexPtr() + 1,
    D.data(), &beta, scores);
#else
  gemv(CblasColMajor, CblasNoTrans,
    model.params.Z.rows(), model.params.Z.cols(),
    1.0, model.params.Z.data(), model.params.Z.rows(),
    D.data(), 1, 0.0, scores, 1);
#endif

  // DO WE NEED TO NORMALIZE SCORES?
}

