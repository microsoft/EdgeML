// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "ProtoNNFunctions.h"
#include <algorithm>

#ifdef LOGGER
#define mm LOG_DIAGNOSTIC_MSG(std::string("Calling mm")); mm
#endif

using namespace EdgeML;

FP_TYPE EdgeML::L(
  const ZMatType& Z, const LabelMatType& Y, const MatrixXuf& D,
  const Eigen::Index begin, const Eigen::Index end)
{
  //tmp = (Y - Z*D').^4;
  assert(end - begin == D.rows());
  MatrixXuf temp = Y.middleCols(begin, end - begin);
  mm(temp, Z, CblasNoTrans, D, CblasTrans, -1.0, 1.0);

#if defined(L2)
  temp = temp.array().square();
#elif defined(L4)
  temp = temp.array().square();
  temp = temp.array().square();
#elif defined(L1)  FP_TYPE* data = temp.data();
  pfor(size_t i = 0; i < temp.rows() * temp.cols(); ++i)
    data[i] = data[i] > 0 ? data[i] : -data[i];
#else
  assert(false);
#endif

  FP_TYPE ret = temp.sum();
  return ret / D.rows();
}

FP_TYPE EdgeML::L(
  const ZMatType& Z, const LabelMatType& Y, const MatrixXuf& D)
{
  return L(Z, Y, D, 0, Y.cols());
}

//FP_TYPE medianHeuristic(const BMatType& B, MatrixXuf WX,
//	FP_TYPE init_guess, FP_TYPE multiplier,
//	Eigen::Index begin, Eigen::Index end)
//{
//	MatrixXuf D = gaussianKernel(B, WX, init_guess, begin, end);
//	FP_TYPE median_ = sequentialQuickSelect(D.data(), D.rows() * D.cols(), (D.rows() * D.cols()) / 2);
//	FP_TYPE eps = 1e-5;
//	assert(median_ > eps);
//	return safeDiv(init_guess*multiplier, sqrt(-log(median_)));
//}

MatrixXuf EdgeML::distanceProjectedPointsToCenters(
  const BMatType& B, const MatrixXuf& WX)
{
  MatrixXuf BColSum = MatrixXuf::Zero(1, B.cols());
  MatrixXuf BAccumulator = MatrixXuf::Constant(1, B.rows(), 1.0);
  BMatType  B_B = B.cwiseProduct(B);

  mm(BColSum, BAccumulator, CblasNoTrans, B_B, CblasNoTrans, 1.0, 0.0L);
  MatrixXuf WXColSum = WX.array().square().colwise().sum();
  MatrixXuf D(WX.cols(), B.cols());

  mm(D,
    WX, CblasTrans,
    B, CblasNoTrans,
    -2.0, 0.0);

  MatrixXuf onesCol = MatrixXuf::Constant(WX.cols(), 1, 1.0);
  mm(D, onesCol, CblasNoTrans, BColSum, CblasNoTrans, 1.0, 1.0);

  MatrixXuf onesRow = MatrixXuf::Constant(1, B.cols(), 1.0);
  mm(D, WXColSum, CblasTrans, onesRow, CblasNoTrans, 1.0, 1.0);

  return D;
}

FP_TYPE EdgeML::medianHeuristic(
  const BMatType& B,
  MatrixXuf WX,
  FP_TYPE multiplier)
{
  MatrixXuf D = distanceProjectedPointsToCenters(B, WX);
  std::sort(D.data(), D.data() + (D.rows() * D.cols()));
  FP_TYPE medianEstimate = sequentialQuickSelect(D.data(), D.rows()*D.cols(), (D.rows() * D.cols()) / 2);
  assert(medianEstimate > 1e-5);
  auto gamma = safeDiv(multiplier, sqrt(medianEstimate));
  return gamma;
}

FP_TYPE EdgeML::batchEvaluate(
  const ZMatType& Z,
  const LabelMatType& Y, const LabelMatType& Yval,
  const BMatType& B,
  const MatrixXuf& WX, const MatrixXuf& WXval,
  const FP_TYPE& gamma,
  const EdgeML::ProblemFormat& problemType,
  FP_TYPE* const stats)
{
  Timer timer("batchEvaluate");
  /*  std::function<FP_TYPE(const MatrixXuf&,
      const MatrixXuf&,
      const MatrixXuf&,
      const dataCount_t*)> metric) {
  */
  FP_TYPE objective = 0.0;

  dataCount_t n = WX.cols();
  dataCount_t nvalid = WXval.cols();
  const size_t maxMem = 0x200000000ULL; // Use a maximum of 8 GiGs memory
  //size_t maxBatch = maxMem/(8LL*B.rows());
  dataCount_t maxBatch = 10000;
  if (maxBatch > n) maxBatch = n;
  if (nvalid > 0) {
    if (maxBatch > nvalid) maxBatch = nvalid;
  }

  dataCount_t bs = maxBatch;
  dataCount_t trainBatches = (n + bs - 1) / bs; //taking ceil
  dataCount_t validationBatches;
  if (nvalid > 0) {
    validationBatches = (nvalid + bs - 1) / bs; //taking ceil
  }
  FP_TYPE accuracyTrain = 0.0;
  FP_TYPE accuracyValidation = 0.0;

  for (dataCount_t i = 0; i < trainBatches; ++i) {
    Eigen::Index idx1 = (i*(Eigen::Index)bs) % n;
    Eigen::Index idx2 = ((i + 1)*(Eigen::Index)bs) % n;
    if (idx2 <= idx1) idx2 = n;

    assert(idx1 < idx2);
    assert(idx2 <= idx1 + (Eigen::Index)maxBatch);

    MatrixXuf D = gaussianKernel(B, WX, gamma, idx1, idx2);
    //LOG_DIAGNOSTIC("idx1, idx2, Y.cols() = " + std::to_string(idx1) + " " + std::to_string(idx2) + " " + std::to_string(Y.cols()));
    assert(idx2 <= Y.cols());
    LabelMatType YBatch = Y.middleCols(idx1, idx2 - idx1);
    objective += (idx2 - idx1) * L(Z, YBatch, D);
    if (problemType == EdgeML::ProblemFormat::binary || problemType == EdgeML::ProblemFormat::multiclass)
      accuracyTrain += (idx2 - idx1) * accuracy(Z, YBatch, D, problemType);
    else if (problemType == EdgeML::ProblemFormat::multilabel)
      accuracyTrain += (idx2 - idx1) * accuracy(Z, YBatch, D, problemType);
  }

  LOG_INFO("Training objective: " + std::to_string(objective));
  stats[0] = objective;
  if (problemType == EdgeML::ProblemFormat::binary || problemType == EdgeML::ProblemFormat::multiclass) {
    LOG_INFO("Training accuracy: " + std::to_string(accuracyTrain / n));
    stats[1] = accuracyTrain / n;
  }
  else if (problemType == EdgeML::ProblemFormat::multilabel) {
    LOG_INFO("Training prec@1: " + std::to_string(accuracyTrain / n));
    stats[1] = accuracyTrain / n;
  }
  if (nvalid > 0) {
    for (dataCount_t i = 0; i < validationBatches; ++i) {
      Eigen::Index idx1 = (i*(Eigen::Index)bs) % nvalid;
      Eigen::Index idx2 = ((i + 1)*(Eigen::Index)bs) % nvalid;
      if (idx2 <= idx1) idx2 = nvalid;

      assert(idx1 < idx2);
      assert(idx2 <= idx1 + (Eigen::Index)maxBatch);

      MatrixXuf D = gaussianKernel(B, WXval, gamma, idx1, idx2);
      //LOG_TRACE("idx1, idx2, Y.cols() = " + std::to_string(idx1) + " " + std::to_string(idx2) + " " + std::to_string(Yval.cols()));
      assert(idx2 <= Y.cols());
      LabelMatType YBatch = Yval.middleCols(idx1, idx2 - idx1);
      if (problemType == EdgeML::ProblemFormat::binary || problemType == EdgeML::ProblemFormat::multiclass)
        accuracyValidation += (idx2 - idx1) * accuracy(Z, YBatch, D, problemType);
      else if (problemType == EdgeML::ProblemFormat::multilabel)
        accuracyValidation += (idx2 - idx1) * accuracy(Z, YBatch, D, problemType);
    }
    if (problemType == EdgeML::ProblemFormat::binary || problemType == EdgeML::ProblemFormat::multiclass) {
      LOG_INFO("Validation accuracy: " + std::to_string(accuracyValidation / nvalid));
      stats[2] = accuracyValidation / nvalid;
    }
    else if (problemType == EdgeML::ProblemFormat::multilabel) {
      LOG_INFO("Validation prec@1: " + std::to_string(accuracyValidation / nvalid));
      stats[2] = accuracyValidation / nvalid;
    }
  }
  else {
    stats[2] = 0.0; // No testing takes place
  }
  return objective;
}

MatrixXuf EdgeML::gaussianKernel(
  const BMatType& B, const MatrixXuf& WX,
  const FP_TYPE gamma,
  const Eigen::Index begin, const Eigen::Index end)
{
  assert(begin < (Eigen::Index)0x7fffffff
    && end < (Eigen::Index)0x7fffffff
    && begin < end);

  Timer timer("gaussianKernel");
  timer.nextTime("starting computation");

  static int debug_int = 0;

  MatrixXuf BColSum = MatrixXuf::Zero(1, B.cols());
  MatrixXuf BAccumulator = MatrixXuf::Constant(1, B.rows(), 1.0);
  BMatType  B_B = B.cwiseProduct(B);

  mm(BColSum, BAccumulator, CblasNoTrans, B_B, CblasNoTrans, 1.0, 0.0L);
  timer.nextTime("BColSum");
  MatrixXuf WXColSum = WX.middleCols(begin, end - begin).array().square().colwise().sum();
  timer.nextTime("WXColSum");
  MatrixXuf D(end - begin, B.cols());

  // D = (2.0 * gamma * gamma) * WX.transpose() * B;
  mm(D,
    WX, CblasTrans,
    B, CblasNoTrans,
    (FP_TYPE)2.0*gamma*gamma, (FP_TYPE)0.0,
    begin, end);

  timer.nextTime("Inner product of B,WX");

  MatrixXuf gammaSqCol = MatrixXuf::Constant(end - begin, 1, -gamma*gamma);
  mm(D, gammaSqCol, CblasNoTrans, BColSum, CblasNoTrans, 1.0, 1.0);

  MatrixXuf gammaSqRow = MatrixXuf::Constant(1, B.cols(), -gamma*gamma);
  mm(D, WXColSum, CblasTrans, gammaSqRow, CblasNoTrans, 1.0, 1.0);

  timer.nextTime("Outer product of WX with constant row");

  parallelExp(D);
  LOG_DIAGNOSTIC(D);
  timer.nextTime("point-wise exponentation");

  return D;
}

MatrixXuf EdgeML::gaussianKernel(
  const BMatType& B, const MatrixXuf& WX, const FP_TYPE gamma)
{
  return gaussianKernel(B, WX, gamma, 0, WX.cols());
}

MatrixXuf EdgeML::gradL_B(
  const BMatType& B, const LabelMatType& Y, const ZMatType& Z,
  const MatrixXuf& WX, const MatrixXuf& D, const FP_TYPE gamma,
  const Eigen::Index begin, const Eigen::Index end)
{
  assert(end - begin == D.rows());
  Timer timer("gradL_B");
  //T = ((Y' - D*Z').^3)*Z;
  MatrixXuf temp = Y.middleCols(begin, end - begin).transpose().eval();
  mm(temp, D, CblasNoTrans, Z, CblasTrans, -1.0, 1.0);
  timer.nextTime("computing temp = Y' - D*Z'");

#if defined(L4)
  temp = temp.array().cube();
  timer.nextTime("cubing temp");
#elif defined(L1)
  //signum function 
  FP_TYPE* data = temp.data();
  pfor(size_t i = 0; i < temp.rows() * temp.cols(); ++i)
    data[i] = data[i] > 0 ? 1 : -1;
  timer.nextTime("taking signum");
#elif defined(L2)
#else
  assert(false);
#endif
  LOG_DIAGNOSTIC(temp);

  MatrixXuf T = MatrixXuf::Zero(D.rows(), D.cols());
  mm(T, temp, CblasNoTrans, Z, CblasNoTrans, 1.0, 0.0L);
  timer.nextTime("computing T = temp * Z");

  //DT = D.*T;
  T.noalias() = T.cwiseProduct(D);
  timer.nextTime("computing T = T .* D");

  //v = 8 * gamma^2 * (B * sparse(1:m, 1:m, sum(DT, 1)) - WX * DT);
  MatrixXuf ret = B;
#if defined(L4)
  VectorXf colMult = 8.0 * gamma * gamma * T.colwise().sum();
#elif defined(L2)
  VectorXf colMult = 4.0 * gamma * gamma * T.colwise().sum();
#elif defined(L1)
  VectorXf colMult = 2.0 * gamma * gamma * T.colwise().sum();
#else
  assert(false);
#endif 

#ifdef ROWMAJOR
  LOG_INFO("Warning: Column-scaling in gradL_B may be slow in rowmajor\n");
#endif
  // TODO: pfor (or map) and vectorize
  pfor(Eigen::Index i = 0; i < B.cols(); ++i)
    ret.col(i).noalias() = ret.col(i) * colMult(i);
  timer.nextTime("multiplying columns of B");

#if defined(L4)
  mm(ret,
    WX, CblasNoTrans,
    T, CblasNoTrans,
    -8.0 * gamma * gamma, 1.0,
    begin, end);
#elif defined(L2)
  mm(ret,
    WX, CblasNoTrans,
    T, CblasNoTrans,
    (FP_TYPE)-4.0 * gamma * gamma, (FP_TYPE)1.0,
    begin, end);
#elif defined(L1)
  mm(ret,
    WX, CblasNoTrans,
    T, CblasNoTrans,
    -2.0 * gamma * gamma, 1.0,
    begin, end);
#else
  assert(false);
#endif
  timer.nextTime("computing WX * T");

  return ret / D.rows();
}

MatrixXuf EdgeML::gradL_B(
  const BMatType& B, const LabelMatType& Y, const ZMatType& Z,
  const MatrixXuf& WX, const MatrixXuf& D, const FP_TYPE gamma)
{
  assert((Y.cols() == WX.cols()) && (D.rows() == WX.cols()));
  return gradL_B(B, Y, Z, WX, D, gamma, 0, WX.cols());
}
/*
  MatrixXuf gradL_Z (const ZMatType& Z, const LabelMatType& Y, const MatrixXuf D,
  const Eigen::Index begin, const Eigen::Index end)
  {
  assert (end-begin == D.rows());
  Timer timer("gradL_Z");
  MatrixXuf temp = Y.middleCols(begin,end-begin);
  mm (temp, Z, CblasNoTrans, D, CblasTrans, -1.0, 1.0);
  timer.nextTime("computing temp = Y - Z*D'");

  #ifdef L4
  temp = temp.array().cube();
  timer.nextTime("cubing temp");
  #else
  #ifdef L1
  //signum function
  FP_TYPE* data = temp.data();
  pfor (size_t i=0; i<temp.rows() * temp.cols(); ++i)
  data[i] = data[i] > 0 ? 1 : -1;
  timer.nextTime("taking signum");
  #endif
  #endif
  LOG_DIAGNOSTIC(temp);

  MatrixXuf ret = MatrixXuf::Zero(Y.rows(), D.cols());
  #ifdef L4
  mm (ret, temp, CblasNoTrans, D, CblasNoTrans, -4.0, 0.0L);
  #else
  #ifdef L2
  mm (ret, temp, CblasNoTrans, D, CblasNoTrans, -2.0, 0.0L);
  #else
  #ifdef L1
  mm (ret, temp, CblasNoTrans, D, CblasNoTrans, -1.0, 0.0L);
  #endif
  #endif
  #endif

  timer.nextTime("computing T = -4.0 * temp * D");

  return ret;
  }
*/

MatrixXuf EdgeML::gradL_Z(
  const ZMatType& Z, const LabelMatType& Y, const MatrixXuf& D)
{
  return gradL_Z(Z, Y, D, 0, Y.cols());
}

MatrixXuf EdgeML::gradL_Z(
  const ZMatType& Z, const LabelMatType& Y, const MatrixXuf& D,
  const Eigen::Index begin, const Eigen::Index end)
{
  assert(end - begin == D.rows());
  Timer timer("gradL_Z");
  LabelMatType YMiddle = Y.middleCols(begin, end - begin);
  MatrixXuf ret(YMiddle.rows(), D.cols());
  mm(ret, YMiddle, CblasNoTrans, D, CblasNoTrans, 1.0, 0.0);
  timer.nextTime("computing ret = Y*D");

  MatrixXuf DtimesD(D.cols(), D.cols());
  mm(DtimesD, D, CblasTrans, D, CblasNoTrans, 1.0, 0.0);
  timer.nextTime("computing DtimesD = D'*D");

#ifndef L2
  assert(false);
#endif

#if defined(L4)
  mm(ret, Z, CblasNoTrans, DtimesD, CblasNoTrans, 4.0, -4.0);
#elif defined(L2)
  mm(ret, Z, CblasNoTrans, DtimesD, CblasNoTrans, 2.0, -2.0);
#elif defined(L1)
  mm(ret, Z, CblasNoTrans, DtimesD, CblasNoTrans, 1.0, -1.0);
#else
  assert(false);
#endif

  timer.nextTime("computing ret = ret - Z*DtimesD");

  return ret / D.rows();
}

MatrixXuf EdgeML::gradL_W(
  const BMatType& B, const LabelMatType& Y, const ZMatType& Z,
  const WMatType& W, const SparseMatrixuf& X, const MatrixXuf& D,
  const FP_TYPE gamma,
  const Eigen::Index begin, const Eigen::Index end)
{
  assert(end - begin == D.rows());
  Timer timer("gradL_W");
  //T = ((Y' - D*Z').^3)*Z;
  MatrixXuf temp = Y.middleCols(begin, end - begin).transpose().eval();
  timer.nextTime("creating temp");
  mm(temp, D, CblasNoTrans, Z, CblasTrans, -1.0, 1.0);
  timer.nextTime("computing temp = Y' - D*Z'");

#if defined(L4)
  temp = temp.array().cube();
  timer.nextTime("cubing temp");
#elif defined(L1)
  //signum function 
  FP_TYPE* data = temp.data();
  pfor(size_t i = 0; i < temp.rows() * temp.cols(); ++i)
    data[i] = data[i] > 0 ? 1 : -1;
  timer.nextTime("taking signum");
#elif defined(L2)
#else
  assert(false);
#endif
  LOG_DIAGNOSTIC(temp);

  MatrixXuf T = MatrixXuf::Zero(D.rows(), D.cols());

#if defined(L4)
  mm(T, temp, CblasNoTrans, Z, CblasNoTrans, (FP_TYPE)8.0*gamma*gamma, (FP_TYPE)0.0);
#elif defined(L2)
  mm(T, temp, CblasNoTrans, Z, CblasNoTrans, (FP_TYPE)4.0*gamma*gamma, (FP_TYPE)0.0);
#elif defined(L1) 
  mm(T, temp, CblasNoTrans, Z, CblasNoTrans, (FP_TYPE)2.0*gamma*gamma, (FP_TYPE)0.0);
#else
  assert(false);
#endif 
  timer.nextTime("computing T = temp * Z");

  //DT = D.*T;
  T.noalias() = T.cwiseProduct(D);
  timer.nextTime("computing T = T .* D");

  //v = -8 * gamma^2 * (B * DT' - W*(X*sparse(1:n, 1:n, sum(DT, 2))))*X;
  // TODO: Fix this, dont automatically cast to dense
  SparseMatrixuf XMiddle = X.middleCols(begin, end - begin);
  VectorXf colMult = T.rowwise().sum();

#ifdef ROWMAJOR
  LOG_INFO("Warning: Column-scaling in gradL_W may be slow in rowmajor\n");
#endif
  // TODO: pfor (or map) and saxpy
  temp = MatrixXuf::Zero(W.rows(), end - begin);
  mm(temp, W, CblasNoTrans, XMiddle, CblasNoTrans, 1.0, 0.0L);
  timer.nextTime("computing temp = W * XMiddle");

  pfor(Eigen::Index i = 0; i < end - begin; ++i)
    temp.col(i).noalias() = temp.col(i) * colMult(i);
  timer.nextTime("multiplying columns of temp");

  mm(temp, B, CblasNoTrans, T, CblasTrans, -1.0, 1.0);
  timer.nextTime("computing temp -= B * T'");

  MatrixXuf ret = MatrixXuf::Zero(W.rows(), W.cols());

  mm(ret,
    temp, CblasNoTrans,
    XMiddle, CblasTrans,
    1.0, 0.0L);

  //ret = temp * XMiddle.transpose();
  timer.nextTime("computing grad_W = temp * X'");

  return ret / (D.rows());
}

MatrixXuf EdgeML::gradL_W(
  const BMatType& B, const LabelMatType& Y, const ZMatType& Z,
  const WMatType& W, const SparseMatrixuf& X, const MatrixXuf& D,
  const FP_TYPE gamma)
{
  assert(Y.cols() == X.cols() && Y.cols() == D.rows());
  return gradL_W(B, Y, Z, W, X, D, gamma, 0, X.cols());
}

void EdgeML::hardThrsd(
  MatrixXuf& mat,
  FP_TYPE sparsity)
{
  Timer timer("hardThrsd");
  assert(sparsity >= 0.0 && sparsity <= 1.0);
  if (sparsity >= 0.999)
    return;
  else;

  const float eps = (FP_TYPE)1e-8;

  assert(sizeof(size_t) == 8);
  size_t matSize = ((size_t)mat.rows()) * ((size_t)mat.cols());
  size_t sampleSize = 10000000;
  sampleSize = std::min(sampleSize, matSize);

  FP_TYPE *data = new FP_TYPE[sampleSize];

  if (sampleSize == matSize) {
    memcpy((void *)data, (void *)mat.data(), sizeof(FP_TYPE)*matSize);
    pfor(std::ptrdiff_t i = 0; i < (std::ptrdiff_t)matSize; ++i) {
      data[i] = std::abs(data[i]);
    }
  }
  else {
    unsigned long long prime = 990377764891511ull;
    assert(prime > matSize);
    unsigned long long seed = (rand() % 100000);
    FP_TYPE* mat_data = mat.data();
    size_t pick;
    for (dataCount_t i = 0; i < sampleSize; ++i) {
      pick = (prime*(i + seed)) % matSize;
      data[i] = std::abs(mat_data[pick]);
    }
  }

  timer.nextTime("starting threshold computation");

  size_t order = (size_t)std::round((1.0 - sparsity)*((FP_TYPE)sampleSize));
  FP_TYPE thresh = sequentialQuickSelect(data, sampleSize, order);

  if (thresh <= eps)thresh = eps;
  delete[] data;
  timer.nextTime("ending threshold computation");

  assert(sizeof(std::ptrdiff_t) == sizeof(size_t));
  data = mat.data();

#ifdef CILK
  cilk::reducer< cilk::op_add<size_t> > nnz(0);
#else
  int *nnz = new int;
  *nnz = 0;
#endif
  pfor(std::ptrdiff_t i = 0; i < (std::ptrdiff_t)matSize; ++i) {
    if (std::abs(data[i]) <= thresh)
      data[i] = 0;
    else
      (*nnz) += 1;
  }
  timer.nextTime("thresholding");
#ifdef CILK
  //LOG_INFO("nnz/numel = " + std::to_string((FP_TYPE)nnz.get_value() / (FP_TYPE)matSize));
#else
  //LOG_INFO("threshold = " + std::to_string((FP_TYPE)(thresh)));
  //LOG_INFO("'nnz/numel = " + std::to_string((FP_TYPE)(*nnz) / (FP_TYPE)matSize));
  delete nnz;
#endif
}

void EdgeML::altMinSGD(
  const EdgeML::Data& data,
  EdgeML::ProtoNN::ProtoNNModel& model,
  FP_TYPE *const stats,
  const std::string& outDir)
{
  // This allows us to make mkl-blas calls on Eigen matrices   
  assert(sizeof(MKL_INT) == sizeof(Eigen::Index));
  Timer timer("altMinSGD");
  assert(sizeof(Eigen::Index) == sizeof(dataCount_t));

  /*
    [~, n] = size(X); [l, m] = size(Z);
    if isempty(iters)
    iters = 50;
    end
    tol = 1e-5;

    % alternating minimization
    counter = 1;epochs = 10; batchSize = 512;

    batchSize = 128; epochs = 3; sgdTol = 0.02;
    learning_rate = 0;
    learning_rate_Z = 0.2; learning_rate_B = 0.2; learning_rate_W = 0.2;
  */

  dataCount_t n = data.Xtrain.cols();
  int         epochs = model.hyperParams.epochs;
  FP_TYPE     sgdTol = (FP_TYPE) 0.02;
  dataCount_t bs = std::min((dataCount_t)model.hyperParams.batchSize, (dataCount_t)n);
#ifdef XML
  dataCount_t hessianbs = std::min((dataCount_t)(1 << 10), bs);
#else
  dataCount_t hessianbs = bs;
#endif 
  //const FP_TYPE hessianAdjustment = 1.0; // (FP_TYPE)hessianbs / (FP_TYPE)bs;
  int     etaUpdate = 0;
  FP_TYPE armijoZ((FP_TYPE)0.2), armijoB((FP_TYPE)0.2), armijoW((FP_TYPE)0.2);
  FP_TYPE fOld, fNew, etaZ(1), etaB(1), etaW(1);

  LOG_INFO("\nComputing model size assuming 4 bytes per entry for matrices with sparsity > 0.5 and 8 bytes per entry for matrices with sparsity <= 0.5 (to store sparse matrices, we require about 4 bytes for the index information)...");
  LOG_INFO("Model size in kB = " + std::to_string(computeModelSizeInkB(model.hyperParams.lambdaW, model.hyperParams.lambdaZ, model.hyperParams.lambdaB, model.params.W, model.params.Z, model.params.B)));

  MatrixXuf WX(model.params.W.rows(), data.Xtrain.cols());
  mm(WX, model.params.W, CblasNoTrans, data.Xtrain, CblasNoTrans, 1.0, 0.0L);

  MatrixXuf WXvalidation(model.params.W.rows(), data.Xvalidation.cols());
  if (data.Xvalidation.cols() > 0) {
    mm(WXvalidation, model.params.W, CblasNoTrans, data.Xvalidation, CblasNoTrans, 1.0, 0.0L);
  }

#ifdef XML
  dataCount_t numEvalTrain = std::min((dataCount_t)20000, (dataCount_t)data.Xtrain.cols());
  MatrixXuf WX_sub(WX.rows(), numEvalTrain);
  SparseMatrixuf Y_sub(data.Ytrain.rows(), numEvalTrain);
  SparseMatrixuf X_sub(data.Xtrain.rows(), numEvalTrain);
  randPick(data.Xtrain, X_sub);
  randPick(data.Ytrain, Y_sub);

  mm(WX_sub, model.params.W, CblasNoTrans, X_sub, CblasNoTrans, 1.0, 0.0L);

  dataCount_t numEvalValidation= std::min((dataCount_t)10000, (dataCount_t)data.Xvalidation.cols());
  MatrixXuf WXvalidation_sub(WX.rows(), numEvalValidation);
  SparseMatrixuf Yvalidation_sub(data.Yvalidation.rows(), numEvalValidation);
  SparseMatrixuf Xvalidation_sub(data.Xvalidation.rows(), numEvalValidation);
  if (data.Xvalidation.cols() > 0) {
    randPick(data.Xvalidation, Xvalidation_sub);
    randPick(data.Yvalidation, Yvalidation_sub);
    mm(WXvalidation_sub, model.params.W, CblasNoTrans, Xvalidation_sub, CblasNoTrans, 1.0, 0.0L);
  }
#endif

  timer.nextTime("starting evaluation");


  LOG_INFO("\nInitial stats...");
#ifdef XML
  fNew = batchEvaluate(model.params.Z, Y_sub, Yvalidation_sub, model.params.B, WX_sub, WXvalidation_sub, model.hyperParams.gamma, model.hyperParams.problemType, stats);
#else 
  fNew = batchEvaluate(model.params.Z, data.Ytrain, data.Yvalidation, model.params.B, WX, WXvalidation, model.hyperParams.gamma, model.hyperParams.problemType, stats);
#endif 
  timer.nextTime("evaluating");

  VectorXf eta = VectorXf::Zero(10, 1);

  MatrixXuf gtmpW(model.params.W.rows(), model.params.W.cols());
  WMatType  Wtmp(model.params.W.rows(), model.params.W.cols());

  MatrixXuf gtmpB(model.params.B.rows(), model.params.B.cols());
  BMatType Btmp(model.params.B.rows(), model.params.B.cols());

  MatrixXuf gtmpZ(model.params.Z.rows(), model.params.Z.cols());
  ZMatType  Ztmp(model.params.Z.rows(), model.params.Z.cols());


#if defined(DUMP) || defined(VERIFY)
  std::ofstream f;
  std::string fileName;
#endif

  LOG_INFO("\nStarting optimization. Number of outer iterations (altMinSGD) = " + std::to_string(model.hyperParams.iters));
  // for i = 1 : iters
  for (int i = 0; i < model.hyperParams.iters; ++i) {
    LOG_INFO(
      "\n=========================== " + std::to_string(i) + "\n"
      + "On iter " + std::to_string(i) + "\n" +
      +"=========================== " + std::to_string(i));
    timer.nextTime("starting optimization w.r.t. W");
    LOG_INFO("Optimizing w.r.t. projection matrix (W)...");

#ifdef BTLS
    etaW = armijoW * btls<WMatType>
      ([&model, &data] (const WMatType& W, const Eigen::Index begin, const Eigen::Index end) ->FP_TYPE {
	MatrixXuf WX = MatrixXuf::Zero(W.rows(), end - begin);
	SparseMatrixuf XMiddle = data.Xtrain.middleCols(begin, end - begin);
	mm(WX, W, CblasNoTrans, XMiddle,
	   CblasNoTrans, 1.0, 0.0L);
	return L(model.params.Z, data.Ytrain,
		 gaussianKernel(model.params.B, WX, model.hyperParams.gamma),
		 begin, end);
      },
	[&model, &data]
	(const WMatType& W, const Eigen::Index begin, const Eigen::Index end)
	->MatrixXuf {
	MatrixXuf WX = MatrixXuf::Zero(W.rows(), end - begin);
	SparseMatrixuf XMiddle = data.Xtrain.middleCols(begin, end - begin);
	mm(WX, W, CblasNoTrans,
	   XMiddle,
	   CblasNoTrans, 1.0, 0.0L);
	return gradL_W(model.params.B, data.Ytrain, model.params.Z, W, data.Xtrain,
		       gaussianKernel(model.params.B, WX, model.hyperParams.gamma),
		       model.hyperParams.gamma, begin, end);
      },
	std::bind(hardThrsd, std::placeholders::_1, model.hyperParams.lambdaW),
	model.params.W, n, bs, (etaW/armijoW)*2);
#else
    for (auto j = 0; j < eta.size(); ++j) {
      Eigen::Index idx1 = (j*(Eigen::Index)hessianbs) % n;
      Eigen::Index idx2 = ((j + 1)*(Eigen::Index)hessianbs) % n;
      //assert (((j+1)*(Eigen::Index)hessianbs) < n);

      if (idx2 <= idx1) idx2 = n;

      gtmpW = gradL_W(model.params.B, data.Ytrain, model.params.Z, model.params.W, data.Xtrain,
        gaussianKernel(model.params.B, WX, model.hyperParams.gamma, idx1, idx2),
        model.hyperParams.gamma, idx1, idx2);

      MatrixXuf gtmpWThresh = gtmpW;
      hardThrsd(gtmpWThresh, model.hyperParams.lambdaW);

      Wtmp = model.params.W
        - 0.001*safeDiv(model.params.W.cwiseAbs().maxCoeff(), gtmpW.cwiseAbs().maxCoeff()) * gtmpWThresh;
      gtmpW -= gradL_W(model.params.B, data.Ytrain, model.params.Z, Wtmp, data.Xtrain,
        gaussianKernel(model.params.B, Wtmp*data.Xtrain.middleCols(idx1, idx2 - idx1), model.hyperParams.gamma),
        model.hyperParams.gamma, idx1, idx2);

      if (gtmpW.norm() <= 1e-20L) {
        LOG_WARNING("Difference between consecutive gradients of W has become really low.");
        eta(j) = 1.0;
      }
      else
        eta(j) = safeDiv((Wtmp - model.params.W).norm(), gtmpW.norm());
    }
    std::sort(eta.data(), eta.data() + eta.size());
    etaW = armijoW * eta(4);
#endif
    //LOG_INFO("Step-length estimate for gradW = " + std::to_string(etaW));

    accProxSGD<WMatType>
      (//[&model.params.Z, &data.Ytrain, &model.params.B, &data.Xtrain, &model.hyperParams] TODO: Figure out the elegant way of getting this to work
        [&model, &data]
    (const WMatType& W, const Eigen::Index begin, const Eigen::Index end)
      ->FP_TYPE {
      MatrixXuf WX = MatrixXuf::Zero(W.rows(), end - begin);
      SparseMatrixuf XMiddle = data.Xtrain.middleCols(begin, end - begin);
      mm(WX, W, CblasNoTrans,
        XMiddle,
        CblasNoTrans, 1.0, 0.0L);
      return L(model.params.Z, data.Ytrain, gaussianKernel(model.params.B, WX, model.hyperParams.gamma), begin, end);
    },
      // [&(model.params.B), &(data.Ytrain), &(model.params.Z), &(data.Xtrain), &(model.hyperParams)]
      [&model, &data]
    (const WMatType& W, const Eigen::Index begin, const Eigen::Index end)
      ->MatrixXuf {
      MatrixXuf WX = MatrixXuf::Zero(W.rows(), end - begin);
      SparseMatrixuf XMiddle = data.Xtrain.middleCols(begin, end - begin);
      mm(WX, W, CblasNoTrans,
        XMiddle,
        CblasNoTrans, 1.0, 0.0L);
      return gradL_W(model.params.B, data.Ytrain, model.params.Z, W, data.Xtrain,
        gaussianKernel(model.params.B, WX, model.hyperParams.gamma),
        model.hyperParams.gamma, begin, end);
    },
      std::bind(hardThrsd, std::placeholders::_1, model.hyperParams.lambdaW),
      model.params.W, epochs, n, bs, etaW, etaUpdate);
    timer.nextTime("ending gradW");
    //LOG_INFO("Final step-length for gradW = " + std::to_string(etaW));

    mm(WX, model.params.W, CblasNoTrans, data.Xtrain, CblasNoTrans, 1.0, 0.0L);
    if (data.Xvalidation.cols() > 0) {
      mm(WXvalidation, model.params.W, CblasNoTrans, data.Xvalidation, CblasNoTrans, 1.0, 0.0L);
    }

    fOld = fNew;
#ifdef XML
    mm(WX_sub, model.params.W, CblasNoTrans, X_sub, CblasNoTrans, 1.0, 0.0L);
    if (data.Xvalidation.cols() > 0) {
      mm(WXvalidation_sub, model.params.W, CblasNoTrans, Xvalidation_sub, CblasNoTrans, 1.0, 0.0L);
    }
    fNew = batchEvaluate(model.params.Z, Y_sub, Yvalidation_sub, model.params.B, WX_sub, WXvalidation_sub, model.hyperParams.gamma, model.hyperParams.problemType, stats + 9 * i + 3);
#else 
    fNew = batchEvaluate(model.params.Z, data.Ytrain, data.Yvalidation, model.params.B, WX, WXvalidation, model.hyperParams.gamma, model.hyperParams.problemType, stats + 9 * i + 3);
#endif 

    if (fNew >= fOld * (1 + safeDiv(sgdTol*(FP_TYPE)log(3), (FP_TYPE)log(2 + i))))
      armijoW *= (FP_TYPE)0.7;
    else if (fNew <= fOld * (1 - safeDiv(3 * sgdTol*(FP_TYPE)log(3), (FP_TYPE)log(2 + i))))
      armijoW *= (FP_TYPE)1.1;
    else;

#ifdef VERIFY
    fileName = outDir + "/verify/W" + std::to_string(i);
    f.open(fileName);
    f << "W_check = [" << model.params.W << "];" << std::endl;
    f.close();
#endif 
#ifdef DUMP 
    fileName = outDir + "/dump/W" + std::to_string(i);
    f.open(fileName);
    f << model.params.W.format(eigen_tsv);
    f.close();
#endif 

    timer.nextTime("starting optimization w.r.t. Z");
    LOG_INFO("Optimizing w.r.t. prototype-label matrix (Z)...");

#ifdef BTLS
    etaZ = armijoZ * btls<ZMatType>
      ([&model, &data, &WX]
	(const ZMatType& Z, const Eigen::Index begin, const Eigen::Index end)
	->FP_TYPE {return L(Z, data.Ytrain, gaussianKernel(model.params.B, WX, model.hyperParams.gamma, begin, end), begin, end); },
	[&model, &data, &WX]
	(const ZMatType& Z, const Eigen::Index begin, const Eigen::Index end)
	->MatrixXuf
      {return gradL_Z(Z, data.Ytrain,
		      gaussianKernel(model.params.B, WX, model.hyperParams.gamma, begin, end),
		      begin, end); },
	std::bind(hardThrsd, std::placeholders::_1, model.hyperParams.lambdaZ),
       model.params.Z, n, bs, (etaZ/armijoZ)*2);
#else
    for (auto j = 0; j < eta.size(); ++j) { //eta.size(); ++j) {
      Eigen::Index idx1 = (j*(Eigen::Index)hessianbs) % n;
      Eigen::Index idx2 = ((j + 1)*(Eigen::Index)hessianbs) % n;
      if (idx2 <= idx1) idx2 = n;

      gtmpZ = gradL_Z(model.params.Z, data.Ytrain,
        gaussianKernel(model.params.B, WX, model.hyperParams.gamma, idx1, idx2),
        idx1, idx2);

      MatrixXuf gtmpZThresh = gtmpZ;
      hardThrsd(gtmpZThresh, model.hyperParams.lambdaZ);

      // Below: Ztmp = Z - 0.001*safeDiv(maxAbsVal(Z), gtmpZ.cwiseAbs().maxCoeff()) * gtmpZThresh;
      gtmpZThresh *= (FP_TYPE)-0.001*safeDiv(maxAbsVal(model.params.Z), gtmpZ.cwiseAbs().maxCoeff());
      typeMismatchAssign(Ztmp, gtmpZThresh);
      Ztmp += model.params.Z;

      gtmpZ -= gradL_Z(Ztmp, data.Ytrain,
        gaussianKernel(model.params.B, WX, model.hyperParams.gamma, idx1, idx2),
        idx1, idx2);

      if (gtmpZ.norm() <= 1e-20L) {
        LOG_WARNING("Difference between consecutive gradients of Z has become really low.");
        eta(j) = 1.0;
      }
      else
        eta(j) = safeDiv((Ztmp - model.params.Z).norm(), gtmpZ.norm());
    }
    std::sort(eta.data(), eta.data() + eta.size());
    etaZ = armijoZ * eta(4);
#endif
    //LOG_INFO("Step-length estimate for gradZ = " + std::to_string(etaZ));
    
    accProxSGD<ZMatType>
      (//[&model.params.B, &data.Ytrain, &WX, &model.hyperParams] 
        [&model, &data, &WX]
    (const ZMatType& Z, const Eigen::Index begin, const Eigen::Index end)
      ->FP_TYPE {return L(Z, data.Ytrain, gaussianKernel(model.params.B, WX, model.hyperParams.gamma, begin, end), begin, end); },
      //[&WX, &data.Ytrain, &model.params.B, &model.hyperParams]
      [&model, &data, &WX]
    (const ZMatType& Z, const Eigen::Index begin, const Eigen::Index end)
      ->MatrixXuf
    {return gradL_Z(Z, data.Ytrain,
      gaussianKernel(model.params.B, WX, model.hyperParams.gamma, begin, end),
      begin, end); },
      std::bind(hardThrsd, std::placeholders::_1, model.hyperParams.lambdaZ),
      model.params.Z, epochs, n, bs, etaZ, etaUpdate);
    timer.nextTime("ending gradZ");
    //LOG_INFO("Final step-length for gradZ = " + std::to_string(etaZ));

    fOld = fNew;
#ifdef XML
    fNew = batchEvaluate(model.params.Z, Y_sub, Yvalidation_sub, model.params.B, WX_sub, WXvalidation_sub, model.hyperParams.gamma, model.hyperParams.problemType, stats + 9 * i + 6);
#else 
    fNew = batchEvaluate(model.params.Z, data.Ytrain, data.Yvalidation, model.params.B, WX, WXvalidation, model.hyperParams.gamma, model.hyperParams.problemType, stats + 9 * i + 6);
#endif

    if (fNew >= fOld * (1 + safeDiv(sgdTol*(FP_TYPE)log(3), (FP_TYPE)log(2 + i))))
      armijoZ *= (FP_TYPE)0.7;
    else if (fNew <= fOld * (1 - safeDiv(3 * (FP_TYPE)sgdTol*(FP_TYPE)log(3), (FP_TYPE)log(2 + i))))
      armijoZ *= (FP_TYPE)1.1;
    else;

#ifdef VERIFY
    fileName = outDir + "/verify/Z" + std::to_string(i);
    f.open(fileName);
    f << "Z_check = [" << model.params.Z << "];" << std::endl;
    f.close();
#endif 
#ifdef DUMP
    fileName = outDir + "/dump/Z" + std::to_string(i);
    f.open(fileName);
    f << model.params.Z.format(eigen_tsv);
    f.close();
#endif 

    timer.nextTime("starting optimization w.r.t. B");
    LOG_INFO("Optimizing w.r.t. prototype matrix (B)...");

#ifdef BTLS
    etaB = armijoB * btls<BMatType>
      ([&model, &data, &WX]
       (const BMatType& B, const Eigen::Index begin, const Eigen::Index end)
       ->FP_TYPE {return L(model.params.Z, data.Ytrain, gaussianKernel(B, WX, model.hyperParams.gamma, begin, end), begin, end); },
       [&model, &data, &WX]
       (const BMatType& B, const Eigen::Index begin, const Eigen::Index end)
       ->MatrixXuf
      {return gradL_B(B, data.Ytrain, model.params.Z, WX,
		      gaussianKernel(B, WX, model.hyperParams.gamma, begin, end),
		      model.hyperParams.gamma, begin, end); },
       std::bind(hardThrsd, std::placeholders::_1, model.hyperParams.lambdaB),
       model.params.B, n, bs, (etaB/armijoB)*2);
#else    
    for (auto j = 0; j < eta.size(); ++j) {
      Eigen::Index idx1 = (j*(Eigen::Index)hessianbs) % n;
      Eigen::Index idx2 = ((j + 1)*(Eigen::Index)hessianbs) % n;
      if (idx2 <= idx1) idx2 = n;

      gtmpB = gradL_B(model.params.B, data.Ytrain, model.params.Z, WX,
        gaussianKernel(model.params.B, WX, model.hyperParams.gamma, idx1, idx2),
        model.hyperParams.gamma, idx1, idx2);

      MatrixXuf gtmpBThresh = gtmpB;
      hardThrsd(gtmpBThresh, model.hyperParams.lambdaB);

      Btmp = model.params.B - 0.001*safeDiv(model.params.B.cwiseAbs().maxCoeff(), gtmpB.cwiseAbs().maxCoeff())*gtmpBThresh;

      gtmpB -= gradL_B(Btmp, data.Ytrain, model.params.Z, WX,
        gaussianKernel(Btmp, WX, model.hyperParams.gamma, idx1, idx2),
        model.hyperParams.gamma, idx1, idx2);

      if (gtmpB.norm() <= 1e-20L) {
        LOG_WARNING("Difference between consecutive gradients of B has become really low.");
        eta(j) = 1.0;
      }
      else
        eta(j) = safeDiv((Btmp - model.params.B).norm(), gtmpB.norm());
    }

    std::sort(eta.data(), eta.data() + eta.size());
    etaB = armijoB * eta(4);
#endif
    //LOG_INFO("Step-length estimate for gradB = " + std::to_string(etaB));

    accProxSGD<BMatType>
      (//[&model.params.Z, &data.Ytrain, &WX, &model.hyperParams] 
        [&model, &data, &WX]
    (const BMatType& B, const Eigen::Index begin, const Eigen::Index end)
      ->FP_TYPE {return L(model.params.Z, data.Ytrain, gaussianKernel(B, WX, model.hyperParams.gamma, begin, end), begin, end); },
      //[&WX, &data.Ytrain, &model.params.Z, &model.hyperParams]
      [&model, &data, &WX]
    (const BMatType& B, const Eigen::Index begin, const Eigen::Index end)
      ->MatrixXuf
    {return gradL_B(B, data.Ytrain, model.params.Z, WX,
      gaussianKernel(B, WX, model.hyperParams.gamma, begin, end),
      model.hyperParams.gamma, begin, end); },
      std::bind(hardThrsd, std::placeholders::_1, model.hyperParams.lambdaB),
      model.params.B, epochs, n, bs, etaB, etaUpdate);
    timer.nextTime("ending gradB");
    //LOG_INFO("Final step-length for gradB = " + std::to_string(etaB));

    fOld = fNew;
#ifdef XML
    fNew = batchEvaluate(model.params.Z, Y_sub, Yvalidation_sub, model.params.B, WX_sub, WXvalidation_sub, model.hyperParams.gamma, model.hyperParams.problemType, stats + 9 * i + 9);
#else 
    fNew = batchEvaluate(model.params.Z, data.Ytrain, data.Yvalidation, model.params.B, WX, WXvalidation, model.hyperParams.gamma, model.hyperParams.problemType, stats + 9 * i + 9);
#endif

    if (fNew >= fOld * (1 + safeDiv(sgdTol*(FP_TYPE)log(3), (FP_TYPE)log(2 + i))))
      armijoB *= (FP_TYPE)0.7;
    else if (fNew <= fOld * (1 - safeDiv(3 * sgdTol*(FP_TYPE)log(3), (FP_TYPE)log(2 + i))))
      armijoB *= (FP_TYPE)1.1;
    else;

#ifdef VERIFY
    fileName = outDir + "/verify/B" + std::to_string(i);
    f.open(fileName);
    f << "B_check = [" << model.params.B << "];" << std::endl;
    f.close();
#endif 
#ifdef DUMP
    fileName = outDir + "/dump/B" + std::to_string(i);
    f.open(fileName);
    f << model.params.B.format(eigen_tsv);
    f.close();
#endif 
  }
}

// function v = accuracy(Ytrue, D, Z, k)
// We have set k = inf permanently
// computes accuracy for binary/multiclass datasets, and prec1, prec3, prec5 for multilabel datasets
FP_TYPE EdgeML::accuracy(
  const ZMatType& Z, const LabelMatType& Y, const MatrixXuf& D,
  const EdgeML::ProblemFormat& problemType)
{
  Timer timer("accuracy");
  assert(Y.cols() == D.rows());
  //Ypred = Z*D';
  MatrixXuf Ytrue(Y);
  MatrixXuf Yscore(Ytrue.rows(), Ytrue.cols());
  mm(Yscore, Z, CblasNoTrans, D, CblasTrans, 1.0, 0.0L);
  timer.nextTime("computing Yscore");
  FP_TYPE ret = 0;

  if (problemType == EdgeML::ProblemFormat::binary || problemType == EdgeML::ProblemFormat::multiclass) {
    //[~,Ytrue] = max(Ytrue);
    //[~,Ypred] = max(Ypred);
    dataCount_t Ytrue_, Ypred_;

    //v = numel(find(Ypred == Ytrue))/numel(Ypred);
    for (Eigen::Index i = 0; i < Ytrue.cols(); ++i) {
      Ytrue.col(i).maxCoeff(&Ytrue_);
      Yscore.col(i).maxCoeff(&Ypred_);

      if (Ytrue_ == Ypred_)
        ret += 1;
    }

    assert(Y.cols() != 0);
    ret = safeDiv(ret, (FP_TYPE)Y.cols());
  }
  else if (problemType == EdgeML::ProblemFormat::multilabel) {
    FP_TYPE prec1(0), prec3(0), prec5(0);
    const labelCount_t k = 5;

    assert(k * Yscore.cols() < 3e9);
    std::vector<labelCount_t> topInd(k * Yscore.cols());
    pfor(Eigen::Index i = 0; i < Ytrue.cols(); ++i) {
      for (Eigen::Index j = 0; j < Ytrue.rows(); ++j) {
        FP_TYPE val = Yscore(j, i);
        if (j >= k && (val < Yscore(topInd[i*k + (k - 1)], i)))
          continue;
        size_t top = std::min(j, (Eigen::Index)k - 1);
        while (top > 0 && (Yscore(topInd[i*k + (top - 1)], i) < val)) {
          topInd[i*k + (top)] = topInd[i*k + (top - 1)];
          top--;
        }
        topInd[i*k + top] = j;
      }
    }

    timer.nextTime("finding max values");

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

    timer.nextTime("precision computation");

    dataCount_t totLabel = Y.cols();
    assert(totLabel != 0);
    ret = prec1;
    ret = safeDiv(ret, (FP_TYPE)totLabel);

    /*LOG_INFO("prec@1: " + std::to_string(safeDiv(prec1, (FP_TYPE) totLabel))
      + "  prec@3: " + std::to_string(safeDiv(prec3, 3 * (FP_TYPE)totLabel))
      + "  prec@5: " + std::to_string(safeDiv(prec5, 5 * (FP_TYPE)totLabel)));
    */
  }

  return ret;
}


template<class ParamType>
FP_TYPE EdgeML::btls(std::function<FP_TYPE(const ParamType&,
  const Eigen::Index, const Eigen::Index)> f,
  std::function<MatrixXuf(const ParamType&,
    const Eigen::Index, const Eigen::Index)> gradf,
  std::function<void(MatrixXuf&)> prox,
  ParamType& param,
  const dataCount_t& n,
  const dataCount_t& bs,
  FP_TYPE initialStepSizeEstimate)
{
  Timer timer("btls");
  Logger logger("btls");

  const FP_TYPE beta = 0.7;
  FP_TYPE stepSize = 0.0; 
  if (initialStepSizeEstimate < 0) stepSize = 1.0;
  else stepSize = initialStepSizeEstimate; 

  if (bs > n) {
    LOG_INFO("btls called with batch-size more than #train points.");
    assert(bs <= n);
  }

  MatrixXuf effectiveStep = MatrixXuf::Zero(param.rows(), param.cols());
  ParamType paramNew = param; 
  FP_TYPE f1, f2, gradIp;
  
  Eigen::Index randStartIndex; 
  if (n > bs) 
    randStartIndex = rand()%(n-bs);
  else
    randStartIndex = 0;
  
  Eigen::Index randEndIndex = randStartIndex + bs; 
  MatrixXuf gradParam = gradf(param, randStartIndex, randEndIndex);
  FP_TYPE fParam = f(param, randStartIndex, randEndIndex); 

  while (true){
    effectiveStep = -stepSize * gradParam;
    effectiveStep += param;
    prox(effectiveStep);
    effectiveStep = param - effectiveStep;
    //effectiveStep += param; 
    //effectiveStep /= stepSize; 

    paramNew = param - effectiveStep; 
    f1 = f(paramNew, randStartIndex, randEndIndex); 
    gradIp = gradParam.cwiseProduct(effectiveStep).sum(); 
    f2 = fParam - gradIp + (0.5/stepSize)*effectiveStep.squaredNorm(); 
    if (f1 <= f2) break;
    else stepSize = beta * stepSize; 
  }

  return stepSize; 
}

template<class ParamType>
void EdgeML::accProxSGD(std::function<FP_TYPE(const ParamType&,
  const Eigen::Index, const Eigen::Index)> f,
  std::function<MatrixXuf(const ParamType&,
    const Eigen::Index, const Eigen::Index)> gradf,
  std::function<void(MatrixXuf&)> prox,
  ParamType& param,
  const int& epochs,
  const dataCount_t& n,
  const dataCount_t& bs,
  FP_TYPE eta,
  const int& etaUpdate)
{
  Timer timer("accProxSGD");
  Logger logger("accProxSGD ");

  ParamType paramTailAverage = param;                              // Stores the tail averaged gradient that is finally returned 
  MatrixXuf temp = MatrixXuf::Zero(param.rows(), param.cols());    // A dense matrix to hold intermediate param-sized matrices
  ParamType currentUpdate;                                         // Stores momentum term for accProxSGD; corresponds to vanilla gradient 
                                                                   // descent update for current iterate
  ParamType prevUpdate = param;                                    // A matrix to hold previous update (ie, currentUpdate value of last iteration)

  int burnPeriod = 50;
  FP_TYPE gamma0 = 1;
  FP_TYPE stepSize, gamma, alpha;
  timer.nextTime("creating matrices");

  if (bs > n) {
    LOG_INFO("accelerated proximal gradient descent called with batch-size more than #train points.");
    assert(bs <= n);
  }
  
  uint64_t iters_ = ((uint64_t)n*(uint64_t)epochs) / (uint64_t)bs;
  assert(iters_ < 0x7fffffff);
  auto iters = iters_;

  for (int i = 0; i < iters; ++i) {
    Eigen::Index idx1 = (i*(Eigen::Index)bs) % n;
    Eigen::Index idx2 = ((i + 1)*(Eigen::Index)bs) % n;
    if (idx2 <= idx1) idx2 = n;

    switch (etaUpdate) {
    case -1:
      stepSize = safeDiv(eta, (1 + (FP_TYPE)0.2 * ((FP_TYPE)i + (FP_TYPE)1.0)));
      break;
    case 0:
      stepSize = safeDiv(eta, pow((FP_TYPE)i + (FP_TYPE)1.0, (FP_TYPE)0.5));
      break;
    }

    gamma = (FP_TYPE)0.5 + (FP_TYPE)0.5 * pow(1 + 4 * gamma0*gamma0, (FP_TYPE)0.5);
    alpha = safeDiv((1 - gamma0), gamma);

    temp = (-stepSize) * gradf(param, idx1, idx2);
    // Store (gradient * step size) in temporary matrix
    timer.nextTime("taking gradient");

    temp += param;
    // temp now holds the destination reached after previous update
    timer.nextTime("computing new destination (temp now stores the new parameter after a vanilla gradient step)");

    prox(temp);
    timer.nextTime("L0 projection (sparsifying parameter matrix after taking densifying gradient step)");

    typeMismatchAssign(currentUpdate, temp);
    // Store the current update in a temporary matrix (required for the next update)
    timer.nextTime("storing current destination in temp matrix");

    temp = (1 - alpha) * temp;
    temp += (alpha * prevUpdate);
    // temp now stores the new update = (momentum term + current update)
    // Momentum term is same as previous update
    timer.nextTime("computing new parameter value taking into account current and previous update (accelerated sgd step)");

    typeMismatchAssign(param, temp);
    timer.nextTime("assigning computed destination to the true parameter variable");

    gamma0 = gamma;
    prevUpdate = currentUpdate;
    timer.nextTime("");
    FP_TYPE tmp = ((i - burnPeriod) > 1) ? (i - burnPeriod) : (FP_TYPE)1.0;
    assert(tmp >= 0.999999);

    paramTailAverage = (safeDiv(tmp - (FP_TYPE)1.0, tmp))*paramTailAverage;
    paramTailAverage += safeDiv(1.0, tmp)*param;
    // Tail averaging
    timer.nextTime("updating paramTailAverage");

    FP_TYPE eps = (FP_TYPE)1e-6;
#ifdef CILK
    cilk::reducer< cilk::op_add<size_t> > nnz(0);
#else
    int *nnz = new int;
    *nnz = 0;
#endif
    pfor(std::ptrdiff_t i = 0; i < param.rows()*param.cols(); ++i) {
      if (std::abs(param.data()[i]) > eps)
        *nnz += 1;
    }
#ifndef CILK
    delete nnz;
#endif
  }

  param = paramTailAverage;
  eta = stepSize;
}
