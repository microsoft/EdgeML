// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "blas_routines.h"
#include "Data.h"
#include "ProtoNNFunctions.h"


using namespace EdgeML;
using namespace EdgeML::ProtoNN;


ProtoNNPredictor::ResultStruct::ResultStruct()
  :
  problemType(undefinedProblem),
  accuracy(0),
  precision1(0),
  precision3(0),
  precision5(0)
{}


inline void ProtoNNPredictor::ResultStruct::scaleAndAdd(ProtoNNPredictor::ResultStruct& a, FP_TYPE scale)
{
  if ((problemType == undefinedProblem) && (a.problemType != undefinedProblem))
    problemType = a.problemType;
  assert(problemType == a.problemType);

  accuracy += scale * a.accuracy;
  precision1 += scale * a.precision1;
  precision3 += scale * a.precision3;
  precision5 += scale * a.precision5;
}

inline void ProtoNNPredictor::ResultStruct::scale(FP_TYPE scale)
{
  accuracy *= scale;
  precision1 *= scale;
  precision3 *= scale;
  precision5 *= scale;
}

ProtoNNPredictor::ProtoNNPredictor(
  const DataIngestType& dataIngestType,
  const int& argc,
  const char ** argv)
{
  assert(dataIngestType == FileIngest);

  commandLine = "";
  for (int i = 0; i < argc; ++i)
    commandLine += (std::string(argv[i]) + " ");
  setFromArgs(argc, argv);

  model = ProtoNNModel(modelFile);
  model.hyperParams.ntest = ntest;
  
  testData = Data(dataIngestType,
                  DataFormatParams{
                  0, // set the ntrain to zero
                  0, // set the nvalidation to zero
                  model.hyperParams.ntest,
                  model.hyperParams.l,
                  model.hyperParams.D });

  createOutputDirs();

#ifdef TIMER
  OPEN_TIMER_LOGFILE(outDir);
#endif

#ifdef LIGHT_LOGGER
  OPEN_DIAGNOSTIC_LOGFILE(outDir);
#endif

  //Assert that the trainFile is properly assigned
  assert(!testFile.empty()); 

  // Pass empty string as train and validation file, since we do not need to load those
  std::string trainFile = "";
  std::string validationFile = "";
  testData.loadDataFromFile(dataformatType,
                trainFile,
                validationFile,
		testFile);
  testData.finalizeData();

  normalize();

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

void ProtoNNPredictor::createOutputDirs()
{
  std::string subdirName = model.hyperParams.subdirName();
  outDir = outDir + "/ProtoNNPredictor_" + subdirName;

  try {
    std::string testcommand = "test -d " + outDir;
    std::string command = "mkdir " + outDir;

#ifdef LINUX
    if (system(testcommand.c_str()) == 0)
      LOG_INFO("Directory " + outDir + " already exists.");
    else
      if (system(command.c_str()) != 0)
        LOG_WARNING("Error in creating directory at this location: " + outDir);
#endif

#ifdef DUMP
    testcommand = "test -d " + outDir + "/dump";
    command = "mkdir " + outDir + "/dump";
#ifdef LINUX
    if (system(testcommand.c_str()) == 0)
      LOG_INFO("Directory " + outDir + "/dump already exists.");
    else
      if (system(command.c_str()) != 0)
        LOG_WARNING("Error in creating directory at this location: " + outDir + "/dump");
#endif
#endif
#ifdef VERIFY
    testcommand = "test -d " + outDir + "/verify";
    command = "mkdir " + outDir + "/verify";
#ifdef LINUX
    if (system(testcommand.c_str()) == 0)
      LOG_INFO("Directory " + outDir + "/verify already exists.");
    else
      if (system(command.c_str()) != 0)
        LOG_WARNING("Error in creating directory at this location: " + outDir + "/verify");
#endif
#endif
  }
  catch (...) {
    LOG_WARNING("Error in creating one of the subdirectories. Some of the output may not be recorded.");
  }
}


void ProtoNNPredictor::setFromArgs(const int argc, const char** argv)
{
  for (int i = 1; i < argc; ++i) {

    if (i % 2 == 1)
      assert(argv[i][0] == '-'); //odd arguments must be specifiers, not values 
    else {
      switch (argv[i - 1][1]) {
        case 'I':
          testFile = argv[i];
          break;

        case 'e':
          ntest = strtol(argv[i], NULL, 0);
          break;

        case 'M':
          modelFile = argv[i];
          break;

        case 'n':
          normParamFile = argv[i];
          break;
  
        case 'O': 
          outDir = argv[i];
          break;

        case 'F':
          if (argv[i][0] == '0') dataformatType = libsvmFormat;
          else if (argv[i][0] == '1') dataformatType = tsvFormat;
          else if (argv[i][0] == '2') dataformatType = MNISTFormat;
          else assert(false); //Format unknown
          break;

        case 'b':
          batchSize = strtol(argv[i], NULL, 0);
          break;

/*
        case 'P':
        case 'C':
        case 'R':
        case 'g':
        case 'r':
        case 'e':
        case 'D':
        case 'l':
        case 'W':
        case 'Z':
        case 'B':
        case 'b':
        case 'd':
        case 'm':
        case 'k':
        case 'T':
        case 'E':
        case 'N':
          break;
*/

      default:
        LOG_INFO("Command line argument not recognized; saw character: " + argv[i - 1][1]);
        assert(false);
        break;
      }
    }
  }
}

ProtoNNPredictor::~ProtoNNPredictor()
{
  if (data)
    delete[] data;
}

FP_TYPE ProtoNNPredictor::testDenseDataPoint(
  const FP_TYPE *const values,
  const labelCount_t *const labels,
  const labelCount_t& num_labels,
  const ProblemFormat& problemType)
{
  //TODO: What about normalization

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
}

void ProtoNNPredictor::scoreBatch(
  MatrixXuf& Yscores,
  dataCount_t startIdx,
  dataCount_t batchSize)
{
  dataCount_t ntest = testData.Xtest.cols();
  assert(ntest > 0);
  assert(startIdx >= 0);
  assert(batchSize > 0);
  assert(startIdx + batchSize <= ntest);

  MatrixXuf curWX = MatrixXuf(model.params.W.rows(), batchSize);
  SparseMatrixuf curTestData = testData.Xtest.middleCols(startIdx, batchSize);
  mm(curWX, model.params.W, CblasNoTrans, curTestData, CblasNoTrans, 1.0, 0.0L);
  
  MatrixXuf curD = gaussianKernel(model.params.B, curWX, model.hyperParams.gamma);

  mm(Yscores, model.params.Z, CblasNoTrans, curD, CblasTrans, 1.0, 0.0L);
}

void ProtoNNPredictor::normalize()
{
  NormalizationFormat normalizationType = model.hyperParams.normalizationType;

  std::string minMaxFile;
  switch (normalizationType) {
    case minMax:
      assert(!normParamFile.empty());
      loadMinMax(testData.min, testData.max, testData.Xtest.rows(), normParamFile);
      minMaxNormalize(testData.Xtest, testData.min, testData.max);
      LOG_INFO("Completed min-max normalization of test data");
      break;

    case l2:
      l2Normalize(testData.Xtest);
      LOG_INFO("Completed l2 normalization of test data");
      break;

    case none:
      break;

    default:
      assert(false);
  }
}

ProtoNNPredictor::ResultStruct ProtoNNPredictor::evaluateBatchWise()
{
  dataCount_t n;
  
  n = testData.Xtest.cols();
  
  assert(n > 0);

  dataCount_t nbatches = n / batchSize + 1;

  ProtoNNPredictor::ResultStruct res, tempRes;
  for (dataCount_t i = 0; i < nbatches; ++i) {
    Eigen::Index startIdx =  i * batchSize;
    dataCount_t curBatchSize = (batchSize < n - startIdx)? batchSize : n - startIdx;
    MatrixXuf Yscores = MatrixXuf::Zero(model.hyperParams.l, curBatchSize);
    scoreBatch(Yscores, startIdx, curBatchSize); 
    
    tempRes = evaluateBatch(Yscores, testData.Ytest.middleCols(startIdx, curBatchSize));
    
    res.scaleAndAdd(tempRes, curBatchSize);
  }
  res.scale(1/(FP_TYPE)n);
  return res;
}

ProtoNNPredictor::ResultStruct ProtoNNPredictor::evaluatePointWise()
{
  dataCount_t n;
  FP_TYPE *scores, *featureValues;
  featureCount_t *featureIndices, numIndices;

  n = testData.testData.cols();
  assert(n > 0);

  scores = new FP_TYPE[model.hyperParams.l];
  Map<MatrixXuf> Yscores(scores, model.hyperParams.l, 1);

  ProtoNNPredictor::ResultStruct res, tempRes;
  for (dataCount_t i = 0; i < n; ++i) {
    FP_TYPE* values;
    
    featureCount_t numIndices = testData.Xtest.middleCols(i, 1).nonZeros();
    featureValues = new FP_TYPE[numIndices];
    featureIndices = new featureCount_t[numIndices];
    
    featureCount_t fIdx = 0;
    for (SparseMatrixuf::InnerIterator it(testData.Xtest, i); it; ++it) {
      featureValues[fIdx] = it.value();
      featureIndices[fIdx] = it.index();
      ++fIdx;
    }
    scoreSparseDataPoint(scores, featureValues, featureIndices, numIndices);
    tempRes = evaluateBatch(Yscores, testData.Ytest.middleCols(i, 1));
       
    delete[] featureValues;
    delete[] featureIndices;
    
    res.scaleAndAdd(tempRes, 1);
  }
  res.scale(1/(FP_TYPE)n);

  delete[] scores;
  
  return res;
}


// computes accuracy for binary/multiclass datasets, and prec1, prec3, prec5 for multilabel datasets
ProtoNNPredictor::ResultStruct ProtoNNPredictor::evaluateBatch(
  const MatrixXuf& Yscores, 
  const LabelMatType& Y)
{
  assert(Yscores.cols() == Y.cols());
  assert(Yscores.rows() == Y.rows());
  MatrixXuf Ytrue(Y);
  MatrixXuf Ypred = Yscores;

  FP_TYPE acc = 0;
  FP_TYPE prec1 = 0;
  FP_TYPE prec3 = 0;
  FP_TYPE prec5 = 0;

  ProblemFormat problemType = model.hyperParams.problemType;
  
  ProtoNNPredictor::ResultStruct res;
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

    dataCount_t totLabel = Y.cols();
    assert(totLabel != 0);
    res.precision1 = (prec1 /= (FP_TYPE)totLabel);
    res.precision3 = (prec3 /= ((FP_TYPE)totLabel)*3);
    res.precision5 = (prec5 /= ((FP_TYPE)totLabel)*5);
  }

  return res;
}



