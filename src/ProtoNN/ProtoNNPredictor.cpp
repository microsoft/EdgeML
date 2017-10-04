// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "blas_routines.h"
#include "Data.h"
#include "ProtoNNFunctions.h"


using namespace EdgeML;
using namespace EdgeML::ProtoNN;

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
  model.hyperParams.batchSize = batchSize;
  testData = Data(dataIngestType,
                  DataFormatParams{
                  0, // set the ntrain to zero
                  0, // set the nvalidation to zero
                  model.hyperParams.ntest,
                  model.hyperParams.l,
                  model.hyperParams.D });

  createOutputDirs();

#ifdef TIMER
  OPEN_TIMER_LOGFILE(outdir);
#endif

#ifdef LIGHT_LOGGER
  OPEN_DIAGNOSTIC_LOGFILE(outdir);
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

  normalize();

  data = NULL; //set to NULL, used for InterfaceIngest
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
  std::string outsubdir = outdir + "/ProtoNNPredictor_" + subdirName;

  try {
    std::string testcommand1 = "test -d " + outdir;
    std::string command1 = "mkdir " + outdir;
    std::string testcommand2 = "test -d " + outsubdir;
    std::string command2 = "mkdir " + outsubdir;

#ifdef LINUX
    if (system(testcommand1.c_str()) == 0)
      LOG_INFO("ProtoNNResults subdirectory exists within data folder");
    else
      if (system(command1.c_str()) != 0)
        LOG_WARNING("Error in creating results subdir");

    if (system(testcommand2.c_str()) == 0)
      LOG_INFO("output subdirectory exists within data folder");
    else
      if (system(command2.c_str()) != 0)
        LOG_WARNING("Error in creating subdir for current hyperParams");
#endif

#ifdef DUMP
    std::string testcommand3 = "test -d " + outdir + "/dump";
    std::string command3 = "mkdir " + outdir + "/dump";
#ifdef LINUX
    if (system(testcommand3.c_str()) == 0)
      LOG_INFO("directory for dump within output directory exists");
    else
      if (system(command3.c_str()) != 0)
        LOG_WARNING("Error in creating subdir for dumping intermediate models");
#endif
#endif
#ifdef VERIFY
    std::string testcommand4 = "test -d " + outdir + "/verify";
    std::string command4 = "mkdir " + outdir + "/verify";
#ifdef LINUX
    if (system(testcommand4.c_str()) == 0)
      LOG_INFO("directory for verification log within output directory exists");
    else
      if (system(command4.c_str()) != 0)
        LOG_WARNING("Error in creating subdir for verification dumps");
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
          ntest = std::stoi(argv[i]);
          break;

        case 'M':
          modelFile = argv[i];
          break;

        case 'n':
          normParamFile = argv[i];
          break;
  
        case 'O': //currently not used
          outdir = argv[i];
          break;

        case 'F':
          if (argv[i][0] == '0') dataformatType = libsvmFormat;
          else if (argv[i][0] == '1') dataformatType = tsvFormat;
          else if (argv[i][0] == '2') dataformatType = MNISTFormat;
          else assert(false); //Format unknown
          break;

        case 'b':
          batchSize = std::stoi(argv[i]);
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

ProtoNNPredictor::ResultStruct ProtoNNPredictor::evaluateScores()
{
  ProtoNNPredictor::ResultStruct res;
  FP_TYPE stats[12]; // currently, stats are not being used

  MatrixXuf WX = MatrixXuf::Zero(model.params.W.rows(), testData.Xtest.cols());
  mm(WX, model.params.W, CblasNoTrans, testData.Xtest, CblasNoTrans, 1.0, 0.0L);
  batchEvaluate(model.params.Z, testData.Ytest, model.params.B, WX, model.hyperParams.gamma, model.hyperParams.problemType, res, stats);

//  delete[] stats;
  return res;
}

void ProtoNNPredictor::normalize()
{
  NormalizationFormat normalizationType = model.hyperParams.normalizationType;
  assert((normalizationType == minMax) || (normalizationType == none));

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


