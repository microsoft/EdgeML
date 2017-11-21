// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "blas_routines.h"
#include "Data.h"
#include "ProtoNNFunctions.h"


using namespace EdgeML;
using namespace EdgeML::ProtoNN;

ProtoNNPredictor::ProtoNNPredictor(
  const int& argc,
  const char ** argv)
{
  // initialize member variables
  batchSize = 0;
  ntest = 0;
  dataformatType = undefinedData; 
  dataPoint = NULL;
  
  commandLine = "";
  for (int i = 0; i < argc; ++i)
    commandLine += (std::string(argv[i]) + " ");
  setFromArgs(argc, argv);

  LOG_INFO("Attempting to load a model from the file: " + modelFile + "\n");
  model = ProtoNNModel(modelFile);
  model.hyperParams.ntest = ntest;

  assert(ntest > 0);
  
  testData = Data(FileIngest,
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

  // if batchSize is not set, then we want to do point-wise prediction
  if (batchSize == 0){
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

    dataPoint = new FP_TYPE[model.hyperParams.D];
  }
  
#ifdef SPARSE_Z_PROTONN
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

  dataPoint = new FP_TYPE[model.hyperParams.D];

#ifdef SPARSE_Z_PROTONN
  ZRows = model.params.Z.rows();
  ZCols = model.params.Z.cols();
  alpha = 1.0;
  beta = 0.0;
#endif
}

void ProtoNNPredictor::createOutputDirs()
{
  std::string subdirName = model.hyperParams.subdirName();

#ifdef LINUX
  outDir = outDir + "/ProtoNNPredictor_" + subdirName;
#endif

#ifdef WINDOWS
  outDir = outDir + "\\ProtoNNPredictor_" + subdirName;
#endif

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

#ifdef WINDOWS
    if (system(command.c_str()) != 0)
        LOG_WARNING("Error in creating directory at this location: " + outDir + " (Directory might already exist)");
#endif


#ifdef DUMP
#ifdef LINUX
    testcommand = "test -d " + outDir + "/dump";
    command = "mkdir " + outDir + "/dump";
    if (system(testcommand.c_str()) == 0)
      LOG_INFO("Directory " + outDir + "/dump already exists.");
    else
      if (system(command.c_str()) != 0)
        LOG_WARNING("Error in creating directory at this location: " + outDir + "/dump");
#endif

#ifdef WINDOWS
    command = "mkdir " + outDir + "\\dump";
    if (system(command.c_str()) != 0)
      LOG_WARNING("Error in creating directory at this location: " + outDir + "\dump" + " (Directory might already exist)");
#endif
#endif

#ifdef VERIFY
#ifdef LINUX
    testcommand = "test -d " + outDir + "/verify";
    command = "mkdir " + outDir + "/verify";
    if (system(testcommand.c_str()) == 0)
      LOG_INFO("Directory " + outDir + "/verify already exists.");
    else
      if (system(command.c_str()) != 0)
        LOG_WARNING("Error in creating directory at this location: " + outDir + "/verify");
#endif

#ifdef WINDOWS
    command = "mkdir " + outDir + "\\verify";
    if (system(command.c_str()) != 0)
      LOG_WARNING("Error in creating directory at this location: " + outDir + "\verify" + " (Directory might already exist)");
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
  if(dataPoint)
    delete[] dataPoint;
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
#ifdef SPARSE_Z_PROTONN
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
}

void ProtoNNPredictor::scoreSparseDataPoint(
  FP_TYPE* scores,
  const FP_TYPE *const values,
  const featureCount_t *indices,
  const featureCount_t numIndices)
  
{
  memset(dataPoint, 0, sizeof(FP_TYPE)*model.hyperParams.D);

  pfor(featureCount_t i = 0; i < numIndices; ++i) {
    assert(indices[i] < model.hyperParams.D);
    dataPoint[indices[i]] = values[i];
  }

  gemv(CblasColMajor, CblasNoTrans,
    model.params.W.rows(), model.params.W.cols(),
    1.0, model.params.W.data(), model.params.W.rows(),
    dataPoint, 1, 0.0, WX.data(), 1);

  //  MatrixXuf D = gaussianKernel(model.params.B, WX, model.hyperParams.gamma);
  RBF();

  //  mm(scoresMat, model.params.Z, CblasNoTrans, D, CblasTrans, 1.0, 0.0L);
#ifdef SPARSE_Z_PROTONN
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
      assert(!normParamFile.empty() && "Normalization parameteres file for min-max normalization needs to be provided");
      loadMinMax(testData.min, testData.max, testData.Xtest.rows(), normParamFile);
      minMaxNormalize(testData.Xtest, testData.min, testData.max);
      LOG_INFO("Completed min-max normalization of test data\n");
      break;

    case l2:
      l2Normalize(testData.Xtest);
      LOG_INFO("Completed l2 normalization of test data\n");
      break;

    case none:
      break;

    default:
      assert(false);
  }
}

EdgeML::ResultStruct ProtoNNPredictor::test()
{
  if(batchSize == 0) return testPointWise();
  else return testBatchWise();
}
  
EdgeML::ResultStruct ProtoNNPredictor::testBatchWise()
{
  dataCount_t n;
  
  n = testData.Xtest.cols();
  
  assert(n > 0);

  dataCount_t nBatches = (n + batchSize - 1)/ batchSize;

  EdgeML::ResultStruct res, tempRes;
  for (dataCount_t i = 0; i < nBatches; ++i) {
    Eigen::Index startIdx =  i * batchSize;
    dataCount_t curBatchSize = (batchSize < n - startIdx)? batchSize : n - startIdx;
    MatrixXuf Yscores = MatrixXuf::Zero(model.hyperParams.l, curBatchSize);
    scoreBatch(Yscores, startIdx, curBatchSize); 
    
    tempRes = evaluate(Yscores, testData.Ytest.middleCols(startIdx, curBatchSize), model.hyperParams.problemType);
    
    res.scaleAndAdd(tempRes, curBatchSize);
  }
  res.scale(1/(FP_TYPE)n);
  return res;
}

EdgeML::ResultStruct ProtoNNPredictor::testPointWise()
{  
  dataCount_t n;
  FP_TYPE *scores, *featureValues;
  featureCount_t *featureIndices, numIndices;

  n = testData.Xtest.cols();
  assert(n > 0);

  scores = new FP_TYPE[model.hyperParams.l];
  Map<MatrixXuf> Yscores(scores, model.hyperParams.l, 1);

  EdgeML::ResultStruct res, tempRes;
  for (dataCount_t i = 0; i < n; ++i) {
	scoreSparseDataPoint(scores,
		(const FP_TYPE*) testData.Xtest.valuePtr() + testData.Xtest.outerIndexPtr()[i],
		(const featureCount_t*) testData.Xtest.innerIndexPtr() + testData.Xtest.outerIndexPtr()[i],
		(featureCount_t) testData.Xtest.outerIndexPtr()[i + 1] - testData.Xtest.outerIndexPtr()[i]);

    tempRes = evaluate(Yscores, testData.Ytest.middleCols(i, 1), model.hyperParams.problemType);
    res.scaleAndAdd(tempRes, 1);
  }
  res.scale(1/(FP_TYPE)n);

  delete[] scores;
  
  return res;
}

void ProtoNNPredictor::saveTopKScores(std::string filename, int topk)
{
  dataCount_t tempBatchSize = batchSize;
  if(tempBatchSize == 0) tempBatchSize = 1; 

  const dataCount_t n = testData.Ytest.cols();
  assert(n > 0);

  if (topk < 1)
    topk = 5;

  if (filename.empty())
      filename = outDir + "/detailedPrediction";
  LOG_INFO("Attempting to open the following file for detailed prediction output: " + filename);
  std::ofstream outfile(filename);
  assert(outfile.is_open());

  dataCount_t nBatches = ((n + tempBatchSize - 1)/ tempBatchSize); 
  MatrixXuf topKindices, topKscores;
  for (dataCount_t i = 0; i < nBatches; ++i) {
    Eigen::Index startIdx =  i * tempBatchSize;
    dataCount_t curBatchSize = (tempBatchSize < n - startIdx)? tempBatchSize : n - startIdx;
    MatrixXuf Yscores = MatrixXuf::Zero(model.hyperParams.l, curBatchSize);
    scoreBatch(Yscores, startIdx, curBatchSize); 
    getTopKScoresBatch(Yscores, topKindices, topKscores, topk); 

    for (Eigen::Index j = 0; j < topKindices.cols(); j++) {
      for (SparseMatrixuf::InnerIterator it(testData.Ytest, i*tempBatchSize+j); it; ++it)
        outfile << it.row() << ",  ";
      for (Eigen::Index k = 0; k < topKindices.rows(); k++) {
        outfile << topKindices(k, j) << ":" << topKscores(k, j) << "  ";
      }
      outfile << std::endl;
    }
  }

  outfile.close();
}
