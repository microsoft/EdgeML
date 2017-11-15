// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "blas_routines.h" 
#include "Bonsai.h"

using namespace EdgeML;
using namespace	EdgeML::Bonsai;

void BonsaiPredictor::exitWithHelp()
{
  LOG_INFO("./BonsaiPredictor [Options] \n");
  LOG_INFO("Options:");

  LOG_INFO("-f    : [Required] Input format. Takes two values [0 and 1]. 0 is for libsvmFormat(default), 1 is for tab/space separated input.");
  LOG_INFO("-N    : [Required] Number of data points in the test data.");
  LOG_INFO("-D    : [Required] Directory of data with test.txt present in it.");
  LOG_INFO("-M    : [Required] Directory of the Model (loadableModel and loadableMeanVar).");
  exit(1);
}


void BonsaiPredictor::setFromArgs(
  const int argc,
  const char** argv)
{
  int tempFormat;
  int required = 0;
  for(int i = 1; i < argc; i++) {
    if(i % 2 == 1)
      assert(argv[i][0] == '-');
    else {
      switch(argv[i-1][1]) {
          case 'f':
            tempFormat = int(atoi(argv[i]));
            required++;
            if (tempFormat == 0) dataformatType = DataFormat::libsvmFormat;
            else if (tempFormat == 1) dataformatType = DataFormat::interfaceIngestFormat;
            else exitWithHelp();
            break;
        case 'N':
          numTest = int(atoi(argv[i]));
          required++;
          break;
        case 'M':
          modelDir = argv[i];
          required++;
          break;
        case 'D':
          dataDir = argv[i];
          required++;
          break;
        default:
          LOG_INFO("Unknown option: " + std::to_string(argv[i - 1][1]));
          exitWithHelp();
          break;
      }
    }
  }
  if(required != 4) exitWithHelp();
}

BonsaiPredictor::BonsaiPredictor(
  const int argc,
  const char** argv)
{
  setFromArgs(argc, argv);
  std::string modelFile = modelDir + "/loadableModel"; 
 
  model = BonsaiModel(modelFile, 1);

  feedDataValBuffer = new FP_TYPE[model.hyperParams.dataDimension];
  feedDataFeatureBuffer = new labelCount_t[model.hyperParams.dataDimension];
  
  mean = MatrixXuf::Zero(model.hyperParams.dataDimension, 1);
  variance = MatrixXuf::Zero(model.hyperParams.dataDimension, 1);

  std::string meanVarFile = modelDir + "/loadableMeanVar"; 
  
  importMeanVar(meanVarFile);

  testData = Data(FileIngest,
    DataFormatParams{0, 0, numTest, model.hyperParams.numClasses, model.hyperParams.dataDimension});

  if(model.hyperParams.dataformatType != dataformatType)
    LOG_INFO("WARNING: The Train and Test input formats don't match.");

  testData.loadDataFromFile(dataformatType, "", "", dataDir + "/test.txt");
  evaluate();
}

BonsaiPredictor::BonsaiPredictor(
  const size_t numBytes,
  const char *const fromModel,
  const bool isDense)
  : model(numBytes, fromModel, isDense)
{
  feedDataValBuffer = new FP_TYPE[model.hyperParams.dataDimension];
  feedDataFeatureBuffer = new labelCount_t[model.hyperParams.dataDimension];

  mean = MatrixXuf::Zero(model.hyperParams.dataDimension, 1);
  variance = MatrixXuf::Zero(model.hyperParams.dataDimension, 1);
}

void BonsaiPredictor::importMeanVar(
  std::string meanVarFile)
{
  size_t meanVarSize;
  std::ifstream infileMeanVar(meanVarFile, std::ios::in|std::ios::binary);
  assert(infileMeanVar.is_open());

  infileMeanVar.read((char*)&meanVarSize, sizeof(meanVarSize));
  infileMeanVar.close();
  
  infileMeanVar.open(meanVarFile, std::ios::in|std::ios::binary);
  
  char* meanVarBuff = new char[meanVarSize];
  infileMeanVar.read((char*)meanVarBuff, meanVarSize);
  infileMeanVar.close();
  
  importMeanVar(meanVarSize, meanVarBuff);
}

void BonsaiPredictor::importMeanVar(
  const size_t numBytes,
  const char *const fromBuffer)
{
  size_t offset = 0;

  size_t meanVarSize;
  memcpy((void *)&meanVarSize, fromBuffer + offset, sizeof(meanVarSize));
  offset += sizeof(meanVarSize);
  memcpy(mean.data(), fromBuffer + offset, sizeof(FP_TYPE) * mean.rows() * mean.cols());
  offset += sizeof(FP_TYPE) * mean.rows() * mean.cols();
  memcpy(variance.data(), fromBuffer + offset, sizeof(FP_TYPE) * variance.rows() * variance.cols());
  offset += sizeof(FP_TYPE) * variance.rows() * variance.cols();

  assert(numBytes == offset);
}


BonsaiPredictor::~BonsaiPredictor()
{
  delete[] feedDataValBuffer;
  delete[] feedDataFeatureBuffer;
}

FP_TYPE BonsaiPredictor::predictionScoreOfClassID(
  const MatrixXuf& ZX,
  const std::vector<int> path,
  const labelCount_t& ClassID)
{
  FP_TYPE score = (FP_TYPE)0.0;
  // Hadamard
  MatrixXuf WZX = MatrixXuf::Zero(1, 1);
  MatrixXuf VZX = MatrixXuf::Zero(1, 1);
  for (int i = 0; i < path.size(); i++) {
    mm(WZX, model.getW(ClassID, path[i]), CblasNoTrans, ZX, CblasNoTrans, (FP_TYPE)1.0, (FP_TYPE)0.0L);
    mm(VZX, model.getV(ClassID, path[i]), CblasNoTrans, ZX, CblasNoTrans, (FP_TYPE)1.0, (FP_TYPE)0.0L);
    score += WZX(0, 0) * tanh(model.hyperParams.Sigma * VZX(0, 0));
  }
  return score;
}

std::vector<int> BonsaiPredictor::treePath(const MatrixXuf& ZX)
{
  std::vector<int> visitedNodesList;
  visitedNodesList.push_back(0);
  int curr_node = 0;
  MatrixXuf ThetaZX = MatrixXuf::Zero(1, 1);
  while (curr_node < model.hyperParams.internalNodes) {
    mm(ThetaZX, model.getTheta(curr_node), CblasNoTrans, ZX, CblasNoTrans, (FP_TYPE)1.0, (FP_TYPE)0.0L);
    curr_node = ThetaZX(0, 0) > (FP_TYPE)0.0 ? 2 * curr_node + 1 : 2 * curr_node + 2;
    visitedNodesList.push_back(curr_node);
  }
  return visitedNodesList;
}

void BonsaiPredictor::predictionScore(
  const MatrixXuf& X,
  FP_TYPE *scores)
{
  assert(X.cols() == 1);
  MatrixXuf ZX = MatrixXuf(model.hyperParams.projectionDimension, 1);

  mm(ZX, model.params.Z, CblasNoTrans, X, CblasNoTrans,
    (FP_TYPE)1.0 / model.hyperParams.projectionDimension, (FP_TYPE)0.0);

  std::vector<int> path = treePath(ZX);
  FP_TYPE ymult = model.hyperParams.internalClasses <= 2 ? (FP_TYPE)-1.0 : (FP_TYPE)1.0;
  for (labelCount_t c = 0; c < model.hyperParams.internalClasses; c++)
    scores[c] = ymult*predictionScoreOfClassID(ZX, path, c);
}

void BonsaiPredictor::predictionSparseScore(
  const SparseMatrixuf& X,
  FP_TYPE *scores)
{
  assert(X.cols() == 1);
  MatrixXuf ZX = MatrixXuf(model.hyperParams.projectionDimension, 1);

  mm(ZX, model.params.Z, CblasNoTrans, MatrixXuf(X), CblasNoTrans,
    (FP_TYPE)1.0 / model.hyperParams.projectionDimension, (FP_TYPE)0.0);

  std::vector<int> path = treePath(ZX);
  FP_TYPE ymult = model.hyperParams.internalClasses <= 2 ? (FP_TYPE)-1.0 : (FP_TYPE)1.0;
  for (labelCount_t c = 0; c < model.hyperParams.internalClasses; c++)
    scores[c] = ymult*predictionScoreOfClassID(ZX, path, c);
}

void BonsaiPredictor::scoreSparseDataPoint(
  FP_TYPE* scores,
  const FP_TYPE *const values,
  const featureCount_t *const indices,
  const featureCount_t& numIndices)
{
  memset(scores, 0, sizeof(FP_TYPE)*model.hyperParams.numClasses);

  MatrixXuf dataPoint = MatrixXuf::Zero(model.hyperParams.dataDimension, 1);

  for (featureCount_t f = 0; f < numIndices; ++f)
    dataPoint(indices[f], 0) = values[f];

  dataPoint -= mean;

  //vDiv(model.hyperParams.dataDimension, dataPoint.data(), variance.data(), dataPoint.data());
  for (featureCount_t f = 0; f < model.hyperParams.dataDimension; f++) {
    dataPoint(f, 0) /= variance(f, 0);
  }

  dataPoint(model.hyperParams.dataDimension - 1, 0) = (FP_TYPE)1.0;

  predictionScore(dataPoint, scores);
}

void BonsaiPredictor::scoreDenseDataPoint(
  FP_TYPE* scores,
  const FP_TYPE *const values)
{
  memset(scores, 0, sizeof(FP_TYPE)*model.hyperParams.numClasses);

  MatrixXuf dataPoint(model.hyperParams.dataDimension, 1);

  memcpy(dataPoint.data(), values, sizeof(FP_TYPE)*(model.hyperParams.dataDimension - 1));

  dataPoint = dataPoint - mean;

  //vDiv(model.hyperParams.dataDimension, dataPoint.data(), variance.data(), dataPoint.data());

  for (featureCount_t f = 0; f < model.hyperParams.dataDimension; f++) {
    dataPoint(f, 0) /= variance(f, 0);
  }

  dataPoint(model.hyperParams.dataDimension - 1, 0) = (FP_TYPE)1.0;

  predictionScore(dataPoint, scores);
}

void BonsaiPredictor::evaluate()
{
  batchEvaluate(testData.Xtest, testData.Ytest, dataDir, modelDir);
}

void BonsaiPredictor::batchEvaluate(
  const SparseMatrixuf& Xtest,
  const SparseMatrixuf& Ytest,
  const std::string& dataDir,
  const std::string& currResultsPath)
{
  std::string predLabelPath = currResultsPath + "/predClassAndScore";
  std::ofstream predwriter(predLabelPath);

  dataCount_t nTest = Xtest.cols();
  featureCount_t dataDim = Xtest.rows();
  labelCount_t nLabels = Ytest.rows();

  FP_TYPE *scoreArray = new FP_TYPE[nLabels];

  FP_TYPE *trainvals = new FP_TYPE[dataDim];
  memset(trainvals, 0, sizeof(FP_TYPE)*(dataDim));

  labelCount_t *label = new labelCount_t[1];

  int correct = 0;
  for (dataCount_t i = 0; i < nTest; ++i) {
    featureCount_t count = 0;
    while (count < dataDim) {
      trainvals[count] = Xtest.coeff(count, i);
      count++;
    }

    scoreDenseDataPoint(scoreArray, trainvals);

    labelCount_t predLabel = 0;
    FP_TYPE maxScore = scoreArray[0];
    for (labelCount_t j = 0; j < nLabels; j++) {
      if (maxScore <= scoreArray[j]) {
        maxScore = scoreArray[j];
        predLabel = j;
      }
      if (Ytest.coeff(j, i) == 1) label[0] = j;
    }

    if (label[0] == predLabel) correct++;
    (model.hyperParams.isOneIndex) ? predLabel++ : predLabel;
    predwriter << predLabel << "\t" << maxScore << "\n";
  }

  predwriter.close();

  FP_TYPE accuracy = (FP_TYPE)(correct) / ((FP_TYPE)nTest);

  LOG_INFO("Final Test Accuracy = " + std::to_string(accuracy));

  dumpRunInfo(currResultsPath, accuracy);

  std::ofstream allDumper(dataDir + "/BonsaiResults" + "/resultDump", std::ofstream::out | std::ofstream::app);
  allDumper << totalNonZeros() << " " << accuracy << " " << currResultsPath << "\n";
  allDumper.close();
}

size_t BonsaiPredictor::totalNonZeros()
{
  return model.totalNonZeros();
}

void BonsaiPredictor::dumpRunInfo(
  const std::string currResultsPath,
  const FP_TYPE& accuracy)
{
  BonsaiModel::BonsaiHyperParams hyperParam = model.hyperParams;

  std::ofstream accuracyWriter(currResultsPath + "/runInfo");
  accuracyWriter << "Final Test Accuracy = " + std::to_string(accuracy) << "\n";
  accuracyWriter << "HyperParams: \n";
  accuracyWriter << "\tTree Depth: " << hyperParam.treeDepth << "\n";
  accuracyWriter << "\tProjected Dimension: " << hyperParam.projectionDimension << "\n";
  accuracyWriter << "\tSparsity Fractions: \n" << "\t\tZ: " << hyperParam.lambdaZ << "\n";
  accuracyWriter << "\t\tW: " << hyperParam.lambdaW << "\n\t\tV: " << hyperParam.lambdaV << "\n";
  accuracyWriter << "\t\tTheta: " << hyperParam.lambdaTheta << "\n";
  accuracyWriter << "\tRegularizers: \n" << "\t\tZ: " << hyperParam.regList.lZ << "\n";
  accuracyWriter << "\t\tW: " << hyperParam.regList.lW << "\n\t\tV: " << hyperParam.regList.lV << "\n";
  accuracyWriter << "\t\tTheta: " << hyperParam.regList.lTheta << "\n";
  accuracyWriter << "\tSigma: " << hyperParam.Sigma << "\n \n";
  accuracyWriter << "\tBatch factor: " << hyperParam.batchFactor << "\n \n";
  accuracyWriter << "\tIters: " << hyperParam.iters << "\n \n";


  accuracyWriter << "\tTotal Nonzeros: " << model.totalNonZeros() << "\n";
  accuracyWriter.close();
}
