// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "BonsaiFunctions.h"

using namespace EdgeML;
using namespace EdgeML::Bonsai;

//
// Model is public in trainer, and hyperparams is public in model, so after loading the model using the constructor
// Change the hyper params as you wish and call train() on this instance of trainer.
// DO NOT touch first 7 Flags in the Help printed on screen.
// You can change hyperParams like iters, batchFactor, sparsity, regularisers and sigma.
//

//
// You also have to use this constructor to load a model and data with ntrain = 0, to just have a prediciton happening on test.txt for a pretrained model
// load the model using BonsaiTrainer(...) and then export model and construct BonsaiPredictor(...) with that model and call batchevaluate
//
BonsaiTrainer::BonsaiTrainer(
  const DataIngestType& dataIngestType,
  const size_t& numBytes,
  const char *const fromModel,
  const std::string& dataDir,
  std::string& currResultsPath,
  const bool isDense)
  :
  model(numBytes, fromModel, isDense),   // Initialize model
  data(dataIngestType,
    DataFormatParams{
      model.hyperParams.ntrain,
        model.hyperParams.ntest,
        model.hyperParams.numClasses,
        model.hyperParams.dataDimension })
{
  assert(dataIngestType == FileIngest);

  createOutputDirs(dataDir, currResultsPath);

#ifdef TIMER
  OPEN_TIMER_LOGFILE(currResultsPath);
#endif

  feedDataValBuffer = new FP_TYPE[5];
  feedDataFeatureBuffer = new featureCount_t[5];

  data.loadDataFromFile(model.hyperParams.dataformatType, dataDir + "/train.txt", dataDir + "/test.txt");
  finalizeData();
}

BonsaiTrainer::BonsaiTrainer(
  const DataIngestType& dataIngestType,
  const int& argc,
  const char** argv,
  std::string& dataDir,
  std::string& currResultsPath)
  :
  model(argc, argv, dataDir),               // Initialize model
  data(dataIngestType,
    DataFormatParams{
      model.hyperParams.ntrain,
  model.hyperParams.ntest,
  model.hyperParams.numClasses,
  model.hyperParams.dataDimension })
{
  assert(dataIngestType == FileIngest);

  createOutputDirs(dataDir, currResultsPath);

#ifdef TIMER
  OPEN_TIMER_LOGFILE(currResultsPath);
#endif

  // not required for this constructor
  feedDataValBuffer = new FP_TYPE[5];
  feedDataFeatureBuffer = new featureCount_t[5];

  data.loadDataFromFile(model.hyperParams.dataformatType, dataDir + "/train.txt", dataDir + "/test.txt");
  finalizeData();

  initializeModel();

  train();
}

BonsaiTrainer::BonsaiTrainer(
  const DataIngestType& dataIngestType,
  const BonsaiModel::BonsaiHyperParams& fromHyperParams)
  : model(fromHyperParams),
  data(dataIngestType,
    DataFormatParams{
         model.hyperParams.ntrain,
     model.hyperParams.ntest,
     model.hyperParams.numClasses,
     model.hyperParams.dataDimension })
{
  assert(dataIngestType == InterfaceIngest);
  assert(model.hyperParams.normalizationType == none);

  feedDataValBuffer = new FP_TYPE[model.hyperParams.dataDimension + 5];
  feedDataFeatureBuffer = new featureCount_t[model.hyperParams.dataDimension + 5];

  initializeModel();
}

BonsaiTrainer::~BonsaiTrainer()
{
  delete[] feedDataValBuffer;
  delete[] feedDataFeatureBuffer;
}

void BonsaiTrainer::feedDenseData(
  const FP_TYPE *const values,
  const labelCount_t *const labels,
  const labelCount_t& num_labels)
{
  memcpy(feedDataValBuffer, values, sizeof(FP_TYPE)* (model.hyperParams.dataDimension - 1));
  // Pad 1.0 for the last feature to enable bias learning
  feedDataValBuffer[model.hyperParams.dataDimension - 1] = (FP_TYPE)1.0;

  data.feedDenseData(DenseDataPoint{ feedDataValBuffer, labels, num_labels });
}

void BonsaiTrainer::feedSparseData(
  const FP_TYPE *const values,
  const featureCount_t *const indices,
  const featureCount_t& numIndices,
  const labelCount_t *const labels,
  const labelCount_t& num_labels)
{
  memcpy(feedDataValBuffer, values, sizeof(FP_TYPE)*numIndices);
  feedDataValBuffer[numIndices] = (FP_TYPE)1.0;

  memcpy(feedDataFeatureBuffer, indices, sizeof(labelCount_t)*numIndices);
  feedDataFeatureBuffer[numIndices] = model.hyperParams.dataDimension - 1;

  data.feedSparseData(SparseDataPoint{ feedDataValBuffer, feedDataFeatureBuffer, numIndices + 1, labels, num_labels });
}

void BonsaiTrainer::finalizeData()
{
  data.finalizeData();
  if (model.hyperParams.ntrain == 0) {
    // This condition means that the ingest type is Interface ingest,
    // hence the number of training points was not known beforehand. 
    model.hyperParams.ntrain = data.Xtrain.cols();
    assert(data.Xtest.cols() == 0);
    model.hyperParams.ntest = 0;
  }
  else {
    assert(model.hyperParams.ntrain == data.Xtrain.cols());
    // assert(model.hyperParams.ntest == data.Xtest.cols());
  }

  // Following asserts can only be made in finalieData since TLC 
  // does not give us number of training points before-hand!
  assert(model.hyperParams.ntrain > 0);

  initializeTrainVariables(data.Ytrain);

  computeMeanVar(data.Xtrain, data.mean, data.variance);
  meanVarNormalize(data.Xtrain, data.mean, data.variance);
}

FP_TYPE BonsaiTrainer::computeObjective(const MatrixXuf& ZX, const LabelMatType& Y)
{
  return computeObjective(MatrixXuf(model.params.Z), MatrixXuf(model.params.W), MatrixXuf(model.params.V), MatrixXuf(model.params.Theta), ZX, Y);
}

FP_TYPE BonsaiTrainer::computeObjective(
  const MatrixXuf& Zmat,
  const MatrixXuf& Wmat,
  const MatrixXuf& Vmat,
  const MatrixXuf& Thetamat,
  const MatrixXuf& ZX,
  const LabelMatType& Y)
{
  int accuracy = 0;
  initializeTrainVariables(Y);
  treeCache.fillNodeProbability(model, Thetamat, ZX);

  MatrixXuf trueBestScore = MatrixXuf::Ones(2, ZX.cols())*(-1000.0L);
  MatrixXufINT trueBestClassIndex = MatrixXufINT::Zero(2, ZX.cols());

  getTrueBestClass(trueBestScore, trueBestClassIndex, Wmat, Vmat, Y, ZX);

  MatrixXuf margin = trueBestScore.row(0) - trueBestScore.row(1);
  FP_TYPE marginLoss = (FP_TYPE)0.0L;
  for (int n = 0; n < ZX.cols(); n++)
  {
    if ((FP_TYPE)1.0 - YMultCoeff(0, n)*margin(0, n) > 0.0)
      marginLoss += (FP_TYPE)1.0 - YMultCoeff(0, n)*margin(0, n);
    if (YMultCoeff(0, n)*margin(0, n) > 0)
      accuracy += 1;
  }

  FP_TYPE normAdd
    = (FP_TYPE)0.5 * ((model.hyperParams.regList.lW)*(Wmat.squaredNorm()) + (model.hyperParams.regList.lV)*(Vmat.squaredNorm())
      + (model.hyperParams.regList.lTheta)*(Thetamat.squaredNorm()) + (model.hyperParams.regList.lZ)*(Zmat.squaredNorm()));

  std::string infoStr
    = "Bonsai Objective value for " + std::to_string(ZX.cols()) + " points: "
    + std::to_string(normAdd) + "+" + std::to_string((FP_TYPE)marginLoss / ZX.cols())
    + " = " + std::to_string(normAdd + (FP_TYPE)marginLoss / ZX.cols())
    + " |  Accuracy: " + std::to_string((FP_TYPE)accuracy / ZX.cols());
  if (ZX.cols() == data.Xtrain.cols())
    LOG_INFO(infoStr);
  /* else
   LOG_TRACE(infoStr);*/

  return normAdd + (FP_TYPE)marginLoss / ZX.cols();
}

void BonsaiTrainer::initializeTrainVariables(const LabelMatType& Y)
{
  dataCount_t _numPoints = Y.cols();

  YMultCoeff = MatrixXuf::Ones(1, _numPoints);

  if ((model.hyperParams.internalClasses) <= 2)
  {
    for (dataCount_t n = 0; n < _numPoints; n++)
      YMultCoeff(0, n) = ((FP_TYPE)-2.0 * Y.coeff(0, n) + (FP_TYPE)1.0);
  }

  treeCache.WXWeight = MatrixXuf::Zero((model.hyperParams.totalNodes) * (model.hyperParams.internalClasses), _numPoints);
  treeCache.tanhVXWeight = MatrixXuf::Zero((model.hyperParams.totalNodes) * (model.hyperParams.internalClasses), _numPoints);
  treeCache.nodeProbability = MatrixXuf::Zero((model.hyperParams.totalNodes), _numPoints);
  treeCache.tanhThetaXCache = MatrixXuf::Zero((model.hyperParams.internalNodes), _numPoints);
  treeCache.partialZGradient = MatrixXuf::Zero((model.hyperParams.projectionDimension), _numPoints);
}

void BonsaiTrainer::train()
{
  assert(data.isDataLoaded == true);
  assert(model.hyperParams.isModelInitialized == true);

  normalize();

  jointSgdBonsai(*this);
}


size_t BonsaiTrainer::getModelSize()
{
  return model.modelStat();
}

size_t BonsaiTrainer::getSparseModelSize()
{
  return model.sparseModelStat();
}

void BonsaiTrainer::exportModel(const size_t& modelSize, char *const buffer)
{
  model.exportModel(modelSize, buffer);
}

void BonsaiTrainer::exportSparseModel(const size_t& modelSize, char *const buffer)
{
  model.exportSparseModel(modelSize, buffer);
}

void BonsaiTrainer::exportModel(
  const size_t& modelSize,
  char *const buffer,
  const std::string& currResultsPath)
{
  std::string loadableModelPath = currResultsPath + "/loadableModel";
  model.exportModel(modelSize, buffer);

  std::ofstream modelExporter(loadableModelPath);
  modelExporter.write(buffer, modelSize);
  modelExporter.close();
}

void BonsaiTrainer::exportSparseModel(
  const size_t& modelSize,
  char *const buffer,
  const std::string& currResultsPath)
{
  std::string loadableModelPath = currResultsPath + "/loadableModel";
  model.exportSparseModel(modelSize, buffer);

  std::ofstream modelExporter(loadableModelPath);
  modelExporter.write(buffer, modelSize);
  modelExporter.close();
}

size_t BonsaiTrainer::getMeanVarSize()
{
  size_t offset = 0;

  offset += sizeof(FP_TYPE) * data.mean.rows() * data.mean.cols();
  offset += sizeof(FP_TYPE) * data.variance.rows() * data.variance.cols();

  return offset;
}

void BonsaiTrainer::exportMeanVar(
  const size_t& meanVarSize,
  char *const buffer)
{
  assert(meanVarSize == getMeanVarSize());

  size_t offset = 0;

  memcpy(buffer + offset, data.mean.data(), sizeof(FP_TYPE) * data.mean.rows() * data.mean.cols());
  offset += sizeof(FP_TYPE) * data.mean.rows() * data.mean.cols();

  memcpy(buffer + offset, data.variance.data(), sizeof(FP_TYPE) * data.variance.rows() * data.variance.cols());
  offset += sizeof(FP_TYPE) * data.variance.rows() * data.variance.cols();

  assert(offset < (size_t)(1 << 31)); // Because we make this promise to TLC.
}

void BonsaiTrainer::exportMeanVar(
  const size_t& meanVarSize,
  char *const buffer,
  const std::string& currResultsPath)
{
  std::string loadableMeanvarPath = currResultsPath + "/loadableMeanVar";
  exportMeanVar(meanVarSize, buffer);

  std::ofstream meanVarExporter(loadableMeanvarPath);
  meanVarExporter.write(buffer, meanVarSize);
  meanVarExporter.close();
}

void BonsaiTrainer::normalize()
{
  if (model.hyperParams.normalizationType == minMax) {
    minMaxNormalize(data.Xtrain, data.Xtest);
  }
  else if (model.hyperParams.normalizationType == l2) {
    l2Normalize(data.Xtrain);
    l2Normalize(data.Xtest);
  }
  else;
}

void BonsaiTrainer::initializeModel()
{
  srand((unsigned int)model.hyperParams.seed);

#ifdef SPARSE_Z_BONSAI
  model.params.Z = (MatrixXuf::Random(model.params.Z.rows(), model.params.Z.cols())).sparseView();
#else
  model.params.Z = MatrixXuf::Random(model.params.Z.rows(), model.params.Z.cols());
#endif

#ifdef SPARSE_W_BONSAI
  model.params.W = (MatrixXuf::Random(model.params.W.rows(), model.params.W.cols())).sparseView();
#else
  model.params.W = MatrixXuf::Random(model.params.W.rows(), model.params.W.cols());
#endif

#ifdef SPARSE_V_BONSAI
  model.params.V = (MatrixXuf::Random(model.params.V.rows(), model.params.V.cols())).sparseView();
#else
  model.params.V = MatrixXuf::Random(model.params.V.rows(), model.params.V.cols());
#endif

#ifdef SPARSE_THETA_BONSAI
  model.params.Theta = (MatrixXuf::Random(model.params.Theta.rows(), model.params.Theta.cols())).sparseView();
#else
  model.params.Theta = MatrixXuf::Random(model.params.Theta.rows(), model.params.Theta.cols());
#endif

  initializeTrainVariables(data.Ytrain);
}

void BonsaiTrainer::computeScoreOfClassID(
  MatrixXuf& Score,
  const MatrixXuf& Wmat,
  const MatrixXuf& Vmat,
  const MatrixXuf& ZX,
  const labelCount_t& classID,
  MatrixXuf& WXClassIDScratch,
  MatrixXuf& VXClassIDScratch)
{
  assert(WXClassIDScratch.rows() == model.hyperParams.totalNodes);
  assert(WXClassIDScratch.cols() == ZX.cols());
  assert(VXClassIDScratch.rows() == model.hyperParams.totalNodes);
  assert(VXClassIDScratch.cols() == ZX.cols());

  mm(WXClassIDScratch,
    MatrixXuf(Wmat.middleRows(model.hyperParams.totalNodes*classID, model.hyperParams.totalNodes)), CblasNoTrans,
    ZX, CblasNoTrans, (FP_TYPE)1.0, (FP_TYPE)0.0L);
  mm(VXClassIDScratch,
    MatrixXuf(Vmat.middleRows(model.hyperParams.totalNodes*classID, model.hyperParams.totalNodes)), CblasNoTrans,
    ZX, CblasNoTrans, (FP_TYPE)1.0, (FP_TYPE)0.0L);

  // The code below this commented section is an optimized version
  /* for (int i = 0; i < VXClassIDScratch.rows(); i++)
  for (int j = 0; j < VXClassIDScratch.cols(); j++)
  Score(0, j) +=
  WXClassIDScratch(i, j)
  * tanh(model.hyperParams.Sigma * VXClassIDScratch(i, j))
  * treeCache.nodeProbability(i, j);*/

  // Scale VXClassIDScratch by scalar Sigma
  scal(VXClassIDScratch.rows()*VXClassIDScratch.cols(), model.hyperParams.Sigma, VXClassIDScratch.data(), 1);
  // compute tanh in place using vector ops  
  vTanh(VXClassIDScratch.rows()*VXClassIDScratch.cols(), VXClassIDScratch.data(), VXClassIDScratch.data());
  // 3-way in-place hadamard into WXClassIDScratch
  hadamard3(WXClassIDScratch, WXClassIDScratch, VXClassIDScratch, treeCache.nodeProbability);
  // This computes the column sums of WXClassIDScratch
  MatrixXuf onesVec = MatrixXuf::Ones(1, WXClassIDScratch.rows());
  mm(Score, onesVec, CblasNoTrans, WXClassIDScratch, CblasNoTrans, (FP_TYPE)1.0, (FP_TYPE)0.0);
}

void BonsaiTrainer::getTrueBestClass(
  MatrixXuf& trueBestScore,
  MatrixXufINT& true_best_classIndex,
  const MatrixXuf& Wmat,
  const MatrixXuf& Vmat,
  const LabelMatType& Y,
  const MatrixXuf& ZX)
{
  MatrixXuf WXClassID = MatrixXuf::Zero(model.hyperParams.totalNodes, ZX.cols());
  MatrixXuf VXClassID = MatrixXuf::Zero(model.hyperParams.totalNodes, ZX.cols());

  // trueClassScore+=ScoreCL o Y.row(class_i);
  // bestClassScore=max(bestClassScore,ScoreCL o (1.0 - Y.row(class_i))) -- incorrect;
  if (model.hyperParams.internalClasses <= 2)
  {
    MatrixXuf ScoreCL = MatrixXuf::Zero(1, ZX.cols());
    computeScoreOfClassID(ScoreCL, Wmat, Vmat, ZX, 0, WXClassID, VXClassID);
    trueBestScore.row(0) = ScoreCL;
    trueBestScore.row(1) = MatrixXuf::Zero(1, trueBestScore.cols());
  }
  else
  {
    for (labelCount_t class_i = 0; class_i < model.hyperParams.internalClasses; class_i++)
    {
      MatrixXuf ScoreCL = MatrixXuf::Zero(1, ZX.cols());
      computeScoreOfClassID(ScoreCL, Wmat, Vmat, ZX, class_i, WXClassID, VXClassID);

      for (int n = 0; n < ZX.cols(); n++)
      {
        if (Y.coeff(class_i, n) == 0 && (ScoreCL(0, n) > trueBestScore(1, n)))
        {
          trueBestScore(1, n) = ScoreCL(0, n);
          true_best_classIndex(1, n) = (FP_TYPE)class_i;
        }
        else if (Y.coeff(class_i, n) == 1)
        {
          trueBestScore(0, n) = ScoreCL(0, n);
          true_best_classIndex(0, n) = (FP_TYPE)class_i;
        }
      }
    }
  }
}

void BonsaiTrainer::fillNodeProbability(const MatrixXuf& ZX)
{
  treeCache.fillNodeProbability(model, MatrixXuf(model.params.Theta), ZX);
}

void BonsaiTrainer::TreeCache::fillNodeProbability(
  const BonsaiModel& model,
  const MatrixXuf& Thetamat,
  const MatrixXuf& Xdata)
{
  tanhThetaXCache = MatrixXuf::Zero(model.hyperParams.internalNodes, Xdata.cols());
  nodeProbability = MatrixXuf::Ones(model.hyperParams.totalNodes, Xdata.cols());

  if(model.hyperParams.internalNodes > 0)
    mm(tanhThetaXCache, Thetamat, CblasNoTrans, Xdata, CblasNoTrans, (FP_TYPE)1.0, (FP_TYPE)0.0L);
  // Scale VXClassIDScratch by scalar sigma_i
  scal(tanhThetaXCache.rows()*tanhThetaXCache.cols(), model.hyperParams.sigma_i, tanhThetaXCache.data(), 1);
  // compute tanh in place using vector ops  
  vTanh(tanhThetaXCache.rows()*tanhThetaXCache.cols(), tanhThetaXCache.data(), tanhThetaXCache.data());

  // Iterate over all nodes and generate probablity for reaching each node. 
  for (int n = 0; n < Xdata.cols(); n++)
  {
    for (int i = 0; i < model.hyperParams.internalNodes; i++)
    {
      nodeProbability(2 * i + 1, n) = nodeProbability(i, n)*((FP_TYPE)1.0 + tanhThetaXCache(i, n)) / (FP_TYPE)2.0;
      nodeProbability(2 * i + 2, n) = nodeProbability(i, n)*((FP_TYPE)1.0 - tanhThetaXCache(i, n)) / (FP_TYPE)2.0;
    }
  }
}

void BonsaiTrainer::fillWX(const MatrixXuf& ZX, const MatrixXufINT& classID)
{
  treeCache.fillWX(model, model.params.W, ZX, classID);
}

void BonsaiTrainer::TreeCache::fillWX(
  const BonsaiModel& model,
  const WMatType& Wmat,
  const MatrixXuf& Xdata,
  const MatrixXufINT& classID)
{
  //treeCache.WXWeight = MatrixXuf::Zero(totalNodes*internalClasses, Xdata.cols());
  for (int n = 0; n < Xdata.cols(); n++)
  {
    MatrixXuf X = Xdata.col(n);
    MatrixXuf WXWeightcolN = MatrixXuf::Zero(model.hyperParams.totalNodes, 1);// treeCache.WXWeight.block(classID(0, n)*totalNodes, n, totalNodes, 1);
    mm(WXWeightcolN, MatrixXuf(Wmat.middleRows(model.hyperParams.totalNodes*(labelCount_t)classID(0, n), model.hyperParams.totalNodes)), CblasNoTrans, X, CblasNoTrans, (FP_TYPE)1.0, (FP_TYPE)0.0L);
    WXWeight.block((labelCount_t)classID(0, n)*model.hyperParams.totalNodes, n, model.hyperParams.totalNodes, 1) = WXWeightcolN;
  }
};

void BonsaiTrainer::fillTanhVX(const MatrixXuf& ZX, const MatrixXufINT& classID)
{
  treeCache.fillTanhVX(model, model.params.V, ZX, classID);
};

void BonsaiTrainer::TreeCache::fillTanhVX(
  const BonsaiModel& model,
  const VMatType& Vmat,
  const MatrixXuf& Xdata,
  const MatrixXufINT& classID)
{
  for (int n = 0; n < Xdata.cols(); n++)
  {
    MatrixXuf X = Xdata.col(n);
    // Need a function which returns it as a reference to make it faster
    MatrixXuf VXWeightcolN = MatrixXuf::Zero(model.hyperParams.totalNodes, 1);;// treeCache.tanhVXWeight.block(classID(0, n)*totalNodes, n, totalNodes, 1);
    mm(VXWeightcolN, MatrixXuf(Vmat.middleRows(model.hyperParams.totalNodes*(labelCount_t)classID(0, n), model.hyperParams.totalNodes)), CblasNoTrans, X, CblasNoTrans, (FP_TYPE)1.0, (FP_TYPE)0.0L);
    tanhVXWeight.block((labelCount_t)classID(0, n)*(model.hyperParams.totalNodes), n, (model.hyperParams.totalNodes), 1) = VXWeightcolN;
  }
};

void BonsaiTrainer::loadModel(const std::string model_path, const size_t modelBytes, const bool isDense)
{
  std::ifstream modelreader(model_path);
  char *fromModel = new char[modelBytes];
  modelreader.read(fromModel, modelBytes);
  model = BonsaiModel(modelBytes, fromModel, isDense);
}
size_t BonsaiTrainer::sizeForExportVSparse()
{
  return sparseExportStat(model.params.V);
}
void BonsaiTrainer::exportVSparse(int bufferSize, char * const buf)
{
  exportSparseMatrix(model.params.V, bufferSize, buf);
}
size_t BonsaiTrainer::sizeForExportWSparse()
{
  return sparseExportStat(model.params.W);
}
void BonsaiTrainer::exportWSparse(int bufferSize, char *const buf)
{
  exportSparseMatrix(model.params.W, bufferSize, buf);
}
size_t BonsaiTrainer::sizeForExportZSparse()
{
  return sparseExportStat(model.params.Z);
}
void BonsaiTrainer::exportZSparse(int bufferSize, char *const buf)
{
  exportSparseMatrix(model.params.Z, bufferSize, buf);
}
size_t BonsaiTrainer::sizeForExportThetaSparse()
{
  return sparseExportStat(model.params.Theta);
}
void BonsaiTrainer::exportThetaSparse(int bufferSize, char *const buf)
{
  exportSparseMatrix(model.params.Theta, bufferSize, buf);
}


size_t BonsaiTrainer::sizeForExportVDense()
{
  return denseExportStat(model.params.V);
}
void BonsaiTrainer::exportVDense(int bufferSize, char *const buf)
{
  exportDenseMatrix(model.params.V, bufferSize, buf);
}
size_t BonsaiTrainer::sizeForExportWDense()
{
  return denseExportStat(model.params.W);
}
void BonsaiTrainer::exportWDense(int bufferSize, char *const buf)
{
  exportDenseMatrix(model.params.W, bufferSize, buf);
}
size_t BonsaiTrainer::sizeForExportZDense()
{
  return denseExportStat(model.params.Z);
}
void BonsaiTrainer::exportZDense(int bufferSize, char *const buf)
{
  exportDenseMatrix(model.params.Z, bufferSize, buf);
}
size_t BonsaiTrainer::sizeForExportThetaDense()
{
  return denseExportStat(model.params.Theta);
}
void BonsaiTrainer::exportThetaDense(int bufferSize, char *const buf)
{
  exportDenseMatrix(model.params.Theta, bufferSize, buf);
}

void BonsaiTrainer::dumpModelMeanVar(const std::string& currResultsPath)
{
  std::string params_path = currResultsPath + "/Params";
  writeMatrixInASCII(MatrixXuf(model.params.Z), params_path, "Z");
  writeMatrixInASCII(MatrixXuf(model.params.W), params_path, "W");
  writeMatrixInASCII(MatrixXuf(model.params.V), params_path, "V");
  writeMatrixInASCII(MatrixXuf(model.params.Theta), params_path, "Theta");

  writeMatrixInASCII(data.mean, params_path, "Mean");
  writeMatrixInASCII(data.variance, params_path, "Variance");
}

size_t BonsaiTrainer::totalNonZeros()
{
  return model.totalNonZeros();
}
