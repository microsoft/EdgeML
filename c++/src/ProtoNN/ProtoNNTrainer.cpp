// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "ProtoNNFunctions.h"

#include "mmaped.h"
using namespace EdgeML;
using namespace EdgeML::ProtoNN;

ProtoNNTrainer::ProtoNNTrainer(
  const int& argc,
  const char ** argv)
  :
  model(argc, argv),               // Initialize model
  data(FileIngest,
    DataFormatParams{
      model.hyperParams.ntrain,
      model.hyperParams.nvalidation,
      0, // Set the number of test points to zero
      model.hyperParams.l,
      model.hyperParams.D }),
      dataformatType(DataFormat::undefinedData)
{
  commandLine = "";
  for (int i = 0; i < argc; ++i)
    commandLine += (std::string(argv[i]) + " ");
  setFromArgs(argc, argv);

  createOutputDirs();

#ifdef TIMER
  OPEN_TIMER_LOGFILE(outDir);
#endif

#ifdef LIGHT_LOGGER
  OPEN_DIAGNOSTIC_LOGFILE(outDir);
#endif

  //Assert that the trainFile is properly assigned
  assert((!trainFile.empty()) && (model.hyperParams.ntrain > 0)); 

  // Pass an empty string as test file, else it will try to load the test data also
  std::string testFile = ""; 
  data.loadDataFromFile(dataformatType,
    trainFile,
    validationFile,
    testFile);
 
  finalizeData();
  normalize();
}

void ProtoNNTrainer::createOutputDirs()
{
  std::string subdirName = model.hyperParams.subdirName();

#ifdef LINUX
  outDir = outDir + "/ProtoNNTrainer_" + subdirName;
#endif

#ifdef WINDOWS
  outDir = outDir + "\\ProtoNNTrainer_" + subdirName;
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

ProtoNNTrainer::ProtoNNTrainer(
  const ProtoNNModel::ProtoNNHyperParams& fromHyperParams)
  :
  model(fromHyperParams),
  data(InterfaceIngest,
    DataFormatParams{
       model.hyperParams.ntrain,
       model.hyperParams.nvalidation,
       model.hyperParams.l,
         model.hyperParams.D }),
         dataformatType(DataFormat::interfaceIngestFormat)
{
  assert(model.hyperParams.normalizationType == none);
}

ProtoNNTrainer::~ProtoNNTrainer() {}

void ProtoNNTrainer::feedDenseData(
  const FP_TYPE *const values,
  const labelCount_t *const labels,
  const labelCount_t& numLabels)
{
  data.feedDenseData(DenseDataPoint{ values, labels, numLabels });
}

void ProtoNNTrainer::feedSparseData(
  const FP_TYPE *const values,
  const featureCount_t *const indices,
  const featureCount_t& numIndices,
  const labelCount_t *const labels,
  const labelCount_t& numLabels)
{
  data.feedSparseData(SparseDataPoint{ values, indices, numIndices, labels, numLabels });
}

void ProtoNNTrainer::finalizeData()
{
  data.finalizeData();
  if (model.hyperParams.ntrain == 0) {
    // This condition means that the ingest type is Interface ingest,
    // hence the number of training points was not known beforehand. 
    model.hyperParams.ntrain = data.Xtrain.cols();
    assert(data.Xtest.cols() == 0);
    model.hyperParams.nvalidation = 0;
  }

  else {
    assert(model.hyperParams.ntrain == data.Xtrain.cols());
    assert(model.hyperParams.nvalidation == data.Xvalidation.cols());
  }

  // Following asserts can only be made in finalieData since TLC 
  // does not give us number of training points before-hand!
  assert(model.hyperParams.ntrain > 0);
  assert(model.hyperParams.m <= model.hyperParams.ntrain);

}

void ProtoNNTrainer::train()
{
  assert(data.isDataLoaded == true);
  assert(model.hyperParams.isHyperParamInitialized == true);

  initializeModel();

  FP_TYPE* stats = new FP_TYPE[model.hyperParams.iters * 9 + 3]; // store output of this run
  altMinSGD(data, model, stats, outDir);

  // Save the parameters of the model in separate files
  writeMatrixInASCII(model.params.W, outDir, "W");
  writeMatrixInASCII(model.params.B, outDir, "B");
  writeMatrixInASCII(model.params.Z, outDir, "Z");
  MatrixXuf gammaMat(1, 1);
  gammaMat(0, 0) = model.hyperParams.gamma;
  writeMatrixInASCII(gammaMat, outDir, "gamma");

  // Save the model in a single file
  size_t modelSize = getModelSize();
  char *buffer = new char[modelSize];
  exportModel((const size_t)modelSize, (char *const)buffer);
  std::ofstream fout(outDir+"/model", std::ios::out|std::ios::binary);
  assert(fout.is_open());
  fout.write((char *const)&modelSize, sizeof(modelSize));
  fout.write((char *const)buffer, modelSize);
  fout.close();
  delete[] buffer;

  std::string outFile = outDir + "/runInfo";
  storeParams(commandLine, stats, outFile);

  // Log and final output
  delete[] stats; // currently, stats are not being stored anywhere
}

size_t ProtoNNTrainer::getModelSize()
{
  size_t modelSize = model.modelStat();
  if (data.getIngestType() == DataIngestType::InterfaceIngest)
    assert(modelSize < (1 << 31)); // Because we make this promise to TLC.

  return modelSize;
}
void ProtoNNTrainer::exportModel(const size_t& modelSize, char *const buffer)
{
  assert(modelSize == getModelSize());

  model.exportModel(modelSize, buffer);
}
size_t ProtoNNTrainer::sizeForExportBSparse()
{
  return sparseExportStat(model.params.B);
}
void ProtoNNTrainer::exportBSparse(int bufferSize, char * const buf)
{
  exportSparseMatrix(model.params.B, bufferSize, buf);
}
size_t ProtoNNTrainer::sizeForExportWSparse()
{
  return sparseExportStat(model.params.W);
}
void ProtoNNTrainer::exportWSparse(int bufferSize, char *const buf)
{
  exportSparseMatrix(model.params.W, bufferSize, buf);
}
size_t ProtoNNTrainer::sizeForExportZSparse()
{
  return sparseExportStat(model.params.Z);
}

void ProtoNNTrainer::exportZSparse(int bufferSize, char *const buf)
{
  exportSparseMatrix(model.params.Z, bufferSize, buf);
}

size_t ProtoNNTrainer::sizeForExportBDense()
{
  return denseExportStat(model.params.B);
}
void ProtoNNTrainer::exportBDense(int bufferSize, char *const buf)
{
  exportDenseMatrix(model.params.B, bufferSize, buf);
}
size_t ProtoNNTrainer::sizeForExportWDense()
{
  return denseExportStat(model.params.W);
}
void ProtoNNTrainer::exportWDense(int bufferSize, char *const buf)
{
  exportDenseMatrix(model.params.W, bufferSize, buf);
}
size_t ProtoNNTrainer::sizeForExportZDense()
{
  return denseExportStat(model.params.Z);
}
void ProtoNNTrainer::exportZDense(int bufferSize, char *const buf)
{
  exportDenseMatrix(model.params.Z, bufferSize, buf);
}

void ProtoNNTrainer::normalize()
{
  NormalizationFormat normalizationType = model.hyperParams.normalizationType;

  switch (normalizationType) {
    case minMax: 
    {
      std::string minMaxFile = outDir + "/minMaxParams";
      computeMinMax(data.Xtrain, data.min, data.max);
      saveMinMax(data.min, data.max, minMaxFile);
      minMaxNormalize(data.Xtrain, data.min, data.max);
      if (data.Xvalidation.cols() > 0)
        minMaxNormalize(data.Xvalidation, data.min, data.max);
      LOG_INFO("Completed min-max normalization of data");
      break;
    }

    case l2:
      l2Normalize(data.Xtrain);
      if (data.Xvalidation.cols() > 0)
        l2Normalize(data.Xvalidation);
      LOG_INFO("Completed l2 normalization of data");
      break;

    case none:
      break;

    default:
      assert(false);
  }
}

void ProtoNNTrainer::initializeModel()
{
  LOG_INFO("    ");

  if (model.hyperParams.initializationType == predefined) {
    LOG_INFO("Loading predefined input files from predefined folder " + modelDir);

    MatrixXuf voidMat;
    DataFormat format = tsvFormat;

    std::string infile = modelDir + "/W";
    FileIO::Data W_(infile,
      model.params.W, voidMat, model.hyperParams.d, -1, 0,
      model.hyperParams.D, model.hyperParams.D, 0, format);

    infile = modelDir + "/Z";
    FileIO::Data Z_(infile,
      model.params.Z, voidMat, model.hyperParams.l, -1, 0,
      model.hyperParams.m, model.hyperParams.m, 0, format);

    infile = modelDir + "/B";
    FileIO::Data B_(infile,
      model.params.B, voidMat, model.hyperParams.d, -1, 0,
      model.hyperParams.m, model.hyperParams.m, 0, format);

    infile = modelDir + "/gamma";
    MatrixXuf gammaMat;
    FileIO::Data Gamma_(infile,
      gammaMat, voidMat, 1, -1, 0,
      1, 1, 0, format);
    model.hyperParams.gamma = gammaMat(0, 0);
    LOG_INFO("Gamma set to " + std::to_string(model.hyperParams.gamma));

    model.params.W = model.params.W.transpose().eval();
    model.params.B = model.params.B.transpose().eval();
    model.params.Z = model.params.Z.transpose().eval();
  }

  else {
    // Initialize W as a random Gaussian matrix 
    LOG_INFO("Initializing projection matrix as a Random Gaussian Matrix (with mean 0 and variance 1). This initialization may not work if the data is not normalized/standardized...");
    FP_TYPE* WPtr = model.params.W.data();
    std::normal_distribution<FP_TYPE> distribution(0, 1);
    std::default_random_engine generator;
    for (Eigen::Index i = 0; i < model.params.W.rows()*model.params.W.cols(); ++i) {
      WPtr[i] = distribution(generator);
    }

    // Initialize B, Z according to what user wants
    if (model.hyperParams.initializationType == sample) {
      for (labelCount_t i = 0; i < model.hyperParams.m; ++i) {
        dataCount_t prot = rand() % data.Xtrain.cols();
        model.params.B.col(i) = model.params.W * data.Xtrain.col(prot);
#ifdef SPARSE_Z_PROTONN
        model.params.Z.col(i) = data.trainLabel.col(prot).sparseView();
#else
        model.params.Z.col(i) = data.trainLabel.col(prot);
#endif
      }
    }

    else if (model.hyperParams.initializationType == perClassKmeans) {
      LOG_INFO("Initializing prototype matrix (B) and prototype-label matrix (Z) by clustering data (in projected space) from each class separately using k-means++... ");

      MatrixXuf WX = MatrixXuf::Zero(model.params.W.rows(), data.Xtrain.cols());
      mm(WX, model.params.W, CblasNoTrans, data.Xtrain, CblasNoTrans, 1.0, 0.0L);

#ifdef SPARSE_Z_PROTONN
      MatrixXuf Z = model.params.Z;
      assert(model.params.B.cols() % data.Ytrain.rows() == 0);
      kmeansLabelwise(data.Ytrain, WX, model.params.B, Z,
        model.params.B.cols() / model.params.Z.rows());
      model.params.Z = Z.sparseView();
#else
      assert(model.params.B.cols() % data.Ytrain.rows() == 0);
      kmeansLabelwise(data.Ytrain, WX, model.params.B, model.params.Z,
        model.params.B.cols() / model.params.Z.rows());
#endif
      model.hyperParams.m = model.params.B.cols();
    }

    else if (model.hyperParams.initializationType == overallKmeans) {
      LOG_INFO("Initializing prototype matrix (B) and prototype-label matrix (Z) by clustering data in projected space using k-means++... ");

      MatrixXuf WX = MatrixXuf::Zero(model.params.W.rows(), data.Xtrain.cols());
      mm(WX, model.params.W, CblasNoTrans, data.Xtrain, CblasNoTrans, 1.0, 0.0L);

#ifdef XML
      dataCount_t numRand = std::min((dataCount_t)100000, (dataCount_t)WX.cols());
      MatrixXuf WXSub(model.params.W.rows(), numRand);
      SparseMatrixuf YTrainSub(data.Ytrain.rows(), numRand);
      randPick(WX, WXSub);
      randPick(data.Ytrain, YTrainSub);
#ifdef SPARSE_Z_PROTONN 
      MatrixXuf Z = model.params.Z;
      kmeansOverall(YTrainSub, WXSub, model.params.B, Z);
      model.params.Z = Z.sparseView();
#else
      kmeansOverall(YTrainSub, WXSub, model.params.B, model.params.Z);
#endif

#else
#ifdef SPARSE_Z_PROTONN
      MatrixXuf Z = model.params.Z;
      kmeansOverall(data.Ytrain, WX, model.params.B, Z);
      model.params.Z = Z.sparseView();
#else
      kmeansOverall(data.Ytrain, WX, model.params.B, model.params.Z);
#endif
#endif
    }

    // Set gamma = model.hyperParams.gammaNumerator * 2.5 / (median b/w B and WX)

    MatrixXuf WX = MatrixXuf::Zero(model.params.W.rows(), data.Xtrain.cols());
    mm(WX, model.params.W, CblasNoTrans, data.Xtrain, CblasNoTrans, 1.0, 0.0L);

    FP_TYPE initGuess = (FP_TYPE)0.005;
    FP_TYPE multiplier = model.hyperParams.gammaNumerator * (FP_TYPE) 2.5;

    if (data.Xtrain.cols() * model.params.B.cols() > 2000000000llu) {
      dataCount_t numRand = std::min((dataCount_t)10000, (dataCount_t)WX.cols());
      MatrixXuf WXSub(model.params.W.rows(), numRand);
      randPick(WX, WXSub);
      //model.hyperParams.gamma = medianHeuristic(model.params.B, WXSub, initGuess, multiplier, 0, WXSub.cols());
      model.hyperParams.gamma = medianHeuristic(model.params.B, WXSub, multiplier);
    }
    else {
      model.hyperParams.gamma = medianHeuristic(model.params.B, WX, multiplier);
    }

    LOG_INFO("Set value of gamma using median heuristic: " + std::to_string(model.hyperParams.gamma));
  }
}

void ProtoNNTrainer::setFromArgs(const int argc, const char** argv)
{
  for (int i = 1; i < argc; ++i) {

    if (i % 2 == 1)
      assert(argv[i][0] == '-'); //odd arguments must be specifiers, not values 
    else {
      switch (argv[i - 1][1]) {
      case 'I':
        trainFile = argv[i];
        break;

      case 'V':
        validationFile = argv[i];
        break;

      case 'O':
	outDir = argv[i];
	break;

      case 'M': 
        modelDir = argv[i];
        break;

      case 'F':
        if (argv[i][0] == '0') dataformatType = libsvmFormat;
        else if (argv[i][0] == '1') dataformatType = tsvFormat;
        else if (argv[i][0] == '2') dataformatType = MNISTFormat;
        else assert(false); //Format unknown
        break;

      case 'P':
      case 'C':
      case 'R':
      case 'g':
      case 'r':
      case 'v':
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

      default:
        LOG_INFO("Command line argument not recognized; saw character: " + argv[i - 1][1]);
        assert(false);
        break;
      }
    }
  }
}




// Store hyperparameters and learnt model along with accuracy values
void ProtoNNTrainer::storeParams(std::string commandLine, FP_TYPE* stats, std::string outFile) {
  std::ofstream f(outFile);
  f << "d = " << model.hyperParams.d << std::endl
    << "k = " << model.hyperParams.k << " (if this value is 0, it means k-means overall was used for initialization)" << std::endl
    << "m = " << model.hyperParams.m << std::endl
    << "lambdaW = " << model.hyperParams.lambdaW << std::endl
    << "lambdaZ = " << model.hyperParams.lambdaZ << std::endl
    << "lambdaB = " << model.hyperParams.lambdaB << std::endl
    << "gammaNumerator = " << model.hyperParams.gammaNumerator << std::endl
    << "gamma = " << model.hyperParams.gamma << std::endl
    << "batch-size = " << model.hyperParams.batchSize << std::endl
    << "epochs = " << model.hyperParams.epochs << std::endl
    << "iters = " << model.hyperParams.iters << std::endl
    << "seed = " << model.hyperParams.seed << std::endl;

  if (model.hyperParams.initializationType == EdgeML::perClassKmeans)
    f << "initializationType = perClassKmeans" << std::endl;
  else if (model.hyperParams.initializationType == EdgeML::overallKmeans)
    f << "initializationType = overallKmeans" << std::endl;
  else if (model.hyperParams.initializationType == EdgeML::sample)
    f << "initializationType = sample" << std::endl;
  else if (model.hyperParams.initializationType == EdgeML::predefined)
    f << "initializationType = predefined" << std::endl;
  else;

  if (model.hyperParams.normalizationType == EdgeML::l2)
    f << "normalizationType = l2-normalization" << std::endl;
  else if (model.hyperParams.normalizationType == EdgeML::minMax)
    f << "normalizationType = minmax-normalization" << std::endl;
  else if (model.hyperParams.normalizationType == EdgeML::none)
    f << "normalizationType = none" << std::endl;
  else;

  f << std::endl;
  f << "Command line call: " << commandLine << std::endl;
  f << std::endl;
  f << "Statistics for current run: " << std::endl;
  f << "param | iter | objective, training accuracy, testing accuracy\n";
  for (int i = 0; i < model.hyperParams.iters * 3 + 1; i++) {
    if (i == 0) f << "init  | ";
    else if (i % 3 == 1) f << "W     | ";
    else if (i % 3 == 2) f << "Z     | ";
    else if (i % 3 == 0) f << "B     | ";
    else;
    f << (i - 1) / 3 << "    | ";
    f << *(stats + i * 3) << ", " << *(stats + i * 3 + 1) << ", " << *(stats + i * 3 + 2) << "\n";
  }
  f << std::endl;

  f.close();
}

