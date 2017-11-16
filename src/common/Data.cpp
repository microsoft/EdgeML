// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "mmaped.h"
#include "Data.h"
#include "blas_routines.h"

using namespace EdgeML;

Data::Data(
  DataIngestType ingestType_,
  DataFormatParams formatParams_)
  : isDataLoaded(false),
  formatParams(formatParams_),
  ingestType(ingestType_),
  numPointsIngested(0)
{
  // if(ingestType_ == InterfaceIngest)
  //   assert(formatParams_.numTestPoints == 0);
}

void Data::loadDataFromFile(
  DataFormat format,
  std::string infileTrain,
  std::string infileValidation,
  std::string infileTest)
{
  assert(isDataLoaded == false);
  assert(ingestType == FileIngest);

  trainData = MatrixXuf(0, 0);
  validationData = MatrixXuf(0, 0);
  testData = MatrixXuf(0, 0);

  Xtest = SparseMatrixuf(0, 0);
  Xvalidation = SparseMatrixuf(0, 0);
  Xtest = SparseMatrixuf(0, 0);

  LOG_INFO("");
  if (format == tsvFormat) {
    if((!infileTrain.empty()) && (formatParams.numTrainPoints > 0)) {
      LOG_INFO("Reading train data...");
      FileIO::Data train(infileTrain,
        trainData, trainLabel,
        formatParams.numTrainPoints, 0, 1,
        formatParams.dimension + 1, formatParams.dimension, formatParams.numLabels,
        format);
    }

    if((!infileValidation.empty()) && (formatParams.numValidationPoints > 0)) {
      LOG_INFO("Reading validation data...");
      FileIO::Data validation(infileValidation,
        validationData, validationLabel,
        formatParams.numValidationPoints, 0, 1,
        formatParams.dimension + 1, formatParams.dimension, formatParams.numLabels,
        format);
    }

    if((!infileTest.empty()) && (formatParams.numTestPoints > 0)) {
      LOG_INFO("Reading test data...");
      FileIO::Data test(infileTest,
        testData, testLabel,
        formatParams.numTestPoints, 0, 1,
        formatParams.dimension + 1, formatParams.dimension, formatParams.numLabels,
        format);
    }
  }
  else if (format == libsvmFormat) {
    if((!infileTrain.empty()) && (formatParams.numTrainPoints > 0)) {
      LOG_INFO("Reading train data...");
      FileIO::Data train(infileTrain,
        Xtrain, Ytrain,
        formatParams.numTrainPoints, -1, -1,
        -1, formatParams.dimension, formatParams.numLabels,
        format);
    }

    if ((!infileValidation.empty()) && (formatParams.numValidationPoints > 0)) {
      LOG_INFO("Reading validation data...");
      FileIO::Data validation(infileValidation,
        Xvalidation, Yvalidation,
        formatParams.numValidationPoints, -1, -1,
        -1, formatParams.dimension, formatParams.numLabels,
        format);
    }

    if ((!infileTest.empty()) && (formatParams.numTestPoints > 0)) {
      LOG_INFO("Reading test data...");
      FileIO::Data test(infileTest,
        Xtest, Ytest,
        formatParams.numTestPoints, -1, -1,
        -1, formatParams.dimension, formatParams.numLabels,
        format);
    }
  }
  else if (format == interfaceIngestFormat) {
    labelCount_t *label = new labelCount_t[1];
    if (!infileTrain.empty()) {
      LOG_INFO("Reading train data...");

      std::ifstream trainreader(infileTrain);
      trainData = MatrixXuf::Zero(formatParams.dimension, formatParams.numTrainPoints);
      trainLabel = MatrixXuf::Zero(formatParams.numLabels, formatParams.numTrainPoints);
      FP_TYPE readLbl;
      FP_TYPE readFeat;
      for (dataCount_t i = 0; i < formatParams.numTrainPoints; ++i) {
        trainreader >> readLbl;
        label[0] = (labelCount_t)readLbl;

#ifdef ZERO_BASED_IO
        trainLabel(label[0], i) = 1;
#else
        trainLabel(label[0] - 1, i) = 1;
#endif

        featureCount_t count = 0;
        while (count < formatParams.dimension - 1) {
          trainreader >> readFeat;
          trainData(count, i) = readFeat;
          count++;
        }
        trainData(count, i) = 1.0;
      }
      trainreader.close();
      Xtrain = trainData.sparseView(); trainData.resize(0, 0);
      Ytrain = trainLabel.sparseView(); trainLabel.resize(0, 0);
    }

    if (!infileValidation.empty()) {
      LOG_INFO("Reading validation data...");

      std::ifstream validationreader(infileValidation);
      validationData = MatrixXuf::Zero(formatParams.dimension, formatParams.numValidationPoints);
      validationLabel = MatrixXuf::Zero(formatParams.numLabels, formatParams.numValidationPoints);
      FP_TYPE readLbl;
      FP_TYPE readFeat;
      for (dataCount_t i = 0; i < formatParams.numValidationPoints; ++i) {
        validationreader >> readLbl;
        label[0] = (labelCount_t)readLbl;

#ifdef ZERO_BASED_IO
        validationLabel(label[0], i) = 1;
#else
        validationLabel(label[0] - 1, i) = 1;
#endif

        featureCount_t count = 0;
        while (count < formatParams.dimension - 1) {
          validationreader >> readFeat;
          validationData(count, i) = readFeat;
          count++;
        }
        validationData(count, i) = 1.0;
      }
      validationreader.close();
      Xvalidation = validationData.sparseView(); validationData.resize(0, 0);
      Yvalidation = validationLabel.sparseView(); validationLabel.resize(0, 0);
    }

    if (!infileTest.empty()) {
      std::ifstream testreader(infileTest);
      LOG_INFO("Reading test data...");

      testData = MatrixXuf::Zero(formatParams.dimension, formatParams.numTestPoints);
      testLabel = MatrixXuf::Zero(formatParams.numLabels, formatParams.numTestPoints);
      FP_TYPE readLbl;
      FP_TYPE readFeat;
      for (dataCount_t i = 0; i < formatParams.numTestPoints; ++i) {
        testreader >> readLbl;
        label[0] = (labelCount_t)readLbl;

#ifdef ZERO_BASED_IO
        testLabel(label[0], i) = 1;
#else
        testLabel(label[0] - 1, i) = 1;
#endif

        featureCount_t count = 0;
        while (count < formatParams.dimension - 1) {
          testreader >> readFeat;
          testData(count, i) = readFeat;
          count++;
        }
        testData(count, i) = 1.0;
      }
      testreader.close();

      Xtest = testData.sparseView(); testData.resize(0, 0);
      Ytest = testLabel.sparseView(); testLabel.resize(0, 0);
    }

    delete[] label;
  }
  else
    assert(false);
}

void Data::finalizeData()
{
  if (ingestType == FileIngest)
    assert(sparseDataHolder.size() == 0 && denseDataHolder.size() == 0);

  else if (ingestType == InterfaceIngest) {
    //assert(!(sparseDataHolder.size() != 0 && denseDataHolder.size() != 0));
    assert(sparseDataHolder.size() > 0 || denseDataHolder.size() >0);

    formatParams.numTrainPoints = numPointsIngested;
    formatParams.numTestPoints = 0;
    formatParams.numValidationPoints = 0;

    if (sparseDataHolder.size() != 0) {
      if (denseDataHolder.size() != 0) {
        dataCount_t numDensePointsIngested = (dataCount_t)(denseDataHolder.size() / formatParams.dimension);
        dataCount_t numSparsePointsIngested = numPointsIngested - numDensePointsIngested;
        for (dataCount_t densePt = 0; densePt < numDensePointsIngested; ++densePt)
          for (featureCount_t dim = 0; dim < formatParams.dimension; ++dim)
            sparseDataHolder.push_back(Trip(dim,
                                            numSparsePointsIngested + densePt,
                                            denseDataHolder[densePt*formatParams.dimension + dim]));
        denseDataHolder.resize(0);
        denseDataHolder.shrink_to_fit();
      }
      Xtrain = SparseMatrixuf(formatParams.dimension, numPointsIngested);
      Xtrain.setFromTriplets(sparseDataHolder.begin(), sparseDataHolder.end());
      sparseDataHolder.resize(0);
      sparseDataHolder.shrink_to_fit();

      Ytrain = SparseMatrixuf(formatParams.numLabels, numPointsIngested);
      Ytrain.setFromTriplets(sparseLabelHolder.begin(), sparseLabelHolder.end());
      sparseLabelHolder.resize(0);
      sparseLabelHolder.shrink_to_fit();
    }
    else if (denseDataHolder.size() != 0) {
      trainData = MatrixXuf(formatParams.dimension, numPointsIngested);
      memcpy(trainData.data(), denseDataHolder.data(), sizeof(FP_TYPE)*trainData.rows()*trainData.cols());
      denseDataHolder.resize(0);
      denseDataHolder.shrink_to_fit();

      Ytrain = SparseMatrixuf(formatParams.numLabels, numPointsIngested);
      Ytrain.setFromTriplets(sparseLabelHolder.begin(), sparseLabelHolder.end());
      sparseLabelHolder.resize(0);
      sparseLabelHolder.shrink_to_fit();
    }
    else assert(false);
  }

  //
  // Following lines check if the data was passed to the Data struct in dense or sparse format
  // If data was passed in dense format, then it is guaranteed that: 
  // 1. getnnzs(Xtest) == 0
  // 2. getnnzs(Xtrain) == 0
  // 3. getnnzs(Ytest) == 0
  // 4. getnnzs(Ytrain) == 0
  // If these conditions are true, then we set these sparse matrices to their dense counter-parts
  // Additionally, we deallocate the dense matrices. ProtoNN only uses the sparse data and label matrices.
  //
  if (getnnzs(Xtest) == 0) {
    Xtest = testData.sparseView();
    testData.resize(0, 0);
  }
  if (getnnzs(Xtrain) == 0) {
    Xtrain = trainData.sparseView();
    trainData.resize(0, 0);
  }
  if (getnnzs(Xvalidation) == 0) {
    Xvalidation = validationData.sparseView();
    validationData.resize(0, 0);
  }

  if (getnnzs(Ytest) == 0) {
    Ytest = testLabel.sparseView();
    testLabel.resize(0, 0);
  }
  if (getnnzs(Ytrain) == 0) {
    Ytrain = trainLabel.sparseView();
    trainLabel.resize(0, 0);
  }
  if (getnnzs(Yvalidation) == 0) {
    Yvalidation = validationLabel.sparseView();
    validationLabel.resize(0, 0);
  }

  min = MatrixXuf::Zero(0, 0);
  max = MatrixXuf::Zero(0, 0);

  /*
  if(Xtest.cols() == 0){
    Xtest = Xtrain;
  }
  if(Ytest.cols() == 0){
    Ytest = Ytrain;
    }*/

  isDataLoaded = true;
}

void Data::feedDenseData(const DenseDataPoint& point)
{
  assert(ingestType == InterfaceIngest);
  assert(isDataLoaded == false);

  denseDataHolder.insert(denseDataHolder.end(), point.values, point.values + formatParams.dimension);

  for (labelCount_t id = 0; id < point.numLabels; id = id + 1) {
    assert(point.labels[id] < formatParams.numLabels);
    sparseLabelHolder.push_back(Trip(point.labels[id], numPointsIngested, 1.0));
  }

  numPointsIngested++;
}

void Data::feedSparseData(const SparseDataPoint& point)
{
  assert(ingestType == InterfaceIngest);
  assert(isDataLoaded == false);

  for (featureCount_t id = 0; id < point.numIndices; id = id + 1) {
    assert(point.indices[id] < formatParams.dimension);
    sparseDataHolder.push_back(Trip(point.indices[id], numPointsIngested, point.values[id]));
  }

  for (labelCount_t id = 0; id < point.numLabels; id = id + 1) {
    assert(point.labels[id] < formatParams.numLabels);
    sparseLabelHolder.push_back(Trip(point.labels[id], numPointsIngested, 1.0));
  }

  numPointsIngested++;
}

void EdgeML::computeMinMax(
  const SparseMatrixuf& dataMatrix,
  MatrixXuf& min,
  MatrixXuf& max)
{
#ifdef ROWMAJOR
  assert(false);
#endif
  FP_TYPE * mn = new FP_TYPE[dataMatrix.rows()];
  FP_TYPE * mx = new FP_TYPE[dataMatrix.rows()];
  for (Eigen::Index i = 0; i < dataMatrix.rows(); ++i) {
    mn[i] = 99999999999.0f;
    mx[i] = -99999999999.0f;
  }

  const FP_TYPE * values = dataMatrix.valuePtr();
  const sparseIndex_t * offsets = dataMatrix.innerIndexPtr();
  Eigen::Index nnz = getnnzs(dataMatrix);

  for (auto i = 0; i < nnz; ++i) {
    mn[offsets[i]] = mn[offsets[i]] < values[i] ? mn[offsets[i]] : values[i];
    mx[offsets[i]] = mx[offsets[i]] > values[i] ? mx[offsets[i]] : values[i];
  }

  featureCount_t zero_feats(0);

  for (auto i = 0; i < dataMatrix.rows(); ++i) {
    if (mn[i] == mx[i]) {
      mn[i] = 0;
    }
    if (mx[i] < mn[i]) {
      zero_feats++;
      mx[i] = 1;
      mn[i] = 0;
    }
  }

  if (zero_feats > 0)
    LOG_WARNING(std::to_string(zero_feats) + " features are always zero. Remove them if possible");

  //Assert that min max parameter is not assigned
  assert(min.rows() == 0);

  //Ok, go ahead
  min = MatrixXuf::Zero(dataMatrix.rows(), 1);
  max = MatrixXuf::Zero(dataMatrix.rows(), 1);
  pfor(featureCount_t i = 0; i < dataMatrix.rows(); ++i)
    min(i, 0) = mn[i];
  pfor(featureCount_t i = 0; i < dataMatrix.rows(); ++i)
    max(i, 0) = mx[i];

  delete[] mn;
  delete[] mx;
}

void EdgeML::minMaxNormalize(
  SparseMatrixuf& dataMatrix,
  const MatrixXuf& min,
  const MatrixXuf& max)
{
#ifdef ROWMAJOR
  assert(false);
#endif
  
  //Assert that min max parameter is already initialized
  assert(min.rows() == dataMatrix.rows());
  assert(max.rows() == dataMatrix.rows());

  FP_TYPE * values = dataMatrix.valuePtr();
  const sparseIndex_t * offsets = dataMatrix.innerIndexPtr();
  Eigen::Index nnz = getnnzs(dataMatrix);

  values = dataMatrix.valuePtr();
  offsets = dataMatrix.innerIndexPtr();
  nnz = getnnzs(dataMatrix);
  for (auto i = 0; i < nnz; ++i) {
    values[i] =
      (values[i] - min(offsets[i], 0)) /
      (max(offsets[i], 0) - min(offsets[i], 0));
  }
}

void EdgeML::l2Normalize(SparseMatrixuf& dataMatrix)
{
#ifdef ROWMAJOR
  assert(false);
#endif
  assert(dataMatrix.outerSize() == dataMatrix.cols());
  for (auto i = 0; i < dataMatrix.outerSize(); ++i) {
    FP_TYPE norm = dataMatrix.col(i).norm();
    for (SparseMatrixuf::InnerIterator it(dataMatrix, i); it; ++it) {
      it.valueRef() = it.value() / norm;
    }
  }
}

void EdgeML::meanVarNormalize(
  SparseMatrixuf& dataMatrix,     //< 
  MatrixXuf& mean,                //< Initialize to vector of size numFeatures
  MatrixXuf& variance)            //< Initialize to vector of size numFeatures
{
  MatrixXuf denseDataMatrix = MatrixXuf(dataMatrix);
  const Eigen::Index numDataPoints = denseDataMatrix.cols();
  const Eigen::Index numFeatures = denseDataMatrix.rows();

  // std::cout<<denseDataMatrix<<std::endl<<std::endl;
  const MatrixXuf onesVec = MatrixXuf::Ones(numDataPoints, 1);

  mm(mean, denseDataMatrix, CblasNoTrans, onesVec, CblasNoTrans, (FP_TYPE)1.0 / numDataPoints, (FP_TYPE)0.0);
  mm(denseDataMatrix, mean, CblasNoTrans, onesVec, CblasTrans, (FP_TYPE)-1.0, (FP_TYPE)1.0);


  denseDataMatrix.transposeInPlace();

  for (Eigen::Index f = 0; f < numFeatures; f++) {
    variance(f, 0) = (FP_TYPE)std::sqrt(dot(numDataPoints, denseDataMatrix.data() + f*numDataPoints, 1,
      denseDataMatrix.data() + f*numDataPoints, 1)
      / numDataPoints);

    if (!(fabs(variance(f, 0)) < (FP_TYPE)1e-7)) {
      scal(numDataPoints, (FP_TYPE)1.0 / variance(f, 0), denseDataMatrix.data() + f*numDataPoints, 1);
    }
    else {
      variance(f, 0) = (FP_TYPE)1.0;
    }
  }

  for (Eigen::Index d = 0; d < numDataPoints; d++) {
    denseDataMatrix(d, numFeatures - 1) = (FP_TYPE)1.0;
  }

  denseDataMatrix.transposeInPlace();

  dataMatrix = denseDataMatrix.sparseView();
}

void EdgeML::saveMinMax(
  const MatrixXuf& min,
  const MatrixXuf& max,
  std::string fileName)
{
#ifdef ROWMAJOR
  assert(false);
#endif

  // Assert the min max params are already assigned
  assert(min.rows() > 0);
  assert(max.rows() > 0);

  std::ofstream out(fileName);
  //out.open(fileName);

  for (auto i = 0; i < min.rows(); i++)
    out << min(i, 0) << '\t';
  out << '\n';
  for (auto i = 0; i < max.rows(); i++)
    out << max(i, 0) << '\t';
  out << '\n';
}

void EdgeML::loadMinMax(
  MatrixXuf& min,
  MatrixXuf& max,
  int dim,
  std::string fileName)
{
#ifdef ROWMAJOR
  assert(false);
#endif

  // Assert that min max params are not assigned
  assert(min.rows() == 0);

  //Ok, go ahead
  min = MatrixXuf::Zero(dim, 1);
  max = MatrixXuf::Zero(dim, 1);

  LOG_INFO("Loading min-max normalization parameters from file: " + fileName);
  std::ifstream in(fileName);
  //in.open(fileName);

  for (auto i = 0; i < dim; i++)
    in >> min(i, 0);
  for (auto i = 0; i < dim; i++)
    in >> max(i, 0);

  in.close();
}



// void EdgeML::meanVarNormalize(
//   MatrixXuf& dataMatrix,     //< 
//   MatrixXuf& mean,                //< Initialize to vector of size numFeatures
//   MatrixXuf& variance)            //< Initialize to vector of size numFeatures
// {

//     const Eigen::Index numDataPoints = dataMatrix.cols();
//     const Eigen::Index numFeatures = dataMatrix.rows();

//     assert(mean.rows() == numFeatures);
//     assert(variance.rows() == numFeatures);

//     const MatrixXuf onesVec = MatrixXuf::Ones(numDataPoints, 1);

//     mm(mean, dataMatrix, CblasNoTrans, onesVec, CblasNoTrans, (FP_TYPE)1.0/numDataPoints, (FP_TYPE)0.0);
//     mm(dataMatrix, mean, CblasNoTrans, onesVec, CblasTrans, (FP_TYPE)-1.0, (FP_TYPE)1.0);


//     dataMatrix.transposeInPlace();

//     for(featureCount_t f = 0; f < numFeatures; f++) {
//       variance(f,0) = (FP_TYPE)std::sqrt(dot(numDataPoints, dataMatrix.data() + f*numDataPoints, 1, 
//                                                    dataMatrix.data() + f*numDataPoints, 1) 
//                                 / numDataPoints);

//       if(!(fabs(variance(f, 0)) < (FP_TYPE)1e-7)) {
//           scal(numDataPoints, (FP_TYPE)1.0/variance(f,0), dataMatrix.data() + f*numDataPoints, 1);
//       }
//       else {
//         variance(f, 0) = (FP_TYPE)1.0;
//       }
//     }

//     for(dataCount_t d = 0; d < numDataPoints; d++) {
//       dataMatrix(d, numFeatures-1) = (FP_TYPE)1.0;
//     }

//     dataMatrix.transposeInPlace();

//     dataMatrix = dataMatrix.sparseView();
// }
