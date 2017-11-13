// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __DATA_H__
#define __DATA_H__

#include "pre_processor.h"

namespace EdgeML
{
  enum DataFormat
  {
    undefinedData, interfaceIngestFormat, MNISTFormat, tsvFormat, libsvmFormat
  };

  enum ProblemFormat
  {
    undefinedProblem, binary, multiclass, multilabel
  };

  enum InitializationFormat
  {
    undefinedInitialization, predefined, perClassKmeans, overallKmeans, sample
  };

  enum NormalizationFormat
  {
    undefinedNormalization, none, l2, minMax
  };

  enum DataIngestType
  {
    FileIngest, InterfaceIngest
  };

  struct DataFormatParams
  {
    dataCount_t numTrainPoints;
    dataCount_t numValidationPoints;
    dataCount_t numTestPoints;
    labelCount_t numLabels;
    featureCount_t dimension;
  };

  struct SparseDataPoint
  {
    const FP_TYPE *values;
    const featureCount_t *indices;
    const featureCount_t numIndices;
    const labelCount_t *labels;
    const labelCount_t numLabels;
  };


  struct DenseDataPoint
  {
    const FP_TYPE *values;
    const labelCount_t *labels;
    const labelCount_t numLabels;
  };

  class Data
  {
    DataFormatParams formatParams;
    DataIngestType ingestType;

    std::vector<Trip> sparseDataHolder;
    std::vector<Trip> sparseLabelHolder;
    std::vector<FP_TYPE> denseDataHolder;
    dataCount_t numPointsIngested;

  public:
    bool isDataLoaded;

    //NormalizationFormat normalizationType;

    MatrixXuf trainData, trainLabel;
    MatrixXuf validationData, validationLabel;
    MatrixXuf testData, testLabel;

    SparseMatrixuf Xtrain, Ytrain;
    SparseMatrixuf Xvalidation, Yvalidation;
    SparseMatrixuf Xtest, Ytest;

    MatrixXuf mean, variance;
    MatrixXuf min, max;

    void loadDataFromFile(
      DataFormat format,
      std::string trainFile,
      std::string validationFile,
      std::string testFile);

    Data() {};

    Data(DataIngestType dataIngestType,
      DataFormatParams formatParams);

    ~Data() {};

    void feedSparseData(const SparseDataPoint& point);
    void feedDenseData(const DenseDataPoint& point);
    void finalizeData();

    inline DataIngestType getIngestType() { return ingestType; }
 };

  void computeMinMax(const SparseMatrixuf& dataMatrix, MatrixXuf& min, MatrixXuf& max);
  void minMaxNormalize(SparseMatrixuf& dataMatrix, const MatrixXuf& min, const MatrixXuf& max);
  void l2Normalize(SparseMatrixuf& dataMatrix);
  void meanVarNormalize(SparseMatrixuf& dataMatrix, MatrixXuf& mean, MatrixXuf& variance);
  void saveMinMax(const MatrixXuf& min, const MatrixXuf& max, std::string fileName);
  void loadMinMax(MatrixXuf& min, MatrixXuf& max, int dim, std::string fileName);
}
#endif
