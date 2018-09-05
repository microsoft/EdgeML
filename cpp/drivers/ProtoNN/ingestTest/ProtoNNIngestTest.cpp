// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "ProtoNN.h"

using namespace EdgeML;
using namespace EdgeML::ProtoNN;

int main()
{
  ProtoNNModel::ProtoNNHyperParams hyperParam;
  
  hyperParam.problem_type = ProblemFormat::multiclass;
  hyperParam.initialization_type = InitializationFormat::overall_kmeans;
  hyperParam.dataformat_type = DataFormat::interface_ingest_format;
  hyperParam.normalization_type = NormalizationFormat::none;
  hyperParam.seed = 41;
  hyperParam.batch_size = 100;
  hyperParam.iters = 5;
  hyperParam.epochs = 2;
  hyperParam.D = 2;
  hyperParam.d = 2;
  hyperParam.m = 4;
  hyperParam.k = 0;
  hyperParam.l = 3;
  hyperParam.gammaNumerator = 1.0;
  hyperParam.lambda_W = 1.0;
  hyperParam.lambda_Z = 1.0;
  hyperParam.lambda_B = 1.0;
  
  hyperParam.finalizeHyperParams();

  // trivial data set
  {
    auto trainer = new ProtoNNTrainer(DataIngestType::InterfaceIngest, hyperParam);

    FP_TYPE trainPts[2*16] = {-1.1, -1.1,
			      -0.9, -1.1,
			      -1.1, -0.9,
			      -0.9, -0.9,
			      1.1, 1.1,
			      0.9, 1.1,
			      1.1, 0.9,
			      0.9, 0.9,
			      -1.1, 1.1,
			      -0.9, 1.1,
			      -1.1, 0.9,
			      -0.9, 0.9,
			      1.1, -1.1,
			      0.9, -1.1,
			      1.1, -0.9,
			      0.9, -0.9};
    labelCount_t labels[3] = {0,1,2};
    for (int i=0; i<4; ++i)
      trainer->feedDenseData (trainPts + 2*i, labels, 1);
    for (int i=4; i<8; ++i)
      trainer->feedDenseData (trainPts + 2*i, labels, 1);
    for (int i=8; i<12; ++i)
      trainer->feedDenseData (trainPts + 2*i, labels+1, 1);
    for (int i=12; i<16; ++i)
      trainer->feedDenseData (trainPts + 2*i, labels+2, 1);
  
    trainer->finalizeData();
  
    trainer->train();

    auto modelBytes = trainer->getModelSize();
    auto model = new char[modelBytes];
  
    trainer->exportModel(modelBytes, model);		       
    auto predictor = new ProtoNNPredictor(modelBytes, model);

    FP_TYPE scoreArray[3] = {0.0, 0.0, 0.0};
    FP_TYPE testPts[2*4] = {-1.0, -1.0,
			    1.0, 1.0,
			    -1.0, 1.0,
			    1.0, -1.0};

    for (int t=0; t<4; ++t) {
      predictor->scoreDenseDataPoint(scoreArray, testPts + 2*t);
      for(int i=0;i<3;++i) std::cout<<scoreArray[i]<<"  ";std::cout<<std::endl;
    }

    delete[] model;
    delete trainer, predictor;
  }

  // Slightly less trivial example
  {
    auto trainer = new ProtoNNTrainer(DataIngestType::InterfaceIngest, hyperParam);

    FP_TYPE trainPts[2*17] = {-1.1, -1.1,
			      -0.9, -1.1,
			      -1.1, -0.9,
			      -0.9, -0.9,
			      1.1, 1.1,
			      0.9, 1.1,
			      1.1, 0.9,
			      0.9, 0.9,
			      -1.1, 1.1,
			      -0.9, 1.1,
			      -1.1, 0.9,
			      -0.9, 0.9,
			      1.1, -1.1,
			      0.9, -1.1,
			      1.1, -0.9,
			      0.9, -0.9,
			      0.0, 0.0}; // Outlier
    labelCount_t labels[3] = {0,1,2};
    for (int i=0; i<3; ++i)
      trainer->feedDenseData (trainPts + 2*i, labels, 1);
    trainer->feedDenseData (trainPts + 6, labels + 1, 1);
    for (int i=4; i<7; ++i)
      trainer->feedDenseData (trainPts + 2*i, labels, 1);
    trainer->feedDenseData (trainPts + 14, labels + 2, 1);
    for (int i=8; i<11; ++i)
      trainer->feedDenseData (trainPts + 2*i, labels+1, 1);
    trainer->feedDenseData (trainPts + 22, labels + 2, 1);
    for (int i=12; i<15; ++i)
      trainer->feedDenseData (trainPts + 2*i, labels+2, 1);
    trainer->feedDenseData (trainPts + 30, labels + 1, 1);

    trainer->feedDenseData (trainPts + 32, labels+2, 1);
  
    trainer->finalizeData();
  
    trainer->train();

    auto modelBytes = trainer->getModelSize();
    auto model = new char[modelBytes];
  
    trainer->exportModel(modelBytes, model);		       
    auto predictor = new ProtoNNPredictor(modelBytes, model);

    FP_TYPE scoreArray[3] = {0.0, 0.0, 0.0};

    FP_TYPE testPts[2*5] = {-1.0, -1.0,
			    1.0, 1.0,
			    -1.0, 1.0,
			    1.0, -1.0,
			    0.5, 0.5};

    for (int t=0; t<5; ++t) {
      predictor->scoreDenseDataPoint(scoreArray, testPts + 2*t);
      for(int i=0;i<3;++i) std::cout<<scoreArray[i]<<"  ";std::cout<<std::endl;
    }

    delete[] model;
    delete trainer, predictor;
  }

  // Slightly less trivial example for sparse data
  {
    auto trainer = new ProtoNNTrainer(DataIngestType::InterfaceIngest, hyperParam);

    featureCount_t indices[2] = {0, 1};
    int numIndices = 2;
    FP_TYPE trainPts[2*17] = {-1.1, -1.1,
			      -0.9, -1.1,
			      -1.1, -0.9,
			      -0.9, -0.9,
			      1.1, 1.1,
			      0.9, 1.1,
			      1.1, 0.9,
			      0.9, 0.9,
			      -1.1, 1.1,
			      -0.9, 1.1,
			      -1.1, 0.9,
			      -0.9, 0.9,
			      1.1, -1.1,
			      0.9, -1.1,
			      1.1, -0.9,
			      0.9, -0.9,
			      0.0, 0.0}; // Outlier
    labelCount_t labels[3] = {0,1,2};
    for (int i=0; i<3; ++i)
      trainer->feedSparseData (trainPts + 2*i, indices, numIndices, labels, 1);
    trainer->feedSparseData (trainPts + 6, indices, numIndices, labels + 1, 1);
    for (int i=4; i<7; ++i)
      trainer->feedSparseData (trainPts + 2*i, indices, numIndices, labels, 1);
    trainer->feedSparseData (trainPts + 14, indices, numIndices, labels + 2, 1);
    for (int i=8; i<11; ++i)
      trainer->feedSparseData (trainPts + 2*i, indices, numIndices, labels+1, 1);
    trainer->feedSparseData (trainPts + 22, indices, numIndices, labels + 2, 1);
    for (int i=12; i<15; ++i)
      trainer->feedSparseData (trainPts + 2*i, indices, numIndices, labels+2, 1);
    trainer->feedSparseData (trainPts + 30, indices, numIndices, labels + 1, 1);

    trainer->feedSparseData (trainPts + 32, indices, numIndices, labels+2, 1);
  
    trainer->finalizeData();
  
    trainer->train();

    auto modelBytes = trainer->getModelSize();
    auto model = new char[modelBytes];
  
    trainer->exportModel(modelBytes, model);		       
    auto predictor = new ProtoNNPredictor(modelBytes, model);

    FP_TYPE scoreArray[3] = {0.0, 0.0, 0.0};

    FP_TYPE testPts[2*5] = {-1.0, -1.0,
			    1.0, 1.0,
			    -1.0, 1.0,
			    1.0, -1.0,
			    0.5, 0.5};

    for (int t=0; t<5; ++t) {
      //predictor->scoreDenseDataPoint(scoreArray, testPts + 2*t);
      // both dense and sparse scoring work
      predictor -> scoreSparseDataPoint(scoreArray, testPts + 2*t, indices, 2);
      for(int i=0;i<3;++i) std::cout<<scoreArray[i]<<"  ";std::cout<<std::endl;
    }

    delete[] model;
    delete trainer, predictor;
  }

}
