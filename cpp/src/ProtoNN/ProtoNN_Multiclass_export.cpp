// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//#include "stdafx.h"

#include "utils.h"
#include "ProtoNN.h"

#ifdef _MSC_VER
#define EXPORT_API(ret) extern "C" __declspec(dllexport) ret __stdcall
#else
#define EXPORT_API(ret) extern "C" __attribute__((visisbility("default"))) ret
#endif

namespace EdgeML
{
  namespace ProtoNN
  {
    //************************************************
    // 1. Create a trainer
    // 2. Feed the data
    // 3. Finalize data
    // 4. Train the model
    // 5. Export the model
    // 6. Destroy trainer
    //************************************************
    EXPORT_API(ProtoNNTrainer*) CreateTrainer(
      labelCount_t numClasses,			// Number of classes (Can you get this parameter in FinalizeData call? I know number of features, before i go throug data, but not labels 
      featureCount_t numFeatures,			// Number of features,
      featureCount_t projectedDimension,	// Hyperparam [default: 10, 5, 50, 100] 
                                          // Dimension of space features are projected in
                                          // Constraint: projectedDimension <= numFeatures
      InitializationFormat clusteringInit,// Hyperparam [default: overallKmeans, perClassKmeans]
                                          // What clustering do you want on points initially
      labelCount_t numPrototypes,			// Hyperparam [default: 40, 20, 100, 200, 500, 1000] 
                                          // Used only if "clusterinInit==overallKmeans"
                                          // Constraint: 	numPrototypes < #trainPoints
      labelCount_t numPrototypesPerClass,	// Hyperparam [default: 10, 1, 5, 20]
                                          // Used only if "clusteringInit==per_class_k_means"
      FP_TYPE gammaNumerator,				// Hyperparam [default: 1.0, 0.1, 10.0]
                                          // RBF kernel  = 2.5*gammaNumerator/median(dist.b/w.points & init prototypes)
      FP_TYPE sparsityW,					// Hyperparam [default: 1.0, 0.001, 0.01, 0.1]
                                          // Sparsity of projection matrix
      FP_TYPE sparsityZ,					// Hyperparam [default: 1.0, 0.001, 0.01, 0.1]
                                          // Sparsity of label matrix
      FP_TYPE sparsityB,					// Hyperparam [default: 1.0, 0.01, 0.1]
                                          // Sparsity of Prototype matrix
      dataCount_t batchSize,				// Hyperparam [default: sqrt(ntrain), 32, 128, 512, 4096] 
                                          // Batch size of stochastic gradient descent steps
      NormalizationFormat normalize,		// Hyperparam [default: none, l2, minMax]
      int iterations,						// Hyperparam [default: 20, 10, 50, 100]
      int epochs,							// Hyperparam [default: 5, 1, 3, 10, 20]
      int seed = 42,						// Hyperparam [default: 42, any random seed]
      ChannelFunc trace_print_func_ = NULL,
      ChannelFunc info_print_func_ = NULL,
      ChannelFunc warning_print_func_ = NULL,
      ChannelFunc error_print_func_ = NULL
    )
    {
      ProtoNNModel::ProtoNNHyperParams hyperParam;

      hyperParam.problemType = ProblemFormat::multiclass;
      hyperParam.initializationType = clusteringInit;
      hyperParam.normalizationType = normalize;
      hyperParam.seed = seed;
      hyperParam.batchSize = batchSize;
      hyperParam.iters = iterations;
      hyperParam.epochs = epochs;
      hyperParam.D = numFeatures;
      hyperParam.d = projectedDimension < numFeatures ? projectedDimension : numFeatures;
      hyperParam.m = numPrototypes;
      hyperParam.k = numPrototypesPerClass;
      hyperParam.l = numClasses;
      hyperParam.nvalidation = 0;
      hyperParam.gammaNumerator = gammaNumerator;
      hyperParam.lambdaW = sparsityW;
      hyperParam.lambdaZ = sparsityZ;
      hyperParam.lambdaB = sparsityB;

      hyperParam.finalizeHyperParams();

      LOG_SET_TRACE_FUNC(trace_print_func_);
      LOG_SET_INFO_FUNC(info_print_func_);
      LOG_SET_WARNING_FUNC(warning_print_func_);
      LOG_SET_ERROR_FUNC(error_print_func_);

      return new ProtoNNTrainer(DataIngestType::InterfaceIngest, hyperParam);
    }

    EXPORT_API(void) DestroyTrainer(
      ProtoNNTrainer* trainer)
    {
      LOG_SET_TRACE_FUNC(NULL);
      LOG_SET_INFO_FUNC(NULL);
      LOG_SET_WARNING_FUNC(NULL);
      LOG_SET_ERROR_FUNC(NULL);
      delete trainer;
    }

    EXPORT_API(void) FeedSparseData(
      ProtoNNTrainer* trainer,
      const featureCount_t numIndices,		// #non-zero features
      const FP_TYPE *const values,			// non-zero feature values
      const featureCount_t *const indices,	// feature index of non-zero values
      const labelCount_t label)				// which class is this point
    {
      trainer->feedSparseData(values, indices, numIndices, &label, 1);
    }

    EXPORT_API(void) FeedDenseData(ProtoNNTrainer* trainer,
      const FP_TYPE *const values,			// feature values
      const labelCount_t label)				// which class is this point
    {
      trainer->feedDenseData(values, &label, 1);
    }

    EXPORT_API(void) FinalizeData(ProtoNNTrainer* trainer)
    {
      trainer->finalizeData();
    }

    EXPORT_API(void) Train(ProtoNNTrainer* trainer)
    {
      trainer->train();
    }

    //************************************************
    // Export model all at once
    //************************************************
    // 
    // Size of the trained model to exported 
    // Guarantees that return is less than 1<<31.
    // 
    EXPORT_API(int) GetModelSize(ProtoNNTrainer* trainer)
    {
      return (int)trainer->getModelSize();
    }
    // 
    // preallocate buffer to modelStat and all exportModel to retrieve trained model
    //
    EXPORT_API(void) ExportModel(
      ProtoNNTrainer* trainer,
      int allocatedBits,
      char *const buffer)
    {
      assert(allocatedBits == trainer->getModelSize());
      trainer->exportModel(allocatedBits, buffer);
    }

    //************************************************
    // Export model in parts in sparse format
    //************************************************

    EXPORT_API(int) sizeForExportBSparse(ProtoNNTrainer* trainer)
    {
      return (int)trainer->sizeForExportBSparse();
    }
    EXPORT_API(void) exportBSparse(
      ProtoNNTrainer* trainer, int bufferSize, char *const buf)
    {
      trainer->exportBSparse(bufferSize, buf);
    }
    EXPORT_API(int) sizeForExportWSparse(ProtoNNTrainer* trainer)
    {
      return (int)trainer->sizeForExportWSparse();
    }
    EXPORT_API(void) exportWSparse(
      ProtoNNTrainer* trainer, int bufferSize, char *const buf)
    {
      trainer->exportWSparse(bufferSize, buf);
    }
    EXPORT_API(int) sizeForExportZSparse(ProtoNNTrainer* trainer)
    {
      return (int)trainer->sizeForExportZSparse();
    }
    EXPORT_API(void) exportZSparse(
      ProtoNNTrainer* trainer, int bufferSize, char *const buf)
    {
      trainer->exportZSparse(bufferSize, buf);
    }

    //************************************************
    // Export model in parts in dense format
    //************************************************
    EXPORT_API(int) sizeForExportBDense(ProtoNNTrainer* trainer)
    {
      return (int)trainer->sizeForExportBDense();
    }
    EXPORT_API(void) exportBDense(
      ProtoNNTrainer* trainer, int bufferSize, char *const buf)
    {
      trainer->exportBDense(bufferSize, buf);
    }
    EXPORT_API(int) sizeForExportWDense(ProtoNNTrainer* trainer)
    {
      return (int)trainer->sizeForExportWDense();
    }
    EXPORT_API(void) exportWDense(
      ProtoNNTrainer* trainer, int bufferSize, char *const buf)
    {
      trainer->exportWDense(bufferSize, buf);
    }
    EXPORT_API(int) sizeForExportZDense(ProtoNNTrainer* trainer)
    {
      return (int)trainer->sizeForExportZDense();
    }
    EXPORT_API(void) exportZDense(
      ProtoNNTrainer* trainer, int bufferSize, char *const buf)
    {
      trainer->exportZDense(bufferSize, buf);
    }



    //***********************************************
    // PREDICTOR CALLS  
    //***********************************************

    EXPORT_API(ProtoNNPredictor*) CreatePredictor(
      const int numBytes,
      const char *const trainedModel)
    {
      return new ProtoNNPredictor(numBytes, trainedModel);
    }

    EXPORT_API(void) ScoreDenseData(
      ProtoNNPredictor* predictor,
      const FP_TYPE *const values,	// The features of test point,
      FP_TYPE *const scoresPerClass)  //  Score per class, should be initialized to length #classes before calling
    {
      predictor->scoreDenseDataPoint(scoresPerClass, values);
    }

    EXPORT_API(void) ScoreSparseData(
      ProtoNNPredictor* predictor,
      const featureCount_t numIndices,		// #non-zero features
      const FP_TYPE *const values,			// non-zero feature values
      const featureCount_t *const indices,	// feature index of non-zero values
      FP_TYPE *const scoresPerClass)			//  Score per class, should be initialized to #classes before calling
    {
      predictor->scoreSparseDataPoint(scoresPerClass, values, indices, numIndices);
    }

    //It's nice to cleanup after yourself.
    EXPORT_API(void) DestroyPredictor(
      ProtoNNPredictor* predictor)
    {
       delete predictor;
    }
  }
}
