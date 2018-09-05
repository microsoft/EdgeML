// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "Bonsai.h"

#ifdef _MSC_VER
#define EXPORT_API(ret) extern "C" __declspec(dllexport) ret __stdcall
#else
#define EXPORT_API(ret) extern "C" __attribute__((visisbility("default"))) ret
#endif

namespace EdgeML
{
  namespace Bonsai
  {
    //************************************************
    // 1. Create a trainer
    // 2. Feed the data
    // 3. Finalize data
    // 4. Train the model
    // 5. Export the model
    // 6. Destroy trainer
    //************************************************
    EXPORT_API(BonsaiTrainer*) CreateTrainer(
      labelCount_t numClasses,			  // Number of classes 
      featureCount_t numFeatures,		  // Number of features,
      featureCount_t projectedDimension,  // Number of dimensions of the space that features are projected into
                                          // Hyperparam [default: 10, 5, 50, 100] 
                                          // Constraint: projectedDimension <= numFeatures
      int treeDepth,					  // Depth of tree to be built
                                          // Hyperparam [default: 3, 2, 5, 10]
      FP_TYPE sigma,					  // Sigmoid scale hyperparameter at each node
                                          // Hyperparam [default: 1.0, 0.1, 10.0] 

      // There are four model parameters Z, W, V, Theta, which are matrices. 
      // Each parametrs gets a regularizer, listed below.
      FP_TYPE regulariserZ,				// Hyperparam [default: 0.00001, 0.001, 0.0001, 0.000001]
      FP_TYPE regulariserW,				// Hyperparam [default: 0.0001, 0.01, 0.001, 0.00001]
      FP_TYPE regulariserV,				// Hyperparam [default: 0.0001, 0.01, 0.001, 0.00001]
      FP_TYPE regulariserTheta,			// Hyperparam [default: 0.0001, 0.01, 0.001, 0.00001]

      // Each parameter also gets a hyperparameter indicating its sparsity
      FP_TYPE sparsityZ,				// Hyperparam for multiclass [default: 0.2, 0.1, 0.3, 0.4, 0.5]
                                        // Hyperparam for binary	 [default: 0.2, 0.1, 0.3, 0.4, 0.5]
      FP_TYPE sparsityW,				// Hyperparam for multiclass [default: 0.2, 0.1, 0.3, 0.4, 0.5]
                                        // Hyperparam for binary	 [default: 1.0]	
      FP_TYPE sparsityV,				// Hyperparam [default: 0.2, 0.1, 0.3, 0.4, 0.5]
                                        // Hyperparam for binary	 [default: 1.0]	
      FP_TYPE sparsityTheta,			// Hyperparam [default: 0.2, 0.1, 0.3, 0.4, 0.5]
                                        // Hyperparam for binary	 [default: 1.0]	

      // Normalization. Currently Mean-var normalization is done internally.
      NormalizationFormat normalize,	// Hyperparam [default: none, l2, minMax]

      // Number of iterations, equally divided between three training phases of the algorithm
      int iterations,					// Hyperparam [default: 20, 10, 30, 50]
      int seed = 42,					// Hyperparam [default: 42, any random seed]
      ChannelFunc trace_print_func = NULL,
      ChannelFunc info_print_func = NULL,
      ChannelFunc warning_print_func = NULL,
      ChannelFunc error_print_func = NULL
    )
    {
      BonsaiModel::BonsaiHyperParams hyperParam;

      hyperParam.problemType = ProblemFormat::multiclass;
      hyperParam.dataformatType = DataFormat::interfaceIngestFormat;

      hyperParam.normalizationType = normalize;

      hyperParam.seed = seed;
      hyperParam.batchSize = 1;
      hyperParam.iters = iterations;

      hyperParam.Sigma = sigma;
      hyperParam.dataDimension = numFeatures;
      hyperParam.projectionDimension = projectedDimension <= numFeatures + 1 ? projectedDimension : numFeatures;
      hyperParam.numClasses = numClasses;
      hyperParam.treeDepth = treeDepth;

      hyperParam.regList.lZ = regulariserZ;
      hyperParam.regList.lW = regulariserW;
      hyperParam.regList.lV = regulariserV;
      hyperParam.regList.lTheta = regulariserTheta;

      hyperParam.lambdaW = sparsityW;
      hyperParam.lambdaV = sparsityV;
      hyperParam.lambdaZ = sparsityZ;
      hyperParam.lambdaTheta = sparsityTheta;

      hyperParam.internalNodes = (1 << treeDepth) - 1;
      hyperParam.totalNodes = 2 * hyperParam.internalNodes + 1;

      hyperParam.finalizeHyperParams();

      /* LOG_SET_TRACE_FUNC(trace_print_func);
       LOG_SET_INFO_FUNC(info_print_func);
       LOG_SET_WARNING_FUNC(warning_print_func);
       LOG_SET_ERROR_FUNC(error_print_func);*/

      return new BonsaiTrainer(DataIngestType::InterfaceIngest, hyperParam);
    }

    EXPORT_API(void) DestroyTrainer(BonsaiTrainer* trainer)
    {
      LOG_SET_TRACE_FUNC(NULL);
      LOG_SET_INFO_FUNC(NULL);
      LOG_SET_WARNING_FUNC(NULL);
      LOG_SET_ERROR_FUNC(NULL);
      delete trainer;
    }

    EXPORT_API(void) FeedSparseData(BonsaiTrainer* trainer,
      const featureCount_t numIndices,// #non-zero features
      const FP_TYPE *const values,// non-zero feature values
      const featureCount_t *const indices,// feature index of non-zero values
      const labelCount_t label)// which class is this point
    {
      trainer->feedSparseData(values, indices, numIndices, &label, 1);
    }

    EXPORT_API(void) FeedDenseData(BonsaiTrainer* trainer,
      const FP_TYPE *const values,// feature values
      const labelCount_t label)// which class is this point
    {
      trainer->feedDenseData(values, &label, 1);
    }

    EXPORT_API(void) FinalizeData(BonsaiTrainer* trainer)
    {
      trainer->finalizeData();
    }

    EXPORT_API(void) Train(BonsaiTrainer* trainer)
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
    EXPORT_API(int) GetModelSize(BonsaiTrainer* trainer)
    {
      auto modelSize = trainer->model.modelStat();
      if (trainer->data.getIngestType() == DataIngestType::InterfaceIngest)
        assert(modelSize < (size_t)(1 << 31)); // Because we make this promise to TLC.

      return (int)modelSize;
    }
    // 
    // preallocate buffer to modelStat and all exportModel to retrieve trained model
    //
    EXPORT_API(void) ExportModel(
      BonsaiTrainer* trainer,
      int allocatedBits,
      char *const buffer)
    {
      assert(allocatedBits == trainer->model.modelStat());
      trainer->model.exportModel(allocatedBits, buffer);
    }

    //
    // Size of the mean-var normalization values to be exported.
    // Call before exportMeanVar
    //
    EXPORT_API(int) GetMeanVarSize(
      BonsaiTrainer* trainer)
    {
      size_t meanVarSize = trainer->getMeanVarSize();
      assert(meanVarSize < (size_t)(1 << 31)); // Because this is expected in TLC
      return (int)meanVarSize;
    }

    //
    // Preallocate to size mean-var normalization
    //
    EXPORT_API(void) ExportMeanVar(
      BonsaiTrainer* trainer,
      const int meanVarSize,
      char *const buffer)
    {
      assert(meanVarSize == trainer->getMeanVarSize());
      trainer->exportMeanVar(meanVarSize, buffer);
    }


    //************************************************
    // Export model in parts in sparse format
    //************************************************

    EXPORT_API(int) sizeForExportVSparse(BonsaiTrainer* trainer)
    {
      return (int)trainer->sizeForExportVSparse();
    }
    EXPORT_API(void) exportVSparse(
      BonsaiTrainer* trainer, int bufferSize, char *const buf)
    {
      trainer->exportVSparse(bufferSize, buf);
    }
    EXPORT_API(int) sizeForExportWSparse(BonsaiTrainer* trainer)
    {
      return (int)trainer->sizeForExportWSparse();
    }
    EXPORT_API(void) exportWSparse(
      BonsaiTrainer* trainer, int bufferSize, char *const buf)
    {
      trainer->exportWSparse(bufferSize, buf);
    }
    EXPORT_API(int) sizeForExportZSparse(BonsaiTrainer* trainer)
    {
      return (int)trainer->sizeForExportZSparse();
    }
    EXPORT_API(void) exportZSparse(
      BonsaiTrainer* trainer, int bufferSize, char *const buf)
    {
      trainer->exportZSparse(bufferSize, buf);
    }
    EXPORT_API(int) sizeForExportThetaSparse(BonsaiTrainer* trainer)
    {
      return (int)trainer->sizeForExportThetaSparse();
    }
    EXPORT_API(void) exportThetaSparse(
      BonsaiTrainer* trainer, int bufferSize, char *const buf)
    {
      trainer->exportThetaSparse(bufferSize, buf);
    }

    //************************************************
    // Export model in parts in dense format
    //************************************************
    EXPORT_API(int) sizeForExportVDense(BonsaiTrainer* trainer)
    {
      return (int)trainer->sizeForExportVDense();
    }
    EXPORT_API(void) exportVDense(
      BonsaiTrainer* trainer, int bufferSize, char *const buf)
    {
      trainer->exportVDense(bufferSize, buf);
    }
    EXPORT_API(int) sizeForExportWDense(BonsaiTrainer* trainer)
    {
      return (int)trainer->sizeForExportWDense();
    }
    EXPORT_API(void) exportWDense(
      BonsaiTrainer* trainer, int bufferSize, char *const buf)
    {
      trainer->exportWDense(bufferSize, buf);
    }
    EXPORT_API(int) sizeForExportZDense(BonsaiTrainer* trainer)
    {
      return (int)trainer->sizeForExportZDense();
    }
    EXPORT_API(void) exportZDense(
      BonsaiTrainer* trainer, int bufferSize, char *const buf)
    {
      trainer->exportZDense(bufferSize, buf);
    }
    EXPORT_API(int) sizeForExportThetaDense(BonsaiTrainer* trainer)
    {
      return (int)trainer->sizeForExportThetaDense();
    }
    EXPORT_API(void) exportThetaDense(
      BonsaiTrainer* trainer, int bufferSize, char *const buf)
    {
      trainer->exportThetaDense(bufferSize, buf);
    }


    //***********************************************
    // PREDICTOR CALLS  
    //***********************************************

    EXPORT_API(BonsaiPredictor*) CreatePredictor(
      const int numBytes,
      const char *const trainedModel,
      const int meanVarSize = 0,
      const char *const buffer = NULL)
    {
      BonsaiPredictor* predictor = new BonsaiPredictor(numBytes, trainedModel);
      predictor->importMeanVar(meanVarSize, buffer);
      return predictor;
    }


    EXPORT_API(void) ScoreDenseData(BonsaiPredictor* predictor,
      const FP_TYPE *const values,	// The features of test point,
      FP_TYPE *const scoresPerClass)  //  Score per class, should be initialized to length #classes before calling
    {
      predictor->scoreDenseDataPoint(scoresPerClass, values);
    }

    //
    // Pre-Allocate scoresPerClass to length #Classes and not numIndices
    //
    EXPORT_API(void) ScoreSparseData(
      BonsaiPredictor* predictor,
      const featureCount_t numIndices,	// #non-zero features
      const FP_TYPE *const values,		// non-zero feature values
      const featureCount_t *const indices,// feature index of non-zero values
      FP_TYPE *const scoresPerClass)		//  Score per class, should be initialized to #classes before calling
    {
      predictor->scoreSparseDataPoint(scoresPerClass, values, indices, numIndices);
    }


    //It's nice to cleanup after yourself.
    EXPORT_API(void) DestroyPredictor(BonsaiPredictor* predictor)
    {
      delete predictor;
    }
  }
}
