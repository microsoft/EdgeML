// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __PROTONN_H__
#define __PROTONN_H__

#include "Data.h"

namespace EdgeML
{
  namespace ProtoNN
  {
    //
    // Should the model parameters and labels be represented as sparse or dense?
    //
#ifdef SPARSE_Z
#define ZMatType SparseMatrixuf
#else
#define ZMatType MatrixXuf
#endif

#ifdef SPARSE_W
#define WMatType SparseMatrixuf
#else
#define WMatType MatrixXuf
#endif

#ifdef SPARSE_B
#define BMatType SparseMatrixuf
#else
#define BMatType MatrixXuf
#endif

#ifdef SPARSE_LABEL
#define LabelMatType SparseMatrixuf
#else
#define LabelMatType MatrixXuf
#endif

    //
    // ProtoNNModel includes hyperparameters, parameters, and some state on initialization
    //    
    class ProtoNNModel
    {

    public:

      struct ProtoNNHyperParams
      {
        int seed;

        dataCount_t ntrain, ntest, batchSize;
        int iters, epochs;

        featureCount_t D, d;
        labelCount_t m, k, l;

        FP_TYPE gammaNumerator, gamma;
        FP_TYPE lambdaW, lambdaZ, lambdaB;

        NormalizationFormat normalizationType;
        ProblemFormat problemType;
        InitializationFormat initializationType;

        bool isHyperParamInitialized;

        ProtoNNHyperParams();
        ~ProtoNNHyperParams();

        void setHyperParamsFromArgs(const int argc, const char** argv);
        void finalizeHyperParams();

        //
        // Create a string with hyperParam settings
        // Will be used for subdir name dumping results
        //
        std::string subdirName() const;
      };

      struct ProtoNNParams
      {
        ZMatType Z;
        WMatType W;
        BMatType B;

        ProtoNNParams();
        ~ProtoNNParams();

        void resizeParamsFromHyperParams(
          const struct ProtoNNHyperParams& hyperParams,
          const bool setMemory = true);
      };

      struct ProtoNNParams params;
      struct ProtoNNHyperParams hyperParams;
      size_t modelStat();

      //
      // exportModel assumes that toModel is allocated with modelStat() bytes.
      //
      void exportModel(const size_t modelSize, char *const toModel);
      void importModel(const size_t numBytes, const char *const fromModel);

      ProtoNNModel(const size_t numBytes, const char *const fromModel);
      ProtoNNModel(const int argc, const char** argv);
      ProtoNNModel(const ProtoNNHyperParams& hyperParams_);
      ~ProtoNNModel();
    };

    class ProtoNNTrainer
    {
      ////////////////////////////////////////////////////////
      // DO NOT REORDER model and data. 
      // They should be in this order for constructors to work
      ProtoNNModel model;
      Data data;
      ////////////////////////////////////////////////////////

      DataFormat dataformatType;
      std::string indir;
      std::string outdir;
      std::string commandLine;

      void normalize();
      void initializeModel();

    public:
        //
        // Call this constructor if:
        // 1. Training data is ingested from the disk
        // 2. You are starting with a new model from scratch
        // Gurantees that finalizeData is called before returning
        //
      ProtoNNTrainer(
        const DataIngestType& dataIngestType,
        const int& argc,
        const char ** argv);

      //
      // Call this constructor if:
      // 1. Training data is ingested using feedData
      // 2. You are starting with a new model from scratch
      // finalizeData is not called inside, it must explicity called after feeding data
      //
      ProtoNNTrainer(
        const DataIngestType& dataIngestType,
        const ProtoNNModel::ProtoNNHyperParams& hyperParams);

      ~ProtoNNTrainer();

      void feedDenseData(
        const FP_TYPE *const values,
        const labelCount_t *const labels,
        const labelCount_t& numLabels);

      void feedSparseData(
        const FP_TYPE *const values,
        const featureCount_t *const indices,
        const featureCount_t& numIndices,
        const labelCount_t *const labels,
        const labelCount_t& numLabels);

      void finalizeData();

      void setFromArgs(const int argc, const char** argv);

      void storeParams(
        std::string commandLine,
        FP_TYPE* stats,
        std::string outFile);

      void createOutputDirs();

      void train();

      // This exports W, B, Z together in a dense format.
      // Call getModelSize, prealloc buffer, and pass it to exportModel.
      size_t getModelSize();
      void exportModel(const size_t& modelSize, char *const buffer);

      // Call these to export W, B, Z separately.
      // call size required, preallocate space required and call export
      size_t sizeForExportBSparse();
      void exportBSparse(int bufferSize, char *const buf);

      size_t sizeForExportWSparse();
      void exportWSparse(int bufferSize, char *const buf);

      size_t sizeForExportZSparse();
      void exportZSparse(int bufferSize, char *const buf);

      size_t sizeForExportBDense();
      void exportBDense(int bufferSize, char *const buf);

      size_t sizeForExportWDense();
      void exportWDense(int bufferSize, char *const buf);

      size_t sizeForExportZDense();
      void exportZDense(int bufferSize, char *const buf);
    };

    class ProtoNNPredictor
    {
      ProtoNNModel model;

      MatrixXuf D, WX, WXColSum;    // Updated within RBF
      MatrixXuf BColSum, BAccumulator, B_B, gammaSqRow, gammaSqCol; // Constants set in constructor
      FP_TYPE gammaSq;

      FP_TYPE* data;	// for scoreSparseDataPoint

#ifdef SPARSE_Z
      // for mkl csc_mv call
      char matdescra[6] = { 'G', 'X', 'X', 'C', 'X', 'X' }; // 'X' means unused
      char transa = 'n';
      MKL_INT ZRows;
      MKL_INT ZCols;
      FP_TYPE alpha, beta;
#endif

      void RBF();

    public:

      ProtoNNPredictor(
        const size_t numBytes,
        const char *const fromModel);

      ~ProtoNNPredictor();

      // Not thread safe      
      FP_TYPE testDenseDataPoint(
        const FP_TYPE *const values,
        const labelCount_t *const labels,
        const labelCount_t& numLabels,
        const EdgeML::ProblemFormat& problemType);


      // Not thread safe
      void scoreDenseDataPoint(
        FP_TYPE* scores,
        const FP_TYPE *const values);

      void scoreSparseDataPoint(
        FP_TYPE* scores,
        const FP_TYPE *const values,
        const featureCount_t *indices,
        const featureCount_t numIndices);

    };
  }
}
#endif
