// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __BONSAI_H__
#define __BONSAI_H__

#include "Data.h"


namespace EdgeML
{
  namespace Bonsai
  {
    // for Bonsai

#ifdef SPARSE_Z_BONSAI
#define ZMatType SparseMatrixuf
#else
#define ZMatType MatrixXuf
#endif

#ifdef SPARSE_W_BONSAI
#define WMatType SparseMatrixuf
#else
#define WMatType MatrixXuf
#endif

#ifdef SPARSE_V_BONSAI
#define VMatType SparseMatrixuf
#else
#define VMatType MatrixXuf
#endif

#ifdef SPARSE_THETA_BONSAI
#define ThetaMatType SparseMatrixuf
#else
#define ThetaMatType MatrixXuf
#endif

#ifdef SPARSE_LABEL_BONSAI
#define LabelMatType SparseMatrixuf
#else
#define LabelMatType MatrixXuf
#endif

#define nodeProbabilityMatType MatrixXuf 
#define tanhThetaXMatType MatrixXuf
#define tanhVXWeightMatType MatrixXuf 
#define WXWeightMatType MatrixXuf 
#define partialZGradientMatType MatrixXuf

    ///
    /// Bonsai Model Class, to store all the params and hyper params required by the model during training
    ///
    class BonsaiModel
    {
    public:
      ///
      /// Stuct to hold all the regularizers
      ///
      struct RegularizerList
      {
        FP_TYPE lZ;
        FP_TYPE lW;
        FP_TYPE lV;
        FP_TYPE lTheta;
      };

      ///
      /// Struct to Hold all the Hyper Parameters
      ///
      struct BonsaiHyperParams
      {

        bool isModelInitialized;
        ProblemFormat problemType;
        DataFormat dataformatType;
        NormalizationFormat normalizationType;


        int seed;
        int iters, epochs;
        dataCount_t ntrain, nvalidation, ntest, batchSize;
        bool isOneIndex;

        FP_TYPE Sigma; ///< Sigmoid parameter for prediction
        FP_TYPE sigma_i;

        int treeDepth;
        int internalNodes;
        int totalNodes;
        FP_TYPE batchFactor;
        featureCount_t projectionDimension; ///< Projection Dimension, where the learning happens

        // Dataset Parameters
        featureCount_t dataDimension;
        labelCount_t numClasses;

        labelCount_t internalClasses;


        RegularizerList regList;

        FP_TYPE lambdaW, lambdaZ, lambdaV, lambdaTheta; ///< Sparsity Params to reduce the Model Size

        BonsaiHyperParams();
        ~BonsaiHyperParams();

        /// 
        /// Function to set Hyper Params from args, currently unused
        ///
        void setHyperParamsFromArgs(const int& argc, const char** argv);

        ///
        /// Function to fix the hyperparams after the input
        ///
        void finalizeHyperParams();

        ///
        /// Generic Mkdir currently unused
        /// 
        void mkdir() const;
      };

      ///
      /// Struct to hold only the Params of the model
      ///
      struct BonsaiParams
      {
        ZMatType Z;
        WMatType W;
        VMatType V;
        ThetaMatType Theta;

        BonsaiParams();
        ~BonsaiParams();

        ///
        /// Allocaating Memory for params with the knowledge of hyperparams
        ///
        void resizeParamsFromHyperParams(
          const struct BonsaiHyperParams& hyperParams,
          const bool setMemory = true);
      };

      ///
      /// Model Constructor from a existing model
      ///
      BonsaiModel(
        const size_t numBytes,
        const char *const fromModel,
        const bool isDense);

      ///
      /// Model Constructor from commandline args
      ///
      BonsaiModel(
        std::string modelFile,
        const bool isDense);

      ///
      /// Model Constructor from commandline args
      ///
      BonsaiModel(
        const int& argc,
        const char** argv,
        std::string& data_dir);

      ///
      /// Model Constructor from HyperParams
      ///
      BonsaiModel(const BonsaiHyperParams& hyperParams_);

      ///
      /// Default Constructor
      ///
      BonsaiModel();

      ///
      /// Destructor
      ///
      ~BonsaiModel();

      struct BonsaiParams params; ///< Object of Params
      struct BonsaiHyperParams hyperParams; ///< Object of Hyper Params

      ///
      /// Function to Compute and return Dense Model Size
      ///
      size_t modelStat();

      ///
      /// Function to Compute and return Sparse Model Size
      ///
      size_t sparseModelStat();

      ///
      /// Function to Export a model in form of Char buffer. Assumes that toModel is allocated with modelStat() bytes.
      ///
      void exportModel(const size_t modelSize, char *const toModel);

      ///
      /// Function to import a model from a Char buffer. Assumes that toModel is allocated with modelStat() bytes.
      ///
      void importModel(const size_t numBytes, const char *const fromModel);

      ///
      /// Function to export a sparse model from a Char buffer. Assumes that toModel is allocated with modelStat() bytes.
      ///
      void exportSparseModel(const size_t modelSize, char *const toModel);

      ///
      /// Function to import a sparse model from a Char buffer. Assumes that toModel is allocated with modelStat() bytes.
      ///
      void importSparseModel(const size_t numBytes, const char *const fromModel);

      ///
      /// Access W of class classID across all nodes. Returns the corresponding W Matrix (numNodesxprojection Dimension)
      ///
      WMatType getW(const labelCount_t& classID);

      ///
      /// Access V of class classID across all nodes. Returns the corresponding V Matrix (numNodesxprojection Dimension)
      ///
      VMatType getV(const labelCount_t& classID);

      ///
      /// Access W of class classID for a given node. Returns the corresponding W Matrix (1xprojection Dimension)
      ///    
      WMatType getW(const labelCount_t& classID,
        const labelCount_t& globalNodeID);

      ///
      /// Access V of class classID for a given node. Returns the corresponding V Matrix (1xprojection Dimension)
      ///
      VMatType getV(const labelCount_t& classID,
        const labelCount_t& globalNodeID);

      ///
      /// Access Theta(Decison Boundary) for a given node.
      ///
      ThetaMatType getTheta(const labelCount_t& globalNodeID);


      ///
      /// Function to Initialise sigma_i which makes indicator function smooth
      ///
      void initializeSigmaI();

      ///
      /// Function to update sigma_i which makes indicator function smooth
      ///
      void updateSigmaI(const MatrixXuf& ZX,
        const int exp_fac);

      ///
      /// Function to Dump a Readable Model
      ///
      void dumpModel(std::string modelPath);

      size_t totalNonZeros();

      // TODO: get gradient fucntions here

    };

    ///
    /// Class to hold the requirements for Training Bonsai Model
    ///
    class BonsaiTrainer
    {
      FP_TYPE* feedDataValBuffer; ///< Buffer to hold incoming Data values
      featureCount_t* feedDataFeatureBuffer; ///< Buffer to hold incoming Label values

      MatrixXuf mean; ///< Object to hold the mean of the train data
      MatrixXuf variance; ///< Object to hold variance of the train data

      ///
      /// Function to support various normalisations
      ///
      void normalize();

      ///
      /// Function to Initilise Model
      ///
      void initializeModel();

    public:

      ///
      /// Struct to Hold the tree cache information while training Bonsai tree to compute Gradients.
      ///
      struct TreeCache
      {
        nodeProbabilityMatType nodeProbability;
        tanhThetaXMatType tanhThetaXCache;
        tanhVXWeightMatType tanhVXWeight;
        WXWeightMatType WXWeight;
        partialZGradientMatType partialZGradient;

        ///
        /// Function to fill the Indicator Values at each node
        ///
        void fillNodeProbability(
          const BonsaiModel& model,
          const MatrixXuf& Theta,
          const MatrixXuf& Xdata);

        ///
        /// Function to fill W'Zx values at each node for a given class
        ///
        void fillWX(
          const BonsaiModel& model,
          const WMatType& W,
          const MatrixXuf& Xdata,
          const MatrixXufINT& classID);

        ///
        /// Function to fill tanh(Sigma*V'Zx) values at each node for a given class
        ///
        void fillTanhVX(
          const BonsaiModel& model,
          const VMatType& V,
          const MatrixXuf& Xdata,
          const MatrixXufINT& classID);
      };

      /// DO NOT REORDER model and data. They should be in this order for constructors to work
      BonsaiModel model; ///< Model Object    
      Data data; ///< Data Object to store the train and test data
      //////////////////// 

      struct TreeCache treeCache; ///< Tree Cache Object
      MatrixXuf YMultCoeff; ///< Object to hold different label convention of Binary classification

      ///
      /// Use this constructor for training 
      /// 1. On data ingested from file
      /// 2. Reloading pre-trained model from disk
      /// Gurantees that finalizeData is called before returning
      /// Read comments in BonsaiTrainer.cpp before using this constructor
      ///
      BonsaiTrainer(
        const DataIngestType& dataIngestType,
        const size_t& numBytes,
        const char *const fromModel,
        const std::string& data_dir,
        std::string& currResultsPath,
        const bool isDense = true);

      ///
      /// Use this constructor for training
      /// 1. On data ingested from the disk
      /// 2. Starting with a new model from scratch.
      /// Gurantees that finalizeData is called before returning
      ///
      BonsaiTrainer(
        const DataIngestType& dataIngestType,
        const int& argc,
        const char ** argv,
        std::string& data_dir,
        std::string& currResultsPath);

      ///
      /// Call this constructor for training
      /// 1. On data ingested using feedData
      /// 2. Starting with a new model from scratch
      /// finalizeData is not called inside, it must explicity called after feeding data
      ///
      BonsaiTrainer(
        const DataIngestType& dataIngestType,
        const BonsaiModel::BonsaiHyperParams& hyperParams);

      ~BonsaiTrainer();

      ///
      /// Function to Feed a single Dense Data point
      ///
      void feedDenseData(
        const FP_TYPE *const values,
        const labelCount_t *const labels,
        const labelCount_t& numLabels);

      ///
      /// Function to Feed a single Sparse Data point
      ///  
      void feedSparseData(
        const FP_TYPE *const values,
        const featureCount_t *const indices,
        const featureCount_t& numIndices,
        const labelCount_t *const labels,
        const labelCount_t& numLabels);

      ///
      /// Function get all the fed data to required Matrix Form
      ///
      void finalizeData();

      ///
      /// Function to Compute the regularized Loss function with inherent model params
      ///
      FP_TYPE computeObjective(
        const MatrixXuf& ZX,
        const LabelMatType& Y);

      ///
      /// Overloaded Function to Compute the regularized Loss function with custom params
      ///
      FP_TYPE computeObjective(
        const MatrixXuf& Z,
        const MatrixXuf& W,
        const MatrixXuf& V,
        const MatrixXuf& Theta,
        const MatrixXuf& ZX,
        const LabelMatType& Y);

      ///
      /// Initialise Tree Cache and Custom Label Matrix
      ///
      void initializeTrainVariables(const LabelMatType& Y);

      ///
      /// Core Train Function which calls required solver
      ///
      void train();

      ///
      /// Compute Score of a given point on a given Class, by having it pass through entire tree
      ///
      void computeScoreOfClassID(
        MatrixXuf& Score,
        const MatrixXuf& W,
        const MatrixXuf& V,
        const MatrixXuf& ZX,
        const labelCount_t& classID,
        MatrixXuf& WXClassIDScratch,
        MatrixXuf& VXClassIDScratch);

      ///
      /// Function to get Score and Index of True Class and Best Non Class. Used in computing margin loss
      ///
      void getTrueBestClass(
        MatrixXuf& true_best_Score,
        MatrixXufINT& true_best_classIndex,
        const MatrixXuf& Wmat,
        const MatrixXuf& Vmat,
        const LabelMatType& Y,
        const MatrixXuf& ZX);

      ///
      /// Function to fill the Indicator Values at each node
      ///
      void fillNodeProbability(const MatrixXuf& ZX);

      ///
      /// Function to fill W'Zx values at each node for a given class
      ///
      void fillWX(const MatrixXuf& ZX,
        const MatrixXufINT& classID);

      ///
      /// Function to fill tanh(Sigma*V'Zx) values at each node for a given class
      ///
      void fillTanhVX(const MatrixXuf& ZX,
        const MatrixXufINT& classID);

      ///
      /// Export Functions
      ///
      void exportModel(
        const size_t& modelSize,
        char *const buffer);
      void exportModel(
        const size_t& modelSize,
        char *const buffer,
        const std::string& currResultsPath);

      void exportSparseModel(
        const size_t& modelSize,
        char *const buffer);
      void exportSparseModel(
        const size_t& modelSize,
        char *const buffer,
        const std::string& currResultsPath);

      size_t getModelSize();
      size_t getSparseModelSize();

      void exportMeanVar(
        const size_t& meanVarSize,
        char *const buffer);
      void exportMeanVar(
        const size_t& meanVarSize,
        char *const buffer,
        const std::string& currResultsPath);

      size_t getMeanVarSize();

      ///
      /// Gives reloadablity for a pretrained model stored after training in the results directory
      ///
      void loadModel(
        const std::string model_path,
        const size_t modelBytes,
        const bool isDense = true);

      // Call these to export V W, Z,Theta separately.
      // call size required, preallocate space required and call export
      size_t sizeForExportVSparse();
      void exportVSparse(int bufferSize, char *const buf);

      size_t sizeForExportWSparse();
      void exportWSparse(int bufferSize, char *const buf);

      size_t sizeForExportZSparse();
      void exportZSparse(int bufferSize, char *const buf);

      size_t sizeForExportThetaSparse();
      void exportThetaSparse(int bufferSize, char *const buf);

      size_t sizeForExportVDense();
      void exportVDense(int bufferSize, char *const buf);

      size_t sizeForExportWDense();
      void exportWDense(int bufferSize, char *const buf);

      size_t sizeForExportZDense();
      void exportZDense(int bufferSize, char *const buf);

      size_t sizeForExportThetaDense();
      void exportThetaDense(int bufferSize, char *const buf);

      ///
      /// Function to Dump Readable Mean, Variance and Model 
      ///
      void dumpModelMeanVar(const std::string& currResultsPath);

      ///
      /// Function to Dump Loadable Mean, Variance and Model 
      ///
      void getLoadableModelMeanVar(char *const modelBuffer, const size_t& modelBytes, char *const meanVarBuffer, const size_t& meanVarBytes, const std::string& currResultsPath);

      size_t totalNonZeros();
    };


    ///
    /// Bonsai Predictor Class to hold relevant information and methods for predictor of Bonsai
    ///
    class BonsaiPredictor
    {
      FP_TYPE* feedDataValBuffer; ///< Buffer to hold incoming Data values
      featureCount_t* feedDataFeatureBuffer; ///< Buffer to hold incoming Label values

      MatrixXuf mean; ///< Object to hold the mean of the train data from imported model
      MatrixXuf variance; ///< Object to hold variance of the train data from imported model

      BonsaiModel model; ///< Object to hold the imported model
      Data testData;
      dataCount_t numTest;
      DataFormat dataformatType;

      std::string dataDir;
      std::string modelDir;
      void setFromArgs(const int argc, const char** argv);
      void exitWithHelp();

    public:
      
      ///
      /// Constructor instantiating model from commandline
      ///
      BonsaiPredictor(const int argc,
        const char** argv);

      ///
      /// Constructor instantiating model from a trained model
      ///
      BonsaiPredictor(const size_t numBytes,
        const char *const fromModel,
        const bool isDense = true);

      ~BonsaiPredictor();

      ///
      /// Function to import stored Mean and Variance
      ///
      void importMeanVar(const size_t numBytes,
        const char *const fromBuffer);

      ///
      /// Function to import stored Mean and Variance
      ///
      void importMeanVar(std::string meanVarFile);

      ///
      /// Function to Score an incoming Dense Data Point.Not thread safe
      ///
      void scoreDenseDataPoint(FP_TYPE* scores,
        const FP_TYPE *const values);

      ///
      /// Function to obtain Prediction score of a given class for a given data point
      ///
      FP_TYPE predictionScoreOfClassID(const MatrixXuf& ZX,
        const std::vector<int> path,
        const labelCount_t& classID);

      ///
      /// Computes and returns the path traversed in Bonsai Tree
      ///
      std::vector<int> treePath(const MatrixXuf& ZX);

      ///
      /// Function to return the scores of all classes for a given Dense Data Point
      ///
      void predictionScore(
        const MatrixXuf& X,
        FP_TYPE *scores);

      ///
      /// Function to return the scores of all classes for a given Sparse Data Point
      ///
      void predictionSparseScore(
        const SparseMatrixuf& X,
        FP_TYPE *scores);

      ///
      /// Function to Score an incoming sparse Data Point.Not thread safe
      ///
      void scoreSparseDataPoint(
        FP_TYPE* scores,
        const FP_TYPE *const values,
        const featureCount_t *const indices,
        const featureCount_t& numIndices);

      ///
      /// Function to return total nonzeros in the model loaded
      ///
      size_t totalNonZeros();

      ///
      /// Function to dump the hyperparams of the model along with prediction stats
      ///
      void dumpRunInfo(
        const std::string currResultsPath,
        const FP_TYPE& correct);

      ///
      /// Function to predict an entire test dataset
      ///
      void batchEvaluate(
        const SparseMatrixuf& Xtest,
        const SparseMatrixuf& Ytest,
        const std::string& dataDir,
        const std::string& currResultsPath);
      
      ///
      /// Function to predict an entire test dataset
      ///
      void evaluate();
    };

  }
}
#endif
