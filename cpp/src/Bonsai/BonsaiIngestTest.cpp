//#include "stdafx.h"

#include "Bonsai.h"

using namespace EdgeML;
using namespace EdgeML::Bonsai;

int main()
{
  BonsaiModel::BonsaiHyperParams hyperParam;

  hyperParam.problemType = ProblemFormat::multiclass;
  hyperParam.dataformatType = DataFormat::interfaceIngestFormat;
  hyperParam.normalizationType = NormalizationFormat::none;

  hyperParam.seed = 41;

  hyperParam.batchSize = 1;
  hyperParam.iters = 20;
  hyperParam.epochs = 1;

  hyperParam.dataDimension = 784;
  hyperParam.projectionDimension = 30;

  hyperParam.numClasses = 10;

  hyperParam.nvalidation = 0;
  hyperParam.ntrain = 5000;

  hyperParam.Sigma = 1.0;
  hyperParam.treeDepth = 3;

  hyperParam.internalNodes = (pow(2, hyperParam.treeDepth) - 1);
  hyperParam.totalNodes = 2 * hyperParam.internalNodes + 1;

  hyperParam.regList.lW = 1.0e-3;
  hyperParam.regList.lZ = 1.0e-5;
  hyperParam.regList.lV = 1.0e-3;
  hyperParam.regList.lTheta = 1.0e-3;


  hyperParam.lambdaW = 10 / 30;
  hyperParam.lambdaZ = 150 / 785;
  hyperParam.lambdaV = 10 / 30;
  hyperParam.lambdaTheta = 10 / 30;

  hyperParam.finalizeHyperParams();

  // trivial data set
  {
	BonsaiTrainer trainer(DataIngestType::InterfaceIngest, hyperParam);

	std::ifstream ifs("/home/t-vekusu/ICMLDatasets/multiclass/mnist/train");
	FP_TYPE *trainvals = new FP_TYPE[hyperParam.dataDimension];
	memset(trainvals, 0, sizeof(FP_TYPE)*(hyperParam.dataDimension));


	FP_TYPE temp;
	// labelCount_t *temp1 = new labelCount_t[hyperParam.numClasses];
	labelCount_t *labve = new labelCount_t[1];
	for (int i = 0; i < hyperParam.ntrain; ++i) {
	  int count = 0;
	  ifs >> temp;
	  // std::cout<<temp<<std::endl;
	  labve[0] = (labelCount_t)temp;
	  while (count < hyperParam.dataDimension) {
		ifs >> temp;
		trainvals[count] = temp;
		count++;
	  }
	  // std::cout<<labve[0]<<std::endl;
	  labve[0]--;
	  // std::cout<<labve[0]<<std::endl;
	  // for(int j=0; j<hyperParam.numClasses; j++) {
	  //   ift >> temp1[j];
	  //   if(temp1[j] != 0) {
	  //     labve[0] = j;
	  //   }
	  // }

	  trainer.feedDenseData(trainvals, labve, 1);
	  // if(i%5000 == 0) std::cout<<i<<std::endl;
	}
	ifs.close();


	trainer.finalizeData();

	trainer.train();

	// auto modelBytes = trainer.getModelSize();
	// auto model = new char[modelBytes];
	auto modelBytes = trainer.getSparseModelSize();
	auto model = new char[modelBytes];

	auto meanVarBytes = trainer.getMeanVarSize();
	auto meanVar = new char[meanVarBytes];

	// trainer.exportModel(modelBytes, model); 
	trainer.exportSparseModel(modelBytes, model);

	trainer.exportMeanVar(meanVarBytes, meanVar);

	// BonsaiPredictor predictor(modelBytes, model);

	BonsaiPredictor predictor(modelBytes, model, false);

	predictor.importMeanVar(meanVarBytes, meanVar);

	FP_TYPE *scoreArray = new FP_TYPE[hyperParam.numClasses];

	std::ifstream ifw("/home/t-vekusu/ICMLDatasets/multiclass/mnist/test");

	int correct = 0;
	for (int i = 0; i < 10; ++i) {
	  int count = 0;
	  ifw >> temp;

	  labve[0] = (labelCount_t)temp;
	  while (count < hyperParam.dataDimension) {
		ifw >> temp;
		trainvals[count] = temp;
		count++;
	  }

	  predictor.scoreDenseDataPoint(scoreArray, trainvals);

	  labelCount_t predLabel = 0;
	  FP_TYPE max_score = scoreArray[0];
	  for (int j = 0; j < hyperParam.numClasses; j++) {
		if (max_score <= scoreArray[j]) {
		  max_score = scoreArray[j];
		  predLabel = j;
		}
	  }


	  labve[0]--;

	  if (labve[0] == predLabel) correct++;
	}
	std::cout << correct << std::endl;
	ifw.close();


	std::cout << "Final Test Accuracy = " << ((FP_TYPE)correct) / ((FP_TYPE)10) << std::endl;
	delete[] scoreArray, trainvals, model, labve;


  }

  // // Slightly less trivial example
  // {
  //   auto trainer = new BonsaiTrainer(DataIngestType::InterfaceIngest, hyperParam);

  //   FP_TYPE trainPts[2*16] = {-1.1, -1.1,
		// 	      0.9, 1.1,
		// 	      1.1, 0.9,
		// 	      -0.9, -0.9,
		// 	      1.1, 1.1,
		// 	      0.9, 1.1,
		// 	      1.1, 0.9,
		// 	      0.9, 0.9,
		// 	      -1.1, 1.1,
		// 	      -0.9, 1.1,
		// 	      -1.1, 0.9,
		// 	      -0.9, 0.9,
		// 	      1.1, -1.1,
		// 	      0.9, -1.1,
		// 	      1.1, -0.9,
		// 	      0.9, -0.9}; // Outlier
  //   labelCount_t labels[3] = {0,1,2};
  //   for (int i=0; i<4; ++i)
  //     trainer->feedDenseData (trainPts + 2*i, labels, 1);

  //   for (int i=4; i<8; ++i)
  //     trainer->feedDenseData (trainPts + 2*i, labels, 1);

  //   for (int i=8; i<12; ++i)
  //     trainer->feedDenseData (trainPts + 2*i, labels+1, 1);

  //   for (int i=12; i<16; ++i)
  //     trainer->feedDenseData (trainPts + 2*i, labels+2, 1);


  //   trainer->finalizeData();

  //   // std::cout<<trainer->mean<<std::endl<<std::endl;
  //   // std::cout<<trainer->variance<<std::endl<<std::endl;

  //   trainer->train();

  //   auto modelBytes = trainer->getModelSize();
  //   auto model = new char[modelBytes];

  //   auto meanVarBytes = trainer->getMeanVarSize();
  //   auto meanVar = new char[meanVarBytes];

  //   trainer->exportModel(modelBytes, model);
  //   trainer->exportMeanVar(meanVarBytes, meanVar);

  //   auto predictor = new BonsaiPredictor(modelBytes, model);
  //   predictor->importMeanVar(meanVarBytes, meanVar);



  //   FP_TYPE scoreArray[3] = {0.0, 0.0, 0.0};

  //   FP_TYPE testPts[2*4] = {-1.0, -1.0,
		// 		   1.0, 1.0,
		// 		   -1.0, 1.0,
		// 		   1.0, -1.0};

  //   for (int t=0; t<4; ++t) {
  //     predictor->scoreDenseDataPoint(scoreArray, testPts + 2*t);
  //     for(int i=0;i<3;++i) std::cout<<scoreArray[i]<<"  ";std::cout<<std::endl;
  //   }

  //   delete[] model;
  //   delete trainer, predictor;
  // }

  // // Slightly less trivial example for sparse data
  // {
  //   auto trainer = new BonsaiTrainer(DataIngestType::InterfaceIngest, hyperParam);

  //   featureCount_t indices[2] = {0, 1};
  //   int numIndices = 2;
  //   FP_TYPE trainPts[2*17] = {-1.1, -1.1,
  //           -0.9, -1.1,
  //           -1.1, -0.9,
  //           -0.9, -0.9,
  //           1.1, 1.1,
  //           0.9, 1.1,
  //           1.1, 0.9,
  //           0.9, 0.9,
  //           -1.1, 1.1,
  //           -0.9, 1.1,
  //           -1.1, 0.9,
  //           -0.9, 0.9,
  //           1.1, -1.1,
  //           0.9, -1.1,
  //           1.1, -0.9,
  //           0.9, -0.9,
  //           0.0, 0.0}; // Outlier
  //   labelCount_t labels[3] = {0,1,2};
  //   for (int i=0; i<3; ++i)
  //     trainer->feedSparseData (trainPts + 2*i, indices, numIndices, labels, 1);
  //   trainer->feedSparseData (trainPts + 6, indices, numIndices, labels + 1, 1);
  //   for (int i=4; i<7; ++i)
  //     trainer->feedSparseData (trainPts + 2*i, indices, numIndices, labels, 1);
  //   trainer->feedSparseData (trainPts + 14, indices, numIndices, labels + 2, 1);
  //   for (int i=8; i<11; ++i)
  //     trainer->feedSparseData (trainPts + 2*i, indices, numIndices, labels+1, 1);
  //   trainer->feedSparseData (trainPts + 22, indices, numIndices, labels + 2, 1);
  //   for (int i=12; i<15; ++i)
  //     trainer->feedSparseData (trainPts + 2*i, indices, numIndices, labels+2, 1);
  //   trainer->feedSparseData (trainPts + 30, indices, numIndices, labels + 1, 1);

  //   trainer->feedSparseData (trainPts + 32, indices, numIndices, labels+2, 1);

  //   trainer->finalizeData();

  //   trainer->train();

  //   auto modelBytes = trainer->getModelSize();
  //   auto model = new char[modelBytes];

  //   trainer->exportModel(modelBytes, model);           
  //   auto predictor = new BonsaiPredictor(modelBytes, model);

  //   FP_TYPE scoreArray[3] = {0.0, 0.0, 0.0};

  //   FP_TYPE testPts[2*5] = {-1.0, -1.0,
  //          1.0, 1.0,
  //          -1.0, 1.0,
  //          1.0, -1.0,
  //          0.5, 0.5};

  //   for (int t=0; t<5; ++t) {
  //     predictor->scoreDenseDataPoint(scoreArray, testPts + 2*t);
  //     for(int i=0;i<3;++i) std::cout<<scoreArray[i]<<"  ";std::cout<<std::endl;
  //   }

  //   delete[] model;
  //   delete trainer, predictor;
  // }
  // // Slightly less trivial example for sparse data
  // {
  //   auto trainer = new BonsaiTrainer(DataIngestType::InterfaceIngest, hyperParam);

  //   featureCount_t indices[2] = {0, 1};
  //   int numIndices = 2;
  //   FP_TYPE trainPts[2*17] = {-1.1, -1.1,
  //           -0.9, -1.1,
  //           -1.1, -0.9,
  //           -0.9, -0.9,
  //           1.1, 1.1,
  //           0.9, 1.1,
  //           1.1, 0.9,
  //           0.9, 0.9,
  //           -1.1, 1.1,
  //           -0.9, 1.1,
  //           -1.1, 0.9,
  //           -0.9, 0.9,
  //           1.1, -1.1,
  //           0.9, -1.1,
  //           1.1, -0.9,
  //           0.9, -0.9,
  //           0.0, 0.0}; // Outlier
  //   labelCount_t labels[3] = {0,1,2};
  //   for (int i=0; i<3; ++i)
  //     trainer->feedSparseData (trainPts + 2*i, indices, numIndices, labels, 1);
  //   trainer->feedSparseData (trainPts + 6, indices, numIndices, labels + 1, 1);
  //   for (int i=4; i<7; ++i)
  //     trainer->feedSparseData (trainPts + 2*i, indices, numIndices, labels, 1);
  //   trainer->feedSparseData (trainPts + 14, indices, numIndices, labels + 2, 1);
  //   for (int i=8; i<11; ++i)
  //     trainer->feedSparseData (trainPts + 2*i, indices, numIndices, labels+1, 1);
  //   trainer->feedSparseData (trainPts + 22, indices, numIndices, labels + 2, 1);
  //   for (int i=12; i<15; ++i)
  //     trainer->feedSparseData (trainPts + 2*i, indices, numIndices, labels+2, 1);
  //   trainer->feedSparseData (trainPts + 30, indices, numIndices, labels + 1, 1);

  //   trainer->feedSparseData (trainPts + 32, indices, numIndices, labels+2, 1);

  //   trainer->finalizeData();

  //   trainer->train();

  //   auto modelBytes = trainer->getModelSize();
  //   auto model = new char[modelBytes];

  //   trainer->exportModel(modelBytes, model);           
  //   auto predictor = new BonsaiPredictor(modelBytes, model);

  //   FP_TYPE scoreArray[3] = {0.0, 0.0, 0.0};

  //   FP_TYPE testPts[2*5] = {-1.0, -1.0,
  //          1.0, 1.0,
  //          -1.0, 1.0,
  //          1.0, -1.0,
  //          0.5, 0.5};

  //   for (int t=0; t<5; ++t) {
  //     predictor->scoreSparseDataPoint(scoreArray, testPts + 2*t, indices, numIndices);
  //     for(int i=0;i<3;++i) std::cout<<scoreArray[i]<<"  ";std::cout<<std::endl;
  //   }

  //   delete[] model;
  //   delete trainer, predictor;
  // }
}
