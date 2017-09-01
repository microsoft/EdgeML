/***************************************************************

BonsaiLocalDriver.cpp
   
Contains main(){ ... }. Entry point for instantiations of Bonsai. 
    
Authors: 
    - Chirag Gupta <t-chgupt@microsoft.com>
    - Aditya Kusupati <t-vekusu@microsoft.com>
    - Harsha Vardhan Simhadri <harshasi@microsoft.com>
    - Prateek Jain <prajain@microsoft.com>

Date: 18 July, 2017

****************************************************************/

#include "stdafx.h"

#include "Bonsai.h"
#include "BonsaiFunctions.h"

using namespace EdgeML;

using namespace EdgeML::Bonsai;

int main(int argc, char **argv)
{
#ifdef LINUX
  trapfpe();
  struct sigaction sa;
  sigemptyset (&sa.sa_mask);
  sa.sa_flags = SA_SIGINFO;
  sa.sa_sigaction = fpehandler;
  sigaction (SIGFPE, &sa, NULL);
#endif

  assert (sizeof(MKL_INT) == 8);
  assert (sizeof(MKL_INT) == sizeof(Eigen::Index));
  std::cout << "Using " << sizeof (MKL_INT) << " bytes for MKL calls and indexing EIGEN matrices \n";


// #ifndef STDERR_ONSCREEN
//   char cerr_file [100];
//   sprintf (cerr_file, "log");
//   std::ofstream cerr_outfile(cerr_file);
//   std::cerr.rdbuf(cerr_outfile.rdbuf());
// #endif

  BonsaiModel::BonsaiHyperParams hyperParam;

  std::string dir_path;
  ParseInput(argc, argv, hyperParam, dir_path);

  std::string train_path = dir_path + "/train";
  std::string test_path = dir_path + "/test";

  std::cout<<train_path<<" "<<test_path<<std::endl;

  hyperParam.finalizeHyperParams();

  BonsaiTrainer trainer(DataIngestType::InterfaceIngest, hyperParam);

  std::ifstream trainreader(train_path);

  FP_TYPE *trainvals = new FP_TYPE[hyperParam.dataDimension];
  memset(trainvals, 0, sizeof(FP_TYPE)*(hyperParam.dataDimension));

  FP_TYPE readerVar;
  labelCount_t *label = new labelCount_t[1];

  for(int i=0; i<hyperParam.ntrain; ++i) {
    int count = 0;
    trainreader >> readerVar;
    label[0] = (labelCount_t)readerVar;

    while(count < hyperParam.dataDimension) {
      trainreader >> readerVar;
      trainvals[count] = readerVar;
      count++;
    }

    label[0]--; // uncomment this is the labels are not zero index

    trainer.feedDenseData(trainvals, label, 1);
    // if(i%1000 == 0) std::cout<<i<<std::endl;
  } 

  trainreader.close();


  trainer.finalizeData();

  trainer.train();


  auto modelBytes = trainer.getModelSize();
  auto model = new char[modelBytes];
  auto meanVarBytes = trainer.getMeanVarSize();
  auto meanVar = new char[meanVarBytes];

  trainer.exportModel(modelBytes, model); 
  trainer.exportMeanVar(meanVarBytes, meanVar);

  BonsaiPredictor predictor(modelBytes, model);
  predictor.importMeanVar(meanVarBytes, meanVar);

  FP_TYPE *scoreArray = new FP_TYPE[hyperParam.numClasses];

  std::ifstream testreader(test_path);

  int correct = 0;
  for(int i=0; i<hyperParam.ntest; ++i) {
    int count = 0;
    testreader >> readerVar;
    label[0] = (labelCount_t)readerVar;
    while(count < hyperParam.dataDimension) {
      testreader >> readerVar;
      trainvals[count] = readerVar;
      count++;
    }

    predictor.scoreDenseDataPoint(scoreArray, trainvals);

    labelCount_t predLabel = 0;
    FP_TYPE max_score = scoreArray[0];
    for(int j=0; j<hyperParam.numClasses; j++) {
      if(max_score <= scoreArray[j]) {
        max_score = scoreArray[j];
        predLabel = j;
      }
    }

    label[0]--; // uncomment this is the labels are not zero index

    if (label[0] == predLabel) correct++;
  }

  testreader.close();
  
  std::cout<<"Final Test Accuracy = "<<((FP_TYPE)correct)/((FP_TYPE)hyperParam.ntest)<<std::endl;
  delete[] scoreArray, trainvals, model, label, meanVar;

  return 0;
}
