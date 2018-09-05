// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "Bonsai.h"

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
  assert (sizeof(MKL_INT) == sizeof(Eigen::Index));
  
  std::string dataDir;
  std::string currResultsPath;
  
  BonsaiTrainer trainer(DataIngestType::FileIngest, argc, (const char**) argv, 
                        dataDir, currResultsPath);
  
  auto modelBytes = trainer.getModelSize(); // This can be changed to getSparseModelSize() if you need to export sparse model
  auto model = new char[modelBytes];
  auto meanVarBytes = trainer.getMeanVarSize();
  auto meanVar = new char[meanVarBytes];
  
  trainer.exportModel(modelBytes, model); // use exportSparseModel(...) if you need sparse model
  trainer.exportMeanVar(meanVarBytes, meanVar);
  trainer.getLoadableModelMeanVar(model, modelBytes, meanVar, meanVarBytes, currResultsPath);
  trainer.dumpModelMeanVar(currResultsPath);
  
  delete[] model, meanVar;
  
  return 0;
}
