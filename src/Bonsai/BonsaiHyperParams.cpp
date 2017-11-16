// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "Bonsai.h"

using namespace EdgeML;
using namespace EdgeML::Bonsai;

// WARNING
// Always ensure that the default values in TLC export are the same as here
//

BonsaiModel::BonsaiHyperParams::BonsaiHyperParams()
{
  dataformatType = undefinedData;
  problemType = undefinedProblem;
  normalizationType = none;


  seed = 42;

  ntrain = 0;
  nvalidation = 0;
  batchSize = 0;

  iters = 0;
  epochs = 0;
  isOneIndex = 0;
  batchFactor = 0.0;

  regList.lZ = (FP_TYPE)-1.0;
  regList.lW = (FP_TYPE)-1.0;
  regList.lV = (FP_TYPE)-1.0;
  regList.lTheta = (FP_TYPE)-1.0;

  Sigma = (FP_TYPE)-1.0;
  sigma_i = (FP_TYPE)-1.0;
  treeDepth = 0;
  internalNodes = 0;
  totalNodes = 0;

  projectionDimension = 0;
  dataDimension = 0;
  numClasses = 0;
  internalClasses = 0;

  lambdaW = (FP_TYPE)1.0;
  lambdaZ = (FP_TYPE)1.0;
  lambdaV = (FP_TYPE)1.0;
  lambdaTheta = (FP_TYPE)1.0;

}

BonsaiModel::BonsaiHyperParams::~BonsaiHyperParams() {}

void BonsaiModel::BonsaiHyperParams::setHyperParamsFromArgs(const int& argc,
  const char** argv)
{
  // parse_args(argc, argv);
}


// TODO: populate mkdir
void BonsaiModel::BonsaiHyperParams::mkdir() const
{
  /*
  sprintf (outDir, "%s/results/%f_%f_%f_%llu_%d",
     indir, lambdaW, lambdaZ, lambda_B, d, m);

  char command [100];

  try {
    sprintf (command, "mkdir %s/results", indir);
    system(command);
    sprintf (command, "mkdir %s", outDir);
    system (command);
#ifdef DUMP
    sprintf (command, "mkdir %s/dump", outDir);
    system (command);
#endif
#ifdef VERIFY
    sprintf (command, "mkdir %s/verify", outDir);
    system (command);
#endif
  }
  catch (...){
    std::cerr << "One of the directories could not be created... " <<std::endl;
  }
  */
}

void BonsaiModel::BonsaiHyperParams::finalizeHyperParams()
{

  assert(treeDepth >= 0);
  assert(internalNodes >= 0);
  assert(totalNodes >= 1);
  assert(iters >= 1);
  assert(projectionDimension > 0);
  assert(dataDimension > 0);
  assert(batchFactor >= 0.0);
  // Following asserts removed to faciliate support for TLC
  // which does not know how many datapoints are going to be fed before-hand!
  // assert(ntrain >= 1);               
  assert(projectionDimension <= dataDimension + 1);
  assert(numClasses > 0);

  assert(lambdaW >= (FP_TYPE)0.0L);
  assert(lambdaZ >= (FP_TYPE)0.0L);
  assert(lambdaV >= (FP_TYPE)0.0L);
  assert(lambdaTheta >= (FP_TYPE)0.0L);

  assert(lambdaW <= (FP_TYPE)1.0);
  assert(lambdaZ <= (FP_TYPE)1.0);
  assert(lambdaV <= (FP_TYPE)1.0);
  assert(lambdaTheta <= (FP_TYPE)1.0);

  assert(problemType != undefinedProblem);
  assert(dataformatType != undefinedData);
  assert(normalizationType != undefinedNormalization);

  srand(seed);
  mkdir();
  internalClasses = (numClasses <= 2) ? 1 : numClasses;
  isModelInitialized = true;
  LOG_INFO("Dataset successfully initialized...");
}
