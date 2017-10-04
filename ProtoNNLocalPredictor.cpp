// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.


#include "ProtoNN.h"
#include "logger.h"
#include <iostream>

using namespace EdgeML::ProtoNN;

int main(int argc, char **argv) {
#ifdef LINUX
  trapfpe();
  struct sigaction sa;
  sigemptyset (&sa.sa_mask);
  sa.sa_flags = SA_SIGINFO;
  sa.sa_sigaction = fpehandler;
  sigaction (SIGFPE, &sa, NULL);
#endif
 
  assert(sizeof(MKL_INT) == 8 && "need large enough indices to store matrices");
  assert(sizeof(MKL_INT) == sizeof(Eigen::Index) && "MKL BLAS routines are called directly on data of an Eigen matrix. Hence, the index sizes should match."); 

  EdgeML::ProtoNN::ProtoNNPredictor predictor(EdgeML::DataIngestType::FileIngest,
                                              argc, (const char**)argv);

  ProtoNNPredictor::ResultStruct res;
  res = predictor.evaluateScores();
  switch(res.problemType) {
    case binary:
    case multiclass:
      std::cout << std::endl << "Accuracy: " << res.accuracy << std::endl;
      break;
    case multilabel:
      std::cout << std::endl << "Prec@1: " << res.precision1 << std::endl
                             << "Prec@3: " << res.precision3 << std::endl
                             << "Prec@5: " << res.precision5 << std::endl;
      break;
    default:
      assert(false);
  }
}
