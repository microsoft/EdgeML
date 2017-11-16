// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.


#include "ProtoNN.h"
#include <iostream>

using namespace EdgeML; 

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

  ProtoNN::ProtoNNPredictor predictor(argc, (const char**)argv);
  EdgeML::ResultStruct res;

  res = predictor.test(); 

  predictor.saveTopKScores();

  switch(res.problemType) {
    case binary:
    case multiclass:
      LOG_INFO("Accuracy: " + std::to_string(res.accuracy));
      break;
    case multilabel:
      LOG_INFO("Prec@1: " + std::to_string(res.precision1));
      LOG_INFO("Prec@3: " + std::to_string(res.precision3));
      LOG_INFO("Prec@5: " + std::to_string(res.precision5));
      break;
    default:
      assert(false);
  }
}
