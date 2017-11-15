// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.


#include "ProtoNN.h"
#include "logger.h"

using namespace EdgeML::ProtoNN;

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
  
  assert(sizeof(MKL_INT) == 8 && "need large enough indices to store matrices");
  assert(sizeof(MKL_INT) == sizeof(Eigen::Index) && "MKL BLAS routines are called directly on data of an Eigen matrix. Hence, the index sizes should match."); 
  EdgeML::ProtoNN::ProtoNNTrainer trainer(argc, (const char**)argv);
  
  trainer.train();

  return 0;
}
