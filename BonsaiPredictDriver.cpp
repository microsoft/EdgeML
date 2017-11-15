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

  BonsaiPredictor predictor(argc, (const char**) argv); // use the constructor predictor(modelBytes, model, false) for loading a sparse model.
  
  return 0;
}
