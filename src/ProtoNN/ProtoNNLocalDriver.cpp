/***************************************************************

ProtoNNLocalDriver.cpp
   
Contains main(){ ... }. Entry point for instantiations of ProtoNN. 
    
Authors: 
    - Chirag Gupta <t-chgupt@microsoft.com>
    - Aditya Kusupati <t-vekusu@microsoft.com>
    - Harsha Vardhan Simhadri <harshasi@microsoft.com>
    - Prateek Jain <prajain@microsoft.com>

Date: 18 July, 2017

****************************************************************/

#include "stdafx.h"

#include "ProtoNN.h"

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

  assert (sizeof(MKL_INT) == 8);
  assert (sizeof(MKL_INT) == sizeof(Eigen::Index));
  std::cout << "Using " << sizeof (MKL_INT) << " bytes for MKL calls and indexing EIGEN matrices \n";

#ifndef STDERR_ONSCREEN
  char cerr_file [100];
  sprintf (cerr_file, "log");
  std::ofstream cerr_outfile(cerr_file);
  std::cerr.rdbuf(cerr_outfile.rdbuf());
#endif

  EdgeML::ProtoNN::ProtoNNTrainer trainer(EdgeML::DataIngestType::FileIngest,
					  argc, (const char**)argv,
					  "/mnt/datasets/eurlex/data/train.txt",
					  "/mnt/datasets/eurlex/data/test.txt");

  trainer.train();

  return 0;
}
