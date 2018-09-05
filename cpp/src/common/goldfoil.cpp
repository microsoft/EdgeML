// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifdef LINUX

#include "goldfoil.h"

void __attribute__((constructor)) trapfpe()
{
  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW); // | FE_UNDERFLOW);
}

void fpehandler(int sig, siginfo_t *info, void *uc)
{
  fprintf(stderr,
    "Caught signal no. %d; code: %d \n",
    sig, info->si_code);
  if (info->si_code == FPE_INTDIV) {
    fprintf(stderr,
      "Integer division by zero.\n");
  }
  if (info->si_code == FPE_FLTUND) {
    fprintf(stderr,
      "Underflow\n");
  }
  if (info->si_code == FPE_FLTOVF) {
    fprintf(stderr,
      "Overflow\n");
  }
  if (info->si_code == FPE_FLTRES) {
    fprintf(stderr,
      "Inexact result\n");
  }
  if (info->si_code == FPE_FLTINV) {
    fprintf(stderr,
      "Invalid exception\n");
  }
  fprintf(stderr,
    "SIGFPE issued, exiting.\n");

  _Exit(0);
}

#endif
