// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifdef LINUX  

#ifndef __GOLDFOIL_H__
#define __GOLDFOIL_H__

#include <cstdlib>
#include <cstdio>
#include <csignal>
#include <cfenv>

void __attribute__((constructor)) trapfpe();

void fpehandler(int sig, siginfo_t *info, void *uc);

#endif
#endif 
