// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

void initializeProfiling();

void updateRange(float x);
void updateRangeOfExp(float x);

void dumpRange(std::string outputFile);

void debug();

void diff(float *A, MYINT *B, MYINT scale, MYINT I, MYINT J);
void diff(float *A, MYINT *B, MYINT scale, MYINT I, MYINT J, MYINT K);

void checkRange2(float* A, int I, int J);
void Profile4(float* A, int I, int J, int K, int L, std::string name);
void Profile3(float* A, int I, int J, int K, std::string name);
void Profile2(float* A, int I, int J, std::string name);
void flushProfile();
void dumpProfile();

extern bool profilingEnabled;