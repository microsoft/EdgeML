// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

// Used by old SeeDot, invoked before the inference code is executed, it is used to
// set the variables used for profiling to default values.
void initializeProfiling();

// Methods used by old SeeDot to capture the range of a variable in floating point mode.
// Used only for exponentiation in old SeeDot.
void updateRange(float x);
void updateRangeOfExp(float x);
// Used by old SeeDot to store the range of exponentiation variable.
void dumpRange(std::string outputFile);

// Store the range of all variables into a file.
void dumpProfile();
// This method is used to take the ranges of variables taken for one datapoint and store them into the global range,
// if the range of exponentiation variables is within an acceptable threshold.
void flushProfile();

void debug();

// Check whether the range of the variable is higher than a threshold, beyond which the datapoint is not considered for profiling.
// Please check OOPSLA'20 paper Section 5.4 for details.
void checkRange2(float* A, int I, int J);

// Methods used to capture the range of a 4-D, 3-D and 2-D variables in the floating point mode which is used for data-driven scaling.
void Profile4(float* A, int I, int J, int K, int L, std::string name);
void Profile3(float* A, int I, int J, int K, std::string name);
void Profile2(float* A, int I, int J, std::string name);

// Used to capture the difference of corresponding variables in floating-point and fixed-point mode.
void diff(float* A, MYINT* B, MYINT scale, MYINT I, MYINT J);
void diff(float* A, MYINT* B, MYINT scale, MYINT I, MYINT J, MYINT K);

extern bool profilingEnabled;
