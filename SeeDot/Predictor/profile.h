// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

void initializeProfiling();

void updateRange(float x);
void updateRangeOfExp(float x);

void dumpRange(std::string outputFile);

void debug();

void diff(float* A, MYINT* B, MYINT scale, MYINT I, MYINT J);
void diff(float* A, MYINT* B, MYINT scale, MYINT I, MYINT J, MYINT K);
