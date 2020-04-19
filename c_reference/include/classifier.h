// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __CLASSIFIER_H__
#define __CLASSIFIER_H__

#include "utils.h"

void FC(const float *const FCweights, const float *const FCbias,
  const float *const input, const unsigned inputLen,
  float* const classScores, const unsigned numClasses) {
  matVec(FCweights, input, numClasses, inputLen, 0.0f, 1.0f, classScores);
  v_add(1.0f, FCbias, 1.0f, classScores, numClasses, classScores);
}

#endif
