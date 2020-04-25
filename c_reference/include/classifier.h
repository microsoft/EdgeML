// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __CLASSIFIER_H__
#define __CLASSIFIER_H__

void FC(const float* const FCweights, const float* const FCbias,
  const float* const input, const unsigned inputLen,
  float* const classScores, const unsigned numClasses);

#endif
