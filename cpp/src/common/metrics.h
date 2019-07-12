// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __METRICS_H__
#define __METRICS_H__

#include "Data.h"
#include "utils.h"

namespace EdgeML
{
  struct ResultStruct {
    ProblemFormat problemType;
    FP_TYPE accuracy;
    FP_TYPE precision1;
    FP_TYPE precision3;
    FP_TYPE precision5;
    
    ResultStruct();
    void scaleAndAdd(ResultStruct& a, FP_TYPE scale);
    void scale(FP_TYPE scale);
  };


  ResultStruct evaluate(
    const MatrixXuf& Yscores,
    const SparseMatrixuf& Y,
    const ProblemFormat problemType);
  

  void getTopKScoresBatch(
    const MatrixXuf& Yscores,
    MatrixXuf& topKIndices,
    MatrixXuf& topKScores,
    int k = 5);
};

#endif 
