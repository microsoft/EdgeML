// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "ProtoNN.h"

using namespace EdgeML::ProtoNN;

ProtoNNModel::ProtoNNParams::ProtoNNParams() {}
ProtoNNModel::ProtoNNParams::~ProtoNNParams() {}

void ProtoNNModel::ProtoNNParams::resizeParamsFromHyperParams(
  const struct ProtoNNModel::ProtoNNHyperParams& hyperParams,
  const bool setMemory)
{
  assert(hyperParams.isHyperParamInitialized == true);
  Z.resize(hyperParams.l, hyperParams.m);
  B.resize(hyperParams.d, hyperParams.m);
  W.resize(hyperParams.d, hyperParams.D);

  // The following non-sense is because Eigen does not document whether resize should set the data to 0
  if (setMemory) {
    memset(Z.data(), 0, sizeof(FP_TYPE)*Z.rows()*Z.cols());
    memset(W.data(), 0, sizeof(FP_TYPE)*W.rows()*W.cols());
    memset(B.data(), 0, sizeof(FP_TYPE)*B.rows()*B.cols());
  }
}

