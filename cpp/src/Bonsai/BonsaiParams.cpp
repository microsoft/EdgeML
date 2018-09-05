// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "Bonsai.h"

using namespace EdgeML;
using namespace EdgeML::Bonsai;

BonsaiModel::BonsaiParams::BonsaiParams() {}
BonsaiModel::BonsaiParams::~BonsaiParams() {}

void BonsaiModel::BonsaiParams::resizeParamsFromHyperParams(
  const struct BonsaiModel::BonsaiHyperParams& hyperParams,
  const bool setMemory)
{
  assert(hyperParams.isModelInitialized == true);
  Z.resize(hyperParams.projectionDimension, hyperParams.dataDimension);
  W.resize((hyperParams.internalClasses) * (hyperParams.totalNodes),
    hyperParams.projectionDimension);
  V.resize((hyperParams.internalClasses) * (hyperParams.totalNodes),
    hyperParams.projectionDimension);
  Theta.resize(hyperParams.internalNodes, hyperParams.projectionDimension);

  // The following non-sense is because Eigen does not document whether resize should set the data to 0
  if (setMemory) {
#ifdef SPARSE_Z_BONSAI
#else
    memset(Z.data(), 0, sizeof(FP_TYPE)*Z.rows()*Z.cols());
#endif

#ifdef SPARSE_W_BONSAI
#else
    memset(W.data(), 0, sizeof(FP_TYPE)*W.rows()*W.cols());
#endif

#ifdef SPARSE_V_BONSAI
#else
    memset(V.data(), 0, sizeof(FP_TYPE)*V.rows()*V.cols());
#endif

#ifdef SPARSE_THETA_BONSAI
#else
    memset(Theta.data(), 0, sizeof(FP_TYPE)*Theta.rows()*Theta.cols());
#endif
  }
}
