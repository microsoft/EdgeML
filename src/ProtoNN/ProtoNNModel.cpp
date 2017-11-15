// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "ProtoNN.h"

using namespace EdgeML;
using namespace EdgeML::ProtoNN;

ProtoNNModel::ProtoNNModel(
  std::string modelFile)
{
  std::ifstream infile(modelFile, std::ios::in|std::ios::binary);
  assert(infile.is_open());
  
  // Read the size of the model
  size_t modelSize;
  infile.read((char*)&modelSize, sizeof(modelSize));

  // Allocate buffer
  char* buff = new char[modelSize];

  //Load model from model file
  infile.read((char*)buff, modelSize);
  infile.close();

  importModel(modelSize, (char *const) buff);

  delete[] buff;
}

ProtoNNModel::ProtoNNModel()
{
}

ProtoNNModel::ProtoNNModel(
  const size_t numBytes,
  const char *const fromModel)
{
  importModel(numBytes, fromModel);
}

ProtoNNModel::ProtoNNModel(const int argc, const char** argv)
{
  hyperParams.setHyperParamsFromArgs(argc, argv);
  params.resizeParamsFromHyperParams(hyperParams);
}


ProtoNNModel::ProtoNNModel(const ProtoNNModel::ProtoNNHyperParams& hyperParams_)
  : hyperParams(hyperParams_)
{
  LOG_INFO("Initialized model hyperparameters");
  params.resizeParamsFromHyperParams(hyperParams_);
  LOG_INFO("Resized model parameters");
}
ProtoNNModel::~ProtoNNModel() {}

size_t ProtoNNModel::modelStat()
{
  size_t offset = 0;
  offset += sizeof(hyperParams);
#ifdef SPARSE_Z_PROTONN
  offset += sizeof(bool);
  offset += sizeof(Eigen::Index);
  Eigen::Index nnz = params.Z.outerIndexPtr()[params.Z.cols()]
    - params.Z.outerIndexPtr()[0];
  offset += sizeof(FP_TYPE) * nnz;
  offset += sizeof(sparseIndex_t) * nnz;
  offset += sizeof(sparseIndex_t) * (params.Z.cols() + 1);
#else
  offset += sizeof(bool);
  offset += sizeof(FP_TYPE) * params.Z.rows() * params.Z.cols();
#endif
  offset += sizeof(FP_TYPE) * params.W.rows() * params.W.cols();
  offset += sizeof(FP_TYPE) * params.B.rows() * params.B.cols();
  return offset;
}

void ProtoNNModel::exportModel(
  const size_t modelSize,
  char *const toModel)
{
  assert(modelSize == modelStat());

  size_t offset(0);
#ifdef Z_SPARSE
  bool isZSparse(true);
#else
  bool isZSparse(false);
#endif

  memcpy(toModel + offset, (void *)&hyperParams, sizeof(hyperParams));
  offset += sizeof(hyperParams);

#ifdef SPARSE_Z_PROTONN
  memcpy(toModel + offset, (void *)&isZSparse, sizeof(bool));
  offset += sizeof(bool);
  offset += sizeof(Eigen::Index);
  Eigen::Index nnz = params.Z.outerIndexPtr()[params.Z.cols()]
    - params.Z.outerIndexPtr()[0];
  offset += sizeof(FP_TYPE) * nnz;
  offset += sizeof(sparseIndex_t) * nnz;
  offset += sizeof(sparseIndex_t) * (params.Z.cols() + 1);
  ///// Resume from here . 
#else
  memcpy(toModel + offset, (void *)&isZSparse, sizeof(bool));
  offset += sizeof(bool);
  memcpy(toModel + offset, params.Z.data(), sizeof(FP_TYPE) * params.Z.rows() * params.Z.cols());
  offset += sizeof(FP_TYPE) * params.Z.rows() * params.Z.cols();
#endif

  memcpy(toModel + offset, params.W.data(), sizeof(FP_TYPE) * params.W.rows() * params.W.cols());
  offset += sizeof(FP_TYPE) * params.W.rows() * params.W.cols();

  memcpy(toModel + offset, params.B.data(), sizeof(FP_TYPE) * params.B.rows() * params.B.cols());
  offset += sizeof(FP_TYPE) * params.B.rows() * params.B.cols();
}

void ProtoNNModel::importModel(const size_t numBytes, const char *const fromModel)
{
  size_t offset = 0;

  memcpy((void *)&hyperParams, fromModel + offset, sizeof(hyperParams));
  offset += sizeof(hyperParams);
  params.resizeParamsFromHyperParams(hyperParams, false); // No need to set to zero.

  bool isZSparse(true);
#ifdef SPARSE_Z_PROTONN
#else
  memcpy((void *)&isZSparse, fromModel + offset, sizeof(bool));
  offset += sizeof(bool);
  assert(isZSparse == false);
  memcpy(params.Z.data(), fromModel + offset, sizeof(FP_TYPE) * params.Z.rows() * params.Z.cols());
  offset += sizeof(FP_TYPE) * params.Z.rows() * params.Z.cols();
#endif

  memcpy(params.W.data(), fromModel + offset, sizeof(FP_TYPE) * params.W.rows() * params.W.cols());
  offset += sizeof(FP_TYPE) * params.W.rows() * params.W.cols();

  memcpy(params.B.data(), fromModel + offset, sizeof(FP_TYPE) * params.B.rows() * params.B.cols());
  offset += sizeof(FP_TYPE) * params.B.rows() * params.B.cols();
}

