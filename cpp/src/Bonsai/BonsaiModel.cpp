// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "Bonsai.h"
#include "blas_routines.h"
#include "utils.h"
#include "BonsaiFunctions.h"

using namespace EdgeML;
using namespace EdgeML::Bonsai;

BonsaiModel::BonsaiModel(
  std::string modelFile, 
  const bool isDense)
{
  std::ifstream infile(modelFile, std::ios::in|std::ios::binary);
  assert(infile.is_open());

  size_t modelSize;
  infile.read((char*)&modelSize, sizeof(modelSize));
  infile.close();
  infile.open(modelFile, std::ios::in|std::ios::binary);
  char* modelBuff = new char[modelSize];
  infile.read((char*)modelBuff, modelSize);
  infile.close();  

  (isDense) ? importModel(modelSize, modelBuff) : importSparseModel(modelSize, modelBuff);
}

BonsaiModel::BonsaiModel(
  const size_t numBytes,
  const char *const fromModel,
  const bool isDense)
{
  (isDense) ? importModel(numBytes, fromModel) : importSparseModel(numBytes, fromModel);
}

BonsaiModel::BonsaiModel() {}
BonsaiModel::BonsaiModel(const int& argc,
  const char** argv,
  std::string& dataDir)
{
  parseInput(argc, argv, hyperParams, dataDir);
  hyperParams.finalizeHyperParams();
  ++(hyperParams.dataDimension);
  params.resizeParamsFromHyperParams(hyperParams);
  initializeSigmaI();
}

// Add an extra bias dimension only went ingested from TLC interface
BonsaiModel::BonsaiModel(const BonsaiModel::BonsaiHyperParams& hyperParams_)
  : hyperParams(hyperParams_)
{
  ++(hyperParams.dataDimension);
  params.resizeParamsFromHyperParams(hyperParams);
  initializeSigmaI();
}

BonsaiModel::~BonsaiModel() {}

size_t BonsaiModel::modelStat()
{
  size_t offset = 0;

  offset += sizeof(offset);
  offset += sizeof(hyperParams);

  offset += sizeof(FP_TYPE) * params.Z.rows() * params.Z.cols();
  offset += sizeof(FP_TYPE) * params.W.rows() * params.W.cols();
  offset += sizeof(FP_TYPE) * params.V.rows() * params.V.cols();
  offset += sizeof(FP_TYPE) * params.Theta.rows() * params.Theta.cols();

  offset += 4 * denseMatrixMetaData::structStat();
  return offset;
}

void BonsaiModel::initializeSigmaI()
{
  hyperParams.sigma_i = 1.0;
}

void BonsaiModel::updateSigmaI(
  const MatrixXuf& ZX,
  const int exp_fac)
{
  FP_TYPE sum_tr = 0.0;
  int numTrials = std::min(100, (int)ZX.cols());

  srand((unsigned int)hyperParams.seed);

  for (int f = 0; f < numTrials; f++)
  {
    int theta_i = rand() % std::max(1, (int)params.Theta.rows());
    int x_i = rand() % ZX.cols();
    MatrixXuf ThetaZX(1, 1);
    if(params.Theta.rows() > 0)
      mm(ThetaZX, MatrixXuf(params.Theta).row(theta_i), CblasNoTrans, ZX.col(x_i), CblasNoTrans, 1.0, 0.0L);
    else
      ThetaZX(0, 0) = 0;
    sum_tr += fabs(ThetaZX(0, 0));
    if (hyperParams.sigma_i > (FP_TYPE)1000) hyperParams.sigma_i = (FP_TYPE)1000;
  }
  sum_tr /= 100.0;
  if(sum_tr != 0.0)
    hyperParams.sigma_i = (FP_TYPE)0.1 / sum_tr;
  else
    hyperParams.sigma_i = (FP_TYPE)0.1;
  hyperParams.sigma_i *= (FP_TYPE)pow(2.0, exp_fac);
  if (hyperParams.sigma_i > (FP_TYPE)1000) hyperParams.sigma_i = (FP_TYPE)1000;
}

void BonsaiModel::exportModel(
  const size_t modelSize,
  char *const toModel)
{
  // TODO: Fix this code and importModel to work with sparse V,W,Z,Theta flags
  assert(modelSize == modelStat());

  size_t offset = 0;
  memcpy(toModel + offset, (void *)&modelSize, sizeof(modelSize));
  offset += sizeof(modelSize);

  memcpy(toModel + offset, (void *)&hyperParams, sizeof(hyperParams));
  offset += sizeof(hyperParams);

#ifdef SPARSE_Z_BONSAI
  MatrixXuf denseZ = MatrixXuf(params.Z);
  offset += exportDenseMatrix(denseZ, denseExportStat(denseZ), toModel + offset);
#else
  offset += exportDenseMatrix(params.Z, denseExportStat(params.Z), toModel + offset);
#endif

#ifdef SPARSE_W_BONSAI
  MatrixXuf denseW = MatrixXuf(params.W);
  offset += exportDenseMatrix(denseW, denseExportStat(denseW), toModel + offset);
#else
  offset += exportDenseMatrix(params.W, denseExportStat(params.W), toModel + offset);
#endif

#ifdef SPARSE_V_BONSAI
  MatrixXuf denseV = MatrixXuf(params.V);
  offset += exportDenseMatrix(denseV, denseExportStat(denseV), toModel + offset);
#else
  offset += exportDenseMatrix(params.V, denseExportStat(params.V), toModel + offset);
#endif

#ifdef SPARSE_THETA_BONSAI
  MatrixXuf denseTheta = MatrixXuf(params.Theta);
  offset += exportDenseMatrix(denseTheta, denseExportStat(denseTheta), toModel + offset);
#else
  offset += exportDenseMatrix(params.Theta, denseExportStat(params.Theta), toModel + offset);
#endif

  assert(offset < (size_t)(1 << 31)); // Because we make this promise to TLC.
 /*
  memcpy(toModel + offset, params.Z.data(), sizeof(FP_TYPE) * params.Z.rows() * params.Z.cols());
  offset += sizeof(FP_TYPE) * params.Z.rows() * params.Z.cols();

  memcpy(toModel + offset, params.W.data(), sizeof(FP_TYPE) * params.W.rows() * params.W.cols());
  offset += sizeof(FP_TYPE) * params.W.rows() * params.W.cols();

  memcpy(toModel + offset, params.V.data(), sizeof(FP_TYPE) * params.V.rows() * params.V.cols());
  offset += sizeof(FP_TYPE) * params.V.rows() * params.V.cols();

  memcpy(toModel + offset, params.Theta.data(), sizeof(FP_TYPE) * params.Theta.rows() * params.Theta.cols());
  offset += sizeof(FP_TYPE) * params.Theta.rows() * params.Theta.cols();
*/

}


void BonsaiModel::importModel(
  const size_t numBytes,
  const char *const fromModel) // TODO: Fill them with elegance
{
  // TODO: Fix this code and exportModel to work with sparse V,W,Z,Theta flags

  size_t offset = 0;

  size_t modelSize;
  memcpy((void *)&modelSize, fromModel + offset, sizeof(modelSize));
  offset += sizeof(modelSize);

  memcpy((void *)&hyperParams, fromModel + offset, sizeof(hyperParams));
  offset += sizeof(hyperParams);

  params.resizeParamsFromHyperParams(hyperParams, false); // No need to set to zero.

#ifdef SPARSE_Z_BONSAI
  MatrixXuf denseZ(params.Z.rows(), params.Z.cols());
  offset += importDenseMatrix(denseZ, denseExportStat(denseZ), fromModel + offset);
  params.Z = denseZ.sparseView();
#else
  offset += importDenseMatrix(params.Z, denseExportStat(params.Z), fromModel + offset);
#endif

#ifdef SPARSE_W_BONSAI
  MatrixXuf denseW(params.W.rows(), params.W.cols());
  offset += importDenseMatrix(denseW, denseExportStat(denseW), fromModel + offset);
  params.W = denseW.sparseView();
#else
  offset += importDenseMatrix(params.W, denseExportStat(params.W), fromModel + offset);
#endif

#ifdef SPARSE_V_BONSAI
  MatrixXuf denseV(params.V.rows(), params.V.cols());
  offset += importDenseMatrix(denseV, denseExportStat(denseV), fromModel + offset);
  params.V = denseV.sparseView();
#else
  offset += importDenseMatrix(params.V, denseExportStat(params.V), fromModel + offset);
#endif

#ifdef SPARSE_THETA_BONSAI
  MatrixXuf denseTheta(params.Theta.rows(), params.Theta.cols());
  offset += importDenseMatrix(denseTheta, denseExportStat(denseTheta), fromModel + offset);
  params.Theta = denseTheta.sparseView();
#else
  offset += importDenseMatrix(params.Theta, denseExportStat(params.Theta), fromModel + offset);
#endif

  /*
  memcpy(params.Z.data(), fromModel + offset, sizeof(FP_TYPE) * params.Z.rows() * params.Z.cols());
  offset += sizeof(FP_TYPE) * params.Z.rows() * params.Z.cols();

  memcpy(params.W.data(), fromModel + offset, sizeof(FP_TYPE) * params.W.rows() * params.W.cols());
  offset += sizeof(FP_TYPE) * params.W.rows() * params.W.cols();

  memcpy(params.V.data(), fromModel + offset, sizeof(FP_TYPE) * params.V.rows() * params.V.cols());
  offset += sizeof(FP_TYPE) * params.V.rows() * params.V.cols();

  memcpy(params.Theta.data(), fromModel + offset, sizeof(FP_TYPE) * params.Theta.rows() * params.Theta.cols());
  offset += sizeof(FP_TYPE) * params.Theta.rows() * params.Theta.cols();
  */
  assert(numBytes == offset);
}

size_t BonsaiModel::sparseModelStat()
{
  return sizeof(size_t) + sizeof(hyperParams) + sparseExportStat(params.Z) + sparseExportStat(params.W) +
    sparseExportStat(params.V) + sparseExportStat(params.Theta);
}

void BonsaiModel::exportSparseModel(
  const size_t modelSize,
  char *const toModel)
{
  assert(modelSize == sparseModelStat());

  size_t offset = 0;

  memcpy(toModel + offset, (void *)&modelSize, sizeof(modelSize));
  offset += sizeof(modelSize);

  memcpy(toModel + offset, (void *)&hyperParams, sizeof(hyperParams));
  offset += sizeof(hyperParams);

  {
#ifdef SPARSE_Z_BONSAI
    offset += exportSparseMatrix(params.Z, sparseExportStat(params.Z), toModel + offset);
#else
    SparseMatrixuf sparseZ = params.Z.sparseView();
    offset += exportSparseMatrix(sparseZ, sparseExportStat(sparseZ), toModel + offset);
#endif
  }

  {
#ifdef SPARSE_W_BONSAI
    offset += exportSparseMatrix(params.W, sparseExportStat(params.W), toModel + offset);
#else
    SparseMatrixuf sparseW = params.W.sparseView();
    offset += exportSparseMatrix(sparseW, sparseExportStat(sparseW), toModel + offset);
#endif
  }

  {
#ifdef SPARSE_V_BONSAI
    offset += exportSparseMatrix(params.V, sparseExportStat(params.V), toModel + offset);
#else
    SparseMatrixuf sparseV = params.V.sparseView();
    offset += exportSparseMatrix(sparseV, sparseExportStat(sparseV), toModel + offset);
#endif
  }

  {
#ifdef SPARSE_THETA_BONSAI
    offset += exportSparseMatrix(params.Theta, sparseExportStat(params.Theta), toModel + offset);
#else
    SparseMatrixuf sparseTheta = params.Theta.sparseView();
    offset += exportSparseMatrix(sparseTheta, sparseExportStat(sparseTheta), toModel + offset);
#endif
  }
  assert(offset < (size_t)(1 << 31));
}


void BonsaiModel::importSparseModel(
  const size_t numBytes,
  const char *const fromModel)
{
  size_t offset = 0;
  size_t modelSize;

  memcpy((void *)&modelSize, fromModel + offset, sizeof(modelSize));
  offset += sizeof(modelSize);

  memcpy((void *)&hyperParams, fromModel + offset, sizeof(hyperParams));
  offset += sizeof(hyperParams);

  params.resizeParamsFromHyperParams(hyperParams, false); // No need to set to zero.

  {
#ifdef SPARSE_Z_BONSAI
    offset += importSparseMatrix(params.Z, fromModel + offset);
#else
    SparseMatrixuf sparseZ(params.Z.rows(), params.Z.cols());
    offset += importSparseMatrix(sparseZ, fromModel + offset);
    params.Z = MatrixXuf(sparseZ);
#endif
  }

  {
#ifdef SPARSE_W_BONSAI
    offset += importSparseMatrix(params.W, fromModel + offset);
#else
    SparseMatrixuf sparseW(params.W.rows(), params.W.cols());
    offset += importSparseMatrix(sparseW, fromModel + offset);
    params.W = MatrixXuf(sparseW);
#endif
  }

  {
#ifdef SPARSE_V_BONSAI
    offset += importSparseMatrix(params.V, fromModel + offset);
#else
    SparseMatrixuf sparseV(params.V.rows(), params.V.cols());
    offset += importSparseMatrix(sparseV, fromModel + offset);
    params.V = MatrixXuf(sparseV);
#endif
  }

  {
#ifdef SPARSE_THETA_BONSAI
    offset += importSparseMatrix(params.Theta, fromModel + offset);
#else
    SparseMatrixuf sparseTheta(params.Theta.rows(), params.Theta.cols());
    offset += importSparseMatrix(sparseTheta, fromModel + offset);
    params.Theta = MatrixXuf(sparseTheta);
#endif
  }

  assert(numBytes == offset);
}

WMatType BonsaiModel::getW(const labelCount_t& classID)
{
  return(params.W.middleRows(hyperParams.totalNodes*classID, hyperParams.totalNodes));
}

VMatType BonsaiModel::getV(const labelCount_t& classID)
{
  return(params.V.middleRows(hyperParams.totalNodes*classID, hyperParams.totalNodes));
}

WMatType BonsaiModel::getW(const labelCount_t& classID, const labelCount_t& globalNodeID)
{
  return(params.W.middleRows(hyperParams.totalNodes*classID + globalNodeID, 1));
}

VMatType BonsaiModel::getV(const labelCount_t& classID, const labelCount_t& globalNodeID)
{
  return(params.V.middleRows(hyperParams.totalNodes*classID + globalNodeID, 1));
}

ThetaMatType BonsaiModel::getTheta(const labelCount_t& globalNodeID)
{
  return(params.Theta.middleRows(globalNodeID, 1));
}

void BonsaiModel::dumpModel(std::string modelPath)
{
  std::ofstream modelDumper(modelPath);

  modelDumper << "HyperParams: \n";
  modelDumper << "\tTree Depth: " << hyperParams.treeDepth << "\n";
  modelDumper << "\tProjected Dimension: " << hyperParams.projectionDimension << "\n";
  modelDumper << "\tSparsity Fractions: \n" << "\t\tZ: " << hyperParams.lambdaZ << "\n";
  modelDumper << "\t\tW: " << hyperParams.lambdaW << "\n\t\tV: " << hyperParams.lambdaV << "\n";
  modelDumper << "\t\tTheta: " << hyperParams.lambdaTheta << "\n";
  modelDumper << "\tRegularizers: \n" << "\t\tZ: " << hyperParams.regList.lZ << "\n";
  modelDumper << "\t\tW: " << hyperParams.regList.lW << "\n\t\tV: " << hyperParams.regList.lV << "\n";
  modelDumper << "\t\tTheta: " << hyperParams.regList.lTheta << "\n";
  modelDumper << "\tSigma: " << hyperParams.Sigma << "\n \n";
  modelDumper << "\tBatch Factor: " << hyperParams.batchFactor << "\n \n";
  modelDumper << "\tIters: " << hyperParams.iters << "\n \n";

  modelDumper << "Z: \n";
  modelDumper << MatrixXuf(params.Z) << "\n \n";

  modelDumper << "W: \n";
  modelDumper << MatrixXuf(params.W) << "\n \n";

  modelDumper << "V: \n";
  modelDumper << MatrixXuf(params.V) << "\n \n";

  modelDumper << "Theta: \n";
  modelDumper << MatrixXuf(params.Theta) << "\n \n";

  modelDumper.close();
}

size_t BonsaiModel::totalNonZeros()
{
  return Bonsai::countnnz(params.Z) + Bonsai::countnnz(params.W) +
    Bonsai::countnnz(params.V) + Bonsai::countnnz(params.Theta);
}