// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"

using namespace EdgeML;

EdgeML::sparseMatrixMetaData::sparseMatrixMetaData()
  :
  nRows(0), nCols(0), nnzs(0)
{}

EdgeML::sparseMatrixMetaData::sparseMatrixMetaData(
  const featureCount_t& nRows_,
  const dataCount_t& nCols_,
  const Eigen::Index& nnzs_)
  :
  nRows(nRows_), nCols(nCols_), nnzs(nnzs_)
{}

size_t EdgeML::sparseMatrixMetaData::structStat()
{
  return sizeof(featureCount_t) + sizeof(dataCount_t) + sizeof(Eigen::Index);
}

size_t EdgeML::sparseMatrixMetaData::exportToBuffer(char *const buffer)
{
  size_t offset = 0;

  *((featureCount_t*)buffer) = nRows; offset += sizeof(featureCount_t);
  *((dataCount_t*)(buffer + offset)) = nCols; offset += sizeof(dataCount_t);
  *((Eigen::Index*)(buffer + offset)) = nnzs; offset += sizeof(Eigen::Index);
  return offset;
}

size_t EdgeML::sparseMatrixMetaData::importFromBuffer(const char *const buffer)
{
  size_t offset = 0;
  nRows = *((featureCount_t*)buffer); offset += sizeof(featureCount_t);
  nCols = *((dataCount_t*)(buffer + offset)); offset += sizeof(dataCount_t);
  nnzs = *((Eigen::Index*)(buffer + offset)); offset += sizeof(Eigen::Index);
  return offset;
}

size_t EdgeML::sparseMatrixMetaData::importSparseMatrixStat()
{
  return
    sizeof(FP_TYPE)*nnzs + sizeof(sparseIndex_t)*nnzs
    + sizeof(sparseIndex_t)*(nCols + 1);
}


EdgeML::denseMatrixMetaData::denseMatrixMetaData()
  :
  nRows(0), nCols(0)
{}

EdgeML::denseMatrixMetaData::denseMatrixMetaData(
  const featureCount_t& nRows_,
  const dataCount_t& nCols_)
  :
  nRows(nRows_), nCols(nCols_)
{}

size_t EdgeML::denseMatrixMetaData::structStat()
{
  return sizeof(featureCount_t) + sizeof(dataCount_t);
}

size_t EdgeML::denseMatrixMetaData::exportToBuffer(char *const buffer)
{
  size_t offset = 0;
  *((featureCount_t*)buffer) = nRows; offset += sizeof(featureCount_t);
  *((dataCount_t*)(buffer + offset)) = nCols; offset += sizeof(dataCount_t);
  return offset;
}

size_t EdgeML::denseMatrixMetaData::importFromBuffer(const char *const buffer)
{
  size_t offset = 0;
  nRows = *((featureCount_t*)buffer); offset += sizeof(featureCount_t);
  nCols = *((dataCount_t*)(buffer + offset)); offset += sizeof(dataCount_t);
  return offset;
}


void EdgeML::checkDenominator(const FP_TYPE &denominator)
{
  assert(!std::isnan(denominator));
  assert(denominator > FP_TYPE_MIN || denominator < -FP_TYPE_MIN);
  if (!(denominator > MIN_DEN || denominator < -MIN_DEN))
    LOG_WARNING("Denominator has become very small");
}

FP_TYPE EdgeML::safeDiv(
  const FP_TYPE &num,
  const FP_TYPE &den)
{
  assert(!std::isnan(num));
  checkDenominator(den);
  FP_TYPE ret = num / den;
  assert(!std::isnan(ret));
  return ret;
}

FP_TYPE EdgeML::computeModelSizeInkB(
  const FP_TYPE& lambdaW,
  const FP_TYPE& lambdaZ,
  const FP_TYPE& lambda_B,
  const MatrixXuf& W,
  const MatrixXuf& Z,
  const MatrixXuf& B)
{
  FP_TYPE ret = 0.0;
  // counting double for sparsely stored arrays
  ret += W.rows()* W.cols()* fmin((FP_TYPE)1.0, 2 * lambdaW);
  ret += Z.rows()* Z.cols()* fmin((FP_TYPE)1.0, 2 * lambdaZ);
  ret += B.rows()* B.cols()* fmin((FP_TYPE)1.0, 2 * lambda_B);

  return (ret / (FP_TYPE)256.0); //4 bytes per entry, report in kb
}

// sequentialQuickSelect is in place; Contents of @data will be corrupted when the function returns. 
// Equivalent to:
// {
//   std::sort(data, data + count);
//   return data[order-1];
// }
FP_TYPE EdgeML::sequentialQuickSelect(
  FP_TYPE* data,
  size_t count,
  size_t order)
{
  assert(count > 0);
  assert(order > 0);
  assert(order <= count);

  if (order == 1)
    return *std::min_element(data, data + count);
  else if (order >= count)
    return *std::max_element(data, data + count);

  FP_TYPE pivot = data[(rand() * rand()) % count];

  size_t left = 0;
  size_t right = count - 1;

  while (left <= right) {
    if (data[left] <= pivot) left++;
    else if (data[right] > pivot) right--;
    else {
      std::swap(data[left], data[right]);
      left++; right--;
    }
  }

  if (left > 0)
    assert(data[left - 1] <= pivot);
  else
    assert(data[0] > pivot);

  if (left < count)
    assert(data[left] > pivot);
  else
    assert(data[left - 1] <= pivot);

  if (right + 1 < count)
    assert(data[right + 1] > pivot);

  if (left == count)
    if (*std::min_element(data, data + count) == *std::max_element(data, data + count))
      return pivot;

  if (left == order)
    return pivot;
  else if (order < left)
    return sequentialQuickSelect(data, left, order);
  else
    return sequentialQuickSelect(data + left, count - left, order - left);
}


void EdgeML::randPick(
  const MatrixXuf& source,
  MatrixXuf& target,
  dataCount_t seed)
{
  assert(target.cols() <= source.cols());
  if (target.cols() == source.cols()) {
    target = source;
    return;
  }

  assert(seed < 1e6);
  unsigned long long prime = 1000000000ULL + 7;
  assert(source.cols() < prime);
  Eigen::Index pick;

  for (dataCount_t i = 0; i < (dataCount_t)target.cols(); ++i) {
    pick = ((prime*(i + seed)) % source.cols());
    target.col(i) = source.col(pick);
  }
}

void EdgeML::randPick(
  const SparseMatrixuf& source,
  SparseMatrixuf& target,
  dataCount_t seed)
{
  assert(target.cols() <= source.cols());
  if (target.cols() == source.cols()) {
    target = source;
    return;
  }

  assert(seed < 1e6);
  unsigned long long prime = 1000000000ULL + 7;
  assert(source.cols() < prime);
  Eigen::Index pick;

  for (dataCount_t i = 0; i < (dataCount_t)target.cols(); ++i) {
    pick = ((prime*(i + seed)) % source.cols());
    MatrixXuf temp = source.col(pick);
    target.col(i) = temp.sparseView(); //source.col(pick);
  }
}

size_t EdgeML::sparseExportStat(const SparseMatrixuf& mat)
{
  return sparseMatrixMetaData::structStat()
    + sizeof(FP_TYPE)*mat.nonZeros()
    + sizeof(sparseIndex_t)*mat.nonZeros()
    + sizeof(sparseIndex_t)*(mat.cols() + 1);
}

size_t EdgeML::sparseExportStat(const MatrixXuf& mat)
{
  SparseMatrixuf sparseMat = mat.sparseView();
  return sparseExportStat(sparseMat);
}

size_t EdgeML::denseExportStat(const MatrixXuf& mat)
{
  return denseMatrixMetaData::structStat()
    + sizeof(FP_TYPE)*mat.rows()*mat.cols();
}

size_t EdgeML::denseExportStat(const SparseMatrixuf& mat)
{
  return denseMatrixMetaData::structStat()
    + sizeof(FP_TYPE)*mat.rows()*mat.cols();
}

size_t EdgeML::exportSparseMatrix(
  const SparseMatrixuf& mat,
  const size_t& bufferSize,
  char *const buffer)
{
  assert(bufferSize == sparseExportStat(mat));

  size_t offset = 0;
  sparseMatrixMetaData metaData((featureCount_t)mat.rows(), (dataCount_t)mat.cols(), mat.nonZeros());
  offset += metaData.exportToBuffer(buffer + offset);

  memcpy(buffer + offset, mat.valuePtr(), sizeof(FP_TYPE) * mat.nonZeros());
  offset += sizeof(FP_TYPE) * mat.nonZeros();

  memcpy(buffer + offset, mat.innerIndexPtr(), sizeof(sparseIndex_t) * mat.nonZeros());
  offset += sizeof(sparseIndex_t) * mat.nonZeros();

  memcpy(buffer + offset, mat.outerIndexPtr(), sizeof(sparseIndex_t) * (mat.cols() + 1));
  offset += sizeof(sparseIndex_t) * (mat.cols() + 1);

  assert(offset < (size_t)(1 << 31));
  assert(offset == bufferSize);
  return offset;
}

size_t EdgeML::exportSparseMatrix(
  const MatrixXuf& mat,
  const size_t& bufferSize,
  char *const buffer)
{
  SparseMatrixuf sparseMat = mat.sparseView();
  return exportSparseMatrix(sparseMat, bufferSize, buffer);
}

size_t EdgeML::exportDenseMatrix(
  const MatrixXuf& mat,
  const size_t& bufferSize,
  char *const buffer)
{
  assert(bufferSize == denseExportStat(mat));

  size_t offset = 0;
  denseMatrixMetaData metaData((featureCount_t)mat.rows(), (dataCount_t)mat.cols());
  offset += metaData.exportToBuffer(buffer + offset);

  memcpy(buffer + offset, mat.data(), sizeof(FP_TYPE) * mat.rows() * mat.cols());
  offset += sizeof(FP_TYPE) * mat.rows() * mat.cols();

  assert(offset < (size_t)(1 << 31));
  assert(offset == bufferSize);
  return offset;
}

size_t EdgeML::exportDenseMatrix(
  const SparseMatrixuf& mat,
  const size_t& bufferSize,
  char *const buffer)
{
  MatrixXuf denseMat(mat);
  return exportDenseMatrix(mat, bufferSize, buffer);
}


size_t EdgeML::importSparseMatrix(
  SparseMatrixuf& mat,
  const char *const buffer)
{
  size_t offset = 0;
  sparseMatrixMetaData metaData;
  offset += metaData.importFromBuffer(buffer);

  mat.resize(metaData.nRows, metaData.nCols);
  mat.reserve(metaData.nnzs);

  memcpy(mat.valuePtr(), buffer + offset, sizeof(FP_TYPE) * metaData.nnzs);
  offset += sizeof(FP_TYPE) * metaData.nnzs;

  memcpy(mat.innerIndexPtr(), buffer + offset, sizeof(sparseIndex_t) * metaData.nnzs);
  offset += sizeof(sparseIndex_t) * metaData.nnzs;

  memcpy(mat.outerIndexPtr(), buffer + offset, sizeof(sparseIndex_t) * (mat.cols() + 1));
  offset += sizeof(sparseIndex_t) * (mat.cols() + 1);


  assert(offset == sparseExportStat(mat));
  return offset;
}

size_t EdgeML::importDenseMatrix(
  MatrixXuf& mat,
  const size_t& bufferSize,
  const char *const buffer)
{
  size_t offset = 0;
  denseMatrixMetaData metaData;
  offset += metaData.importFromBuffer(buffer);

  mat.resize(metaData.nRows, metaData.nCols);

  memcpy(mat.data(), buffer + offset, sizeof(FP_TYPE) * mat.rows() * mat.cols());
  offset += sizeof(FP_TYPE) * mat.rows() * mat.cols();

  assert(bufferSize == offset);
  assert(offset == bufferSize);
  return offset;
}

void EdgeML::writeMatrixInASCII(
  const MatrixXuf& mat,
  const std::string& outDir,
  const std::string& fileName)
{
  std::string filePath = outDir + "/" + fileName;
  std::ofstream f(filePath, std::ofstream::out);
  f << mat.format(eigen_tsv);
  f.close();
}

void EdgeML::writeSparseMatrixInASCII(
  const SparseMatrixuf& mat,
  const std::string& outDir,
  const std::string& fileName)
{
  writeMatrixInASCII(MatrixXuf(mat), outDir, fileName);
}
