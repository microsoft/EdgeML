// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "cluster.h"

using namespace EdgeML;

void sparsekmeans::computePointsL2Sq(
  const SparseMatrixuf& pointsMatrix,
  FP_TYPE *const pointsL2Sq)
{
  assert(!pointsMatrix.IsRowMajor);
  assert(pointsL2Sq != NULL);

  memset((void *)pointsL2Sq, 0, sizeof(FP_TYPE) * pointsMatrix.cols());

  const FP_TYPE* valsCSC = pointsMatrix.valuePtr();
  const sparseIndex_t* rowsCSC = pointsMatrix.innerIndexPtr();
  const sparseIndex_t* offsetsCSC = pointsMatrix.outerIndexPtr();

  pfor(int64_t d = 0; d < pointsMatrix.cols(); ++d)
    for (auto row = offsetsCSC[d]; row < offsetsCSC[d + 1]; ++row)
      pointsL2Sq[d] += valsCSC[row] * valsCSC[row];
}

inline FP_TYPE sparsekmeans::distSqDocToPt(
  const SparseMatrixuf& pointsMatrix,
  const dataCount_t point,
  const FP_TYPE *const center)
{
  const FP_TYPE* valsCSC = pointsMatrix.valuePtr();
  const sparseIndex_t* rowsCSC = pointsMatrix.innerIndexPtr();
  const sparseIndex_t* offsetsCSC = pointsMatrix.outerIndexPtr();

  const FP_TYPE centerL2Sq = dot(pointsMatrix.rows(), center, 1, center, 1);

  FP_TYPE ret = centerL2Sq
    + dot(offsetsCSC[point + 1] - offsetsCSC[point],
      valsCSC + offsetsCSC[point], 1,
      valsCSC + offsetsCSC[point], 1);
  for (auto row = offsetsCSC[point]; row < offsetsCSC[point + 1]; ++row)
    ret -= 2 * valsCSC[row] * center[rowsCSC[row]];
  return ret;
}

void sparsekmeans::distsqAllpointsToCenters(
  const SparseMatrixuf& pointsMatrix,
  const FP_TYPE *const pointsL2Sq,
  dataCount_t numCenters,
  FP_TYPE *const centersCoords,
  const FP_TYPE *const centersL2Sq,
  FP_TYPE *const distMatrix)
{
  const MKL_INT dim = pointsMatrix.rows();
  const MKL_INT numPoints = pointsMatrix.cols();
  const FP_TYPE* valsCSC = pointsMatrix.valuePtr();
  const sparseIndex_t* rowsCSC = pointsMatrix.innerIndexPtr();
  const sparseIndex_t* offsetsCSC = pointsMatrix.outerIndexPtr();

  FP_TYPE *onesVec = new FP_TYPE[numPoints > (MKL_INT)numCenters ? numPoints : (MKL_INT)numCenters];
  std::fill_n(onesVec, numPoints > (MKL_INT)numCenters ? numPoints : (MKL_INT)numCenters, (FP_TYPE)1.0);

  FP_TYPE *centersTranspose = new FP_TYPE[numCenters*dim];

  // TODO: mkl_?omatcopy doesn't seem to work. 
  /*omatcopy(CblasColMajor, 'T',
    dim, numCenters,
    1.0, centers, dim, centersTranspose, numCenters);*/
    // Improve this
  for (int64_t r = 0; r < dim; ++r)
    for (int64_t c = 0; c < numCenters; ++c)
      centersTranspose[c + r*numCenters] = centersCoords[r + c*dim];

  const char transa = 'N';
  const MKL_INT m = numPoints;
  const MKL_INT n = numCenters;
  const MKL_INT k = dim;
  const char matdescra[6] = { 'G',0,0,'C',0,0 };
  FP_TYPE alpha = -2.0; FP_TYPE beta = 0.0;

  csrmm(&transa, &m, &n, &k, &alpha, matdescra,
    valsCSC, (const sparseIndex_t*)rowsCSC,
    (const sparseIndex_t*)offsetsCSC, (const sparseIndex_t*)(offsetsCSC + 1),
    centersTranspose, &n,
    &beta, distMatrix, &n);
  gemm(CblasRowMajor, CblasTrans, CblasNoTrans,
    numPoints, numCenters, 1,
    (FP_TYPE)1.0, onesVec, numPoints, centersL2Sq, numCenters,
    (FP_TYPE)1.0, distMatrix, numCenters);
  gemm(CblasRowMajor, CblasTrans, CblasNoTrans,
    numPoints, numCenters, 1,
    (FP_TYPE)1.0, pointsL2Sq, numPoints, onesVec, numCenters,
    (FP_TYPE)1.0, distMatrix, numCenters);
  delete[] onesVec;
  delete[] centersTranspose;
}


void sparsekmeans::computeClosestCenters(
  const SparseMatrixuf& pointsMatrix,
  const FP_TYPE *const pointsL2Sq,
  const dataCount_t numCenters,
  MatrixXuf& centersMatrix,
  dataCount_t *closestCenter,
  FP_TYPE *const distMatrix) // Initialized to numCenters*numPoints size 
{
  assert(pointsMatrix.rows() == centersMatrix.rows());
  FP_TYPE *centers = centersMatrix.data();
  MKL_INT dim = centersMatrix.rows();

  FP_TYPE *const centersL2Sq = new FP_TYPE[numCenters];
  pfor(int64_t c = 0; c < numCenters; ++c)
    centersL2Sq[c] = dot(dim,
      centers + c*dim, 1,
      centers + c*dim, 1);
  distsqAllpointsToCenters(pointsMatrix, pointsL2Sq,
    numCenters, centersMatrix.data(), centersL2Sq,
    distMatrix);
  pfor(int64_t d = 0; d < pointsMatrix.cols(); ++d)
    closestCenter[d] = (dataCount_t)imin(numCenters, distMatrix + d*numCenters, 1);
  delete[] centersL2Sq;
}

FP_TYPE sparsekmeans::lloydsIter(
  const SparseMatrixuf& pointsMatrix,
  const dataCount_t numCenters,
  const FP_TYPE *const pointsL2Sq,
  MatrixXuf& centersMatrix,
  dataCount_t *const closestCenter,
  const bool computeResidual = false)
{
  const auto numPoints = pointsMatrix.cols();
  const auto dim = pointsMatrix.rows();
  const FP_TYPE* valsCSC = pointsMatrix.valuePtr();
  const sparseIndex_t* rowsCSC = pointsMatrix.innerIndexPtr();
  const sparseIndex_t* offsetsCSC = pointsMatrix.outerIndexPtr();
  FP_TYPE* centers = centersMatrix.data();

  FP_TYPE *distMatrix = new FP_TYPE[numCenters * numPoints];

  computeClosestCenters(pointsMatrix, pointsL2Sq, numCenters, centersMatrix, closestCenter, distMatrix);

  auto closestDocs = new std::vector<dataCount_t>[numCenters];
  for (Eigen::Index d = 0; d < numPoints; ++d)
    closestDocs[closestCenter[d]].push_back(d);

  memset((void *)centers, 0, sizeof(FP_TYPE) * numCenters*dim);

  pfor(dataCount_t c = 0; c < numCenters; ++c) {
    auto center = centers + c*dim;
    auto div = (FP_TYPE)closestDocs[c].size();
    for (auto diter = closestDocs[c].begin(); diter != closestDocs[c].end(); ++diter)
      for (auto row = offsetsCSC[*diter]; row < offsetsCSC[1 + (*diter)]; ++row)
        *(center + rowsCSC[row]) += valsCSC[row] / div;
  }

  FP_TYPE residual = 0.0;
  if (computeResidual) {
    int BUF_PAD = 32;
    int CHUNK_SIZE = 8192;
    int nchunks = numPoints / CHUNK_SIZE + (numPoints % CHUNK_SIZE == 0 ? 0 : 1);
    std::vector<FP_TYPE> residuals(nchunks*BUF_PAD, 0.0);

    pfor(int chunk = 0; chunk < nchunks; ++chunk)
      for (dataCount_t d = (dataCount_t)chunk*CHUNK_SIZE;
        d < (dataCount_t)numPoints && d < (dataCount_t)(chunk + 1)*CHUNK_SIZE; ++d)
        residuals[chunk*BUF_PAD] += distSqDocToPt(pointsMatrix, d,
          centers + closestCenter[d] * pointsMatrix.rows());

    for (int chunk = 0; chunk < nchunks; ++chunk)
      residual += residuals[chunk*BUF_PAD];
  }

  delete[] closestDocs;
  delete[] distMatrix;
  return residual;
}


void sparsekmeans::updateMinDistSqToCenters(
  const SparseMatrixuf& pointsMatrix,
  const FP_TYPE *const pointsL2Sq,
  const dataCount_t numNewCenters,
  FP_TYPE *const centers,
  FP_TYPE *const minDist,
  FP_TYPE *const distScratch) // preallocated scratch of space num_docs
{
  Eigen::Index dim = pointsMatrix.rows();
  const Eigen::Index numPoints = pointsMatrix.cols();

  FP_TYPE *const centersL2Sq = new FP_TYPE[numNewCenters];
  for (dataCount_t c = 0; c < numNewCenters; ++c)
    centersL2Sq[c] = dot(dim, centers + c*dim, 1, centers + c*dim, 1);
  distsqAllpointsToCenters(pointsMatrix, pointsL2Sq,
    numNewCenters, centers, centersL2Sq,
    distScratch);
  pfor(Eigen::Index d = 0; d < numPoints; ++d) {
    distScratch[d] = distScratch[d] > (FP_TYPE)0.0 ? distScratch[d] : (FP_TYPE)0.0;

    if (numNewCenters == 1)
      minDist[d] = minDist[d] > distScratch[d] ? distScratch[d] : minDist[d];
    else {
      FP_TYPE min = FP_TYPE_MAX;
      for (dataCount_t c = 0; c < numNewCenters; ++c)
        if (distScratch[c + d*numNewCenters] < min)
          min = distScratch[c + d*numNewCenters];
      minDist[d] = min > (FP_TYPE)0.0 ? min : (FP_TYPE)0.0;
    }
  }
  delete[] centersL2Sq;
}


FP_TYPE sparsekmeans::kmeanspp(
  const SparseMatrixuf& pointsMatrix,
  const FP_TYPE *const pointsL2Sq,
  MatrixXuf& centersMatrix)
{
  const MKL_INT numCenters = centersMatrix.cols();
  std::vector<dataCount_t> centers;
  MKL_INT numPoints = pointsMatrix.cols();
  MKL_INT dim = pointsMatrix.rows();
  const FP_TYPE *const valsCSC = pointsMatrix.valuePtr();
  const sparseIndex_t* offsetsCSC = pointsMatrix.outerIndexPtr();
  const sparseIndex_t* rowsCSC = pointsMatrix.innerIndexPtr();

  FP_TYPE *const centersL2Sq = new FP_TYPE[numCenters];
  FP_TYPE *const minDist = new FP_TYPE[numPoints];
  FP_TYPE *const centersCoords = centersMatrix.data();
  FP_TYPE *const distScratchSpace = new FP_TYPE[numPoints];

  std::vector<FP_TYPE> distCumul(numPoints + 1);

  memset(centersCoords, 0, sizeof(FP_TYPE)*numCenters*dim);
  std::fill_n(minDist, numPoints, FP_TYPE_MAX);

  centers.push_back((dataCount_t)(((long long)rand() * 84619573LL) % numPoints));
  centersL2Sq[0] = dot(offsetsCSC[centers[0] + 1] - offsetsCSC[centers[0]],
    valsCSC + offsetsCSC[centers[0]], 1,
    valsCSC + offsetsCSC[centers[0]], 1);
  for (auto idx = offsetsCSC[centers[0]]; idx < offsetsCSC[centers[0] + 1]; ++idx)
    * (centersCoords + rowsCSC[idx]) = valsCSC[idx];

  while (centers.size() < numCenters) {
    LOG_TRACE("centers size = " + std::to_string(centers.size()));
    updateMinDistSqToCenters(pointsMatrix, pointsL2Sq,
      1, centersCoords + (centers.size() - 1)*dim,
      minDist, distScratchSpace);
    distCumul[0] = 0;
    for (int point = 0; point < numPoints; ++point)
      distCumul[point + 1] = distCumul[point] + minDist[point];
    for (auto iter = centers.begin(); iter != centers.end(); ++iter) {
      // Disance from center to its closest center == 0
      assert(abs(distCumul[(*iter) + 1] - distCumul[*iter]) < 1e-4);
      // Center is not replicated
      assert(std::find(centers.begin(), centers.end(), *iter) == iter);
      assert(std::find(iter + 1, centers.end(), *iter) == centers.end());
    }

    auto diceThrow = distCumul[numPoints] * rand_fraction();
    assert(diceThrow < distCumul[numPoints]);
    dataCount_t newCenter = (dataCount_t)(std::upper_bound(distCumul.begin(), distCumul.end(), diceThrow)
      - 1 - distCumul.begin());
    assert(newCenter < (dataCount_t)numPoints);
    centersL2Sq[centers.size()] = dot(offsetsCSC[newCenter + 1] - offsetsCSC[newCenter],
      valsCSC + offsetsCSC[newCenter], 1,
      valsCSC + offsetsCSC[newCenter], 1);
    for (auto idx = offsetsCSC[newCenter]; idx < offsetsCSC[newCenter + 1]; ++idx)
      *(centersCoords + centers.size() * dim + rowsCSC[idx]) = valsCSC[idx];

    centers.push_back(newCenter);
  }
  delete[] distScratchSpace;
  delete[] centersL2Sq;
  delete[] minDist;
  return distCumul[numPoints - 1];
}



// data is CSC with each column being a point
// Points are clustered.
FP_TYPE sparsekmeans::kmeans(
  const SparseMatrixuf& pointsMatrix,
  MatrixXuf& centersMatrix,
  const int numIters,
  dataCount_t *const closestCenter)
{
  assert(!pointsMatrix.IsRowMajor);
  assert(!centersMatrix.IsRowMajor);
  assert(pointsMatrix.rows() == centersMatrix.rows());
  assert(pointsMatrix.cols() >= centersMatrix.cols());
  const MKL_INT numCenters = centersMatrix.cols();

  FP_TYPE residual = (FP_TYPE)0.0;

  FP_TYPE *pointsL2Sq = new FP_TYPE[pointsMatrix.cols()];
  computePointsL2Sq(pointsMatrix, pointsL2Sq);
  memset(centersMatrix.data(), 0, sizeof(FP_TYPE)*centersMatrix.rows()*centersMatrix.cols());
  kmeanspp(pointsMatrix, pointsL2Sq, centersMatrix);

  for (int i = 0; i < numIters; ++i) {
    residual = lloydsIter(pointsMatrix, numCenters, pointsL2Sq, centersMatrix, closestCenter, true);
    LOG_TRACE("Lloyd's iter " + std::to_string(i) + "  dist_sq residual: " + std::to_string(std::sqrt(residual)));
  }

  delete[] pointsL2Sq;
  return residual;
}

void densekmeans::distsqAllpointsToCenters(
  const MatrixXuf& pointsMatrix,
  const FP_TYPE *const pointsL2Sq,
  const dataCount_t numCenters,
  const FP_TYPE *const centers,
  const FP_TYPE *const centersL2Sq,
  FP_TYPE *const distMatrix)
{
  MKL_INT numPoints = pointsMatrix.cols();
  MKL_INT dim = pointsMatrix.rows();
  const FP_TYPE *const points = pointsMatrix.data();

  auto onesVec = new FP_TYPE[numPoints > (MKL_INT)numCenters ? numPoints : (MKL_INT)numCenters];
  std::fill_n(onesVec, numPoints > (MKL_INT)numCenters ? numPoints : (MKL_INT)numCenters, (FP_TYPE)1.0);

  gemm(CblasColMajor, CblasTrans, CblasNoTrans,
    numCenters, numPoints, dim,
    (FP_TYPE)-2.0, centers, dim, points, dim,
    (FP_TYPE)0.0, distMatrix, numCenters);
  gemm(CblasColMajor, CblasNoTrans, CblasTrans,
    numCenters, numPoints, 1,
    (FP_TYPE)1.0, centersL2Sq, numCenters, onesVec, numPoints,
    (FP_TYPE)1.0, distMatrix, numCenters);
  gemm(CblasColMajor, CblasNoTrans, CblasTrans,
    numCenters, numPoints, 1,
    (FP_TYPE)1.0, onesVec, numCenters, pointsL2Sq, numPoints,
    (FP_TYPE)1.0, distMatrix, numCenters);
  delete[] onesVec;
}

void densekmeans::computeClosestCenters(
  const MatrixXuf& pointsMatrix,
  const FP_TYPE *const pointsL2Sq,
  const MKL_INT numCenters,
  FP_TYPE *const centers,
  dataCount_t *closestCenter,
  FP_TYPE *const distMatrix)
{
  MKL_INT dim = pointsMatrix.rows();
  MKL_INT numPoints = pointsMatrix.cols();
  const FP_TYPE* const points = pointsMatrix.data();

  FP_TYPE *const centersL2Sq = new FP_TYPE[numCenters];
  pfor(int64_t c = 0; c < numCenters; ++c)
    centersL2Sq[c] = dot(dim,
      centers + c*dim, 1,
      centers + c*dim, 1);
  distsqAllpointsToCenters(pointsMatrix, pointsL2Sq,
    numCenters, centers, centersL2Sq,
    distMatrix);
  pfor(int64_t d = 0; d < numPoints; ++d)
    closestCenter[d] = (dataCount_t)imin(numCenters, distMatrix + d*numCenters, 1);
  delete[] centersL2Sq;
}

FP_TYPE densekmeans::distsq(
  const FP_TYPE *const p1Coords,
  const FP_TYPE *const p2Coords,
  const MKL_INT dim)
{
  return dot(dim, p1Coords, 1, p1Coords, 1)
    + dot(dim, p2Coords, 1, p2Coords, 1)
    - 2 * dot(dim, p1Coords, 1, p2Coords, 1);
}


void densekmeans::computePointsL2Sq(
  const MatrixXuf& pointsMatrix,
  FP_TYPE *const pointsL2Sq)
{
  assert(pointsL2Sq != NULL);
  MKL_INT dim = pointsMatrix.rows();
  MKL_INT numPoints = pointsMatrix.cols();
  const FP_TYPE *const data = pointsMatrix.data();
  pfor(int64_t d = 0; d < numPoints; ++d)
    pointsL2Sq[d] = dot(dim,
      data + d*dim, 1,
      data + d*dim, 1);
}


FP_TYPE densekmeans::lloydsIter(
  const MatrixXuf& pointsMatrix,
  const FP_TYPE *const pointsL2Sq,
  MatrixXuf& centersMatrix,
  dataCount_t *const closestCenter,
  const bool computeResidual = false)
{
  const MKL_INT numCenters = centersMatrix.cols();
  const MKL_INT numPoints = pointsMatrix.cols();
  const MKL_INT dim = pointsMatrix.rows();
  const FP_TYPE *const points = pointsMatrix.data();
  FP_TYPE *const centers = centersMatrix.data();

  FP_TYPE *const distMatrix = new FP_TYPE[numCenters * numPoints];

  computeClosestCenters(pointsMatrix, pointsL2Sq,
    numCenters, centers, closestCenter,
    distMatrix);

  auto closestPoints = new std::vector<dataCount_t>[numCenters];
  for (int d = 0; d < numPoints; ++d)
    closestPoints[closestCenter[d]].push_back(d);

  memset(centers, 0, sizeof(FP_TYPE)*numCenters*dim);

  pfor(int64_t c = 0; c < numCenters; ++c)
    for (auto iter = closestPoints[c].begin(); iter != closestPoints[c].end(); ++iter)
      axpy(dim, (FP_TYPE)(1.0) / closestPoints[c].size(),
        points + (*iter)*dim, 1, centers + c*dim, 1);


  int BUF_PAD = 32;
  int CHUNK_SIZE = 8196;
  int nchunks = numPoints / CHUNK_SIZE + (numPoints % CHUNK_SIZE == 0 ? 0 : 1);
  std::vector<FP_TYPE> residuals(nchunks*BUF_PAD, 0.0);

  pfor(int chunk = 0; chunk < nchunks; ++chunk)
    for (dataCount_t d = (dataCount_t)chunk*CHUNK_SIZE;
      d < (dataCount_t)numPoints && d < (dataCount_t)(chunk + 1)*CHUNK_SIZE; ++d)
      residuals[chunk*BUF_PAD] += distsq(points + d*dim,
        centers + closestCenter[d] * dim,
        dim);
  delete[] closestPoints;
  delete[] distMatrix;

  FP_TYPE residual = 0.0;
  for (int chunk = 0; chunk < nchunks; ++chunk)
    residual += residuals[chunk*BUF_PAD];
  return residual;
}

void densekmeans::updateMinDistSqToCenters(
  const MatrixXuf& pointsMatrix,
  const FP_TYPE *const pointsL2Sq,
  const dataCount_t numNewCenters,
  const FP_TYPE *const centers,
  FP_TYPE *const minDist,
  FP_TYPE *const distScratch) // preallocated scratch of space num_docs
{
  MKL_INT dim = pointsMatrix.rows();
  const MKL_INT numPoints = pointsMatrix.cols();

  FP_TYPE *const centersL2Sq = new FP_TYPE[numNewCenters];
  for (dataCount_t c = 0; c < numNewCenters; ++c)
    centersL2Sq[c] = dot(dim, centers + c*dim, 1, centers + c*dim, 1);
  distsqAllpointsToCenters(pointsMatrix, pointsL2Sq,
    numNewCenters, centers, centersL2Sq,
    distScratch);
  pfor(int d = 0; d < numPoints; ++d) {
    distScratch[d] = distScratch[d] > (FP_TYPE)0.0 ? distScratch[d] : (FP_TYPE)0.0;

    if (numNewCenters == 1)
      minDist[d] = minDist[d] > distScratch[d] ? distScratch[d] : minDist[d];
    else {
      FP_TYPE min = FP_TYPE_MAX;
      for (dataCount_t c = 0; c < numNewCenters; ++c)
        if (distScratch[c + d*numNewCenters] < min)
          min = distScratch[c + d*numNewCenters];
      minDist[d] = min > (FP_TYPE)0.0 ? min : (FP_TYPE)0.0;
    }
  }
  delete[] centersL2Sq;
}

FP_TYPE densekmeans::kmeanspp(
  const MatrixXuf& pointsMatrix,
  const FP_TYPE *const pointsL2Sq,
  MatrixXuf& centersMatrix)
{
  const MKL_INT numCenters = centersMatrix.cols();
  std::vector<dataCount_t> centers;
  MKL_INT numPoints = pointsMatrix.cols();
  MKL_INT dim = pointsMatrix.rows();
  const FP_TYPE *const points = pointsMatrix.data();

  FP_TYPE *const centersL2Sq = new FP_TYPE[numCenters];
  FP_TYPE *const minDist = new FP_TYPE[numPoints];
  FP_TYPE *const centersCoords = centersMatrix.data();
  FP_TYPE *const distScratchSpace = new FP_TYPE[numPoints];

  std::vector<FP_TYPE> distCumul(numPoints + 1);

  memset(centersCoords, 0, sizeof(FP_TYPE)*numCenters*dim);
  std::fill_n(minDist, numPoints, FP_TYPE_MAX);

  //centers.push_back((dataCount_t)(rand() * 84619573 % numPoints));
  centers.push_back((dataCount_t)(((long long)rand() * 84619573LL) % numPoints));
  centersL2Sq[0] = dot(dim,
    points + centers[0] * dim, 1,
    points + centers[0] * dim, 1);
  memcpy(centersCoords,
    points + centers[0] * dim,
    dim * sizeof(FP_TYPE));

  while (centers.size() < numCenters) {
    LOG_TRACE("k-means++, centers added: " + std::to_string(centers.size()));
    updateMinDistSqToCenters(pointsMatrix, pointsL2Sq,
      1, centersCoords + (centers.size() - 1)*dim,
      minDist, distScratchSpace);
    distCumul[0] = 0;
    for (int point = 0; point < numPoints; ++point)
      distCumul[point + 1] = distCumul[point] + minDist[point];
    for (auto iter = centers.begin(); iter != centers.end(); ++iter) {
      // Disance from center to its closest center == 0
        assert(abs(distCumul[(*iter) + 1] - distCumul[*iter]) < 1e-2 *distCumul[numPoints]/numPoints);
      // Center is not replicated
      assert(std::find(centers.begin(), centers.end(), *iter) == iter);
      assert(std::find(iter + 1, centers.end(), *iter) == centers.end());
    }

    auto diceThrow = distCumul[numPoints] * rand_fraction();
    assert(diceThrow < distCumul[numPoints]);
    dataCount_t newCenter = (dataCount_t)(std::upper_bound(distCumul.begin(), distCumul.end(), diceThrow)
      - 1 - distCumul.begin());
    assert(newCenter < (dataCount_t)numPoints);
    centersL2Sq[centers.size()] = dot(dim,
      points + newCenter * dim, 1,
      points + newCenter * dim, 1);
    memcpy(centersCoords + centers.size() * dim,
      points + newCenter * dim,
      dim * sizeof(FP_TYPE));

    centers.push_back(newCenter);
  }
  delete[] distScratchSpace;
  delete[] centersL2Sq;
  delete[] minDist;
  return distCumul[numPoints - 1];
}

FP_TYPE densekmeans::kmeans(
  const MatrixXuf& pointsMatrix,
  MatrixXuf& centersMatrix,
  const int numIters,
  dataCount_t *const closestCenter)
{
  assert(pointsMatrix.rows() == centersMatrix.rows());
  const MKL_INT numPoints = pointsMatrix.cols();
  const dataCount_t numCenters = centersMatrix.cols();
  FP_TYPE residual = (FP_TYPE)0.0;

  FP_TYPE *pointsL2Sq = new FP_TYPE[numPoints];
  computePointsL2Sq(pointsMatrix, pointsL2Sq);
  memset(centersMatrix.data(), 0, sizeof(FP_TYPE)*centersMatrix.rows()*centersMatrix.cols());
  kmeanspp(pointsMatrix, pointsL2Sq, centersMatrix);

  for (int i = 0; i < numIters; ++i) {
    residual = lloydsIter(pointsMatrix, pointsL2Sq,
      centersMatrix, closestCenter,
      true);
    LOG_TRACE("Lloyd's iter " + std::to_string(i) + "  dist_sq residual: " + std::to_string(std::sqrt(residual)));
  }

  delete[] pointsL2Sq;
  return residual;
}


void EdgeML::labelSpaceClustering(
  SparseMatrixuf& labels,
  int numClusters)
{
  Timer timer("labelSpaceClustering");
  labelCount_t L = labels.rows();
  dataCount_t N = labels.cols();
  assert(N > (dataCount_t)numClusters);

  SparseMatrixuf labels_transpose = labels.transpose().eval();
  timer.nextTime("transposing the label matrix (library expects N X L)");

  auto clusterIdentities = new dataCount_t[L];
  MatrixXuf clusterCenters(N, numClusters);
  assert(clusterIdentities != NULL);

  sparsekmeans::kmeans(labels_transpose, clusterCenters,
    20, clusterIdentities);
  /*
  RunKMeans(labels.data(),
        L, N, numClusters,
        100, rand()%1000, initialization_heuristic,
        clusterCenters, clusterIdentities);
  */

  std::ofstream f("label_identities");
  for (labelCount_t i = 0; i < L; ++i) f << clusterIdentities[i] << std::endl;

  delete[] clusterIdentities;
}

void EdgeML::kmeansLabelwise(
  const LabelMatType& Y,
  const MatrixXuf& WX,
  MatrixXuf& B,
  MatrixXuf& Z,
  const int KPerClass)
{
  assert(KPerClass*Y.rows() == B.cols());
  assert(Y.cols() == WX.cols());
  Timer timer("kmeans initialization");

  int nonZeroLabels = 0;
  const FP_TYPE* WXData = WX.data();
  FP_TYPE eps = 1.0e-2f;
  B = MatrixXuf::Zero(B.rows(), B.cols());
  Z = MatrixXuf::Zero(Z.rows(), Z.cols());

  for (Eigen::Index i = 0; i < Y.rows(); ++i) {
    std::vector <FP_TYPE> protData;
    dataCount_t numP = 0;
    for (Eigen::Index j = 0; j < Y.cols(); ++j) {
      if (Y.coeff(i, j) > 1 - eps) {
#ifdef ROWMAJOR
        assert(false); //Following line does not work if in rowmajor
#endif
        protData.insert(protData.end(), WXData + j*WX.rows(), WXData + (j + 1)*WX.rows());
        numP++;
      }
    }

    if (numP == 0) continue;
    else nonZeroLabels++;

    timer.nextTime("collecting points that belong to the current class");
    MatrixXuf clusterPoints = MatrixXuf::Zero(WX.rows(), numP);
    assert(protData.size() == WX.rows() * numP);
    memcpy(clusterPoints.data(), protData.data(), sizeof(FP_TYPE) * protData.size());
    timer.nextTime("converting data sub-matrix into appropriate format to pass");

    dataCount_t* clusterIdentities = new dataCount_t[numP];
    MatrixXuf BProt = MatrixXuf(B.rows(), KPerClass);

    densekmeans::kmeans(clusterPoints, BProt,
      20, clusterIdentities);

    for (int j = 0; j < KPerClass; ++j) {
      B.col((nonZeroLabels - 1)*KPerClass + j) = BProt.col(j);
      Z(i, (nonZeroLabels - 1)*KPerClass + j) = 1.0;
    }
    delete[] clusterIdentities;
    if (i % 100 == 99)
      LOG_TRACE("Completed 100 labels.");
  }

  if (Z.rows() != nonZeroLabels)
    LOG_INFO("Some labels have no data-points. #labels with at least one data-point = " + std::to_string(nonZeroLabels));
  B.conservativeResize(B.rows(), nonZeroLabels*KPerClass);
  Z.conservativeResize(Z.rows(), nonZeroLabels*KPerClass);
}

void EdgeML::kmeansOverall(
  const LabelMatType& Y,
  const MatrixXuf& WX,
  MatrixXuf& B,
  MatrixXuf& Z)
{
  assert(B.cols() == Z.cols());
  assert(Y.cols() == WX.cols());
  Timer timer("kmeans initialization");

  dataCount_t numP = WX.cols();
  dataCount_t* clusterIdentities = new dataCount_t[numP];
  dataCount_t* clusterDensity = new dataCount_t[numP];

  assert(clusterIdentities != NULL && clusterDensity != NULL);
  for (Eigen::Index i = 0; i < Z.cols(); ++i) clusterDensity[i] = 0;

  densekmeans::kmeans(WX, B, 20, clusterIdentities);

  //B = B_transpose.cast <FP_TYPE>().transpose().eval();
  Z = MatrixXuf::Zero(Z.rows(), Z.cols());
  for (Eigen::Index i = 0; i < WX.cols(); ++i) {
    clusterDensity[(int)std::round(clusterIdentities[i])] ++;
    assert((clusterIdentities[i] >= 0) && (clusterIdentities[i] < (dataCount_t)Z.cols()));
    Z.col(clusterIdentities[i]) += Y.col(i);
  }
  for (Eigen::Index i = 0; i < Z.cols(); ++i) {
    assert(clusterDensity[i] > 0);
    Z.col(i) = Z.col(i) / clusterDensity[i];
  }

  delete[] clusterIdentities;
  delete[] clusterDensity;
}
