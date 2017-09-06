// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __CLUSTER_H__
#define __CLUSTER_H__

#include "utils.h"
#include "ProtoNN.h"

namespace EdgeML
{
  void labelSpaceClustering(
    SparseMatrixuf& labels,
    int numClusters);

  void kmeansLabelwise(
    const LabelMatType& Y,
    const MatrixXuf& WX,
    MatrixXuf& B,
    MatrixXuf& Z,
    const int KPerClass);

  void kmeansOverall(
    const LabelMatType& Y,
    const MatrixXuf& WX,
    MatrixXuf& B,
    MatrixXuf& Z);


  namespace sparsekmeans
  {
    void computePointsL2Sq(
      const SparseMatrixuf& pointsMatrix,
      FP_TYPE *const pointsL2Sq);

    inline FP_TYPE distSqDocToPt(
      const SparseMatrixuf& pointsMatrix,
      const dataCount_t point,
      const FP_TYPE *const center);

    void distsqAllpointsToCenters(
      const SparseMatrixuf& pointsMatrix,
      const FP_TYPE *const pointsL2Sq,
      dataCount_t numCenters,
      FP_TYPE *const centersCoords,
      const FP_TYPE *const centersL2Sq,
      FP_TYPE *const distMatrix);

    void computeClosestCenters(
      const SparseMatrixuf& pointsMatrix,
      const FP_TYPE *const pointsL2Sq,
      const dataCount_t numCenters,
      MatrixXuf& centersMatrix,
      dataCount_t *closestCenter,
      FP_TYPE *const distMatrix);

    FP_TYPE lloydsIter(
      const SparseMatrixuf& pointsMatrix,
      const dataCount_t numCenters,
      const FP_TYPE *const pointsL2Sq,
      MatrixXuf& centersMatrix,
      dataCount_t *const closestCenter,
      const bool compute_residual);

    void updateMinDistSqToCenters(
      const SparseMatrixuf& pointsMatrix,
      const FP_TYPE *const pointsL2Sq,
      const dataCount_t numNewCenters,
      FP_TYPE *const centers,
      FP_TYPE *const minDist,
      FP_TYPE *const distScratch);

    FP_TYPE kmeanspp(
      const SparseMatrixuf& pointsMatrix,
      const FP_TYPE *const pointsL2Sq,
      MatrixXuf& centersMatrix);

    // data is CSC with each column being a point
    // Points are clustered.
    FP_TYPE kmeans(
      const SparseMatrixuf& pointsMatrix,
      MatrixXuf& centersMatrix,
      const int numIterations,
      dataCount_t *const closestCenter);
  };


  namespace densekmeans
  {

    void distsqAllpointsToCenters(
      const MatrixXuf& pointsMatrix,
      const FP_TYPE *const pointsL2Sq,
      const dataCount_t numCenters,
      const FP_TYPE *const centers,
      const FP_TYPE *const centersL2Sq,
      FP_TYPE *const distMatrix);

    void computeClosestCenters(
      const MatrixXuf& pointsMatrix,
      const FP_TYPE *const pointsL2Sq,
      const MKL_INT numCenters,
      FP_TYPE *const centers,
      dataCount_t *closestCenter,
      FP_TYPE *const distMatrix);

    void computePointsL2Sq(
      const MatrixXuf& pointsMatrix,
      FP_TYPE *const pointsL2Sq);

    FP_TYPE lloydsIter(
      const MatrixXuf& pointsMatrix,
      const FP_TYPE *const pointsL2Sq,
      MatrixXuf& centersMatrix,
      dataCount_t *const closestCenter,
      const bool compute_residual);

    void updateMinDistSqToCenters(
      const MatrixXuf& pointsMatrix,
      const FP_TYPE *const pointsL2Sq,
      const dataCount_t numNewCenters,
      const FP_TYPE *const centers,
      FP_TYPE *const minDist,
      FP_TYPE *const distScratch);

    FP_TYPE distsq(
      const FP_TYPE *const p1Coords,
      const FP_TYPE *const p2Coords,
      const MKL_INT dim);

    FP_TYPE kmeanspp(
      const MatrixXuf& pointsMatrix,
      const FP_TYPE *const pointsL2Sq,
      MatrixXuf& centersMatrix);

    FP_TYPE kmeans(
      const MatrixXuf& pointsMatrix,
      MatrixXuf& centersMatrix,
      const int numIterations,
      dataCount_t *const closestCenter);
  };
};
#endif
