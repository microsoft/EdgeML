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
    const LabelMatType& Y, const MatrixXuf& WX,
    MatrixXuf& B, MatrixXuf& Z,
    const int K_per_class);

  void kmeansOverall(
    const LabelMatType& Y,
    const MatrixXuf& WX, MatrixXuf& B, MatrixXuf& Z);


  namespace sparsekmeans
  {
    void computePointsL2Sq(
      const SparseMatrixuf& pointsMatrix,
      FP_TYPE *const points_l2sq);

    inline FP_TYPE distSqDocToPt(
      const SparseMatrixuf& pointsMatrix,
      const dataCount_t point,
      const FP_TYPE *const center);

    void distsqAllpointsToCenters(
      const SparseMatrixuf& pointsMatrix,
      const FP_TYPE *const points_l2sq,
      dataCount_t num_centers,
      FP_TYPE *const centers_coords,
      const FP_TYPE *const centers_l2sq,
      FP_TYPE *const dist_matrix);

    void computeClosestCenters(
      const SparseMatrixuf& pointsMatrix,
      const FP_TYPE *const points_l2sq,
      const dataCount_t num_centers,
      MatrixXuf& centersMatrix,
      dataCount_t *closest_center,
      FP_TYPE *const dist_matrix);

    FP_TYPE lloydsIter(
      const SparseMatrixuf& pointsMatrix,
      const dataCount_t num_centers,
      const FP_TYPE *const points_l2sq,
      MatrixXuf& centersMatrix,
      dataCount_t *const closest_center,
      const bool compute_residual);

    void updateMinDistSqToCenters(
      const SparseMatrixuf& pointsMatrix,
      const FP_TYPE *const points_l2sq,
      const dataCount_t num_new_centers,
      FP_TYPE *const centers,
      FP_TYPE *const min_dist,
      FP_TYPE *const dist_scratch);

    FP_TYPE kmeanspp(
      const SparseMatrixuf& pointsMatrix,
      const FP_TYPE *const points_l2sq,
      MatrixXuf& centersMatrix);

    // data is CSC with each column being a point
    // Points are clustered.
    FP_TYPE kmeans(
      const SparseMatrixuf& pointsMatrix,
      MatrixXuf& centersMatrix,
      const int num_iterations,
      dataCount_t *const closest_center);
  };


  namespace densekmeans
  {

    void distsqAllpointsToCenters(
      const MatrixXuf& pointsMatrix,
      const FP_TYPE *const points_l2sq,
      const dataCount_t num_centers,
      const FP_TYPE *const centers,
      const FP_TYPE *const centers_l2sq,
      FP_TYPE *const dist_matrix);

    void computeClosestCenters(
      const MatrixXuf& pointsMatrix,
      const FP_TYPE *const points_l2sq,
      const MKL_INT num_centers,
      FP_TYPE *const centers,
      dataCount_t *closest_center,
      FP_TYPE *const dist_matrix);

    void computePointsL2Sq(
      const MatrixXuf& pointsMatrix,
      FP_TYPE *const points_l2sq);

    FP_TYPE lloydsIter(
      const MatrixXuf& pointsMatrix,
      const FP_TYPE *const points_l2sq,
      MatrixXuf& centersMatrix,
      dataCount_t *const closest_center,
      const bool compute_residual);

    void updateMinDistSqToCenters(
      const MatrixXuf& pointsMatrix,
      const FP_TYPE *const points_l2sq,
      const dataCount_t num_new_centers,
      const FP_TYPE *const centers,
      FP_TYPE *const min_dist,
      FP_TYPE *const dist_scratch);

    FP_TYPE distsq(
      const FP_TYPE *const p1_coords,
      const FP_TYPE *const p2_coords,
      const MKL_INT dim);

    FP_TYPE kmeanspp(
      const MatrixXuf& pointsMatrix,
      const FP_TYPE *const points_l2sq,
      MatrixXuf& centersMatrix);

    FP_TYPE kmeans(
      const MatrixXuf& pointsMatrix,
      MatrixXuf& centersMatrix,
      const int num_iterations,
      dataCount_t *const closest_center);
  };
};
#endif
