// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __FUNCTIONS_H_
#define __FUNCTIONS_H_

#include "utils.h"
#include "blas_routines.h"
#include "par_utils.h"
#include "cluster.h"
#include "ProtoNN.h"

namespace EdgeML
{
  FP_TYPE medianHeuristic(
    const BMatType& B,
    MatrixXuf WX,
    FP_TYPE multiplier);

  FP_TYPE batchEvaluate(
    const ZMatType& Z,
    const LabelMatType& Y,
    const LabelMatType& Yval,
    const BMatType& B,
    const MatrixXuf& WX,
    const MatrixXuf& WXval,
    const FP_TYPE& gamma,
    const EdgeML::ProblemFormat& problemType,
    FP_TYPE * const stats);

  FP_TYPE batchEvaluate(
    const ZMatType& Z,
    const LabelMatType& Y,
    const BMatType& B,
    const MatrixXuf& WX,
    const FP_TYPE& gamma,
    const EdgeML::ProblemFormat& problemType,
    EdgeML::ResultStruct& res,
    FP_TYPE* const stats);

  //
  // Returns accuracy with respect to current parameters
  // Input: @Z: sparse, @Y: sparse, @D: dense
  //
  FP_TYPE accuracy(
    const ZMatType& Z,
    const LabelMatType& Y,
    const MatrixXuf& D,
    const EdgeML::ProblemFormat& problem);

  void accuracy(
    const ZMatType& Z, const LabelMatType& Y, const MatrixXuf& D,
    const EdgeML::ProblemFormat& problemType, 
    EdgeML::ResultStruct& res);


  //
  // Returns loss with respect to current parameters
  // Input: @Z: sparse, @Y: sparse, @D: dense
  //
  FP_TYPE L(
    const ZMatType& Z,
    const LabelMatType& Y,
    const MatrixXuf& D);

  FP_TYPE L(
    const ZMatType& Z,
    const LabelMatType& Y,
    const MatrixXuf& D,
    const Eigen::Index begin,
    const Eigen::Index end);

  MatrixXuf distanceProjectedPointsToCenters(
    const BMatType& B,
    const MatrixXuf& WX);

  //
  // Returns matrix with Ret[i,j] = exp(-gamma^2 * || WX[i,:] - B[j,;] ||^2_2)
  // Input: @B: CSR format,@WX: Dense (can be row or column major based on inernal flag)
  // @B: prototype matrix of size d X m
  // @WX: W*X where W is the weight vector
  // @X: the data matrix. Size is d X n
  //
  MatrixXuf gaussianKernel(
    const BMatType& B,
    const MatrixXuf& WX,
    const FP_TYPE gamma,
    const Eigen::Index begin,
    const Eigen::Index end);

  MatrixXuf gaussianKernel(
    const BMatType& B,
    const MatrixXuf& WX,
    const FP_TYPE gamma);

  //
  // Returns the gradient of @B
  // Input: @B, @Y, @Z can be CSR or CSC
  // Input: @WX is either row or column major
  //
  MatrixXuf gradL_B(
    const BMatType& B,
    const LabelMatType& Y,
    const ZMatType& Z,
    const MatrixXuf& WX,
    const MatrixXuf& D,
    const FP_TYPE gamma,
    const Eigen::Index begin,
    const Eigen::Index end);

  MatrixXuf gradL_B(
    const BMatType& B,
    const LabelMatType& Y,
    const ZMatType& Z,
    const MatrixXuf& WX,
    const MatrixXuf& D,
    const FP_TYPE gamma);

  //
  // Returns the gradient of @Z
  // Input: @D=gaussianKernel
  // Input: @Y, @Z can be CSR or CSC
  //
  MatrixXuf gradL_Z(
    const ZMatType& Z,
    const LabelMatType& Y,
    const MatrixXuf& D,
    const Eigen::Index begin,
    const Eigen::Index end);

  MatrixXuf gradL_Z(
    const ZMatType& Z,
    const LabelMatType& Y,
    const MatrixXuf& D);

  //
  // Returns the gradient of @B
  // Input: @B, @Y, @Z, @X, @W can be CSR or CSC
  // Input: @D=gaussianKernel
  //
  MatrixXuf gradL_W(
    const BMatType& B,
    const LabelMatType& Y,
    const ZMatType& Z,
    const WMatType& W,
    const SparseMatrixuf& X,
    const MatrixXuf& D,
    const FP_TYPE gamma,
    const Eigen::Index begin,
    const Eigen::Index end);

  MatrixXuf gradL_W(
    const BMatType& B,
    const LabelMatType& Y,
    const ZMatType& Z,
    const WMatType& W,
    const SparseMatrixuf& X,
    const MatrixXuf& D,
    const FP_TYPE gamma);

  //
  // Returns sparsified version of @mat, retaining only the top sparsity-many values
  // @mat: Matrix to be thresholded and returned
  // @sparsity: ratio of non-zero entries to be retained
  //
  void hardThrsd(MatrixXuf& mat, FP_TYPE sparsity);


  // uses accelerated proximal stochastic gradient descent
  void altMinSGD(
    const EdgeML::Data& data,
    EdgeML::ProtoNN::ProtoNNModel& model,
    FP_TYPE *const stats,
    const std::string& outDir);

  // ParamType is either MatrixXuf or SparseMatrixuf
  template <class ParamType>
  void accProxSGD(
    std::function<FP_TYPE(const ParamType&,
      const Eigen::Index, const Eigen::Index)> f,
    std::function<MatrixXuf(const ParamType&,
      const Eigen::Index, const Eigen::Index)> gradf,
    std::function<void(MatrixXuf&)> prox,
    ParamType& param,
    const int& epochs,
    const dataCount_t& n,
    const dataCount_t& bs,
    FP_TYPE eta,
    const int& etaUpdate);

  template<class ParamType>
    FP_TYPE btls(std::function<FP_TYPE(const ParamType&,
      const Eigen::Index, const Eigen::Index)> f,
    std::function<MatrixXuf(const ParamType&,
      const Eigen::Index, const Eigen::Index)> gradf,
    std::function<void(MatrixXuf&)> prox,
    ParamType& param,
    const dataCount_t& n,
    const dataCount_t& bs,
    FP_TYPE initialStepSizeEstimate);
}
#endif
