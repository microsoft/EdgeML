// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __FUNCTIONS_H_
#define __FUNCTIONS_H_

#include "utils.h"
#include "blas_routines.h"
#include "par_utils.h"
#include "Bonsai.h"


//Bonsai Calls
namespace EdgeML
{
  namespace Bonsai
  {
    ///
    /// Solver taking Trainer Object and Gives a converged Model
    ///
    void jointSgdBonsai(EdgeML::Bonsai::BonsaiTrainer& trainer);

    ///
    /// Function to Compute 2-way Hadamard product
    ///
    void hadamard2(MatrixXuf& M12,
      const MatrixXuf& M1,
      const MatrixXuf& M2);

    ///
    /// Function to Compute 3-way Hadamard product
    ///
    void hadamard3(MatrixXuf& M123,
      const MatrixXuf& M1,
      const MatrixXuf& M2,
      const MatrixXuf& M3);

    ///
    /// Function to Compute Coefficient Obtained during GradYhatW
    ///
    void gradWCoeff(MatrixXuf& CoeffMat,
      const MatrixXuf& ZX, EdgeML::Bonsai::BonsaiTrainer& trainer,
      const MatrixXufINT& classLst,
      const MatrixXuf& margin);

    ///
    /// Function to Compute Coefficient Obtained during GradYhatV
    ///
    void gradVCoeff(MatrixXuf& CoeffMat,
      const MatrixXuf& ZX,
      EdgeML::Bonsai::BonsaiTrainer& trainer,
      const MatrixXufINT& classLst,
      const MatrixXuf& margin);

    ///
    /// Function to Compute Coefficient Obtained during GradYhatTheta
    ///
    void gradThetaCoeff(MatrixXuf& ThetaCoeffMat,
      const MatrixXuf& ZX,
      const EdgeML::Bonsai::BonsaiTrainer& trainer,
      const MatrixXufINT& classLst,
      const MatrixXuf& margin);

    ///
    /// Function to Compute Gradient of prediction Function wrt W
    ///
    void gradYhatW(MatrixXuf& gradOut,
      const LabelMatType& Y,
      const SparseMatrixuf& X,
      const MatrixXuf& ZX,
      EdgeML::Bonsai::BonsaiTrainer& trainer,
      const MatrixXufINT& classLst,
      const MatrixXuf& margin);

    ///
    /// Function to Compute Gradient of prediction Function wrt V
    ///
    void gradYhatV(MatrixXuf& gradOut,
      const LabelMatType& Y,
      const SparseMatrixuf& X,
      const MatrixXuf& ZX,
      EdgeML::Bonsai::BonsaiTrainer& trainer,
      const MatrixXufINT& classLst,
      const MatrixXuf& margin);

    ///
    /// Function to Compute Gradient of prediction Function wrt Theta
    ///
    void gradYhatTheta(MatrixXuf& gradOut,
      const LabelMatType& Y,
      const SparseMatrixuf& X,
      const MatrixXuf& ZX,
      EdgeML::Bonsai::BonsaiTrainer& trainer,
      const MatrixXufINT& classLst,
      const MatrixXuf& margin);

    ///
    /// Function to Compute Gradient of prediction Function wrt Z
    ///
    void gradYhatZ(MatrixXuf& gradOut,
      const LabelMatType& Y,
      const SparseMatrixuf& X,
      const MatrixXuf& ZX,
      EdgeML::Bonsai::BonsaiTrainer& trainer,
      const MatrixXufINT& classLst,
      const MatrixXuf& margin);

    ///
    /// Typedef for elegant passing of functions with same signature
    ///
    typedef void(*grad_y_param_fun)(MatrixXuf& gradOut,
      const LabelMatType&,
      const SparseMatrixuf&,
      const MatrixXuf&,
      EdgeML::Bonsai::BonsaiTrainer&,
      const MatrixXufINT&,
      const MatrixXuf&);

    ///
    /// Function to Compute Gradient of Entire Optimisation Function wrt a given Dense Param(one of Z, W, V, Theta) and its gradYhatParam
    ///
    void gradLossParam(
      MatrixXuf& gradOut,
      const grad_y_param_fun grad_y_param,
      const MatrixXuf& Param,
      const FP_TYPE& regularizer,
      const LabelMatType& Y,
      const SparseMatrixuf& X,
      const MatrixXuf& ZX,
      EdgeML::Bonsai::BonsaiTrainer& trainer,
      const  MatrixXuf& margin,
      const MatrixXufINT& trueBestClassIndex);

    ///
    /// Function to Compute Gradient of Entire Optimisation Function wrt a given Sparse Param(one of Z, W, V, Theta) and its gradYhatParam
    ///
    void gradLossParam(
      MatrixXuf& gradOut,
      const grad_y_param_fun gradYParam,
      const SparseMatrixuf& param,
      const FP_TYPE& regularizer,
      const LabelMatType& Y,
      const SparseMatrixuf& X,
      const MatrixXuf& ZX,
      EdgeML::Bonsai::BonsaiTrainer& trainer,
      const  MatrixXuf& margin,
      const MatrixXufINT& trueBestClassIndex);

    ///
    /// Function to Compute Gradient of Entire Optimisation Function wrt W
    ///
    void gradLW(MatrixXuf& gradOut,
      const WMatType& W,
      const FP_TYPE& lW,
      const LabelMatType& Y,
      const SparseMatrixuf& X,
      const MatrixXuf& ZX,
      EdgeML::Bonsai::BonsaiTrainer& trainer);

    ///
    /// Function to Compute Gradient of Entire Optimisation Function wrt V
    ///
    void gradLV(MatrixXuf& gradOut,
      const VMatType& V,
      const FP_TYPE& lV,
      const LabelMatType& Y,
      const SparseMatrixuf& X,
      const MatrixXuf& ZX,
      EdgeML::Bonsai::BonsaiTrainer& trainer);

    ///
    /// Function to Compute Gradient of Entire Optimisation Function wrt Theta
    ///
    void gradLTheta(MatrixXuf& gradOut,
      const ThetaMatType& Theta,
      const FP_TYPE& lTheta,
      const LabelMatType& Y,
      const SparseMatrixuf& X,
      const MatrixXuf& ZX,
      EdgeML::Bonsai::BonsaiTrainer& trainer);

    ///
    /// Function to Compute Gradient of Entire Optimisation Function wrt Z
    ///
    void gradLZ(MatrixXuf& gradOut,
      const ZMatType& Z,
      const FP_TYPE& lZ,
      const LabelMatType& Y,
      const SparseMatrixuf& X,
      const MatrixXuf& ZX,
      EdgeML::Bonsai::BonsaiTrainer& trainer);


    ///
    /// Function to obtain Step Size using Armijo Rule
    ///
    template<class ParamType>
    MatrixXuf Armijo(std::function<FP_TYPE(const MatrixXuf&)> Loss,
      ParamType &param,
      MatrixXuf &grad,
      FP_TYPE target_sparsity,
      int iter);


    ///
    /// Function to Copy Support for Sparse Matrices, to be filled
    ///
    void copySupport(SparseMatrixuf& dst,
      const SparseMatrixuf& src);

    ///
    /// Function to Copy Support for Dense Matrices
    ///
    void copySupport(MatrixXuf& dst,
      const MatrixXuf& src);

    // Input - 
    // @mat: Matrix to be thresholded
    // @sparsity: ratio of non-zero entries to be retained
    // Returns sparsified version of @mat, retaining only the top sparsity-many values
    void hardThrsd(MatrixXuf& mat,
      FP_TYPE sparsity);


    // ParamType is either MatrixXuf or SparseMatrixuf
    template <class ParamType>
    void accproxsgd(std::function<FP_TYPE(const ParamType&,
      const Eigen::Index, const Eigen::Index)> f,
      std::function<MatrixXuf(const ParamType&,
        const Eigen::Index, const Eigen::Index)> gradf,
      std::function<void(MatrixXuf&)> prox,
      ParamType& param,
      const int& epochs,
      const dataCount_t& n,
      const dataCount_t& bs,
      FP_TYPE& eta,
      const int& eta_update);

    ///
    /// Returns nnzs in a dense Matrix as good as A.nonZeros()
    ///
    int countnnz(const MatrixXuf& A);

    ///
    /// Auxiliary function to give out usage infor when imput is wrong
    ///
    void exitWithHelp();

    ///
    /// Parser of Input Args to get the Program up and running from command line
    ///
    void parseInput(const int& argc,
      const char** argv,
      EdgeML::Bonsai::BonsaiModel::BonsaiHyperParams& hyperParam,
      std::string& dataDir);

    ///
    /// creates required subdirs and files for the data directory
    ///
    void createOutputDirs(const std::string& dataDir,
      std::string& currResultsPath);
  }
}

#endif
