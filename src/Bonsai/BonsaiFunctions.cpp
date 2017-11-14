// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "BonsaiFunctions.h"
// Bonsai Functions

using namespace EdgeML;

int Bonsai::countnnz(const MatrixXuf& A)
{
  int nnz = 0;
  for (int i = 0; i < A.rows(); i++)
	for (int j = 0; j < A.cols(); j++)
	  if (A(i, j) != (FP_TYPE)0.0)
		++nnz;
  return nnz;
}

void Bonsai::hadamard2(MatrixXuf& M12, const MatrixXuf& M1, const MatrixXuf& M2)
{
  vMul(M12.rows()*M12.cols(), M1.data(), M2.data(), M12.data());
}

void Bonsai::hadamard3(MatrixXuf& M123, const MatrixXuf& M1, const MatrixXuf& M2, const MatrixXuf& M3)
{
  vMul(M123.rows()*M123.cols(), M1.data(), M2.data(), M123.data());
  vMul(M123.rows()*M123.cols(), M3.data(), M123.data(), M123.data());
}

void Bonsai::gradWCoeff(MatrixXuf& CoeffMat, const MatrixXuf& ZX, EdgeML::Bonsai::BonsaiTrainer& trainer,
  const MatrixXufINT &classLst, const MatrixXuf &margin)
{
  for (int j = 0; j < CoeffMat.cols(); j++)
  {
	if (trainer.YMultCoeff(0, j) *margin(0, j) < (FP_TYPE)1.0)
	{
	  int classNodesStart = (labelCount_t)classLst(0, j)*trainer.model.hyperParams.totalNodes;
	  for (int i = classNodesStart; i < classNodesStart + trainer.model.hyperParams.totalNodes; i++)
	  {
		// dot product
		CoeffMat(i, j) = trainer.YMultCoeff(0, j)
		  *tanh(trainer.model.hyperParams.Sigma * trainer.treeCache.tanhVXWeight(i, j))
		  *trainer.treeCache.nodeProbability(i%trainer.model.hyperParams.totalNodes, j);
	  }
	}
  }
}

void Bonsai::gradVCoeff(MatrixXuf& CoeffMat, const MatrixXuf& ZX, EdgeML::Bonsai::BonsaiTrainer& trainer,
  const MatrixXufINT &classLst, const MatrixXuf &margin)
{
  for (int j = 0; j < CoeffMat.cols(); j++)
  {
	if (trainer.YMultCoeff(0, j) *margin(0, j) < (FP_TYPE)1.0)
	{
	  int classNodesStart = (labelCount_t)classLst(0, j)*trainer.model.hyperParams.totalNodes;
	  for (int i = classNodesStart; i < classNodesStart + trainer.model.hyperParams.totalNodes; i++)
	  {
		CoeffMat(i, j) = trainer.YMultCoeff(0, j) * trainer.treeCache.WXWeight(i, j)
		  * ((FP_TYPE)1.0 - pow(tanh(trainer.model.hyperParams.Sigma * trainer.treeCache.tanhVXWeight(i, j)), (FP_TYPE)2.0))
		  *trainer.treeCache.nodeProbability(i%trainer.model.hyperParams.totalNodes, j);
	  }
	}
  }
}

void Bonsai::gradThetaCoeff(MatrixXuf &ThetaCoeffMat, const MatrixXuf& ZX, const EdgeML::Bonsai::BonsaiTrainer& trainer,
  const MatrixXufINT &classLst, const MatrixXuf& margin)
{
  for (int n = 0; n < ZX.cols(); n++)
  {
	if (trainer.YMultCoeff(0, n) *margin(0, n) < (FP_TYPE)1.0)
	{

	  int classNodesStart = (labelCount_t)classLst(0, n)*trainer.model.hyperParams.totalNodes;
	  for (int i = classNodesStart; i < classNodesStart + trainer.model.hyperParams.totalNodes; i++)
	  {
		FP_TYPE tempGrad = trainer.YMultCoeff(0, n)
		  *trainer.treeCache.nodeProbability(i%trainer.model.hyperParams.totalNodes, n)
		  *tanh(trainer.model.hyperParams.Sigma * trainer.treeCache.tanhVXWeight(i, n))
		  *trainer.treeCache.WXWeight(i, n);

		int r = i%trainer.model.hyperParams.totalNodes;
		while (r > 0)
		{
		  int t = (r - 1) / 2;
		  ThetaCoeffMat(t, n) += tempGrad*(r % 2 == 1 ? (1 - trainer.treeCache.tanhThetaXCache(t, n)) : (-1 - trainer.treeCache.tanhThetaXCache(t, n)));
		  r = t;
		}
	  }
	}
  }
}

void Bonsai::gradYhatW(
  MatrixXuf& gradOut,
  const LabelMatType& Y,
  const SparseMatrixuf& X,
  const MatrixXuf& ZX,
  EdgeML::Bonsai::BonsaiTrainer& trainer,
  const MatrixXufINT &classLst,
  const MatrixXuf &margin)
{
  assert(gradOut.rows() == (trainer.model.hyperParams.totalNodes) * (trainer.model.hyperParams.internalClasses));
  assert(gradOut.cols() == trainer.model.hyperParams.projectionDimension);

  trainer.treeCache.partialZGradient = MatrixXuf::Zero(trainer.model.hyperParams.projectionDimension, ZX.cols());
  MatrixXuf CoeffMat = MatrixXuf::Zero((trainer.model.hyperParams.internalClasses) * (trainer.model.hyperParams.totalNodes), ZX.cols());

  gradWCoeff(CoeffMat, ZX, trainer, classLst, margin);
  mm(gradOut, CoeffMat, CblasNoTrans, ZX, CblasTrans, (FP_TYPE)1.0, (FP_TYPE)0.0);
}

void Bonsai::gradYhatV(
  MatrixXuf& gradOut,
  const LabelMatType& Y,
  const SparseMatrixuf& X,
  const MatrixXuf& ZX,
  EdgeML::Bonsai::BonsaiTrainer& trainer,
  const MatrixXufINT &classLst,
  const MatrixXuf &margin)
{
  assert(gradOut.rows() == (trainer.model.hyperParams.totalNodes) * (trainer.model.hyperParams.internalClasses));
  assert(gradOut.cols() == trainer.model.hyperParams.projectionDimension);

  MatrixXuf CoeffMat = MatrixXuf::Zero((trainer.model.hyperParams.internalClasses) * (trainer.model.hyperParams.totalNodes), ZX.cols());
  gradVCoeff(CoeffMat, ZX, trainer, classLst, margin);

  mm(gradOut, CoeffMat, CblasNoTrans, ZX, CblasTrans, (FP_TYPE)1.0, (FP_TYPE)0.0);
};


void Bonsai::gradYhatTheta(MatrixXuf& gradOut,
  const LabelMatType& Y, const SparseMatrixuf& X,
  const MatrixXuf& ZX, EdgeML::Bonsai::BonsaiTrainer& trainer, const MatrixXufINT &classLst, const MatrixXuf &margin)
{
  assert(gradOut.rows() == trainer.model.hyperParams.internalNodes);
  assert(gradOut.cols() == trainer.model.hyperParams.projectionDimension);

  MatrixXuf CoeffMat = MatrixXuf::Zero(trainer.model.hyperParams.internalNodes, ZX.cols());
  gradThetaCoeff(CoeffMat, ZX, trainer, classLst, margin);

  if(trainer.model.hyperParams.internalNodes > 0)
    mm(gradOut, CoeffMat, CblasNoTrans, ZX, CblasTrans, (FP_TYPE)1.0, (FP_TYPE)0.0);
};


void Bonsai::gradYhatZ(
  MatrixXuf& gradOut,
  const LabelMatType& Y, const SparseMatrixuf& X,
  const MatrixXuf& ZX,
  EdgeML::Bonsai::BonsaiTrainer& trainer,
  const MatrixXufINT &classLst,
  const MatrixXuf &margin)
{
  assert(gradOut.rows() == trainer.model.hyperParams.projectionDimension);
  assert(gradOut.cols() == trainer.model.hyperParams.dataDimension);

  MatrixXuf CoeffMatW = MatrixXuf::Zero(trainer.model.hyperParams.internalClasses * trainer.model.hyperParams.totalNodes, ZX.cols());
  MatrixXuf CoeffMatV = MatrixXuf::Zero(trainer.model.hyperParams.internalClasses * trainer.model.hyperParams.totalNodes, ZX.cols());
  MatrixXuf CoeffMatTheta = MatrixXuf::Zero(trainer.model.hyperParams.internalNodes, ZX.cols());

  gradWCoeff(CoeffMatW, ZX, trainer, classLst, margin);
  gradVCoeff(CoeffMatV, ZX, trainer, classLst, margin);
  gradThetaCoeff(CoeffMatTheta, ZX, trainer, classLst, margin);

  for (int n = 0; n < ZX.cols(); n++)
  {
	MatrixXuf partialZGradientColN = MatrixXuf::Zero(trainer.model.hyperParams.projectionDimension, 1);

	mm(partialZGradientColN, trainer.model.getW((labelCount_t)classLst(0, n)),
	  CblasTrans, CoeffMatW.block((labelCount_t)classLst(0, n)*trainer.model.hyperParams.totalNodes, n, trainer.model.hyperParams.totalNodes, 1),
	  CblasNoTrans, (FP_TYPE)1.0, (FP_TYPE)1.0);

	mm(partialZGradientColN, trainer.model.getV((labelCount_t)classLst(0, n)),
	  CblasTrans, CoeffMatV.block((labelCount_t)classLst(0, n)*trainer.model.hyperParams.totalNodes, n, trainer.model.hyperParams.totalNodes, 1),
	  CblasNoTrans, (FP_TYPE)1.0, (FP_TYPE)1.0);

	trainer.treeCache.partialZGradient.col(n) = partialZGradientColN;
  }

  if(trainer.model.hyperParams.internalNodes > 0)
    mm(trainer.treeCache.partialZGradient, trainer.model.params.Theta, CblasTrans,
  	CoeffMatTheta, CblasNoTrans, (FP_TYPE)1.0, (FP_TYPE)1.0);

  mm(gradOut, trainer.treeCache.partialZGradient, CblasNoTrans, X, CblasTrans, (FP_TYPE)1.0, (FP_TYPE)0.0L);
};


void Bonsai::gradLossParam(
  MatrixXuf& gradOut,
  const grad_y_param_fun gradYParam,
  const MatrixXuf& param,
  const FP_TYPE& regularizer,
  const LabelMatType& Y, const SparseMatrixuf& X, const MatrixXuf& ZX,
  EdgeML::Bonsai::BonsaiTrainer& trainer,
  const  MatrixXuf& margin, const MatrixXufINT& trueBestClassIndex)
{
  assert(gradOut.rows() == param.rows());
  assert(gradOut.cols() == param.cols());

  gradYParam(gradOut, Y, X, ZX, trainer, trueBestClassIndex.row(0), margin);

  if (trainer.model.hyperParams.numClasses > 2)
  {
	MatrixXuf gradBestClass(gradOut.rows(), gradOut.cols());
	gradYParam(gradBestClass, Y, X, ZX, trainer, trueBestClassIndex.row(1), margin);

	gradOut -= gradBestClass;
	gradOut *= (FP_TYPE)-1.0 / (FP_TYPE)ZX.cols();
	gradOut += regularizer*param;
  }
  else
  {
	gradOut *= (FP_TYPE)-1.0 / (FP_TYPE)ZX.cols();
	gradOut += regularizer*param;
  }
};

void Bonsai::gradLossParam(
  MatrixXuf& gradOut,
  const grad_y_param_fun gradYParam,
  const SparseMatrixuf& param,
  const FP_TYPE& regularizer,
  const LabelMatType& Y, const SparseMatrixuf& X, const MatrixXuf& ZX,
  EdgeML::Bonsai::BonsaiTrainer& trainer,
  const  MatrixXuf& margin, const MatrixXufINT& trueBestClassIndex)
{
  assert(gradOut.rows() == param.rows());
  assert(gradOut.cols() == param.cols());

  gradYParam(gradOut, Y, X, ZX, trainer, trueBestClassIndex.row(0), margin);

  if (trainer.model.hyperParams.numClasses > 2)
  {
	MatrixXuf gradBestClass(gradOut.rows(), gradOut.cols());
	gradYParam(gradBestClass, Y, X, ZX, trainer, trueBestClassIndex.row(1), margin);

	gradOut -= gradBestClass;
	gradOut *= (FP_TYPE)-1.0 / (FP_TYPE)ZX.cols();
	gradOut += MatrixXuf(regularizer*param);
  }
  else
  {
	gradOut *= (FP_TYPE)-1.0 / (FP_TYPE)ZX.cols();
	gradOut += MatrixXuf(regularizer*param);
  }
};

void Bonsai::gradLW(
  MatrixXuf& gradOut,
  const WMatType& W, const FP_TYPE& lW, const LabelMatType& Y,
  const SparseMatrixuf& X, const MatrixXuf& ZX,
  EdgeML::Bonsai::BonsaiTrainer& trainer)
{
  //std::cout << "Starting Gradient: " << std::endl;
  trainer.initializeTrainVariables(Y);
  trainer.fillNodeProbability(ZX);

  MatrixXuf trueBestScore = MatrixXuf::Ones(2, ZX.cols())*(-1000.0L);
  MatrixXufINT trueBestClassIndex = MatrixXufINT::Zero(2, ZX.cols());

  trainer.getTrueBestClass(trueBestScore, trueBestClassIndex, W, trainer.model.params.V, Y, ZX);

  trainer.fillWX(ZX, trueBestClassIndex.row(0));
  trainer.fillTanhVX(ZX, trueBestClassIndex.row(0));

  // 2nd term not needed for binary classification
  MatrixXuf margin = trueBestScore.row(0) - trueBestScore.row(1);

  if (trainer.model.hyperParams.internalClasses > 2)
  {
	trainer.fillWX(ZX, trueBestClassIndex.row(1));
	trainer.fillTanhVX(ZX, trueBestClassIndex.row(1));
  }

  gradLossParam(gradOut, &gradYhatW, W, lW, Y, X, ZX, trainer, margin, trueBestClassIndex);
}

void Bonsai::gradLV(
  MatrixXuf& gradOut,
  const VMatType& V, const FP_TYPE& lV, const LabelMatType& Y,
  const SparseMatrixuf& X, const MatrixXuf& ZX, EdgeML::Bonsai::BonsaiTrainer& trainer)
{

  trainer.initializeTrainVariables(Y);
  trainer.fillNodeProbability(ZX);

  MatrixXuf trueBestScore = MatrixXuf::Ones(2, ZX.cols())*(-1000.0L);
  MatrixXufINT trueBestClassIndex = MatrixXufINT::Zero(2, ZX.cols());

  trainer.getTrueBestClass(trueBestScore, trueBestClassIndex, trainer.model.params.W, V, Y, ZX);

  trainer.fillWX(ZX, trueBestClassIndex.row(0));
  trainer.fillTanhVX(ZX, trueBestClassIndex.row(0));

  // 2nd term not needed for binary classification
  MatrixXuf margin = trueBestScore.row(0) - trueBestScore.row(1);

  if (trainer.model.hyperParams.internalClasses > 2)
  {
	trainer.fillWX(ZX, trueBestClassIndex.row(1));
	trainer.fillTanhVX(ZX, trueBestClassIndex.row(1));
  }
  gradLossParam(gradOut, &gradYhatV, V, lV, Y, X, ZX, trainer, margin, trueBestClassIndex);
};

void Bonsai::gradLTheta(
  MatrixXuf& gradOut,
  const ThetaMatType& Theta, const FP_TYPE& lTheta, const LabelMatType& Y,
  const SparseMatrixuf& X, const MatrixXuf& ZX, EdgeML::Bonsai::BonsaiTrainer& trainer)
{
  trainer.initializeTrainVariables(Y);
  trainer.fillNodeProbability(ZX);

  MatrixXuf trueBestScore = MatrixXuf::Ones(2, ZX.cols())*(-1000.0L);
  MatrixXufINT trueBestClassIndex = MatrixXufINT::Zero(2, ZX.cols());

  trainer.getTrueBestClass(trueBestScore, trueBestClassIndex, trainer.model.params.W, trainer.model.params.V, Y, ZX);

  trainer.fillWX(ZX, trueBestClassIndex.row(0));
  trainer.fillTanhVX(ZX, trueBestClassIndex.row(0));

  // 2nd term not needed for binary classification
  MatrixXuf margin = trueBestScore.row(0) - trueBestScore.row(1);

  if (trainer.model.hyperParams.numClasses > 2)
  {
	trainer.fillWX(ZX, trueBestClassIndex.row(1));
	trainer.fillTanhVX(ZX, trueBestClassIndex.row(1));
  }

  gradLossParam(gradOut, &gradYhatTheta, Theta, lTheta, Y, X, ZX, trainer, margin, trueBestClassIndex);
};

void Bonsai::gradLZ(
  MatrixXuf& gradOut,
  const ZMatType& Z, const FP_TYPE& lZ, const LabelMatType& Y,
  const SparseMatrixuf& X, const MatrixXuf& ZX, EdgeML::Bonsai::BonsaiTrainer& trainer)
{
  trainer.initializeTrainVariables(Y);
  trainer.fillNodeProbability(ZX);

  MatrixXuf trueBestScore = MatrixXuf::Ones(2, ZX.cols())*(-1000.0L);
  MatrixXufINT trueBestClassIndex = MatrixXufINT::Zero(2, ZX.cols());

  trainer.getTrueBestClass(trueBestScore, trueBestClassIndex, trainer.model.params.W, trainer.model.params.V, Y, ZX);

  trainer.fillWX(ZX, trueBestClassIndex.row(0));
  trainer.fillTanhVX(ZX, trueBestClassIndex.row(0));

  // 2nd term not needed for binary classification
  MatrixXuf margin = trueBestScore.row(0) - trueBestScore.row(1);

  if (trainer.model.hyperParams.numClasses > 2)
  {
	trainer.fillWX(ZX, trueBestClassIndex.row(1));
	trainer.fillTanhVX(ZX, trueBestClassIndex.row(1));
  }

  gradLossParam(gradOut, &gradYhatZ, Z, lZ, Y, X, ZX, trainer, margin, trueBestClassIndex);
};

template<class ParamType>
MatrixXuf Bonsai::Armijo(std::function<FP_TYPE(const MatrixXuf&)> Loss,
  ParamType &param, MatrixXuf &grad, FP_TYPE targetSparsity, int iter)
{
  FP_TYPE baseOffset = (FP_TYPE)0.01 * grad.squaredNorm();
  FP_TYPE s = (FP_TYPE)1.0;
  FP_TYPE beta = (FP_TYPE)0.5;

  FP_TYPE initLoss = Loss(MatrixXuf(param));

  MatrixXuf paramPlusSGrad(param.rows(), param.cols());
  FP_TYPE curLoss;

  int runCount = 0;
  do {
	paramPlusSGrad = MatrixXuf(param) - s*grad;
	hardThrsd(paramPlusSGrad, targetSparsity);
	curLoss = Loss(paramPlusSGrad);
	s *= beta;
  } while (curLoss > initLoss - (s / beta)*baseOffset && runCount++ < 21);

  return paramPlusSGrad;
}


void Bonsai::jointSgdBonsai(EdgeML::Bonsai::BonsaiTrainer& trainer)
{
  Logger logger("jointSgdBonsai");
  Timer timer("jointSgdBonsai");

  enum training_phase {
	DENSE_TRAIN, CORE_IHT_THRESH, CORE_IHT_FC, SPARSE_RETRAIN
  };
  training_phase trainFlag = DENSE_TRAIN;
  int trimLevel = (trainer.model.hyperParams.numClasses <= 2) ? 5 : 15;

  dataCount_t n = trainer.data.Xtrain.cols();
  int         epochs = trainer.model.hyperParams.epochs;
  //int print_interval = 100;
  FP_TYPE eta_Z((FP_TYPE)0.01), eta_V((FP_TYPE)0.01), eta_W((FP_TYPE)0.01), eta_Theta((FP_TYPE)0.01);

  FP_TYPE sparsity_Z = (FP_TYPE)1.0;
  FP_TYPE sparsity_W = (FP_TYPE)1.0;
  FP_TYPE sparsity_V = (FP_TYPE)1.0;
  FP_TYPE sparsity_Theta = (FP_TYPE)1.0;

  int batchSize = std::max(100, 1 + (int)((trainer.model.hyperParams.batchFactor)*sqrt(trainer.data.Xtrain.cols())));
  // int batchSize = trainer.data.Xtrain.cols();
  if (batchSize > trainer.data.Xtrain.cols()) batchSize = trainer.data.Xtrain.cols();

  Eigen::Index begin = 0;
  Eigen::Index end = begin + batchSize;

  MatrixXuf ZX = MatrixXuf::Zero(trainer.model.params.Z.rows(), trainer.data.Xtrain.cols());
  int iterations_within_phase = 0;

  int batchesPerIter =
	(trainer.data.Xtrain.cols() / batchSize == 0 ? trainer.data.Xtrain.cols() / batchSize
	  : 1 + (trainer.data.Xtrain.cols() / batchSize));
  int numBatches = trainer.model.hyperParams.iters * batchesPerIter;

  MatrixXuf gradZ(trainer.model.params.Z.rows(), trainer.model.params.Z.cols());
  MatrixXuf gradV(trainer.model.params.V.rows(), trainer.model.params.V.cols());
  MatrixXuf gradW(trainer.model.params.W.rows(), trainer.model.params.W.cols());
  MatrixXuf gradTheta(trainer.model.params.Theta.rows(), trainer.model.params.Theta.cols());

  // TODO: update the hyperParams.iter to *= sqrt(ntrain).
  // TODO: Ask for more sensible default iteration parameters
  for (int i = 0; i < numBatches; ++i)
  {
	if (end == trainer.data.Xtrain.cols())  end = 0;
	begin = (i == 0) ? 0 : end;
	end = std::min(begin + batchSize, trainer.data.Xtrain.cols());

	if (begin == 0)
	  LOG_INFO("=========================== \n On iter "
		+ std::to_string(i / batchesPerIter) + "\n"
		+ "=========================== ");
	LOG_INFO("points: (" + std::to_string(begin) + "," + std::to_string(end) + ")");

	// Move to outside the loop
	MatrixXuf ZX_i = MatrixXuf::Zero(trainer.model.params.Z.rows(), end - begin);
	SparseMatrixuf X_sliced = trainer.data.Xtrain.middleCols(begin, end - begin);
	LabelMatType Y_sliced = trainer.data.Ytrain.middleCols(begin, end - begin);

	//  1st 1/3rd iterations are for dense training, 
	//  2nd 1/3rd are for the Core IHT algorithm
	//  3rd 1/3rd is sparse retraining with fixed support 
	if (i >= 1 * (numBatches) / 3
	  && i < 2 * (numBatches) / 3)
	{
	  // For Core IHT, we threshod every trimLevel'th iteration,
	  // and proceed with fuly corrective step in between.
	  if (i%trimLevel == 0)
		trainFlag = CORE_IHT_THRESH;
	  else
		trainFlag = CORE_IHT_FC;
	}
	else if (i >= 2 * (numBatches) / 3
	  && i < (numBatches)) {
	  trainFlag = SPARSE_RETRAIN;
	}

	timer.nextTime("starting gradZ");

	mm(ZX_i, MatrixXuf(trainer.model.params.Z), CblasNoTrans,
	  X_sliced, CblasNoTrans, (FP_TYPE)1.0 / trainer.model.hyperParams.projectionDimension, (FP_TYPE)0.0L);

	if (i == 0 || i == 2 * numBatches / 3 || i == 1 * numBatches / 3)
	{
	  trainer.model.initializeSigmaI();
	  iterations_within_phase = 0;
	}
	else if (iterations_within_phase % 100 == 0)
	{
	  int exp_fac = iterations_within_phase / (numBatches / 30);
	  trainer.model.updateSigmaI(ZX_i, exp_fac);
	}

	gradLZ(gradZ, trainer.model.params.Z,
	  trainer.model.hyperParams.regList.lZ, Y_sliced,
	  X_sliced, ZX_i, trainer);

	gradLW(gradW, trainer.model.params.W,
	  trainer.model.hyperParams.regList.lW, Y_sliced,
	  X_sliced, ZX_i, trainer);

	gradLV(gradV, trainer.model.params.V,
	  trainer.model.hyperParams.regList.lV, Y_sliced,
	  X_sliced, ZX_i, trainer);

	gradLTheta(gradTheta, trainer.model.params.Theta,
	  trainer.model.hyperParams.regList.lTheta, Y_sliced,
	  X_sliced, ZX_i, trainer);

	if (trainFlag == SPARSE_RETRAIN || trainFlag == CORE_IHT_FC)
	{
	  // SPARSE_RETRAIN and CORE_IHT_FC
	  // freeze the support and do gradient updates
	  copySupport(gradZ, trainer.model.params.Z);
	  copySupport(gradW, trainer.model.params.W);
	  copySupport(gradV, trainer.model.params.V);
	  copySupport(gradTheta, trainer.model.params.Theta);
	}
	if (trainFlag == SPARSE_RETRAIN || trainFlag == CORE_IHT_FC || trainFlag == DENSE_TRAIN)
	{
	  // setting sparsity = 1.0 ensures that no thresholding occurs
	  sparsity_Z = (FP_TYPE)1.0;
	  sparsity_W = (FP_TYPE)1.0;
	  sparsity_V = (FP_TYPE)1.0;
	  sparsity_Theta = (FP_TYPE)1.0;
	}
	if (trainFlag == CORE_IHT_THRESH)
	{
	  sparsity_Z = trainer.model.hyperParams.lambdaZ;
	  sparsity_W = trainer.model.hyperParams.lambdaW;
	  sparsity_V = trainer.model.hyperParams.lambdaV;
	  sparsity_Theta = trainer.model.hyperParams.lambdaTheta;
	}

	MatrixXuf Wupdated = Armijo<WMatType>(
	  [&trainer, &X_sliced, &Y_sliced, &ZX_i](const MatrixXuf &W)->FP_TYPE
	{
	  FP_TYPE obj_val = trainer.computeObjective(MatrixXuf(trainer.model.params.Z), W,
		MatrixXuf(trainer.model.params.V), MatrixXuf(trainer.model.params.Theta),
		ZX_i, Y_sliced);
	  return obj_val;
	}
	, trainer.model.params.W, gradW, sparsity_W, i);

#ifdef SPARSE_W_BONSAI
	trainer.model.params.W = Wupdated.sparseView();
#else
	trainer.model.params.W = Wupdated;
#endif


	MatrixXuf Vupdated = Armijo<VMatType>(
	  [&trainer, &X_sliced, &Y_sliced, &ZX_i](const MatrixXuf &V)->FP_TYPE
	{
	  FP_TYPE obj_val = trainer.computeObjective(MatrixXuf(trainer.model.params.Z), MatrixXuf(trainer.model.params.W),
		V, MatrixXuf(trainer.model.params.Theta),
		ZX_i, Y_sliced);
	  return obj_val;
	}
	, trainer.model.params.V, gradV, sparsity_V, i);

#ifdef SPARSE_V_BONSAI
	trainer.model.params.V = Vupdated.sparseView();
#else
	trainer.model.params.V = Vupdated;
#endif


	MatrixXuf Thetaupdated = Armijo<ThetaMatType>(
	  [&trainer, &X_sliced, &Y_sliced, &ZX_i](const MatrixXuf &Theta)->FP_TYPE
	{
	  FP_TYPE obj_val = trainer.computeObjective(MatrixXuf(trainer.model.params.Z), MatrixXuf(trainer.model.params.W),
		MatrixXuf(trainer.model.params.V), Theta,
		ZX_i, Y_sliced);
	  return obj_val;
	}
	, trainer.model.params.Theta, gradTheta, sparsity_Theta, i);

#ifdef SPARSE_THETA_BONSAI
	trainer.model.params.Theta = Thetaupdated.sparseView();
#else
	trainer.model.params.Theta = Thetaupdated;
#endif

	MatrixXuf Zupdated = Armijo<ZMatType>(
	  [&trainer, &X_sliced, &Y_sliced, &ZX_i](const MatrixXuf &Z)->FP_TYPE
	{
	  mm(ZX_i, Z, CblasNoTrans, X_sliced, CblasNoTrans, (FP_TYPE)1.0L / trainer.model.hyperParams.projectionDimension, (FP_TYPE)0.0L);
	  FP_TYPE obj_val = trainer.computeObjective(Z, MatrixXuf(trainer.model.params.W),
		MatrixXuf(trainer.model.params.V), MatrixXuf(trainer.model.params.Theta),
		ZX_i, Y_sliced);
	  return obj_val;
	}
	, trainer.model.params.Z, gradZ, sparsity_Z, i);

#ifdef SPARSE_Z_BONSAI
	trainer.model.params.Z = Zupdated.sparseView();
#else
	trainer.model.params.Z = Zupdated;
#endif


	if (end >= trainer.data.Xtrain.cols())
	{
	  mm(ZX, MatrixXuf(trainer.model.params.Z), CblasNoTrans, trainer.data.Xtrain, CblasNoTrans, (FP_TYPE)1.0 / trainer.model.hyperParams.projectionDimension, (FP_TYPE)0.0L);
	  FP_TYPE objval = trainer.computeObjective(ZX, trainer.data.Ytrain);

	  LOG_INFO("Finished Iter:" + std::to_string(i / batchesPerIter) + "  "
		+ "nnz(W): " + std::to_string(countnnz(trainer.model.params.W)) + "/" + std::to_string(trainer.model.params.W.rows()*trainer.model.params.W.cols()) + "  " +
		+"nnz(V): " + std::to_string(countnnz(trainer.model.params.V)) + "/" + std::to_string(trainer.model.params.V.rows()*trainer.model.params.V.cols()) + "  " +
		+"nnz(Theta): " + std::to_string(countnnz(trainer.model.params.Theta)) + "/" + std::to_string(trainer.model.params.Theta.rows()*trainer.model.params.Theta.cols()) + "  " +
		+"nnz(Z): " + std::to_string(countnnz(trainer.model.params.Z)) + "/" + std::to_string(trainer.model.params.Z.rows()*trainer.model.params.Z.cols()));
	}
	iterations_within_phase++;
  }
}

void Bonsai::copySupport(SparseMatrixuf& dst, const SparseMatrixuf& src)
{
  assert(false);
  // Fill this in.
}

void Bonsai::copySupport(MatrixXuf& dst, const MatrixXuf& src)
{
  assert(dst.rows() == src.rows());
  assert(dst.cols() == src.cols());
  for (int j = 0; j < dst.cols(); j++)
	for (int i = 0; i < dst.rows(); i++)
	  if (src(i, j) == (FP_TYPE)0.0)
		dst(i, j) = (FP_TYPE)0.0;
}

void Bonsai::hardThrsd(MatrixXuf& mat, FP_TYPE sparsity)
{
  Timer timer("hardThrsd");
  assert(sparsity >= (FP_TYPE)0.0 && sparsity <= (FP_TYPE)1.0);
  if (sparsity >= (FP_TYPE)0.999 || (mat.rows()*mat.cols() == 0))
	return;
  else;

  const FP_TYPE eps = (FP_TYPE)1e-8;

  assert(sizeof(size_t) == 8);
  size_t mat_size = ((size_t)mat.rows()) * ((size_t)mat.cols());
  size_t sample_size = 10000000;
  sample_size = std::min(sample_size, mat_size);

  FP_TYPE *data = new FP_TYPE[sample_size];

  if (sample_size == mat_size) {
	memcpy((void *)data, (void *)mat.data(), sizeof(FP_TYPE)*mat_size);
	for (size_t i = 0; i < mat_size; ++i) data[i] = std::abs(data[i]);
  }
  else {
	unsigned long long prime = 990377764891511ull;
	assert(prime > mat_size);
	unsigned long long seed = (rand() % 100000);
	FP_TYPE* matData = mat.data();
	size_t pick;
	for (dataCount_t i = 0; i < sample_size; ++i) {
	  pick = (prime*(i + seed)) % mat_size;
	  data[i] = std::abs(matData[pick]);
	}
  }
  timer.nextTime("allocating and initializing memory");


  timer.nextTime("starting threshold computation");
  /*std::sort (data, data + mat_size,
  [](FP_TYPE i, FP_TYPE j) {return std::abs(i) > std::abs(j);});*/
  //sampleSort<FP_TYPE, std::greater<FP_TYPE>, size_t> (data, mat_size, std::greater<FP_TYPE>());
  //FP_TYPE thresh = std::abs (data[(size_t)((sparsity*mat_size) - 1)]);
  size_t order = (size_t)std::round((1.0 - sparsity)*((FP_TYPE)sample_size)) + (size_t)1;
  if (order > sample_size) order = sample_size;
  FP_TYPE thresh = sequentialQuickSelect(data, sample_size, order);

  if (thresh <= eps)thresh = eps;
  delete[] data;
  timer.nextTime("ending threshold computation");

  assert(sizeof(std::ptrdiff_t) == sizeof(size_t));
  data = mat.data();

#ifdef CILK
  cilk::reducer< cilk::op_add<size_t> > nnz(0);
#else
  int *nnz = new int;
  *nnz = 0;
#endif
  for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)mat_size; ++i) {
	if (std::abs(data[i]) < thresh)
	  data[i] = 0;
	else
	  *nnz += 1;
  }
  timer.nextTime("thresholding");
#ifdef CILK
  LOG_TRACE("nnz/numel = " + std::to_string((FP_TYPE)nnz.get_value() / (FP_TYPE)mat_size));
#else
  LOG_TRACE("nnz/numel = " + std::to_string((FP_TYPE)(*nnz) / (FP_TYPE)mat_size));
  delete nnz;
#endif
}

// function xb = accproxsgd(f, gradf, prox, x, batchSize, epochs, n, eta, learning_rate)
// returns the latest step-length used in case it has to be restarted. 
template<class ParamType>
void Bonsai::accproxsgd(std::function<FP_TYPE(const ParamType&,
  const Eigen::Index, const Eigen::Index)> f,
  std::function<MatrixXuf(const ParamType&,
	const Eigen::Index, const Eigen::Index)> gradf,
  std::function<void(MatrixXuf&)> prox,
  ParamType& param,
  const int& epochs,
  const dataCount_t& n,
  const dataCount_t& bs,
  FP_TYPE& eta,
  const int& eta_update)
{
  Timer timer("accproxsgd");
  Logger local_logger("accproxsgd ");

  /*
  xb = sparse(zeros(size(x)));
  burnPeriod = 50;
  iters_per_epoch = round(n*1.0/batchSize);
  gammaO = 1; x0 = x;
  */

  MatrixXuf paramt = MatrixXuf::Zero(param.rows(), param.cols());
  ParamType paramb = ParamType(param.rows(), param.cols());
  MatrixXuf gf = MatrixXuf::Zero(param.rows(), param.cols());
  ParamType param0 = param;
  dataCount_t bs_ = bs;
  int burnPeriod = 50;
  FP_TYPE gamma0 = 1;
  FP_TYPE t, gamma, alpha;
  timer.nextTime("creating matrices");

  if (bs_ > n) {
	bs_ = n;
	//std::cerr << bs_ << " " << n << "\n";
	//assert (bs_ <= n);
  }
  uint64_t iters = ((uint64_t)n*(uint64_t)epochs) / (uint64_t)bs_;
  assert(iters < 0x7fffffff);

  /*
  i = 1;
  while i <= epochs
  j = 1;
  while j <= iters_per_epoch
  iter = (i-1)*iters_per_epoch + j;
  */
  for (int i = 0; i < iters; ++i) {
	LOG_DIAGNOSTIC(i);
	Eigen::Index idx1 = (i*(Eigen::Index)bs_) % n;
	Eigen::Index idx2 = ((i + 1)*(Eigen::Index)bs_) % n;
	if (idx2 <= idx1) idx2 = n;

	/*
	 if learning_rate == -1
	 t = eta/(1+0.2*iter);
	 elseif learning_rate == 0
	 t = eta/sqrt(iter);
	 end
	*/
	switch (eta_update) {
	case -1:
	  t = safeDiv(eta, (1.0f + 0.2f*(i + 1.0f)));
	  break;
	case 0:
	  t = safeDiv(eta, pow(i + 1.0f, 0.5f));
	  break;
	}
	//std::cerr << "eta = " << t <<"\n";

	/*
	gamma = 0.5 + 0.5 * sqrt(1 + 4 * gammaO^2);
	alpha = (1 - gammaO)/gamma;
	xt = prox(x - t * gradf(x, unique(randi(n, 1, batchSize))), t);
	x = (1 - alpha) * xt + alpha * x0;
	*/
	gamma = (FP_TYPE)0.5 + (FP_TYPE)0.5 * pow((FP_TYPE)1.0 + (FP_TYPE)4.0 * gamma0*gamma0, (FP_TYPE)0.5);
	alpha = safeDiv((1 - gamma0), gamma);
	gf = gradf(param, idx1, idx2);
	timer.nextTime("taking gradient");
	paramt = param - t * gf;
	timer.nextTime("computing new paramt");
	prox(paramt);
	timer.nextTime("L0 projection (sparsifying paramt)");

	ParamType paramt_;
	typeMismatchAssign(paramt_, paramt);

	param = ((1 - alpha) * paramt_) + (alpha * param0);
	timer.nextTime("updating param");

	/*
	  % save variables for next iteration
	  gammaO = gam
	  ma;
	  x0 = xt;

	  % update return value taking into account previous estimate
	  tmp = max(1, iter - burnPeriod);
	  xb = (1.0/tmp)*x + ((tmp-1)*1.0/tmp)*xb;
	*/
	gamma0 = gamma;
	param0 = paramt_;
	FP_TYPE tmp = ((i - burnPeriod) > 1) ? (i - burnPeriod) : 1.0;
	assert(tmp >= (FP_TYPE)0.999999);

	paramb = (safeDiv(tmp - (FP_TYPE)1.0, tmp))*paramb;
	paramb += safeDiv((FP_TYPE)1.0, tmp)*param;
	timer.nextTime("updating paramb");

	/*
	  if (eta > 0.3) {
	  std::cerr << "----------------i = " << i << std::endl;
	  std::cerr << idx1 << " " << idx2 << std::endl;
	  std::cerr << gf.minCoeff() << " " << gf.maxCoeff() <<"\n";
	  std::cerr << param.minCoeff() << " " << param.maxCoeff() <<"\n";
	  std::cerr << paramt.minCoeff() << " " << paramt.maxCoeff() <<"\n";
	  if (i == 10) exit(1);
	  }
	*/

	/*
	FP_TYPE eps = 1e-6;
	cilk::reducer< cilk::op_add<size_t> > nnz(0);
	for (std::ptrdiff_t i=0;i<paramb.rows()*paramb.cols();++i) {
	  if (std::abs(paramb.data()[i]) > eps)
	*nnz += 1;
	}
	std::cerr << "nnz/numel = " << (FP_TYPE)nnz.get_value()/(FP_TYPE)(paramb.rows() * paramb.cols()) << "\n";
	*/
	LOG_TRACE("norm(param) = " + std::to_string(paramb.norm()));
  }
  //typeMismatchAssign(param, paramb);
  param = paramb;
  eta = t;
}

void Bonsai::createOutputDirs(const std::string& dataDir, std::string& currResultsPath)
{
  time_t now = time(0);
  tm *ltm = localtime(&now);

  std::string append_path 
	= std::to_string(ltm->tm_hour) + "_" + std::to_string(ltm->tm_min)
	+ "_" + std::to_string(ltm->tm_sec) + "_" + std::to_string(ltm->tm_mday)
	+ "_" + std::to_string(1 + ltm->tm_mon);
  std::string resultsPath = dataDir + "/BonsaiResults/";
  currResultsPath = resultsPath + append_path;
  std::string paramsPath = currResultsPath + "/Params";

#if defined(_WIN32)
  _mkdir(resultsPath.c_str());
  _mkdir(currResultsPath.c_str());
  _mkdir(paramsPath.c_str());
#else 
  mkdir(resultsPath.c_str(), 0777);
  mkdir(currResultsPath.c_str(), 0777);
  mkdir(paramsPath.c_str(), 0777);
#endif
}

void Bonsai::exitWithHelp()
{
  LOG_INFO("./Bonsai [Options] DataFolder \n");
  LOG_INFO("Options:");

  LOG_INFO("-F    : [Required] Number of features in the data.");
  LOG_INFO("-C    : [Required] Number of Classification Classes/Labels.");
  LOG_INFO("-nT   : [Required] Number of training examples.");
  LOG_INFO("-nE   : [Required] Number of examples in test file.");
  LOG_INFO("-f    : [Optional] Input format. Takes two values [0 and 1]. 0 is for libsvmFormat(default), 1 is for tab/space separated input.");

  LOG_INFO("-P   : [Optional] Projection Dimension. (Default: 10 Try: [5, 20, 30, 50]) ");
  LOG_INFO("-D   : [Optional] Depth of the Bonsai tree. (Default: 3 Try: [2, 4, 5])");
  LOG_INFO("-S   : [Optional] \\sigma = parameter for sigmoid sharpness  (Default: 1.0 Try: [3.0, 0.05, 0.005] ).");

  LOG_INFO("-lW  : [Optional] lW = regularizer for classifier parameter W  (Default: 0.0001 Try: [0.01, 0.001, 0.00001]).");
  LOG_INFO("-lT  : [Optional] lTheta = regularizer for kernel parameter Theta  (Default: 0.0001 Try: [0.01, 0.001, 0.00001]).");
  LOG_INFO("-lV  : [Optional] lV = regularizer for kernel parameters V  (Default: 0.0001 Try: [0.01, 0.001, 0.00001]).");
  LOG_INFO("-lZ  : [Optional] lZ = regularizer for kernel parameters Z  (Default: 0.00001 Try: [0.001, 0.0001, 0.000001]).");

  LOG_INFO("Use Sparsity Params to vary your model Size");
  LOG_INFO("-sW  : [Optional] lambdaW = sparsity for classifier parameter W  (Default: For Binary 1.0 else 0.2 Try: [0.1, 0.3, 0.4, 0.5]).");
  LOG_INFO("-sT  : [Optional] lambdaTheta = sparsity for kernel parameter Theta  (Default: For Binary 1.0 else 0.2 Try: [0.1, 0.3, 0.4, 0.5]).");
  LOG_INFO("-sV  : [Optional] lambdaV = sparsity for kernel parameters V  (Default: For Binary 1.0 else 0.2 Try: [0.1, 0.3, 0.4, 0.5]).");
  LOG_INFO("-sZ  : [Optional] lambdaZ = sparsity for kernel parameters Z  (Default: 0.2 Try: [0.1, 0.3, 0.4, 0.5]).");

  LOG_INFO("-I   : [Optional] [Default: 42 Try: [100, 30, 60]] Number of passes through the dataset.");
  LOG_INFO("-B   : [Optional] Batch Factor [Default: 1 Try: [2.5, 10, 100]] Float Factor to multiply with sqrt(ntrain) to make the batchSize = min(max(100, B*sqrt(nT)), nT).");
  LOG_INFO("DataFolder : [Required] Path to folder containing data with filenames being 'train.txt' and 'test.txt' in the folder.");
  LOG_INFO("\ntrain.txt is train data file with label followed by features, test.txt is test data file with label followed by features");
  LOG_INFO("Try to shuffle the 'train.txt' file before feeding it in.");
  LOG_INFO("Note  : Both libsvmFormat and Space/Tab separated format can be either Zero or One Indexed in labels. To use Zero Index enable ZERO_BASED_IO flag in config.mk and recompile Bonsai");
  exit(1);
}

void Bonsai::parseInput(const int& argc, const char** argv,
  EdgeML::Bonsai::BonsaiModel::BonsaiHyperParams& hyperParam, std::string& dataDir)
{
  if (argc < 10)
	exitWithHelp();
  int i;
  hyperParam.problemType = ProblemFormat::multiclass;
  hyperParam.dataformatType = DataFormat::libsvmFormat;
  hyperParam.normalizationType = NormalizationFormat::none;

  hyperParam.seed = 41;

  hyperParam.batchSize = 1;
  hyperParam.iters = 42;
  hyperParam.epochs = 1;
  hyperParam.batchFactor = 1.0;

  hyperParam.projectionDimension = 10;

  hyperParam.Sigma = 1;
  hyperParam.treeDepth = 3;

  int required = 0;

  hyperParam.regList.lW = (FP_TYPE)1.0e-4;
  hyperParam.regList.lZ = (FP_TYPE)1.0e-5;
  hyperParam.regList.lV = (FP_TYPE)1.0e-4;
  hyperParam.regList.lTheta = (FP_TYPE)1.0e-4;


  hyperParam.lambdaZ = (FP_TYPE)0.2;

  int tempFormat = -1;
  for (i = 1; i < argc; i++) {
	if (argv[i][0] != '-')
	  break;
	if (++i >= argc)
	  exitWithHelp();
	switch (argv[i - 1][1]) {
	case 'f':
	  tempFormat = int(atoi(argv[i]));
	  if (tempFormat == 0) hyperParam.dataformatType = DataFormat::libsvmFormat;
	  else if (tempFormat == 1) hyperParam.dataformatType = DataFormat::interfaceIngestFormat;
	  else exitWithHelp();
	  break;
	case 'B':
	  hyperParam.batchFactor = (FP_TYPE)atof(argv[i]);
	  break;
	case 'F':
	  hyperParam.dataDimension = int(atoi(argv[i]));
	  required++;
	  break;
	case 'C':
	  hyperParam.numClasses = int(atoi(argv[i]));
	  hyperParam.lambdaW = (hyperParam.numClasses == 2) ? (FP_TYPE)1.0 : hyperParam.lambdaZ;
	  hyperParam.lambdaV = hyperParam.lambdaW;
	  hyperParam.lambdaTheta = hyperParam.lambdaV;
	  required++;
	  break;
	case 'n':
	  switch (argv[i - 1][2]) {
	  case 'T':
		hyperParam.ntrain = int(atoi(argv[i]));
		required++;
		break;
	  case 'E':
		hyperParam.nvalidation = int(atoi(argv[i]));
		required++;
    hyperParam.ntest = 0;
		break;
	  }
	  break;
	case 'P':
	  hyperParam.projectionDimension = int(atoi(argv[i]));
	  break;
	case 'D':
	  hyperParam.treeDepth = int(atoi(argv[i]));
	  break;
	case 'S':
	  hyperParam.Sigma = (FP_TYPE)atof(argv[i]);
	  break;

	case 'I':
	  hyperParam.iters = atoi(argv[i]);
	  break;

	case 'l':
	  switch (argv[i - 1][2]) {
	  case 'T':
		hyperParam.regList.lTheta = (FP_TYPE)atof(argv[i]);
		break;
	  case 'W':
		hyperParam.regList.lW = (FP_TYPE)atof(argv[i]);
		break;
	  case 'V':
		hyperParam.regList.lV = (FP_TYPE)atof(argv[i]);
		break;
	  case 'Z':
		hyperParam.regList.lZ = (FP_TYPE)atof(argv[i]);
		break;
	  default:
		LOG_INFO("Unknown option: -%c\n" + std::to_string(argv[i - 1][2]));
		exitWithHelp();
		break;
	  }
	  break;

	case 's':
	  switch (argv[i - 1][2]) {
	  case 'T':
		hyperParam.lambdaTheta = (FP_TYPE)atof(argv[i]);
		break;
	  case 'W':
		hyperParam.lambdaW = (FP_TYPE)atof(argv[i]);
		break;
	  case 'V':
		hyperParam.lambdaV = (FP_TYPE)atof(argv[i]);
		break;
	  case 'Z':
		hyperParam.lambdaZ = (FP_TYPE)atof(argv[i]);
		break;
	  default:
		LOG_INFO("Unknown option: -%c\n" + std::to_string(argv[i - 1][2]));
		exitWithHelp();
		break;
	  }
	  break;

	default:
	  LOG_INFO("Unknown option: " + std::to_string(argv[i - 1][1]));
	  exitWithHelp();
	  break;
	}
  }
  if (i >= argc)
	exitWithHelp();
  dataDir = std::string(argv[i]);
  required++;

  if (required != 5) exitWithHelp();

#ifdef ZERO_BASED_IO
  hyperParam.isOneIndex = true;
#else
  hyperParam.isOneIndex = false;
#endif

  hyperParam.internalNodes = (1 << hyperParam.treeDepth) - 1;
  hyperParam.totalNodes = 2 * hyperParam.internalNodes + 1;
}
