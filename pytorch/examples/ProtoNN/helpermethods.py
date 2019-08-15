# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import print_function
import sys
import os
import numpy as np
import pytorch_edgeml.utils as utils
import argparse


def getModelSize(matrixList, sparcityList, expected=True, bytesPerVar=4):
    '''
    expected: Expected size according to the parameters set. The number of
        zeros could actually be more than that is required to satisfy the
        sparsity constraint.
    '''
    nnzList, sizeList, isSparseList = [], [], []
    hasSparse = False
    for i in range(len(matrixList)):
        A, s = matrixList[i], sparcityList[i]
        assert A.ndim == 2
        assert s >= 0
        assert s <= 1
        nnz, size, sparse = utils.countnnZ(A, s, bytesPerVar=bytesPerVar)
        nnzList.append(nnz)
        sizeList.append(size)
        hasSparse = (hasSparse or sparse)

    totalnnZ = np.sum(nnzList)
    totalSize = np.sum(sizeList)
    if expected:
        return totalnnZ, totalSize, hasSparse
    numNonZero = 0
    totalSize = 0
    hasSparse = False
    for i in range(len(matrixList)):
        A, s = matrixList[i], sparcityList[i]
        numNonZero_ = np.count_nonzero(A)
        numNonZero += numNonZero_
        hasSparse = (hasSparse or (s < 0.5))
        if s <= 0.5:
            totalSize += numNonZero_ * 2 * bytesPerVar
        else:
            totalSize += A.size * bytesPerVar
    return numNonZero, totalSize, hasSparse


def getGamma(gammaInit, projectionDim, dataDim, numPrototypes, x_train):
    if gammaInit is None:
        print("Using median heuristic to estimate gamma.")
        gamma, W, B = utils.medianHeuristic(x_train, projectionDim,
                                            numPrototypes)
        print("Gamma estimate is: %f" % gamma)
        return W, B, gamma
    return None, None, gammaInit

def to_onehot(y, numClasses, minlabel = None):
    '''
    If the y labelling does not contain the minimum label info, use min-label to
    provide this value.
    '''
    lab = y.astype('uint8')
    if minlabel is None:
        minlabel = np.min(lab)
    minlabel = int(minlabel)
    lab = np.array(lab) - minlabel
    lab_ = np.zeros((y.shape[0], numClasses))
    lab_[np.arange(y.shape[0]), lab] = 1
    return lab_

def preprocessData(train, test):
    '''
    Loads data from the dataDir and does some initial preprocessing
    steps. Data is assumed to be contained in two files,
    train.npy and test.npy. Each containing a 2D numpy array of dimension
    [numberOfExamples, numberOfFeatures + 1]. The first column of each
    matrix is assumed to contain label information.

    For an N-Class problem, we assume the labels are integers from 0 through
    N-1.
    '''
    dataDimension = int(train.shape[1]) - 1
    x_train = train[:, 1:dataDimension + 1]
    y_train_ = train[:, 0]
    x_test = test[:, 1:dataDimension + 1]
    y_test_ = test[:, 0]

    numClasses = max(y_train_) - min(y_train_) + 1
    numClasses = max(numClasses, max(y_test_) - min(y_test_) + 1)
    numClasses = int(numClasses)

    # mean-var
    mean = np.mean(x_train, 0)
    std = np.std(x_train, 0)
    std[std[:] < 0.000001] = 1
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # one hot y-train
    lab = y_train_.astype('uint8')
    lab = np.array(lab) - min(lab)
    lab_ = np.zeros((x_train.shape[0], numClasses))
    lab_[np.arange(x_train.shape[0]), lab] = 1
    y_train = lab_

    # one hot y-test
    lab = y_test_.astype('uint8')
    lab = np.array(lab) - min(lab)
    lab_ = np.zeros((x_test.shape[0], numClasses))
    lab_[np.arange(x_test.shape[0]), lab] = 1
    y_test = lab_

    return dataDimension, numClasses, x_train, y_train, x_test, y_test



def getProtoNNArgs():
    def checkIntPos(value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(
                "%s is an invalid positive int value" % value)
        return ivalue

    def checkIntNneg(value):
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError(
                "%s is an invalid non-neg int value" % value)
        return ivalue

    def checkFloatNneg(value):
        fvalue = float(value)
        if fvalue < 0:
            raise argparse.ArgumentTypeError(
                "%s is an invalid non-neg float value" % value)
        return fvalue

    def checkFloatPos(value):
        fvalue = float(value)
        if fvalue <= 0:
            raise argparse.ArgumentTypeError(
                "%s is an invalid positive float value" % value)
        return fvalue

    '''
    Parse protoNN commandline arguments
    '''
    parser = argparse.ArgumentParser(
        description='Hyperparameters for ProtoNN Algorithm')

    msg = 'Data directory containing train and test data. The '
    msg += 'data is assumed to be saved as 2-D numpy matrices with '
    msg += 'names `train.npy` and `test.npy`, of dimensions\n'
    msg += '\t[numberOfInstances, numberOfFeatures + 1].\n'
    msg += 'The first column of each file is assumed to contain label information.'
    msg += ' For a N-class problem, labels are assumed to be integers from 0 to'
    msg += ' N-1 (inclusive).'
    parser.add_argument('-d', '--data-dir', required=True, help=msg)
    parser.add_argument('-l', '--projection-dim', type=checkIntPos, default=10,
                        help='Projection Dimension.')
    parser.add_argument('-p', '--num-prototypes', type=checkIntPos, default=20,
                        help='Number of prototypes.')
    parser.add_argument('-g', '--gamma', type=checkFloatPos, default=None,
                        help='Gamma for Gaussian kernel. If not provided, ' +
                        'median heuristic will be used to estimate gamma.')

    parser.add_argument('-e', '--epochs', type=checkIntPos, default=100,
                        help='Total training epochs.')
    parser.add_argument('-b', '--batch-size', type=checkIntPos, default=32,
                        help='Batch size for each pass.')
    parser.add_argument('-r', '--learning-rate', type=checkFloatPos,
                        default=0.001,
                        help='Initial Learning rate for ADAM Optimizer.')

    parser.add_argument('-rW', type=float, default=0.000,
                        help='Coefficient for l2 regularizer for predictor' +
                        ' parameter W ' + '(default = 0.0).')
    parser.add_argument('-rB', type=float, default=0.00,
                        help='Coefficient for l2 regularizer for predictor' +
                        ' parameter B ' + '(default = 0.0).')
    parser.add_argument('-rZ', type=float, default=0.00,
                        help='Coefficient for l2 regularizer for predictor' +
                        'parameter Z ' +
                        '(default = 0.0).')

    parser.add_argument('-sW', type=float, default=1.000,
                        help='Sparsity constraint for predictor parameter W ' +
                        '(default = 1.0, i.e. dense matrix).')
    parser.add_argument('-sB', type=float, default=1.00,
                        help='Sparsity constraint for predictor parameter B ' +
                        '(default = 1.0, i.e. dense matrix).')
    parser.add_argument('-sZ', type=float, default=1.00,
                        help='Sparsity constraint for predictor parameter Z ' +
                        '(default = 1.0, i.e. dense matrix).')
    parser.add_argument('-pS', '--print-step', type=int, default=200,
                        help='The number of update steps between print ' +
                        'calls to console.')
    parser.add_argument('-vS', '--val-step', type=int, default=3,
                        help='The number of epochs between validation' +
                        'performance evaluation')
    parser.add_argument('-o', '--output-dir', type=str, default='./',
                        help='Output directory to dump model matrices.')
    return parser.parse_args()
