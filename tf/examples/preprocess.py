# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''
 Functions to check sanity of input arguments
 for the example script.
'''
import argparse
import numpy as np
import datetime
import os


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


def getBonsaiArgs():
    '''
    Function to parse arguments for Bonsai Algorithm
    '''
    parser = argparse.ArgumentParser(
        description='HyperParams for Bonsai Algorithm')
    parser.add_argument('-dir', '--data-dir', required=True,
                        help='Data directory containing' +
                        'train.npy and test.npy')

    parser.add_argument('-d', '--depth', type=checkIntNneg, default=2,
                        help='Depth of Bonsai Tree (default: 2 try: [0, 1, 3])')
    parser.add_argument('-p', '--proj-dim', type=checkIntPos, default=10,
                        help='Projection Dimension (default: 20 try: [5, 10, 30])')
    parser.add_argument('-s', '--sigma', type=float, default=1.0,
                        help='Parameter for sigmoid sharpness (default: 1.0 try: [3.0, 0.05, 0.1]')
    parser.add_argument('-e', '--epochs', type=checkIntPos, default=42,
                        help='Total Epochs (default: 42 try:[100, 150, 60])')
    parser.add_argument('-b', '--batch-size', type=checkIntPos,
                        help='Batch Size to be used (default: max(100, sqrt(train_samples)))')
    parser.add_argument('-lr', '--learning-rate', type=checkFloatPos, default=0.01,
                        help='Initial Learning rate for Adam Optimizer (default: 0.01)')

    parser.add_argument('-rW', type=float, default=0.0001,
                        help='Regularizer for predictor parameter W  (default: 0.0001 try: [0.01, 0.001, 0.00001])')
    parser.add_argument('-rV', type=float, default=0.0001,
                        help='Regularizer for predictor parameter V  (default: 0.0001 try: [0.01, 0.001, 0.00001])')
    parser.add_argument('-rT', type=float, default=0.0001,
                        help='Regularizer for branching parameter Theta  (default: 0.0001 try: [0.01, 0.001, 0.00001])')
    parser.add_argument('-rZ', type=float, default=0.00001,
                        help='Regularizer for projection parameter Z  (default: 0.00001 try: [0.001, 0.0001, 0.000001])')

    parser.add_argument('-sW', type=checkFloatPos,
                        help='Sparsity for predictor parameter W  (default: For Binary classification 1.0 else 0.2 try: [0.1, 0.3, 0.5])')
    parser.add_argument('-sV', type=checkFloatPos,
                        help='Sparsity for predictor parameter V  (default: For Binary classification 1.0 else 0.2 try: [0.1, 0.3, 0.5])')
    parser.add_argument('-sT', type=checkFloatPos,
                        help='Sparsity for branching parameter Theta  (default: For Binary classification 1.0 else 0.2 try: [0.1, 0.3, 0.5])')
    parser.add_argument('-sZ', type=checkFloatPos, default=0.2,
                        help='Sparsity for projection parameter Z  (default: 0.2 try: [0.1, 0.3, 0.5])')
    parser.add_argument('-oF', '--output-file', default=None,
                        help='Output file for dumping the program output, (default: stdout)')

    return parser.parse_args()


def getProtoNNArgs():
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
                        help='Coefficient for l2 regularizer for predictor parameter W ' +
                        '(default = 0.0).')
    parser.add_argument('-rB', type=float, default=0.00,
                        help='Coefficient for l2 regularizer for predictor parameter B ' +
                        '(default = 0.0).')
    parser.add_argument('-rZ', type=float, default=0.00,
                        help='Coefficient for l2 regularizer for predictor parameter Z ' +
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
    return parser.parse_args()


def createTimeStampDir(dataDir):
    '''
    Creates a Directory with timestamp as it's name
    '''
    currDir = datetime.datetime.now().strftime("%H_%M_%S_%d_%m_%y")
    if os.path.isdir(dataDir + '/' + currDir) is False:
        try:
            os.mkdir(dataDir + '/' + currDir)
        except OSError:
            print("Creation of the directory %s failed" %
                  dataDir + '/' + currDir)
        else:
            return (dataDir + '/' + currDir)
    return None
