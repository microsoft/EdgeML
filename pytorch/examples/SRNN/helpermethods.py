# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse

def getSRNN2Args():
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

    parser = argparse.ArgumentParser(
        description='Hyperparameters for 2 layer SRNN Algorithm')

    parser.add_argument('-d', '--data-dir', required=True,
                        help='Directory containing processed data.')
    parser.add_argument('-h0', '--hidden-dim0', type=checkIntPos, default=64,
                        help='Hidden dimension of lower layer RNN cell.')
    parser.add_argument('-h1', '--hidden-dim1', type=checkIntPos, default=32,
                        help='Hidden dimension of upper layer RNN cell.')
    parser.add_argument('-bz', '--brick-size', type=checkIntPos, required=True,
                        help='Brick size to be used at the lower layer.')
    parser.add_argument('-c', '--cell-type', default='LSTM',
                        help='Type of RNN cell to use among [LSTM, FastRNN, ' +
                        'FastGRNN')

    parser.add_argument('-p', '--num-prototypes', type=checkIntPos, default=20,
                        help='Number of prototypes.')
    parser.add_argument('-g', '--gamma', type=checkFloatPos, default=None,
                        help='Gamma for Gaussian kernel. If not provided, ' +
                        'median heuristic will be used to estimate gamma.')

    parser.add_argument('-e', '--epochs', type=checkIntPos, default=10,
                        help='Total training epochs.')
    parser.add_argument('-b', '--batch-size', type=checkIntPos, default=128,
                        help='Batch size for each pass.')
    parser.add_argument('-r', '--learning-rate', type=checkFloatPos,
                        default=0.01,
                        help='Learning rate for ADAM Optimizer.')

    parser.add_argument('-pS', '--print-step', type=int, default=200,
                        help='The number of update steps between print ' +
                        'calls to console.')
    parser.add_argument('-vS', '--val-step', type=int, default=5,
                        help='The number of epochs between validation' +
                        'performance evaluation')
    return parser.parse_args()
