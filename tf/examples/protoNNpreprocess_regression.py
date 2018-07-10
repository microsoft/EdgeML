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
    if fvalue < 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive float value" % value)
    return fvalue

def getArgs():
    '''
    Function to parse arguments
    '''
    parser = argparse.ArgumentParser(
        description='HyperParams for ProtoNN Algorithm')
    parser.add_argument('-dir', '--data_dir', required=True,
                        help='Data directory containing' +
                        'train.npy and test.npy')

    parser.add_argument('-p', '--projDim', type=checkIntPos, default=5,
                        help='Projection Dimension (default: 5 try: [5, 20, 30])')

    parser.add_argument('-np', '--num_proto', type=int, default=80,
                        help='Parameter for number of prototypes. (default: 60 try: [45,75,100]')

    parser.add_argument('-g', '--gamma', type=float, default = 0.0015,
                        help='Gamma (default: 0.0015)')

    parser.add_argument('-e', '--num_epochs', type=int, default=100,
                        help='Num of epochs to be used (default : 200)')

    parser.add_argument('-lr', '--learningRate', type=checkFloatPos, default=0.05,
                        help='Initial Learning rate for Adam Optimizer (default: 0.05)')

    parser.add_argument('-b', '--batchSize', type=checkIntPos, default = 32,
                            help='Batch Size to be used (default: 32)')

    parser.add_argument('-rW', type=float, default=0.0,
                        help='Regularizer for W  (default: 0.0001 try: [0.01, 0.001, 0.00001])')

    parser.add_argument('-rB', type=float, default=0.0,
                        help='Regularizer for B  (default: 0.0001 try: [0.01, 0.001, 0.00001])')

    parser.add_argument('-rZ', type=float, default=0.0,
                        help='Regularizer for Z  (default: 0.00001 try: [0.001, 0.0001, 0.000001])')

    return parser.parse_args()
