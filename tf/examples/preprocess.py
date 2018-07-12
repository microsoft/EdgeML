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
    parser.add_argument('-dir', '--data_dir', required=True,
                        help='Data directory containing' +
                        'train.npy and test.npy')

    parser.add_argument('-d', '--depth', type=checkIntNneg, default=2,
                        help='Depth of Bonsai Tree (default: 2 try: [0, 1, 3])')
    parser.add_argument('-p', '--projDim', type=checkIntPos, default=10,
                        help='Projection Dimension (default: 20 try: [5, 10, 30])')
    parser.add_argument('-s', '--sigma', type=float, default=1.0,
                        help='Parameter for sigmoid sharpness (default: 1.0 try: [3.0, 0.05, 0.1]')
    parser.add_argument('-e', '--epochs', type=checkIntPos, default=42,
                        help='Total Epochs (default: 42 try:[100, 150, 60])')
    parser.add_argument('-b', '--batchSize', type=checkIntPos,
                        help='Batch Size to be used (default: max(100, sqrt(train_samples)))')
    parser.add_argument('-lr', '--learningRate', type=checkFloatPos, default=0.01,
                        help='Initial Learning rate for Adam Oprimizer (default: 0.01)')

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
    parser.add_argument('-oF', '--output_file', default=None,
                        help='Output file for dumping the program output, (default: stdout)')

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
