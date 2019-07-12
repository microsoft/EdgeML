# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''
 Functions to check sanity of input arguments
 for the example script.
'''
import argparse
import bz2
import datetime
import json
import os

import numpy as np
import requests


def decompress(filepath):
    print("extracting: ", filepath)
    zipfile = bz2.BZ2File(filepath)  # open the file
    data = zipfile.read()  # get the decompressed data
    newfilepath = os.path.splitext(filepath)[0]  # assuming the filepath ends with .bz2
    with open(newfilepath, 'wb') as f:
        f.write(data)  # write a uncompressed file
    return newfilepath


def download_file(url, local_folder=None):
    """Downloads file pointed to by `url`.
    If `local_folder` is not supplied, downloads to the current folder.
    """
    filename = os.path.basename(url)
    if local_folder:
        filename = os.path.join(local_folder, filename)

    # Download the file
    print("Downloading: " + url)
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception("download file failed with status code: %d, fetching url '%s'" % (response.status_code, url))

    # Write the file to disk
    with open(filename, "wb") as handle:
        handle.write(response.content)
    return filename


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


def getArgs():
    '''
    Function to parse arguments for FastCells
    '''
    parser = argparse.ArgumentParser(
        description='HyperParams for Fast(G)RNN')
    parser.add_argument('-dir', '--data-dir', required=True,
                        help='Data directory containing' +
                        'train.npy and test.npy')

    parser.add_argument('-c', '--cell', type=str, default="FastGRNN",
                        help='Choose between [FastGRNN, FastRNN, UGRNN' +
                        ', GRU, LSTM], default: FastGRNN')

    parser.add_argument('-id', '--input-dim', type=checkIntNneg, required=True,
                        help='Input Dimension of RNN, each timestep will ' +
                        'feed input-dim features to RNN. ' +
                        'Total Feature length = Input Dim * Total Timestep')
    parser.add_argument('-hd', '--hidden-dim', type=checkIntNneg,
                        required=True, help='Hidden Dimension of RNN')

    parser.add_argument('-e', '--epochs', type=checkIntPos, default=300,
                        help='Total Epochs (default: 300 try:[100, 150, 600])')
    parser.add_argument('-b', '--batch-size', type=checkIntPos, default=100,
                        help='Batch Size to be used (default: 100)')
    parser.add_argument('-lr', '--learning-rate', type=checkFloatPos,
                        default=0.01, help='Initial Learning rate for ' +
                        'Adam Optimizer (default: 0.01)')

    parser.add_argument('-rW', '--wRank', type=checkIntPos, default=None,
                        help='Rank for the low-rank parameterisation of W, ' +
                        'None => Full Rank')
    parser.add_argument('-rU', '--uRank', type=checkIntPos, default=None,
                        help='Rank for the low-rank parameterisation of U, ' +
                        'None => Full Rank')

    parser.add_argument('-sW', type=checkFloatPos, default=1.0,
                        help='Sparsity for predictor parameter W(and both ' +
                        'W1 and W2 in low-rank)  ' +
                        '(default: 1.0(Dense) try: [0.1, 0.2, 0.3])')
    parser.add_argument('-sU', type=checkFloatPos, default=1.0,
                        help='Sparsity for predictor parameter U(and both ' +
                        'U1 and U2 in low-rank)  ' +
                        '(default: 1.0(Dense) try: [0.1, 0.2, 0.3])')

    parser.add_argument('-unl', '--update-nl', type=str, default="tanh",
                        help='Update non linearity. Choose between ' +
                        '[tanh, sigmoid, relu, quantTanh, quantSigm]. ' +
                        'default => tanh. Can add more in edgeml/graph/rnn.py')
    parser.add_argument('-gnl', '--gate-nl', type=str, default="sigmoid",
                        help='Gate non linearity. Choose between ' +
                        '[tanh, sigmoid, relu, quantTanh, quantSigm]. ' +
                        'default => sigmoid. Can add more in ' +
                        'edgeml/graph/rnn.py. Only Applicable to FastGRNN')

    parser.add_argument('-dS', '--decay-step', type=checkIntPos, default=200,
                        help='The interval (in epochs) after which the ' +
                        'learning rate should decay. ' +
                        'Default is 200 for 300 epochs')

    parser.add_argument('-dR', '--decay-rate', type=checkFloatPos, default=0.1,
                        help='The factor by which learning rate ' +
                        'should decay after each interval. Default 0.1')

    parser.add_argument('-oF', '--output-file', default=None,
                        help='Output file for dumping the program output, ' +
                        '(default: stdout)')

    return parser.parse_args()


def getQuantArgs():
    '''
    Function to parse arguments for Model Quantisation
    '''
    parser = argparse.ArgumentParser(
        description='Arguments for quantizing Fast models. ' +
        'Works only for piece-wise linear non-linearities, ' +
        'like relu, quantTanh, quantSigm (check rnn.py for the definitions)')
    parser.add_argument('-dir', '--model-dir', required=True,
                        help='model directory containing' +
                        '*.npy weight files dumped from the trained model')
    parser.add_argument('-m', '--max-val', type=checkIntNneg, default=127,
                        help='this represents the maximum possible value ' +
                        'in model, essentially the byte complexity, ' +
                        '127=> 1 byte is default')
    parser.add_argument('-s', '--scalar-scale', type=checkIntNneg,
                        default=1000, help='maximum granularity/decimals ' +
                        'you wish to get when quantising simple sclars ' +
                        'involved. Default is 1000')

    return parser.parse_args()


def createTimeStampDir(dataDir, cell):
    '''
    Creates a Directory with timestamp as it's name
    '''
    if os.path.isdir(os.path.join(dataDir, str(cell) + 'Results')) is False:
        try:
            os.mkdir(os.path.join(dataDir, str(cell) + 'Results'))
        except OSError:
            print("Creation of the directory %s failed" %
                  os.path.join(dataDir, str(cell) + 'Results'))

    currDir = os.path.join(str(cell) + 'Results',
        datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    if os.path.isdir(os.path.join(dataDir, currDir)) is False:
        try:
            os.mkdir(os.path.join(dataDir, currDir))
        except OSError:
            print("Creation of the directory %s failed" %
                  os.path.join(dataDir, currDir))
        else:
            return (os.path.join(dataDir, currDir))
    return None


def preProcessData(dataDir):
    '''
    Function to pre-process input data

    Expects a .npy file of form [lbl feats] for each datapoint,
    feats is timesteps*inputDims, flattened across timestep dimension.
    So input of 1st timestep followed by second and so on.

    Outputs train and test set datapoints
    dataDimension, numClasses are inferred directly
    '''
    train = np.load(os.path.join(dataDir, 'train.npy'))
    test = np.load(os.path.join(dataDir, 'test.npy'))

    dataDimension = int(train.shape[1]) - 1

    Xtrain = train[:, 1:dataDimension + 1]
    Ytrain_ = train[:, 0]
    numClasses = max(Ytrain_) - min(Ytrain_) + 1

    Xtest = test[:, 1:dataDimension + 1]
    Ytest_ = test[:, 0]

    numClasses = int(max(numClasses, max(Ytest_) - min(Ytest_) + 1))

    # Mean Var Normalisation
    mean = np.mean(Xtrain, 0)
    std = np.std(Xtrain, 0)
    std[std[:] < 0.000001] = 1
    Xtrain = (Xtrain - mean) / std

    Xtest = (Xtest - mean) / std
    # End Mean Var normalisation

    lab = Ytrain_.astype('uint8')
    lab = np.array(lab) - min(lab)

    lab_ = np.zeros((Xtrain.shape[0], numClasses))
    lab_[np.arange(Xtrain.shape[0]), lab] = 1
    Ytrain = lab_

    lab = Ytest_.astype('uint8')
    lab = np.array(lab) - min(lab)

    lab_ = np.zeros((Xtest.shape[0], numClasses))
    lab_[np.arange(Xtest.shape[0]), lab] = 1
    Ytest = lab_

    return dataDimension, numClasses, Xtrain, Ytrain, Xtest, Ytest, mean, std


def dumpCommand(list, currDir):
    '''
    Dumps the current command to a file for further use
    '''
    commandFile = open(os.path.join(currDir, 'command.txt'), 'w')
    command = "python"

    command = command + " " + ' '.join(list)
    commandFile.write(command)

    commandFile.flush()
    commandFile.close()


def saveMeanStd(mean, std, currDir):
    '''
    Function to save Mean and Std vectors
    '''
    np.save(os.path.join(currDir, 'mean.npy'), mean)
    np.save(os.path.join(currDir, 'std.npy'), std)


def saveJSon(data, filename):
    with open(filename, "w") as outfile:
        json.dump(data, outfile, indent=2)
