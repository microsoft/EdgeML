# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''
 Functions to check sanity of input arguments
 for the example script.
'''
import argparse
import datetime
import os
import numpy as np


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


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def getArgs():
    '''
    Function to parse arguments for Bonsai Algorithm
    '''
    parser = argparse.ArgumentParser(
        description='HyperParams for Bonsai Algorithm')
    parser.add_argument('-dir', '--data-dir', required=True,
                        help='Data directory containing' +
                        'train.npy and test.npy')

    parser.add_argument('-d', '--depth', type=checkIntNneg, default=2,
                        help='Depth of Bonsai Tree ' +
                        '(default: 2 try: [0, 1, 3])')
    parser.add_argument('-p', '--proj-dim', type=checkIntPos, default=10,
                        help='Projection Dimension ' +
                        '(default: 20 try: [5, 10, 30])')
    parser.add_argument('-s', '--sigma', type=float, default=1.0,
                        help='Parameter for sigmoid sharpness ' +
                        '(default: 1.0 try: [3.0, 0.05, 0.1]')
    parser.add_argument('-e', '--epochs', type=checkIntPos, default=42,
                        help='Total Epochs (default: 42 try:[100, 150, 60])')
    parser.add_argument('-b', '--batch-size', type=checkIntPos,
                        help='Batch Size to be used ' +
                        '(default: max(100, sqrt(train_samples)))')
    parser.add_argument('-lr', '--learning-rate', type=checkFloatPos,
                        default=0.01, help='Initial Learning rate for ' +
                        'Adam Optimizer (default: 0.01)')

    parser.add_argument('-rW', type=float, default=0.0001,
                        help='Regularizer for predictor parameter W  ' +
                        '(default: 0.0001 try: [0.01, 0.001, 0.00001])')
    parser.add_argument('-rV', type=float, default=0.0001,
                        help='Regularizer for predictor parameter V  ' +
                        '(default: 0.0001 try: [0.01, 0.001, 0.00001])')
    parser.add_argument('-rT', type=float, default=0.0001,
                        help='Regularizer for branching parameter Theta  ' +
                        '(default: 0.0001 try: [0.01, 0.001, 0.00001])')
    parser.add_argument('-rZ', type=float, default=0.00001,
                        help='Regularizer for projection parameter Z  ' +
                        '(default: 0.00001 try: [0.001, 0.0001, 0.000001])')

    parser.add_argument('-sW', type=checkFloatPos,
                        help='Sparsity for predictor parameter W  ' +
                        '(default: For Binary classification 1.0 else 0.2 ' +
                        'try: [0.1, 0.3, 0.5])')
    parser.add_argument('-sV', type=checkFloatPos,
                        help='Sparsity for predictor parameter V  ' +
                        '(default: For Binary classification 1.0 else 0.2 ' +
                        'try: [0.1, 0.3, 0.5])')
    parser.add_argument('-sT', type=checkFloatPos,
                        help='Sparsity for branching parameter Theta  ' +
                        '(default: For Binary classification 1.0 else 0.2 ' +
                        'try: [0.1, 0.3, 0.5])')
    parser.add_argument('-sZ', type=checkFloatPos, default=0.2,
                        help='Sparsity for projection parameter Z  ' +
                        '(default: 0.2 try: [0.1, 0.3, 0.5])')
    parser.add_argument('-oF', '--output-file', default=None,
                        help='Output file for dumping the program output, ' +
                        '(default: stdout)')

    parser.add_argument('-regression', type=str2bool, default=False,
                        help='boolean argument which controls whether to perform ' +
                        'regression or classification.' +
                        'default : False (Classification) values: [True, False]')

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

    return parser.parse_args()


def createTimeStampDir(dataDir):
    '''
    Creates a Directory with timestamp as it's name
    '''
    if os.path.isdir(dataDir + '/TFBonsaiResults') is False:
        try:
            os.mkdir(dataDir + '/TFBonsaiResults')
        except OSError:
            print("Creation of the directory %s failed" %
                  dataDir + '/TFBonsaiResults')

    currDir = 'TFBonsaiResults/' + datetime.datetime.now().strftime("%H_%M_%S_%d_%m_%y")
    if os.path.isdir(dataDir + '/' + currDir) is False:
        try:
            os.mkdir(dataDir + '/' + currDir)
        except OSError:
            print("Creation of the directory %s failed" %
                  dataDir + '/' + currDir)
        else:
            return (dataDir + '/' + currDir)
    return None


def preProcessData(dataDir, isRegression=False):
    '''
    Function to pre-process input data
    Expects a .npy file of form [lbl feats] for each datapoint
    Outputs a train and test set datapoints appended with 1 for Bias induction
    dataDimension, numClasses are inferred directly
    '''
    train = np.load(dataDir + '/train.npy')
    test = np.load(dataDir + '/test.npy')

    dataDimension = int(train.shape[1]) - 1

    Xtrain = train[:, 1:dataDimension + 1]
    Ytrain_ = train[:, 0]

    Xtest = test[:, 1:dataDimension + 1]
    Ytest_ = test[:, 0]

    # Mean Var Normalisation
    mean = np.mean(Xtrain, 0)
    std = np.std(Xtrain, 0)
    std[std[:] < 0.000001] = 1
    Xtrain = (Xtrain - mean) / std
    Xtest = (Xtest - mean) / std
    # End Mean Var normalisation

    # Classification.
    if (isRegression == False):
        numClasses = max(Ytrain_) - min(Ytrain_) + 1
        numClasses = int(max(numClasses, max(Ytest_) - min(Ytest_) + 1))

        lab = Ytrain_.astype('uint8')
        lab = np.array(lab) - min(lab)

        lab_ = np.zeros((Xtrain.shape[0], numClasses))
        lab_[np.arange(Xtrain.shape[0]), lab] = 1
        if (numClasses == 2):
            Ytrain = np.reshape(lab, [-1, 1])
        else:
            Ytrain = lab_

        lab = Ytest_.astype('uint8')
        lab = np.array(lab) - min(lab)

        lab_ = np.zeros((Xtest.shape[0], numClasses))
        lab_[np.arange(Xtest.shape[0]), lab] = 1
        if (numClasses == 2):
            Ytest = np.reshape(lab, [-1, 1])
        else:
            Ytest = lab_

    elif (isRegression == True):
        # The number of classes is always 1, for regression.
        numClasses = 1
        Ytrain = Ytrain_
        Ytest = Ytest_

    trainBias = np.ones([Xtrain.shape[0], 1])
    Xtrain = np.append(Xtrain, trainBias, axis=1)
    testBias = np.ones([Xtest.shape[0], 1])
    Xtest = np.append(Xtest, testBias, axis=1)

    mean = np.append(mean, np.array([0]))
    std = np.append(std, np.array([1]))

    if (isRegression == False):
        return dataDimension + 1, numClasses, Xtrain, Ytrain, Xtest, Ytest, mean, std
    elif (isRegression == True):
        return dataDimension + 1, numClasses, Xtrain, Ytrain.reshape((-1, 1)), Xtest, Ytest.reshape((-1, 1)), mean, std


def dumpCommand(list, currDir):
    '''
    Dumps the current command to a file for further use
    '''
    commandFile = open(currDir + '/command.txt', 'w')
    command = "python"

    command = command + " " + ' '.join(list)
    commandFile.write(command)

    commandFile.flush()
    commandFile.close()


def saveMeanStd(mean, std, currDir):
    '''
    Function to save Mean and Std vectors
    '''
    np.save(currDir + '/mean.npy', mean)
    np.save(currDir + '/std.npy', std)
    saveMeanStdSeeDot(mean, std, currDir + "/SeeDot")


def saveMeanStdSeeDot(mean, std, seeDotDir):
    '''
    Function to save Mean and Std vectors
    '''
    if os.path.isdir(seeDotDir) is False:
        try:
            os.mkdir(seeDotDir)
        except OSError:
            print("Creation of the directory %s failed" %
                  seeDotDir)
    np.savetxt(seeDotDir + '/Mean', mean, delimiter="\t")
    np.savetxt(seeDotDir + '/Std', std, delimiter="\t")
