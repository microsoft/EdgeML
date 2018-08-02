# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import preprocess
import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '../')

from edgeml.trainer.FastTrainer import FastTrainer
from edgeml.graph.rnn import FastGRNNCell
from edgeml.graph.rnn import FastRNNCell


def preProcessData(dataDir):
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

    return dataDimension, numClasses, Xtrain, Ytrain, Xtest, Ytest


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

# Fixing seeds for reproducibility
tf.set_random_seed(42)
np.random.seed(42)

# Hyper Param pre-processing
args = preprocess.getFastArgs()

dataDir = args.data_dir
cell = args.cell
inputDims = args.input_dim
hiddenDims = args.hidden_dim

totalEpochs = args.epochs
learningRate = args.learning_rate
outFile = args.output_file
batchSize = args.batch_size

wRank = args.wRank
uRank = args.uRank

sW = args.sW
sU = args.sU

update_non_linearity = args.update_nl
gate_non_linearity = args.gate_nl

(dataDimension, numClasses,
    Xtrain, Ytrain, Xtest, Ytest) = preProcessData(dataDir)
batchSize = 100

assert dataDimension % inputDims == 0, "Infeasible per step input, Timesteps have to be integer"

X = tf.placeholder("float", [None, int(dataDimension / inputDims), inputDims])
Y = tf.placeholder("float", [None, numClasses])

currDir = preprocess.createTimeStampDirFast(dataDir, cell)

dumpCommand(sys.argv, currDir)

if cell == "FastGRNN":
    FastCell = FastGRNNCell(hiddenDims, gate_non_linearity=gate_non_linearity,
                            update_non_linearity=update_non_linearity,
                            wRank=wRank, uRank=uRank)
elif cell == "FastRNN":
    FastCell = FastRNNCell(hiddenDims, update_non_linearity=update_non_linearity,
                           wRank=wRank, uRank=uRank)
else:
    sys.exit('Exiting: No Such Cell as ' + cell)


FastCellTrainer = FastTrainer(
    FastCell, X, Y, sW=sW, sU=sU, learningRate=learningRate, outFile=outFile)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

FastCellTrainer.train(batchSize, totalEpochs, sess, Xtrain, Xtest,
                      Ytrain, Ytest, dataDir, currDir)
