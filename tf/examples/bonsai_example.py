from __future__ import print_function
import preprocess
import tensorflow as tf
import numpy as np
import multiprocessing
import os
import sys
import pickle
sys.path.insert(0, '../')

from edgeml.trainer.bonsaiTrainer import BonsaiTrainer
from edgeml.graph.bonsai import Bonsai

# Fixing seeds for reproducibility
tf.set_random_seed(42)
np.random.seed(42)


def preProcessData(dataDir,isRegression = False):
    '''
        Function to pre-process input data
        Expects a .npy file of form [lbl feats] for each datapoint
        Outputs a train and test set datapoints appended with 1 for Bias induction
        dataDimension, numClasses are inferred directly
    '''
    train = np.load(data_dir + '/train.npy')
    test = np.load(data_dir + '/test.npy')

    dataDimension = int(train.shape[1]) - 1

    Xtrain = train[:, 1:dataDimension + 1]
    Ytrain_ = train[:, 0]
    Ytrain = Ytrain_

    Xtest = test[:, 1:dataDimension + 1]
    Ytest_ = test[:, 0]
    Ytest = Ytest_

    # Mean Var Normalisation
    mean = np.mean(Xtrain, 0)
    std = np.std(Xtrain, 0)
    std[std[:] < 0.000001] = 1
    Xtrain = (Xtrain - mean) / std
    Xtest = (Xtest - mean) / std
    # End Mean Var normalisation

    #Classification.
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
        numClasses = 1
        Ytrain = Ytrain_
        Ytest = Ytest_

    trainBias = np.ones([Xtrain.shape[0], 1])
    Xtrain = np.append(Xtrain, trainBias, axis=1)
    testBias = np.ones([Xtest.shape[0], 1])
    Xtest = np.append(Xtest, testBias, axis=1)

    if (isRegression == False):
        return dataDimension + 1, numClasses, Xtrain, Ytrain, Xtest, Ytest, mean, std
    elif (isRegression ==True):
        return dataDimension + 1, numClasses, Xtrain, Ytrain.reshape((-1,1)), Xtest, Ytest.reshape((-1,1)), mean, std


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

# Hyper Param pre-processing
args = preprocess.bonsai_getArgs()
isRegression = None

# The only argument that needs to be changed to change between regression and classification is , 'isRegression'.
if (args.regression is not None):
    isRegression = args.regression
else:
    isRegression = False

sigma = args.sigma
depth = args.depth

projectionDimension = args.projDim
regZ = args.rZ
#regT = args.rT
regT = args.rW

regW = args.rW

#regV = args.rV
regV = args.rW

totalEpochs = args.epochs

learningRate = args.learningRate

data_dir = args.data_dir

outFile = args.output_file
print ("outFile : ",outFile)

(dataDimension, numClasses,
    Xtrain, Ytrain, Xtest, Ytest,mean,std) = preProcessData(data_dir,isRegression)

sparZ = args.sZ

if numClasses > 2:
    sparW = 0.2
    sparV = 0.2
    sparT = 0.2
else:
    sparW = 1
    sparV = 1
    sparT = 1

if args.sW is not None:
    sparW = args.sW
if args.sV is not None:
    sparV = args.sV
if args.sT is not None:
    sparT = args.sT

if args.batchSize is None:
    batchSize = np.maximum(100, int(np.ceil(np.sqrt(Ytrain.shape[0]))))
else:
    batchSize = args.batchSize

useMCHLoss = False

loss = args.loss
print ("Loss : ",loss)

if numClasses == 2:
    numClasses = 1

X = tf.placeholder("float32", [None, dataDimension])
Y = tf.placeholder("float32", [None, numClasses])

#currDir = bonsaipreprocess.createDir(data_dir)
currDir = data_dir

print ("Num of classes : ",numClasses)
# numClasses = 1 for binary case
bonsaiObj = Bonsai(numClasses, dataDimension,
                   projectionDimension, depth, sigma,isRegression = isRegression)


split = int(0.8*(np.vstack((Xtrain,Xtest)).shape[0]))
print ("Total Size : ",np.vstack((Xtrain,Xtest)).shape)
print ("Split : ",split,"\n\n")

bonsaiTrainer = BonsaiTrainer(bonsaiObj,
                              regW, regT, regV, regZ,
                              sparW, sparT, sparV, sparZ,
                              learningRate, X, Y, split,useMCHLoss, outFile, isRegression = isRegression,reg_loss = loss)

sess = tf.InteractiveSession()
sess.run(tf.group(tf.initialize_all_variables(),
                  tf.initialize_variables(tf.local_variables())))
saver = tf.train.Saver()

print("DONE")
dict = bonsaiTrainer.train(batchSize, totalEpochs, sess,
                    Xtrain, Xtest, Ytrain, Ytest, split,data_dir, currDir,mean,std)


sess.close()
sys.stdout.close()

#bonsaiObj.saveModel(currDir)

# For the following command:
# Data - Curet
# python train.py -dir ./curet/ -d 2 -p 22 -rW 0.00001 -rZ 0.0000001 -rV 0.00001 -rT 0.000001 -sZ 0.4 -sW 0.5 -sV 0.5 -sT 1 -e 300 -s 0.1 -b 20
# Final Output
# Maximum Test accuracy at compressed model size(including early stopping): 0.94583 at Epoch: 157
# Final Test Accuracy: 0.92516
# Non-Zeros: 118696 Model Size: 115.9140625 KB

# Data - usps2
# python train.py -dir ../../../../../../deepBonsai/DeepBonsai/Bonsai_tf/data/usps2/ -d 2 -p 22 -rW 0.00001 -rZ 0.0000001 -rV 0.00001 -rT 0.000001 -sZ 0.4 -sW 0.5 -sV 0.5 -sT 1 -e 300 -s 0.1 -b 20
# Maximum Test accuracy at compressed model size(including early stopping): 0.960638 at Epoch: 249
# Final Test Accuracy: 0.951171
# Non-Zeros: 19592.0 Model Size: 19.1328125 KB hasSparse: True
