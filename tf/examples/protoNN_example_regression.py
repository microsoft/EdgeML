import argparse
import protoNNpreprocess_regression
import datetime
import pickle
import sys
import os
import pandas as pd
sys.path.insert(0, '../')
import numpy as np
import tensorflow as tf
from edgeml.trainer.protoNNTrainer_regression import ProtoNNTrainer
from edgeml.graph.protoNN_regression import ProtoNN
import matplotlib.pyplot as plt
import edgeml.utils as utils

np.random.seed(42)

def getModelSize(matrixList, sparcityList, expected=True, bytesPerVar=4):
    '''
    expected: Expected size according to the parameters set. The number of
              zeros could actually be more than the applied sparsity constraint.
    '''
    nnzList, sizeList, isSparseList = [], [], []
    hasSparse = False
    for i in range(len(matrixList)):
        A, s = matrixList[i], sparcityList[i]
        assert A.ndim == 2
        # 's - sparsity factor' should be between 0 and 1.
        assert s >= 0
        assert s <= 1
        nnz, size, sparse = utils.countnnZ(A, s, bytesPerVar=bytesPerVar)
        nnzList.append(nnz)
        sizeList.append(size)
        hasSparse = (hasSparse or sparse)

    totalnnZ = np.sum(nnzList)
    totalSize = np.sum(sizeList)
    #By default, go into this part, fails only when the bytesPerVar need to be changed or the sparsity factor.
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
            totalSize += A.size* bytesPerVar
    return numNonZero, totalSize, hasSparse


def loadData(dataDir):
    train = np.load(dataDir + '/train.npy')
    test = np.load(dataDir + '/test.npy')

    dataDimension = int(train.shape[1]) - 1
    x_train = train[:, 1:dataDimension + 1]
    y_train_ = train[:, 0]
    x_test = test[:, 1:dataDimension + 1]
    y_test_ = test[:, 0]

    #To use as a regressor.
    numClasses = 1

    # mean-var normalization.
    mean = np.mean(x_train, 0)
    std = np.std(x_train, 0)
    std[std[:] < 0.000001] = 1
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return dataDimension, numClasses, x_train, y_train_.reshape((-1,1)), x_test, y_test_.reshape((-1,1)),mean,std


def main(**kwargs):
    # -----------------
    # Configuration
    # -----------------
    #Get the directory path, as a command line argument.

    args = protoNNpreprocess_regression.getArgs()
    DATA_DIR = args.data_dir

    PROJECTION_DIM = args.projDim
    NUM_PROTOTYPES = args.num_proto
    GAMMA = args.gamma

    REG_W = args.rW
    REG_B = args.rB
    REG_Z = args.rZ

    # 1.0 implies dense matrix.
    SPAR_W = 1.0
    SPAR_B = 1.0
    SPAR_Z = 1.0
    batchSize = args.batchSize

    LEARNING_RATE = args.learningRate
    NUM_EPOCHS = args.num_epochs
    # -----------------
    # End configuration
    # -----------------

    out = loadData(DATA_DIR)
    dataDimension = out[0]
    numClasses = out[1]
    x_train, y_train = out[2], out[3]
    x_test, y_test = out[4], out[5]

    X = tf.placeholder(tf.float32, [None, dataDimension], name='X')
    Y = tf.placeholder(tf.float32, [None, numClasses], name='Y')

    protoNN = ProtoNN(dataDimension, PROJECTION_DIM,
                      NUM_PROTOTYPES, numClasses,
                      GAMMA)

    trainer = ProtoNNTrainer(DATA_DIR,protoNN, REG_W, REG_B, REG_Z,
                             SPAR_W, SPAR_B, SPAR_Z,
                             LEARNING_RATE, X, Y,lossType='l2')
    sess = tf.Session()
    sess.run(tf.group(tf.initialize_all_variables(),
                      tf.initialize_variables(tf.local_variables())))

    ndict = trainer.train(DATA_DIR,batchSize, NUM_EPOCHS, sess, x_train, x_test, y_train, y_test,
                  DATA_DIR,printStep=200)
    acc,g0 = sess.run([protoNN.accuracy,protoNN.gamma], feed_dict={X: x_test, Y:y_test})

    W, B, Z, _ = protoNN.getModelMatrices()
    print ("Final value of gamma : ",g0)
    matrixList = sess.run([W, B, Z])
    sparcityList = [SPAR_W, SPAR_B, SPAR_Z]
    nnz, size, sparse = getModelSize(matrixList, sparcityList)
    print("Final test accuracy", acc)
    print("Model size constraint (Bytes): ", size)
    print("Number of non-zeros: ", nnz)
    nnz, size, sparse = getModelSize(matrixList, sparcityList, expected=False)
    print("Actual model size: (KB) ", size/1024.0)
    print("Actual non-zeros: ", nnz)

if __name__ == '__main__':
    main()

'''
NOTES:
    1. Curet:
        Data dimension: 610
        num Classes: 61
        Reasonable parameters:
            Projection_Dim = 60
            PROJECTION_DIM = 60
            NUM_PROTOTYPES = 80
            GAMMA = 0.0015
            REG_W = 0.000
            REG_B = 0.000
            REG_Z = 0.000
            SPAR_W = 1.0
            SPAR_B = 1.0
            SPAR_Z = 1.0
            LEARNING_RATE = 0.1
            Expected test accuracy: 87-88%

            PROJECTION_DIM = 60
            NUM_PROTOTYPES = 60
            GAMMA = 0.0015
            REG_W = 0.000005
            REG_B = 0.000
            REG_Z = 0.00005
            SPAR_W = .8
            SPAR_B = 1.0
            SPAR_Z = 1.0
            LEARNING_RATE = 0.05
            Expected test accuracy: 89-90%
'''
