# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import print_function
import sys
import os
import numpy as np
import tensorflow as tf
sys.path.insert(0, '../../')
from edgeml.trainer.protoNNTrainer import ProtoNNTrainer
from edgeml.graph.protoNN import ProtoNN
import edgeml.utils as utils
import helpermethods as helper


def main():
    config = helper.getProtoNNArgs()
    # Get hyper parameters
    DATA_DIR = config.data_dir
    PROJECTION_DIM = config.projection_dim
    NUM_PROTOTYPES = config.num_prototypes
    REG_W = config.rW
    REG_B = config.rB
    REG_Z = config.rZ
    SPAR_W = config.sW
    SPAR_B = config.sB
    SPAR_Z = config.sZ
    LEARNING_RATE = config.learning_rate
    NUM_EPOCHS = config.epochs
    BATCH_SIZE = config.batch_size
    PRINT_STEP = config.print_step
    VAL_STEP = config.val_step
    OUT_DIR = config.out_dir

    # Load data
    x_train = np.load(DATA_DIR + '/x_train.npy')
    y_train = np.load(DATA_DIR + '/y_train.npy')
    x_test = np.load(DATA_DIR + '/x_test.npy')
    y_test = np.load(DATA_DIR + '/y_test.npy')
    dataDimension = x_train.shape[1]
    numClasses = y_train.shape[1]
    W, B, gamma = helper.getGamma(config.gamma, PROJECTION_DIM, dataDimension,
                                  NUM_PROTOTYPES, x_train)

    # Setup input and train protoNN
    X = tf.placeholder(tf.float32, [None, dataDimension], name='X')
    Y = tf.placeholder(tf.float32, [None, numClasses], name='Y')
    protoNN = ProtoNN(dataDimension, PROJECTION_DIM,
                      NUM_PROTOTYPES, numClasses,
                      gamma, W=W, B=B)
    trainer = ProtoNNTrainer(protoNN, REG_W, REG_B, REG_Z,
                             SPAR_W, SPAR_B, SPAR_Z,
                             LEARNING_RATE, X, Y, lossType='xentropy')
    sess = tf.Session()
    trainer.train(BATCH_SIZE, NUM_EPOCHS, sess, x_train, x_test,
                  y_train, y_test, printStep=PRINT_STEP, valStep=VAL_STEP)

    # Print some summary metrics
    acc = sess.run(protoNN.accuracy, feed_dict={X: x_test, Y: y_test})
    # W, B, Z are tensorflow graph nodes
    W, B, Z, gamma  = protoNN.getModelMatrices()
    matrixList = sess.run([W, B, Z])
    sparcityList = [SPAR_W, SPAR_B, SPAR_Z]
    nnz, size, sparse = helper.getModelSize(matrixList, sparcityList)
    print("Final test accuracy", acc)
    print("Model size constraint (Bytes): ", size)
    print("Number of non-zeros: ", nnz)
    nnz, size, sparse = helper.getModelSize(matrixList, sparcityList,
                                            expected=False)
    print("Actual model size: ", size)
    print("Actual non-zeros: ", nnz)
    print("Saving model matrices to: ", OUT_DIR)
    np.save(OUT_DIR + '/W.npy', W)
    np.save(OUT_DIR + '/B.npy', B)
    np.save(OUT_DIR + '/Z.npy', Z)
    np.save(OUT_DIR + '/gamma.npy', gamma)


if __name__ == '__main__':
    main()
