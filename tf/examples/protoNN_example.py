import sys
import os
import numpy as np
import tensorflow as tf
sys.path.insert(0, '../')
from edgeml.trainer.protoNNTrainer import ProtoNNTrainer
from edgeml.graph.protoNN import ProtoNN
import edgeml.utils as utils
import preprocess


def getModelSize(matrixList, sparcityList, expected=True, bytesPerVar=4):
    '''
    expected: Expected size according to the parameters set. The number of
        zeros could actually be more than that is requied to satisfy the
        sparcity constraint.
    '''
    nnzList, sizeList, isSparseList = [], [], []
    hasSparse = False
    for i in range(len(matrixList)):
        A, s = matrixList[i], sparcityList[i]
        assert A.ndim == 2
        assert s >= 0
        assert s <= 1
        nnz, size, sparse = utils.countnnZ(A, s, bytesPerVar=bytesPerVar)
        nnzList.append(nnz)
        sizeList.append(size)
        hasSparse = (hasSparse or sparse)

    totalnnZ = np.sum(nnzList)
    totalSize = np.sum(sizeList)
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


def getGamma(gammaInit, projectionDim, dataDim, numPrototypes, x_train):
    if gammaInit is None:
        if projectionDim > dataDim:
            print("Warning: Projection dimension > data dimension. Gamma")
            print("\t estimation due to median heuristic could fail.")
            print("\tTo retain the projection dataDimension, provide")
            print("\ta value for gamma.")
        print("Using median heuristc to estimate gamma.")
        gamma, W, B = utils.medianHeuristic(x_train, projectionDim,
                                            numPrototypes)
        print("Gamma estimate is: %f" % gamma)
        return W, B, gamma
    return None, None, gammaInit

def loadData(dataDir):
    '''
    Loads data from the dataDir and does some initial preprocessing
    steps. Data is assumed to be contained in two files,
    train.npy and test.npy. Each containing a 2D numpy array of dimension
    [numberOfExamples, numberOfFeatures + 1]. The first column of each
    matrix is assumed to contain label information.

    For an N-Class problem, we assume the labels are integers from 0 through
    N-1.
    '''
    train = np.load(dataDir + '/train.npy')
    test = np.load(dataDir + '/test.npy')

    dataDimension = int(train.shape[1]) - 1
    x_train = train[:, 1:dataDimension + 1]
    y_train_ = train[:, 0]
    x_test = test[:, 1:dataDimension + 1]
    y_test_ = test[:, 0]

    numClasses = max(y_train_) - min(y_train_) + 1
    numClasses = max(numClasses, max(y_test_) - min(y_test_) + 1)
    numClasses = int(numClasses)

    # mean-var
    mean = np.mean(x_train, 0)
    std = np.std(x_train, 0)
    std[std[:] < 0.000001] = 1
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # one hot y-train
    lab = y_train_.astype('uint8')
    lab = np.array(lab) - min(lab)
    lab_ = np.zeros((x_train.shape[0], numClasses))
    lab_[np.arange(x_train.shape[0]), lab] = 1
    y_train = lab_

    # one hot y-test
    lab = y_test_.astype('uint8')
    lab = np.array(lab) - min(lab)
    lab_ = np.zeros((x_test.shape[0], numClasses))
    lab_[np.arange(x_test.shape[0]), lab] = 1
    y_test = lab_

    return dataDimension, numClasses, x_train, y_train, x_test, y_test


def main():
    config = preprocess.getProtoNNArgs()
    # Get hyper parameters
    DATA_DIR =  config.data_dir
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

    # Load data
    out = loadData(DATA_DIR)
    dataDimension = out[0]
    numClasses = out[1]
    x_train, y_train = out[2], out[3]
    x_test, y_test = out[4], out[5]

    W, B, gamma = getGamma(config.gamma, PROJECTION_DIM, dataDimension,
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
    trainer.train(16, NUM_EPOCHS, sess, x_train, x_test, y_train, y_test,
                  printStep=200)

    # Print some summary metrics
    acc = sess.run(protoNN.accuracy, feed_dict={X: x_test, Y:y_test})
    # W, B, Z are tensorflow graph nodes
    W, B, Z, _ = protoNN.getModelMatrices()
    matrixList = sess.run([W, B, Z])
    sparcityList = [SPAR_W, SPAR_B, SPAR_Z]
    nnz, size, sparse = getModelSize(matrixList, sparcityList)
    print("Final test accuracy", acc)
    print("Model size constraint (Bytes): ", size)
    print("Number of non-zeros: ", nnz)
    nnz, size, sparse = getModelSize(matrixList, sparcityList, expected=False)
    print("Actual model size: ", size)
    print("Actual non-zeros: ", nnz)


if __name__ == '__main__':
    main()

'''
NOTES:
    1. Curet:
        python protoNN_example.py \
            --data-dir ./curet \
            --projection-dim 60 --num-prototypes 80 --gamma 0.0015 \
            --learning-rate 0.1 --epochs 200
        Expected test accuracy: 88%

        python protoNN_example.py \
            --data-dir ./curet \
            --projection-dim 60 --num-prototypes 60 --gamma 0.0015 \
            -rW 0.000005  -rB 0.0 -rZ 0.00005 -sW 0.8 \
            -sB 1.0 -sZ 1.0 \
            --learning-rate 0.05 --epochs 800
        Expected test accuracy: 89-90%
'''
