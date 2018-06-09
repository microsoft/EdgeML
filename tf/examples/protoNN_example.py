import sys
import os
sys.path.insert(0, '../')
import numpy as np
import tensorflow as tf
from edgeml.trainer.protoNNTrainer import ProtoNNTrainer
from edgeml.graph.protoNN import ProtoNN

os.environ["CUDA_VISIBLE_DEVICES"] = ''


def loadData(dataDir):
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
    # -----------------
    # Configuration
    # -----------------
    DATA_DIR = './curet/'

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
    trainer = ProtoNNTrainer(protoNN, REG_W, REG_B, REG_Z,
                             SPAR_W, SPAR_B, SPAR_Z,
                             LEARNING_RATE, X, Y, lossType='xentropy')
    sess = tf.Session()
    trainer.train(16, 150, sess, x_train, x_test, y_train, y_test,
                  printStep=100)


if __name__ == '__main__':
    main()

'''
NOTES:
    1. Curet:
        Data dimension: 610
        num Classes: 61
'''
