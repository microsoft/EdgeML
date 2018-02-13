import tensorflow as tf
import numpy as np
import argparse

# Functions to check sanity of input arguments


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

# Function to parse the input arguments


def getArgs():
    parser = argparse.ArgumentParser(
        description='HyperParams for Bonsai Algorithm')
    parser.add_argument('-dir', '--data_dir', required=True,
                        help='Data directory containing train.npy and test.npy')

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

    return parser.parse_args()

# Function for Multi Class Hinge Loss : TF has no internal implementation


def multiClassHingeLoss(logits, label, batch_th):
    flatLogits = tf.reshape(logits, [-1, ])
    correctId = tf.range(0, batch_th) * logits.shape[1] + label
    correctLogit = tf.gather(flatLogits, correctId)

    maxLabel = tf.argmax(logits, 1)
    top2, _ = tf.nn.top_k(logits, k=2, sorted=True)

    wrongMaxLogit = tf.where(tf.equal(maxLabel, label), top2[:, 1], top2[:, 0])

    return tf.reduce_mean(tf.nn.relu(1. + wrongMaxLogit - correctLogit))

# Function for cross entropy loss in multiclass case (for faster convergence in joint training)


def crossEntropyLoss(logits, label):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))

# Hard Thresholding Function


def hardThreshold(A, s):
    A_ = np.copy(A)
    A_ = A_.ravel()
    if len(A_) > 0:
        th = np.percentile(np.abs(A_), (1 - s) * 100.0, interpolation='higher')
        A_[np.abs(A_) < th] = 0.0
    A_ = A_.reshape(A.shape)
    return A_

# Copy the support of src onto dest tensor


def copySupport(src, dest):
    support = np.nonzero(src)
    dest_ = dest
    dest = np.zeros(dest_.shape)
    dest[support] = dest_[support]
    return dest

# Function to read and pre-process data


def preProcessData(data_dir):
    train = np.load(data_dir + '/train.npy')
    test = np.load(data_dir + '/test.npy')

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

    trainBias = np.ones([Xtrain.shape[0], 1])
    Xtrain = np.append(Xtrain, trainBias, axis=1)
    testBias = np.ones([Xtest.shape[0], 1])
    Xtest = np.append(Xtest, testBias, axis=1)

    if numClasses == 2:
        numClasses = 1

    return dataDimension + 1, numClasses, Xtrain, Ytrain, Xtest, Ytest
