# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import scipy.cluster
import scipy.spatial
import os


def medianHeuristic(data, projectionDimension, numPrototypes, W_init=None):
    '''
    This method can be used to estimate gamma for ProtoNN. An approximation to
    median heuristic is used here.
    1. First the data is collapsed into the projectionDimension by W_init. If
    W_init is not provided, it is initialized from a random normal(0, 1). Hence
    data normalization is essential.
    2. Prototype are computed by running a  k-means clustering on the projected
    data.
    3. The median distance is then estimated by calculating median distance
    between prototypes and projected data points.

    data needs to be [-1, numFeats]
    If using this method to initialize gamma, please use the W and B as well.

    TODO: Return estimate of Z (prototype labels) based on cluster centroids
    andand labels

    TODO: Clustering fails due to singularity error if projecting upwards

    W [dxd_cap]
    B [d_cap, m]
    returns gamma, W, B
    '''
    assert data.ndim == 2
    X = data
    featDim = data.shape[1]
    if projectionDimension > featDim:
        print("Warning: Projection dimension > feature dimension. Gamma")
        print("\t estimation due to median heuristic could fail.")
        print("\tTo retain the projection dataDimension, provide")
        print("\ta value for gamma.")

    if W_init is None:
        W_init = np.random.normal(size=[featDim, projectionDimension])
    W = W_init
    XW = np.matmul(X, W)
    assert XW.shape[1] == projectionDimension
    assert XW.shape[0] == len(X)
    # Requires [N x d_cap] data matrix of N observations of d_cap-dimension and
    # the number of centroids m. Returns, [n x d_cap] centroids and
    # elementwise center information.
    B, centers = scipy.cluster.vq.kmeans2(XW, numPrototypes)
    # Requires two matrices. Number of observations x dimension of observation
    # space. Distances[i,j] is the distance between XW[i] and B[j]
    distances = scipy.spatial.distance.cdist(XW, B, metric='euclidean')
    distances = np.reshape(distances, [-1])
    gamma = np.median(distances)
    gamma = 1 / (2.5 * gamma)
    return gamma.astype('float32'), W.astype('float32'), B.T.astype('float32')


def multiClassHingeLoss(logits, label, batch_th):
    '''
    MultiClassHingeLoss to match C++ Version - No TF internal version
    '''
    flatLogits = tf.reshape(logits, [-1, ])
    label_ = tf.argmax(label, 1)

    correctId = tf.range(0, batch_th) * label.shape[1] + label_
    correctLogit = tf.gather(flatLogits, correctId)

    maxLabel = tf.argmax(logits, 1)
    top2, _ = tf.nn.top_k(logits, k=2, sorted=True)

    wrongMaxLogit = tf.where(
        tf.equal(maxLabel, label_), top2[:, 1], top2[:, 0])

    return tf.reduce_mean(tf.nn.relu(1. + wrongMaxLogit - correctLogit))


def crossEntropyLoss(logits, label):
    '''
    Cross Entropy loss for MultiClass case in joint training for
    faster convergence
    '''
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                   labels=tf.stop_gradient(label)))


def mean_absolute_error(logits, label):
    '''
    Function to compute the mean absolute error.
    '''
    return tf.reduce_mean(tf.abs(tf.subtract(logits, label)))


def hardThreshold(A, s):
    '''
    Hard thresholding function on Tensor A with sparsity s
    '''
    A_ = np.copy(A)
    A_ = A_.ravel()
    if len(A_) > 0:
        th = np.percentile(np.abs(A_), (1 - s) * 100.0, interpolation='higher')
        A_[np.abs(A_) < th] = 0.0
    A_ = A_.reshape(A.shape)
    return A_


def copySupport(src, dest):
    '''
    copy support of src tensor to dest tensor
    '''
    support = np.nonzero(src)
    dest_ = dest
    dest = np.zeros(dest_.shape)
    dest[support] = dest_[support]
    return dest


def countnnZ(A, s, bytesPerVar=4):
    '''
    Returns # of non-zeros and representative size of the tensor
    Uses dense for s >= 0.5 - 4 byte
    Else uses sparse - 8 byte
    '''
    params = 1
    hasSparse = False
    for i in range(0, len(A.shape)):
        params *= int(A.shape[i])
    if s < 0.5:
        nnZ = np.ceil(params * s)
        hasSparse = True
        return nnZ, nnZ * 2 * bytesPerVar, hasSparse
    else:
        nnZ = params
        return nnZ, nnZ * bytesPerVar, hasSparse


def getConfusionMatrix(predicted, target, numClasses):
    '''
    Returns a confusion matrix for a multiclass classification
    problem. `predicted` is a 1-D array of integers representing
    the predicted classes and `target` is the target classes.

    confusion[i][j]: Number of elements of class j
        predicted as class i
    Labels are assumed to be in range(0, numClasses)
    Use`printFormattedConfusionMatrix` to echo the confusion matrix
    in a user friendly form.
    '''
    assert(predicted.ndim == 1)
    assert(target.ndim == 1)
    arr = np.zeros([numClasses, numClasses])

    for i in range(len(predicted)):
        arr[predicted[i]][target[i]] += 1
    return arr


def printFormattedConfusionMatrix(matrix):
    '''
    Given a 2D confusion matrix, prints it in a human readable way.
    The confusion matrix is expected to be a 2D numpy array with
    square dimensions
    '''
    assert(matrix.ndim == 2)
    assert(matrix.shape[0] == matrix.shape[1])
    RECALL = 'Recall'
    PRECISION = 'PRECISION'
    print("|%s|" % ('True->'), end='')
    for i in range(matrix.shape[0]):
        print("%7d|" % i, end='')
    print("%s|" % 'Precision')

    print("|%s|" % ('-' * len(RECALL)), end='')
    for i in range(matrix.shape[0]):
        print("%s|" % ('-' * 7), end='')
    print("%s|" % ('-' * len(PRECISION)))

    precisionlist = np.sum(matrix, axis=1)
    recalllist = np.sum(matrix, axis=0)
    precisionlist = [matrix[i][i] / x if x !=
                     0 else -1 for i, x in enumerate(precisionlist)]
    recalllist = [matrix[i][i] / x if x !=
                  0 else -1 for i, x in enumerate(recalllist)]
    for i in range(matrix.shape[0]):
        # len recall = 6
        print("|%6d|" % (i), end='')
        for j in range(matrix.shape[0]):
            print("%7d|" % (matrix[i][j]), end='')
        print("%s" % (" " * (len(PRECISION) - 7)), end='')
        if precisionlist[i] != -1:
            print("%1.5f|" % precisionlist[i])
        else:
            print("%7s|" % "nan")

    print("|%s|" % ('-' * len(RECALL)), end='')
    for i in range(matrix.shape[0]):
        print("%s|" % ('-' * 7), end='')
    print("%s|" % ('-' * len(PRECISION)))
    print("|%s|" % ('Recall'), end='')

    for i in range(matrix.shape[0]):
        if recalllist[i] != -1:
            print("%1.5f|" % (recalllist[i]), end='')
        else:
            print("%7s|" % "nan", end='')

    print('%s|' % (' ' * len(PRECISION)))


def getPrecisionRecall(cmatrix, label=1):
    trueP = cmatrix[label][label]
    denom = np.sum(cmatrix, axis=0)[label]
    if denom == 0:
        denom = 1
    recall = trueP / denom
    denom = np.sum(cmatrix, axis=1)[label]
    if denom == 0:
        denom = 1
    precision = trueP / denom
    return precision, recall


def getMacroPrecisionRecall(cmatrix):
    # TP + FP
    precisionlist = np.sum(cmatrix, axis=1)
    # TP + FN
    recalllist = np.sum(cmatrix, axis=0)
    precisionlist__ = [cmatrix[i][i] / x if x !=
                       0 else 0 for i, x in enumerate(precisionlist)]
    recalllist__ = [cmatrix[i][i] / x if x !=
                    0 else 0 for i, x in enumerate(recalllist)]
    precision = np.sum(precisionlist__)
    precision /= len(precisionlist__)
    recall = np.sum(recalllist__)
    recall /= len(recalllist__)
    return precision, recall


def getMicroPrecisionRecall(cmatrix):
    # TP + FP
    precisionlist = np.sum(cmatrix, axis=1)
    # TP + FN
    recalllist = np.sum(cmatrix, axis=0)
    num = 0.0
    for i in range(len(cmatrix)):
        num += cmatrix[i][i]

    precision = num / np.sum(precisionlist)
    recall = num / np.sum(recalllist)
    return precision, recall


def getMacroMicroFScore(cmatrix):
    '''
    Returns macro and micro f-scores.
    Refer: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.8244&rep=rep1&type=pdf
    '''
    precisionlist = np.sum(cmatrix, axis=1)
    recalllist = np.sum(cmatrix, axis=0)
    precisionlist__ = [cmatrix[i][i] / x if x !=
                       0 else 0 for i, x in enumerate(precisionlist)]
    recalllist__ = [cmatrix[i][i] / x if x !=
                    0 else 0 for i, x in enumerate(recalllist)]
    macro = 0.0
    for i in range(len(precisionlist)):
        denom = precisionlist__[i] + recalllist__[i]
        numer = precisionlist__[i] * recalllist__[i] * 2
        if denom == 0:
            denom = 1
        macro += numer / denom
    macro /= len(precisionlist)

    num = 0.0
    for i in range(len(precisionlist)):
        num += cmatrix[i][i]

    denom1 = np.sum(precisionlist)
    denom2 = np.sum(recalllist)
    pi = num / denom1
    rho = num / denom2
    denom = pi + rho
    if denom == 0:
        denom = 1
    micro = 2 * pi * rho / denom
    return macro, micro


def restructreMatrixBonsaiSeeDot(A, nClasses, nNodes):
    '''
    Restructures a matrix from [nNodes*nClasses, Proj] to 
    [nClasses*nNodes, Proj] for SeeDot
    '''
    tempMatrix = np.zeros(A.shape)
    rowIndex = 0

    for i in range(0, nClasses):
        for j in range(0, nNodes):
            tempMatrix[rowIndex] = A[j * nClasses + i]
            rowIndex += 1

    return tempMatrix


class GraphManager:
    '''
    Manages saving and restoring graphs. Designed to be used with EMI-RNN
    though is general enough to be useful otherwise as well.
    '''

    def __init__(self):
        pass

    def checkpointModel(self, saver, sess, modelPrefix,
                        globalStep=1000, redirFile=None):
        saver.save(sess, modelPrefix, global_step=globalStep)
        print('Model saved to %s, global_step %d' % (modelPrefix, globalStep),
              file=redirFile)

    def loadCheckpoint(self, sess, modelPrefix, globalStep,
                       redirFile=None):
        metaname = modelPrefix + '-%d.meta' % globalStep
        basename = os.path.basename(metaname)
        fileList = os.listdir(os.path.dirname(modelPrefix))
        fileList = [x for x in fileList if x.startswith(basename)]
        assert len(fileList) > 0, 'Checkpoint file not found'
        msg = 'Too many or too few checkpoint files for globalStep: %d' % globalStep
        assert len(fileList) is 1, msg
        chkpt = basename + '/' + fileList[0]
        saver = tf.train.import_meta_graph(metaname)
        metaname = metaname[:-5]
        saver.restore(sess, metaname)
        graph = tf.get_default_graph()
        return graph
