from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
import numpy as np
import scipy.cluster
import scipy.spatial


def medianHeuristic(data, projectionDimension, numPrototypes, W_init=None):
    '''
    This method can be used to estimate gamma for ProtoNN. An approximation to
    median heuristic is used here.
    1. First the data is collapsed into the projectionDimension by W_init. If
    W_init is not provided, it is initiallzed from a random normal(0, 1). Hence
    data normalization is essential.
    2. Prototype are computed by running a  k-means clustering on the projected
    data.
    3. The median distance is then estimated by calculating median distance
    between prototypes and projected data points.

    data needs to be [-1, numFeats]
    If using this method to initialize gamma, please use the W and B as well.

    TODO: Return estimate of Z (prototype labels) based on cluster centroids
    andand labels

    TODO: Cluestering fails due to singularity error if projecting upwards

    W [dxd_cap]
    B [d_cap, m]
    returns gamma, W, B
    '''
    assert data.ndim == 2
    X = data
    featDim = data.shape[1]
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


def hardThreshold(A, s):
    '''
    Hard Thresholding function on Tensor A with sparsity s
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
    Returns # of nonzeros and represnetative size of the tensor
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
