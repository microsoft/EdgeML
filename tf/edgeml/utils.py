import tensorflow as tf
import numpy as np


def multiClassHingeLoss(logits, label, batch_th):
    '''
    MultiClassHingeLoss to match C++ Version - No TF internal version
    '''
    flatLogits = tf.reshape(logits, [-1, ])
    correctId = tf.range(0, batch_th) * logits.shape[1] + label
    correctLogit = tf.gather(flatLogits, correctId)

    maxLabel = tf.argmax(logits, 1)
    top2, _ = tf.nn.top_k(logits, k=2, sorted=True)

    wrongMaxLogit = tf.where(tf.equal(maxLabel, label), top2[:, 1], top2[:, 0])

    return tf.reduce_mean(tf.nn.relu(1. + wrongMaxLogit - correctLogit))


def crossEntropyLoss(logits, label):
    '''
    Cross Entropy loss for MultiClass case in joint training for faster convergence
    '''
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))


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


def countnnZ(A, s, bytesPerVar = 4):
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
