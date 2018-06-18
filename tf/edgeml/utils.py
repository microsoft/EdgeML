import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops


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
    Cross Entropy loss for MultiClass case in joint training for
    faster convergence
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


def gen_non_linearity(A, non_linearity):
    '''
    Returns required activation for a tensor based on the inputs
    '''
    if non_linearity == "tanh":
        return math_ops.tanh(A)
    elif non_linearity == "sigmoid":
        return math_ops.sigmoid(A)
    elif non_linearity == "relu":
        return gen_math_ops.maximum(A, 0.0)
    elif non_linearity == "quantTanh":
        return gen_math_ops.maximum(gen_math_ops.minimum(A, 1.0), -1.0)
    elif non_linearity == "quantSigm":
        A = (A + 1.0) / 2.0
        return gen_math_ops.maximum(gen_math_ops.minimum(A, 1.0), 0.0)
    else:
        return math_ops.tanh(A)


# Auxiliary methods for EMI-RNN
# Will probably be moved out 

def getConfusionMatrix(predicted, target, numClasses):
    '''
    confusion[i][j]: Number of elements of class j
        predicted as class i
    '''
    assert(predicted.ndim == 1)
    assert(target.ndim == 1)
    arr = np.zeros([numClasses, numClasses])
    
    for i in range(len(predicted)):
        arr[predicted[i]][target[i]] += 1
    return arr

def printFormattedConfusionMatrix(matrix):
    '''
    Given a 2D confusion matrix, prints it in a formatte
    way
    '''
    assert(matrix.ndim == 2)
    assert(matrix.shape[0] == matrix.shape[1])
    RECALL = 'Recall'
    PRECISION = 'PRECISION'
    print("|%s|"% ('True->'), end='')
    for i in range(matrix.shape[0]):
        print("%7d|" % i, end='')
    print("%s|" % 'Precision')
    
    print("|%s|"% ('-'* len(RECALL)), end='')
    for i in range(matrix.shape[0]):
        print("%s|" % ('-'* 7), end='')
    print("%s|" % ('-'* len(PRECISION)))
    
    precisionlist = np.sum(matrix, axis=1)
    recalllist = np.sum(matrix, axis=0) 
    precisionlist = [matrix[i][i]/ x if x != 0 else -1 for i,x in enumerate(precisionlist)]
    recalllist = [matrix[i][i]/x if x != 0 else -1for i,x in enumerate(recalllist)]
    for i in range(matrix.shape[0]):
        # len recall = 6
        print("|%6d|"% (i), end='')
        for j in range(matrix.shape[0]):
            print("%7d|" % (matrix[i][j]), end='')
        print("%s" % (" " * (len(PRECISION) - 7)), end='')
        if precisionlist[i] != -1:
            print("%1.5f|" % precisionlist[i])
        else:
            print("%7s|" % "nan")
    
    print("|%s|"% ('-'* len(RECALL)), end='')
    for i in range(matrix.shape[0]):
        print("%s|" % ('-'* 7), end='')
    print("%s|" % ('-'*len(PRECISION)))
    print("|%s|"% ('Recall'), end='')
    
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
    precisionlist__ = [cmatrix[i][i]/ x if x!= 0 else 0 for i,x in enumerate(precisionlist)]
    recalllist__ = [cmatrix[i][i]/x if x!=0 else 0 for i,x in enumerate(recalllist)]
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
    num =0.0
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
    precisionlist__ = [cmatrix[i][i]/ x if x != 0 else 0 for i,x in enumerate(precisionlist)]
    recalllist__ = [cmatrix[i][i]/x if x != 0 else 0 for i,x in enumerate(recalllist)]
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
