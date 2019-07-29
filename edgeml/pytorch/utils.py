# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


def multiClassHingeLoss(logits, labels):
    '''
    MultiClassHingeLoss to match C++ Version - No pytorch internal version
    '''
    flatLogits = torch.reshape(logits, [-1, ])
    labels_ = labels.argmax(dim=1)

    correctId = torch.arange(labels.shape[0]).to(
        logits.device) * labels.shape[1] + labels_
    correctLogit = torch.gather(flatLogits, 0, correctId)

    maxLabel = logits.argmax(dim=1)
    top2, _ = torch.topk(logits, k=2, sorted=True)

    wrongMaxLogit = torch.where((maxLabel == labels_), top2[:, 1], top2[:, 0])

    return torch.mean(F.relu(1. + wrongMaxLogit - correctLogit))


def crossEntropyLoss(logits, labels):
    '''
    Cross Entropy loss for MultiClass case in joint training for
    faster convergence
    '''
    return F.cross_entropy(logits, labels.argmax(dim=1))


def binaryHingeLoss(logits, labels):
    '''
    BinaryHingeLoss to match C++ Version - No pytorch internal version
    '''
    return torch.mean(F.relu(1.0 - (2 * labels - 1) * logits))


def hardThreshold(A: torch.nn.Parameter, s):
    '''
    Hard thresholds and modifies in-palce nn.Parameter A with sparsity s 
    '''
    #PyTorch disallows numpy access/copy to tensors in graph.
    #.detach() creates a new tensor not attached to the graph.
    A_ = A.detach().numpy()
    
    A_ = A_.ravel()
    if len(A_) > 0:
        th = np.percentile(np.abs(A_), (1 - s) * 100.0, interpolation='higher')
        A_[np.abs(A_) < th] = 0.0
    A_ = A_.reshape(A.shape)
    
    A.data = torch.tensor(A_, requires_grad=True)


def copySupport(src, dest):
    '''
    copy support of src tensor to dest tensor
    '''
    zeroSupport = (src.view(-1) == 0.0).nonzero()
    dest = dest.reshape(-1)
    dest[zeroSupport] = 0
    dest = dest.reshape(src.shape)
    del zeroSupport
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

class TriangularLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, stepsize, lr_min, lr_max, gamma):
        self.stepsize = stepsize
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.gamma = gamma
        super(TriangularLR, self).__init__(optimizer)

    def get_lr(self):
        it = self.last_epoch
        cycle = math.floor(1 + it / (2 * self.stepsize))
        x = abs(it / self.stepsize - 2 * cycle + 1)
        decayed_range = (self.lr_max - self.lr_min) * self.gamma ** (it / 3)
        lr = self.lr_min + decayed_range * x
        return [lr]

class ExponentialResettingLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, gamma, reset_epoch):
        self.gamma = gamma
        self.reset_epoch = int(reset_epoch)
        super(ExponentialResettingLR, self).__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch > self.reset_epoch:
            epoch -= self.reset_epoch
        return [base_lr * self.gamma ** epoch
                for base_lr in self.base_lrs]
