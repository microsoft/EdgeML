# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import numpy as np


class Bonsai(nn.Module):

    def __init__(self, numClasses, dataDimension, projectionDimension,
                 treeDepth, sigma, W=None, T=None, V=None, Z=None):
        super(Bonsai, self).__init__()
        '''
        Expected Dimensions:

        Bonsai Params // Optional
        W [numClasses*totalNodes, projectionDimension]
        V [numClasses*totalNodes, projectionDimension]
        Z [projectionDimension, dataDimension + 1]
        T [internalNodes, projectionDimension]

        internalNodes = 2**treeDepth - 1
        totalNodes = 2*internalNodes + 1

        sigma - tanh non-linearity
        sigmaI - Indicator function for node probabilities
        sigmaI - has to be set to infinity(1e9 for practice)
        while doing testing/inference
        numClasses will be reset to 1 in binary case
        '''

        self.dataDimension = dataDimension
        self.projectionDimension = projectionDimension

        if numClasses == 2:
            self.numClasses = 1
        else:
            self.numClasses = numClasses

        self.treeDepth = treeDepth
        self.sigma = sigma

        self.internalNodes = 2**self.treeDepth - 1
        self.totalNodes = 2 * self.internalNodes + 1

        self.W = self.initW(W)
        self.V = self.initV(V)
        self.T = self.initT(T)
        self.Z = self.initZ(Z)

        self.assertInit()

    def initZ(self, Z):
        if Z is None:
            Z = torch.randn([self.projectionDimension, self.dataDimension])
            Z = nn.Parameter(Z)
        else:
            Z = torch.from_numpy(Z.astype(np.float32))
            Z = nn.Parameter(Z)
        return Z

    def initW(self, W):
        if W is None:
            W = torch.randn(
                [self.numClasses * self.totalNodes, self.projectionDimension])
            W = nn.Parameter(W)
        else:
            W = torch.from_numpy(W.astype(np.float32))
            W = nn.Parameter(W)
        return W

    def initV(self, V):
        if V is None:
            V = torch.randn(
                [self.numClasses * self.totalNodes, self.projectionDimension])
            V = nn.Parameter(V)
        else:
            V = torch.from_numpy(V.astype(np.float32))
            V = nn.Parameter(V)
        return V

    def initT(self, T):
        if T is None:
            T = torch.randn([self.internalNodes, self.projectionDimension])
            T = nn.Parameter(T)
        else:
            T = torch.from_numpy(T.astype(np.float32))
            T = nn.Parameter(T)
        return T

    def forward(self, X, sigmaI):
        '''
        Function to build/exxecute the Bonsai Tree graph
        Expected Dimensions

        X is [batchSize, self.dataDimension]
        sigmaI is constant
        '''
        X_ = torch.matmul(self.Z, torch.t(X)) / self.projectionDimension
        W_ = self.W[0:(self.numClasses)]
        V_ = self.V[0:(self.numClasses)]
        self.__nodeProb = []
        self.__nodeProb.append(1)
        score_ = self.__nodeProb[0] * (torch.matmul(W_, X_) *
                                       torch.tanh(self.sigma *
                                                  torch.matmul(V_, X_)))
        for i in range(1, self.totalNodes):
            W_ = self.W[i * self.numClasses:((i + 1) * self.numClasses)]
            V_ = self.V[i * self.numClasses:((i + 1) * self.numClasses)]

            T_ = torch.reshape(self.T[int(np.ceil(i / 2.0) - 1.0)],
                               [-1, self.projectionDimension])
            prob = (1 + ((-1)**(i + 1)) *
                    torch.tanh(sigmaI * torch.matmul(T_, X_)))

            prob = prob / 2.0
            prob = self.__nodeProb[int(np.ceil(i / 2.0) - 1.0)] * prob
            self.__nodeProb.append(prob)
            score_ += self.__nodeProb[i] * (torch.matmul(W_, X_) *
                                            torch.tanh(self.sigma *
                                                       torch.matmul(V_, X_)))

        self.score = score_
        self.X_ = X_
        return torch.t(self.score), self.X_

    def assertInit(self):
        errRank = "All Parameters must has only two dimensions shape = [a, b]"
        assert len(self.W.shape) == len(self.Z.shape), errRank
        assert len(self.W.shape) == len(self.T.shape), errRank
        assert len(self.W.shape) == 2, errRank
        msg = "W and V should be of same Dimensions"
        assert self.W.shape == self.V.shape, msg
        errW = "W and V are [numClasses*totalNodes, projectionDimension]"
        assert self.W.shape[0] == self.numClasses * self.totalNodes, errW
        assert self.W.shape[1] == self.projectionDimension, errW
        errZ = "Z is [projectionDimension, dataDimension]"
        assert self.Z.shape[0] == self.projectionDimension, errZ
        assert self.Z.shape[1] == self.dataDimension, errZ
        errT = "T is [internalNodes, projectionDimension]"
        assert self.T.shape[0] == self.internalNodes, errT
        assert self.T.shape[1] == self.projectionDimension, errT
        assert int(self.numClasses) > 0, "numClasses should be > 1"
        msg = "# of features in data should be > 0"
        assert int(self.dataDimension) > 0, msg
        msg = "Projection should be  > 0 dims"
        assert int(self.projectionDimension) > 0, msg
        msg = "treeDepth should be >= 0"
        assert int(self.treeDepth) >= 0, msg
