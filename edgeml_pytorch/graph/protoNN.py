# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import numpy as np


class ProtoNN(nn.Module):
    def __init__(self, inputDimension, projectionDimension, numPrototypes,
                 numOutputLabels, gamma, W=None, B=None, Z=None):
        '''
        Forward computation graph for ProtoNN.

        inputDimension: Input data dimension or feature dimension.
        projectionDimension: hyperparameter
        numPrototypes: hyperparameter
        numOutputLabels: The number of output labels or classes
        W, B, Z: Numpy matrices that can be used to initialize
            projection matrix(W), prototype matrix (B) and prototype labels
            matrix (B).
            Expected Dimensions:
                W   inputDimension (d) x projectionDimension (d_cap)
                B   projectionDimension (d_cap) x numPrototypes (m)
                Z   numOutputLabels (L) x numPrototypes (m)
        '''
        super(ProtoNN, self).__init__()
        self.__d = inputDimension
        self.__d_cap = projectionDimension
        self.__m = numPrototypes
        self.__L = numOutputLabels

        self.W, self.B, self.Z = None, None, None
        self.gamma = gamma

        self.__validInit = False
        self.__initWBZ(W, B, Z)
        self.__validateInit()

    def __validateInit(self):
        self.__validinit = False
        errmsg = "Dimensions mismatch! Should be W[d, d_cap]"
        errmsg+= ", B[d_cap, m] and Z[L, m]"
        d, d_cap, m, L, _ = self.getHyperParams()
        assert self.W.shape[0] == d, errmsg
        assert self.W.shape[1] == d_cap, errmsg
        assert self.B.shape[0] == d_cap, errmsg
        assert self.B.shape[1] == m, errmsg
        assert self.Z.shape[0] == L, errmsg
        assert self.Z.shape[1] == m, errmsg
        self.__validInit = True

    def __initWBZ(self, inW, inB, inZ):
        if inW is None:
            self.W = torch.randn([self.__d, self.__d_cap])
            self.W = nn.Parameter(self.W)
        else:
            self.W = nn.Parameter(torch.from_numpy(inW.astype(np.float32)))

        if inB is None:
            self.B = torch.randn([self.__d_cap, self.__m])
            self.B = nn.Parameter(self.B)
        else:
            self.B = nn.Parameter(torch.from_numpy(inB.astype(np.float32)))

        if inZ is None:
            self.Z = torch.randn([self.__L, self.__m])
            self.Z = nn.Parameter(self.Z)
        else:
            self.Z = nn.Parameter(torch.from_numpy(inZ.astype(np.float32)))

    def getHyperParams(self):
        '''
        Returns the model hyperparameters:
            [inputDimension, projectionDimension, numPrototypes,
            numOutputLabels, gamma]
        '''
        d =  self.__d
        dcap = self.__d_cap
        m = self.__m
        L = self.__L
        return d, dcap, m, L, self.gamma

    def getModelMatrices(self):
        '''
        Returns model matrices, which can then be evaluated to obtain
        corresponding numpy arrays.  These can then be exported as part of
        other implementations of ProtonNN, for instance a C++ implementation or
        pure python implementation.
        Returns
            [ProjectionMatrix (W), prototypeMatrix (B),
             prototypeLabelsMatrix (Z), gamma]
        '''
        return self.W, self.B, self.Z, self.gamma

    def forward(self, X):
        '''
        This method is responsible for construction of the forward computation
        graph. The end point of the computation graph, or in other words the
        output operator for the forward computation is returned.

        X: Input of shape [-1, inputDimension]
        returns: The forward computation outputs, self.protoNNOut
        '''
        assert self.__validInit is True, "Initialization failed!"

        W, B, Z, gamma = self.W, self.B, self.Z, self.gamma
        WX = torch.matmul(X, W)
        dim = [-1, WX.shape[1], 1]
        WX = torch.reshape(WX, dim)
        dim = [1, B.shape[0], -1]
        B_ = torch.reshape(B, dim)
        l2sim = B_ - WX
        l2sim = torch.pow(l2sim, 2)
        l2sim = torch.sum(l2sim, dim=1, keepdim=True)
        self.l2sim = l2sim
        gammal2sim = (-1 * gamma * gamma) * l2sim
        M = torch.exp(gammal2sim)
        dim = [1] + list(Z.shape)
        Z_ = torch.reshape(Z, dim)
        y = Z_ * M
        y = torch.sum(y, dim=2)
        return y


