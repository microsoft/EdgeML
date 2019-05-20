# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import tensorflow as tf


class ProtoNN:
    def __init__(self, inputDimension, projectionDimension, numPrototypes,
                 numOutputLabels, gamma,
                 W = None, B = None, Z = None):
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
        with tf.compat.v1.name_scope('protoNN') as ns:
            self.__nscope = ns
        self.__d = inputDimension
        self.__d_cap = projectionDimension
        self.__m = numPrototypes
        self.__L = numOutputLabels

        self.__inW = W
        self.__inB = B
        self.__inZ = Z
        self.__inGamma = gamma
        self.W, self.B, self.Z = None, None, None
        self.gamma = None

        self.__validInit = False
        self.__initWBZ()
        self.__initGamma()
        self.__validateInit()
        self.protoNNOut = None
        self.predictions = None
        self.accuracy = None

    def __validateInit(self):
        self.__validInit = False
        errmsg = "Dimensions mismatch! Should be W[d, d_cap]"
        errmsg += ", B[d_cap, m] and Z[L, m]"
        d, d_cap, m, L, _ = self.getHyperParams()
        assert self.W.shape[0] == d, errmsg
        assert self.W.shape[1] == d_cap, errmsg
        assert self.B.shape[0] == d_cap, errmsg
        assert self.B.shape[1] == m, errmsg
        assert self.Z.shape[0] == L, errmsg
        assert self.Z.shape[1] == m, errmsg
        self.__validInit = True

    def __initWBZ(self):
        with tf.compat.v1.name_scope(self.__nscope):
            W = self.__inW
            if W is None:
                W = tf.compat.v1.initializers.random_normal()
                W = W([self.__d, self.__d_cap])
            self.W = tf.Variable(W, name='W', dtype=tf.float32)

            B = self.__inB
            if B is None:
                B = tf.compat.v1.initializers.random_uniform()
                B = B([self.__d_cap, self.__m])
            self.B = tf.Variable(B, name='B', dtype=tf.float32)

            Z = self.__inZ
            if Z is None:
                Z = tf.compat.v1.initializers.random_normal()
                Z = Z([self.__L, self.__m])
            Z = tf.Variable(Z, name='Z', dtype=tf.float32)
            self.Z = Z
        return self.W, self.B, self.Z

    def __initGamma(self):
        with tf.compat.v1.name_scope(self.__nscope):
            gamma = self.__inGamma
            self.gamma = tf.constant(gamma, name='gamma')

    def getHyperParams(self):
        '''
        Returns the model hyperparameters:
            [inputDimension, projectionDimension,
            numPrototypes, numOutputLabels, gamma]
        '''
        d = self.__d
        dcap = self.__d_cap
        m = self.__m
        L = self.__L
        return d, dcap, m, L, self.gamma

    def getModelMatrices(self):
        '''
        Returns Tensorflow tensors of the model matrices, which
        can then be evaluated to obtain corresponding numpy arrays.

        These can then be exported as part of other implementations of
        ProtonNN, for instance a C++ implementation or pure python
        implementation.
        Returns
            [ProjectionMatrix (W), prototypeMatrix (B),
             prototypeLabelsMatrix (Z), gamma]
        '''
        return self.W, self.B, self.Z, self.gamma

    def __call__(self, X, Y=None):
        '''
        This method is responsible for construction of the forward computation
        graph. The end point of the computation graph, or in other words the
        output operator for the forward computation is returned. Additionally,
        if the argument Y is provided, a classification accuracy operator with
        Y as target will also be created. For this, Y is assumed to in one-hot
        encoded format and the class with the maximum prediction score is
        compared to the encoded class in Y.  This accuracy operator is returned
        by getAccuracyOp() method. If a different accuracyOp is required, it
        can be defined by overriding the createAccOp(protoNNScoresOut, Y)
        method.

        X: Input tensor or placeholder of shape [-1, inputDimension]
        Y: Optional tensor or placeholder for targets (labels or classes).
            Expected shape is [-1, numOutputLabels].
        returns: The forward computation outputs, self.protoNNOut
        '''
        # This should never execute
        assert self.__validInit is True, "Initialization failed!"
        if self.protoNNOut is not None:
            return self.protoNNOut

        W, B, Z, gamma = self.W, self.B, self.Z, self.gamma
        with tf.compat.v1.name_scope(self.__nscope):
            WX = tf.matmul(X, W)
            # Convert WX to tensor so that broadcasting can work
            dim = [-1, WX.shape.as_list()[1], 1]
            WX = tf.reshape(WX, dim)
            dim = [1, B.shape.as_list()[0], -1]
            B_ = tf.reshape(B, dim)
            l2sim = B_ - WX
            l2sim = tf.pow(l2sim, 2)
            l2sim = tf.reduce_sum(input_tensor=l2sim, axis=1, keepdims=True)
            self.l2sim = l2sim
            gammal2sim = (-1 * gamma * gamma) * l2sim
            M = tf.exp(gammal2sim)
            dim = [1] + Z.shape.as_list()
            Z_ = tf.reshape(Z, dim)
            y = tf.multiply(Z_, M)
            y = tf.reduce_sum(input_tensor=y, axis=2, name='protoNNScoreOut')
            self.protoNNOut = y
            self.predictions = tf.argmax(input=y, axis=1, name='protoNNPredictions')
            if Y is not None:
                self.createAccOp(self.protoNNOut, Y)
        return y

    def createAccOp(self, outputs, target):
        '''
        Define an accuracy operation on ProtoNN's output scores and targets.
        Here a simple classification accuracy operator is defined. More
        complicated operators (for multiple label problems and so forth) can be
        defined by overriding this method
        '''
        assert self.predictions is not None
        target = tf.argmax(input=target, axis=1)
        correctPrediction = tf.equal(self.predictions, target)
        acc = tf.reduce_mean(input_tensor=tf.cast(correctPrediction, tf.float32),
                             name='protoNNAccuracy')
        self.accuracy = acc

    def getPredictionsOp(self):
        '''
        The predictions operator is defined as argmax(protoNNScores) for each
        prediction.
        '''
        return self.predictions

    def getAccuracyOp(self):
        '''
        returns accuracyOp as defined by createAccOp. It defaults to
        multi-class classification accuracy.
        '''
        msg = "Accuracy operator not defined in graph. Did you provide Y as an"
        msg += " argument to _call_?"
        assert self.accuracy is not None, msg
        return self.accuracy
