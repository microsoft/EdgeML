# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import tensorflow as tf
from tensorflow import keras

class BonsaiLayer(keras.layers.Layer):
    def __init__(self, numClasses, dataDimension, projectionDimension, treeDepth, sigma, isRegression=False):
        super(BonsaiLayer, self).__init__()

        self.dataDimension = dataDimension
        self.projectionDimension = projectionDimension
        self.isRegression = isRegression

        if numClasses == 2:
            self.numClasses = 1
        else:
            self.numClasses = numClasses

        self.treeDepth = treeDepth
        self.sigma = sigma

        self.internalNodes = 2**self.treeDepth - 1
        self.totalNodes = 2 * self.internalNodes + 1

    def build(self, input_shape):
        self.Z = self.add_weight(shape=(self.projectionDimension, self.dataDimension))
        self.W = self.add_weight(shape=(self.numClasses * self.totalNodes, self.projectionDimension))
        self.V = self.add_weight(shape=(self.numClasses * self.totalNodes, self.projectionDimension))
        self.T = self.add_weight(shape=(self.internalNodes, self.projectionDimension))

    def call(self, X):
        sigmaI = 1e9
        errmsg = "Dimension Mismatch, X is [_, self.dataDimension]"
        assert (len(X.shape) == 2 and int(X.shape[1]) == self.dataDimension), errmsg

        X_ = tf.divide(tf.matmul(self.Z, X, transpose_b=True), self.projectionDimension)

        W_ = self.W[0:(self.numClasses)]
        V_ = self.V[0:(self.numClasses)]

        __nodeProb = []
        __nodeProb.append(1)

        score_ = __nodeProb[0] * tf.multiply(tf.matmul(W_, X_), tf.tanh(self.sigma * tf.matmul(V_, X_)))
        for i in range(1, self.totalNodes):
            W_ = self.W[i * self.numClasses:((i + 1) * self.numClasses)]
            V_ = self.V[i * self.numClasses:((i + 1) * self.numClasses)]

            T_ = tf.reshape(self.T[int(np.ceil(i / 2.0) - 1.0)], [-1, self.projectionDimension])
            prob = (1 + ((-1)**(i + 1)) * tf.tanh(tf.multiply(sigmaI, tf.matmul(T_, X_))))

            prob = tf.divide(prob, 2.0)
            prob = __nodeProb[int(np.ceil(i / 2.0) - 1.0)] * prob
            __nodeProb.append(prob)
            score_ += __nodeProb[i] * tf.multiply(tf.matmul(W_, X_), tf.tanh(self.sigma * tf.matmul(V_, X_)))

        self.score = score_
        #return score_
        # Classification.
        if (self.isRegression == False):
            if self.numClasses > 2:
                self.prediction = tf.argmax(tf.transpose(self.score), 1)
            else:
                self.prediction = tf.argmax(
                    tf.concat([tf.transpose(self.score),
                               0 * tf.transpose(self.score)], 1), 1)
        # Regression.
        elif (self.isRegression == True):
            # For regression , scores are the actual predictions, just return them.
            self.prediction = self.score

        return self.prediction



