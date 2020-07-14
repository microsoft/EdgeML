# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow import keras

class ProtoNNLayer(keras.layers.Layer):
    def __init__(self, inputDimension, projectionDimension, numPrototypes, numOutputLabels, gamma):
        super(ProtoNNLayer, self).__init__()

        self.__d     = inputDimension
        self.__d_cap = projectionDimension
        self.__m     = numPrototypes
        self.__L     = numOutputLabels
        self.gamma   = gamma

    def build(self, input_shape):
        d = self.__d
        d_cap = self.__d_cap
        m = self.__m
        L = self.__L

        self.W = self.add_weight(shape=(d, d_cap))
        self.B = self.add_weight(shape=(d_cap, m))
        self.Z = self.add_weight(shape=(L, m))

    def call(self, X):
        W, B, Z, gamma = self.W, self.B, self.Z, self.gamma

        WX = tf.matmul(X, W)
        dim = [-1, WX.shape.as_list()[1], 1]
        WX = tf.reshape(WX, dim)
        dim = [1, B.shape.as_list()[0], -1]
        B_ = tf.reshape(B, dim)
        l2sim = B_ - WX
        l2sim = tf.pow(l2sim, 2)
        l2sim = tf.reduce_sum(l2sim, 1, keepdims=True)
        self.l2sim = l2sim
        gammal2sim = (-1 * gamma * gamma) * l2sim
        M = tf.exp(gammal2sim)
        dim = [1] + Z.shape.as_list()
        Z_ = tf.reshape(Z, dim)
        y = tf.multiply(Z_, M)
        y = tf.reduce_sum( y, 2 )
        return y


