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
    
    def weighted_call(self, X):
        '''
        Original bonsai learns a single, shallow sparse tree whose predictions for a 
        point x are given by
        y(x) = Σ(k) I(k) (W(k).Transpose Zx) ◦ tanh(σV(k).Transpose Zx)

        Proposed function to include path smoothing. (Accuracy improvement for deeper trees)
        To calculate a weighted average with respect to each node from the root node to
        the leaf node. Each node thus contributes to the weighted average. 

        This way the contribution should increase as depth increases and as the Bonsai tree model is balanced
        the weight for each path will remain dependent on the number of classes and depth.
        This way even if a leaf has few observations the non leaf nodes will correct the target distribution
        by smoothing it out. Think of it as a heirarchical prior assignment.
        This way we can still increase depth and remove the problem of overfitting trees.
        y(x) = (Σ(k) I(k) hierarchical_prior(k) (W(k).Transpose Zx) ◦ tanh(V(k).Transpose Zx))/k
        Path to which node to visit is determined by heirarchical_prior(k)
        heirarchical_prior(k) = (n_classes * (1+σ)**depth)/n_classes
        σ = sigma (positive number, [0.1,0.01] are good values)
        Future work: Tree pruning can be added to the model to reduce the number of nodes, 
        Use CUDA or Dask for distributed training.
        '''
        sigmaI = self.sigma
        errmsg = "Dimension Mismatch, X is [_, self.dataDimension]"
        assert (len(X.shape) == 2 and int(X.shape[1]) == self.dataDimension), errmsg
        X_ = tf.divide(tf.matmul(self.Z, X, transpose_b=True), self.projectionDimension)
        W_ = self.W[0:(self.numClasses)]
        V_ = self.V[0:(self.numClasses)]
        __nodeProb = []
        __nodeProb.append(1)
        # Node count starts from 1 to avoid div by 0
        heirarchical_prior = (self.numClasses * (1 + sigmaI)) / self.numClasses
        score_ = __nodeProb[0] * heirarchical_prior * tf.multiply(tf.matmul(W_, X_), tf.tanh(tf.matmul(V_, X_)))
        for i in range(1, self.totalNodes):
            W_ = self.W[i * self.numClasses:((i + 1) * self.numClasses)]
            V_ = self.V[i * self.numClasses:((i + 1) * self.numClasses)]
            T_ = tf.reshape(self.T[int(np.ceil(i / 2.0) - 1.0)], [-1, self.projectionDimension])
            prob = (1 + ((-1) ** (i + 1)) * tf.tanh(tf.multiply(sigmaI, tf.matmul(T_, X_)))) #Indicator function
            prob = tf.divide(prob, 2.0)
            prob = __nodeProb[int(np.ceil(i / 2.0) - 1.0)] * prob
            __nodeProb.append(prob)
            heirarchical_prior = (self.numClasses * (1 + sigmaI) ** np.log2([i+1])) / self.numClasses #Weighted prior
            score_ += prob * heirarchical_prior * tf.multiply(tf.matmul(W_, X_), tf.tanh(tf.matmul(V_, X_)))
        self.score = score_
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


        

