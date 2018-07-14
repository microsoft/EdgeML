# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import edgeml.utils as utils

class EMI_RNNTrainer:
    def __init__(self, predicted, X, Y, lossIndicator, numTimeSteps,
                 numOutput, stepSize=0.001, lossType='l2', optimizer='Adam',
                 automode=True):
        '''
        FIX DOC
        emi_graph: The forward pass graph that includes EMI_RNN implementation.
        X, Y : are inputs in appropriate shapes (TODO: Define)
        lossType: ['l2', 'xentropy']
        lossIndicator: TODO

        automode (for the lack of better terminology) takes care of most of the
        training procesure automatically. In certain cases though, the user
        would want to change certain aspects of the graph - like adding more
        regregularization terms to the loss operation. In such cases, automode
        should be turned off and the following method should be called after
        each graph modification.
            createOpCollections()

        X, Y are outputs from some iterator. We assume to be able to get a
        iterator like behaviour on multiple calls to sess.run(). Specically, we
        expect a tf.erors.OutOfRangeError at the end of iterations
        '''
        self.predicted = predicted
        self.lossType = lossType
        self.X = X
        self.Y = Y
        self.lossIndicator = lossIndicator
        self.numTimeSteps = numTimeSteps
        self.numOutput = numOutput
        self.stepSize = stepSize
        self.automode = automode
        self.__validInit = False
        self.optimizer = optimizer
        # Operations
        self.lossOp = None
        self.trainOp = None
        # Input validation
        self.supportedLosses = ['xentropy', 'l2']
        self.supportedOptimizers = ['Adam']
        assert lossType in self.supportedLosses
        assert optimizer in self.supportedOptimizers
        self.__validInit = self.__validateInit()
        # Construct the remaining graph
        self.target = self.__transformY(self.Y)
        self.lossOp = self.__createLossOp(self.predicted, self.target)
        self.trainOp = self.__createTrainOp()
        if self.automode:
            self.createOpCollections()

    def __validateInit(self):
        # We need not make any assumptions about X, the only
        # think we need is that the final outputs are comparable
        msg = 'Number of classes are different'
        assert self.predicted.shape[-1] == self.Y.shape[-1], msg
        # TODO: Validate arguments to init
        idx = self.__findIndicatorStart(self.lossIndicator)
        assert idx >= 0, "invalid lossIndicator passed"
        self.__verifyLossIndicator(self.lossIndicator, idx)
        return True

    def __findIndicatorStart(self, lossIndicator):
        assert lossIndicator.ndim == 2
        for i in range(lossIndicator.shape[0]):
            if lossIndicator[i, 0] == 1:
                return i
        return -1

    def __transformY(self, Y):
        '''
        Because we need output from each step and not just the last step
        '''
        A_ = tf.expand_dims(Y, axis=2)
        A__ = tf.tile(A_, [1, 1, self.numTimeSteps, 1])
        return A__

    def __verifyLossIndicator(self, lossIndicator, idx):
        numOutput = lossIndicator.shape[1]
        assert lossIndicator.ndim == 2
        assert np.sum(lossIndicator[:idx]) == 0
        assert np.sum(lossIndicator[idx:]) == (len(lossIndicator) - idx) * numOutput

    def __createLossOp(self, predicted, target):
        '''
        We are dropping support for regularization parameter beta
        and asymmetric parameter alpha. Since these parts are very
        much dependent on the graph, we will move it outside the trainer.
        That is, this method will create a lossOp method which the user
        can modify externally.

        TODO: Figure out how to handle saving/restoring in such a case.
        For now I'll assume that the user maintains the LossOp name and  will
        restore accordingly
        '''
        assert self.__validInit is True, 'Initialization failure'
        self.lossIndicator = self.lossIndicator.astype('float32')
        self.lossIndicatorTensor = tf.Variable(self.lossIndicator,
                                               name='lossIndicator',
                                               trainable=False)

        # predicted of dim [-1, numSubinstance, numTimeSteps, numOutput]
        logits__ = tf.reshape(predicted, [-1, self.numTimeSteps, self.numOutput])
        labels__ = tf.reshape(target, [-1, self.numTimeSteps, self.numOutput])
        diff = (logits__ - labels__)
        diff = tf.multiply(self.lossIndicatorTensor, diff)

        # take loss only for the timesteps indicated by lossIndicator for softmax
        idx = self.__findIndicatorStart(self.lossIndicator)
        logits__ = logits__[:, idx:, :]
        labels__ = labels__[:, idx:, :]
        logits__ = tf.reshape(logits__, [-1, self.numOutput])
        labels__ = tf.reshape(labels__, [-1, self.numOutput])

        # Regular softmax
        if self.lossType == 'xentropy':
            softmax1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels__,
                                                                  logits=logits__)
            lossOp = tf.reduce_mean(softmax1, name='xentropy-loss')
        elif self.lossType == 'l2':
            lossOp = tf.nn.l2_loss(diff, name='l2-loss')
        return lossOp

    def __createTrainOp(self):
        tst = tf.train.AdamOptimizer(self.stepSize).minimize(self.lossOp)
        return tst

    def createOpCollections(self):
        tf.add_to_collection('train-op', self.trainOp)
        tf.add_to_collection('loss-op', self.lossOp)

    def __echoCB(self, sess, feedDict, currentBatch, redirFile, **kwargs):
        _, loss = sess.run([self.trainOp, self.lossOp],
                                feed_dict=feedDict)
        print("\rBatch %5d Loss %2.5f" % (currentBatch, loss),
              end='', file=redirFile)

    def trainModel(self, sess, redirFile=None, reuse=False,
                   echoInterval=15, echoCB=None,
                   feedDict=None, **kwargs):
        # TODO: IMPORTANT: Case about self.initVarList Line 608
        # TODO: IMPORTANT: Implement accuracy printing
        init = tf.global_variables_initializer()
        if reuse is False:
            sess.run(init)
        else:
            print("Reuse is True. Not performing initialization.", file=redirFile)
        if echoCB is None:
            echoCB = self.__echoCB

        currentBatch = 0
        while True:
            try:
                if currentBatch % echoInterval == 0:
                    echoCB(sess, feedDict, currentBatch, redirFile, **kwargs)
                else:
                    sess.run([self.trainOp], feed_dict=feedDict)
                currentBatch += 1
            except tf.errors.OutOfRangeError:
                break


