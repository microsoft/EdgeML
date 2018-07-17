# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import edgeml.utils as utils

class EMI_Trainer:
    def __init__(self, numTimeSteps, numOutput, graph=None,
                 stepSize=0.001, lossType='l2', optimizer='Adam',
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
        Mention about automode
        '''
        self.numTimeSteps = numTimeSteps
        self.numOutput = numOutput
        self.graph = graph
        self.stepSize = stepSize
        self.lossType = lossType
        self.optimizer = optimizer
        self.automode = automode
        self.__validInit = False
        self.graphCreated = False
        # Operations to be restored
        self.lossOp = None
        self.trainOp = None
        self.lossIndicatorTensor = None
        self.softmaxPredictions = None
        self.accTilda = None
        # Input validation
        self.supportedLosses = ['xentropy', 'l2']
        self.supportedOptimizers = ['Adam']
        assert lossType in self.supportedLosses
        assert optimizer in self.supportedOptimizers
        # Internal
        self.scope = 'EMI/Trainer/'

    def __findIndicatorStart(self, lossIndicator):
        assert lossIndicator.ndim == 2
        for i in range(lossIndicator.shape[0]):
            if lossIndicator[i, 0] == 1:
                return i
        return -1

    def __validateInit(self, predicted, target, lossIndicator):
        msg = 'Predicted/Target tensors have incorrect dimension'
        assert len(predicted.shape) == 4, msg
        assert predicted.shape[3] == self.numOutput, msg
        assert predicted.shape[2] == self.numTimeSteps, msg
        assert predicted.shape[1] == target.shape[1], msg
        assert len(target.shape) == 3
        assert target.shape[2] == self.numOutput
        msg = "Invalid lossIndicator"
        if lossIndicator is None:
            # Loss indicator can be None when restoring
            self.__validInit = True
            return
        assert lossIndicator.ndim == 2, msg
        idx = self.__findIndicatorStart(lossIndicator)
        assert idx >= 0, msg
        assert np.sum(lossIndicator[:idx]) == 0, msg
        expected = (len(lossIndicator) - idx) * self.numOutput
        assert np.sum(lossIndicator[idx:]) == expected, msg
        self.__validInit = True

    def __call__(self, predicted, target, lossIndicator):
        if self.graphCreated is True:
            assert self.lossOp is not None
            assert self.trainOp is not None
            return self.lossOp, self.trainOp
        self.__validateInit(predicted, target, lossIndicator)
        assert self.__validInit is True
        if self.graph is None:
            assert lossIndicator is not None, 'Invalid lossindicator'
            self._createGraph(predicted, target, lossIndicator)
        else:
            self._restoreGraph(predicted, target, lossIndicator)
        assert self.graphCreated == True
        return self.lossOp, self.trainOp

    def __transformY(self, target):
        '''
        Because we need output from each step and not just the last step.
        Currently we just tile the target to each step. This method can be
        exteneded/overridden to allow more complex behaviours
        '''
        with tf.name_scope(self.scope):
            A_ = tf.expand_dims(target, axis=2)
            A__ = tf.tile(A_, [1, 1, self.numTimeSteps, 1])
        return A__

    def __createLossOp(self, predicted, target, lossIndicator):
        assert self.__validInit is True, 'Initialization failure'
        with tf.name_scope(self.scope):
            lossIndicator = lossIndicator.astype('float32')
            self.lossIndicatorTensor = tf.Variable(lossIndicator,
                                                   name='lossIndicator',
                                                   trainable=False)
            # predicted of dim [-1, numSubinstance, numTimeSteps, numOutput]
            dims = [-1, self.numTimeSteps, self.numOutput]
            logits__ = tf.reshape(predicted, dims)
            labels__ = tf.reshape(target, dims)
            diff = (logits__ - labels__)
            diff = tf.multiply(self.lossIndicatorTensor, diff)
            # take loss only for the timesteps indicated by lossIndicator for softmax
            idx = self.__findIndicatorStart(lossIndicator)
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
        with tf.name_scope(self.scope):
            tst = tf.train.AdamOptimizer(self.stepSize).minimize(self.lossOp)
        return tst

    def _createGraph(self, predicted, target, lossIndicator):
        target = self.__transformY(target)
        with tf.name_scope(self.scope):
            self.softmaxPredictions = tf.nn.softmax(predicted, axis=3,
                                                    name='softmaxed-prediction')
            pred = self.softmaxPredictions[:, :, -1, :]
            actu = target[:, :, -1, :]
            resPred = tf.reshape(pred, [-1, self.numOutput])
            resActu = tf.reshape(actu, [-1, self.numOutput])
            equal = tf.equal(tf.argmax(resPred, axis=1), tf.argmax(resActu,
                                                                   axis=1))
            self.accTilda = tf.reduce_mean(tf.cast(equal, tf.float32),
                                           name='acc-tilda')

        self.lossOp = self.__createLossOp(predicted, target, lossIndicator)
        self.trainOp = self.__createTrainOp()
        if self.automode:
            self.createOpCollections()
        self.graphCreated = True

    def _restoreGraph(self, predicted, target, lossIndicator=None):
        assert self.graphCreated is False
        self.__validateInit(predicted, target, lossIndicator)
        assert self.__validInit is True
        scope = self.scope
        graph = self.graph
        self.trainOp = tf.get_collection('EMI-train-op')
        self.lossOp = tf.get_collection('EMI-loss-op')
        self.lossIndicatorTensor = graph.get_tensor_by_name(scope +
                                                            'lossIndicator:0')
        msg = 'Multiple tensors with the same name in the graph. Are you not'
        msg +=' resetting your graph?'
        assert len(self.trainOp) == 1, msg
        assert len(self.lossOp) == 1, msg
        self.trainOp = self.trainOp[0]
        self.lossOp = self.lossOp[0]
        name = scope + 'softmaxed-prediction:0'
        self.softmaxPredictions = graph.get_tensor_by_name(name)
        name = scope + 'acc-tilda:0'
        self.accTilda = graph.get_tensor_by_name(name)
        self.graphCreated = True

    def createOpCollections(self):
        tf.add_to_collection('EMI-train-op', self.trainOp)
        tf.add_to_collection('EMI-loss-op', self.lossOp)

    def __echoCB(self, sess, feedDict, currentBatch, redirFile, **kwargs):
        _, loss = sess.run([self.trainOp, self.lossOp],
                                feed_dict=feedDict)
        print("\rBatch %5d Loss %2.5f" % (currentBatch, loss),
              end='', file=redirFile)

    def trainModel(self, sess, redirFile=None, echoInterval=15,
                   echoCB=None, feedDict=None, **kwargs):

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

class EMI_Driver:
    def __init__(self, emiDataPipeline, emiTrainer,
                 max_to_keep=1000, globalStepStart=1000):
        self.__dataPipe = emiDataPipeline
        self.__emiTrainer = emiTrainer
        # assert False, 'Not implemented: Check all three objects are properly'
        # 'conconstructed'
        self.__globalStep = globalStepStart
        self.__saver = tf.train.Saver(max_to_keep=max_to_keep,
                                      save_relative_paths=True)
        self.__graphManager = utils.GraphManager()

    def fancyEcho(self, sess, feedDict, currentBatch, redirFile, **kwargs):
        _, loss, acc = sess.run([self.__emiTrainer.trainOp,
                                 self.__emiTrainer.lossOp,
                                 self.__emiTrainer.accTilda],
                                feed_dict=feedDict)
        print("\rBatch %5d Loss %2.5f Acc %2.5f" % (currentBatch, loss, acc),
              end='', file=redirFile)


    def run(self, sess, x_train, y_train, x_val, y_val, numIter,
            numRounds, batchSize, numEpochs, updatePolicy=None,
            echoCB=None, redirFile=None, modelPrefix='/tmp/model'):
        '''
        TODO: Check that max_to_keep > numIter
        TODO: Check that no model with globalStepStart exists; or, print a
        warning that we override these methods.

        Automatically run MI-RNN for 70%% of the rounds and emi for the rest
        Allow option for only MI mode
        '''
        for cround in range(numRounds):
            print("Round: %d" % cround, file=redirFile)
            for citer in range(numIter):
                self.__dataPipe.runInitializer(sess, x_train, y_train, batchSize,
                                               numEpochs)
                # TODO: Implement custo print function 
                self.__emiTrainer.trainModel(sess, echoCB=self.fancyEcho)
                print(file=redirFile)
                self.__graphManager.checkpointModel(self.__saver, sess,
                                                    modelPrefix,
                                                    self.__globalStep,
                                                    redirFile=redirFile)
                self.__globalStep += 1

                '''print some stats'''
                ''' save model'''
                '''keep track of saved models'''
            '''restore best model'''
            '''reinitialize graph and stuff'''
