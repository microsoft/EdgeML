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
        assert self.__validInit is True
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

    def _restoreGraph(self, predicted, target,
                      lossIndicator=None):
        assert self.graphCreated is False
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
        self.__validInit = True

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

    def restoreFromGraph(self, graph):
        self.graphCreated = False
        self.lossOp = None
        self.trainOp = None
        self.lossIndicatorTensor = None
        self.softmaxPredictions = None
        self.accTilda = None
        self.graph = graph
        self.__validInit = True
        assert self.graphCreated is False
        self._restoreGraph(None, None, None)
        assert self.graphCreated is True


class EMI_Driver:
    def __init__(self, emiDataPipeline, emiGraph, emiTrainer,
                 max_to_keep=1000, globalStepStart=1000):
        self.__dataPipe = emiDataPipeline
        self.__emiGraph = emiGraph
        self.__emiTrainer = emiTrainer
        # assert False, 'Not implemented: Check all three objects are properly'
        # 'conconstructed' according to the assumptions made in rnn.py.
        # Specifically amake sure that the graphs are valid with all the
        # variables filled up with corresponding operations
        self.__globalStep = globalStepStart
        self.__saver = tf.train.Saver(max_to_keep=max_to_keep,
                                      save_relative_paths=True)
        self.__graphManager = utils.GraphManager()
        self.__sess = None

    def fancyEcho(self, sess, feedDict, currentBatch, redirFile,
                  numBatches=None):
        _, loss, acc = sess.run([self.__emiTrainer.trainOp,
                                 self.__emiTrainer.lossOp,
                                 self.__emiTrainer.accTilda],
                                feed_dict=feedDict)
        epoch = int(currentBatch /  numBatches)
        batch = int(currentBatch % max(numBatches, 1))
        print("\rEpoch %3d Batch %5d (%5d) Loss %2.5f Acc %2.5f |" %
              (epoch, batch, currentBatch, loss, acc),
              end='', file=redirFile)

    def assignToGraph(self, initVarList):
        '''
        This method should deal with restoring the entire grpah
        now'''
        raise NotImplementedError()

    def initializeSession(self, graph, reuse=False, feedDict=None):
        sess = self.__sess
        if sess is not None:
           sess.close()
        with graph.as_default():
            sess = tf.Session()
        if reuse is False:
            with graph.as_default():
                init = tf.global_variables_initializer()
            sess.run(init)
        self.__sess = sess

    def getCurrentSession(self):
        return self.__sess

    def setSession(self, sess):
        self.__sess = sess

    def runOps(self, sess, opList, X, Y, batchSize, feedDict=None):
        self.__dataPipe.runInitializer(sess, X, Y, batchSize,
                                       numEpochs=1)
        outList = []
        while True:
            try:
                resList = sess.run(opList, feed_dict=feedDict)
                outList.append(resList)
            except tf.errors.OutOfRangeError:
                break
        return outList

    def run(self, numClasses, x_train, y_train, bag_train, x_val, y_val, bag_val,
            numIter, numRounds, batchSize, numEpochs,
            feedDict=None, infFeedDict=None, choCB=None,
            redirFile=None, modelPrefix='/tmp/model',
            updatePolicy='top-k', *args, **kwargs):
        '''
        TODO: Check that max_to_keep > numIter
        TODO: Check that no model with globalStepStart exists; or, print a
        warning that we override these methods.

        Automatically run MI-RNN for 70%% of the rounds and emi for the rest
        Allow option for only MI mode
        '''
        assert self.__sess is not None, 'No sessions initialized'
        sess = self.__sess
        assert updatePolicy in ['prune-ends', 'top-k']
        if updatePolicy == 'top-k':
            updatePolicyFunc = self.__policyTopK
        else:
            updatePolicyFunc = self.__policyPrune

        if infFeedDict is None:
            infFeedDict = feedDict
        curr_y = np.array(y_train)
        for cround in range(numRounds):
            valAccList, globalStepList = [], []
            print("Round: %d" % cround, file=redirFile)
            # Train the best model for the current round
            for citer in range(numIter):
                self.__dataPipe.runInitializer(sess, x_train, curr_y,
                                               batchSize, numEpochs)
                numBatches = int(np.ceil(len(x_train) / batchSize))
                self.__emiTrainer.trainModel(sess, echoCB=self.fancyEcho,
                                             numBatches=numBatches,
                                             feedDict=feedDict)
                acc = self.runOps(sess, [self.__emiTrainer.accTilda],
                                  x_val, y_val, batchSize, feedDict)
                acc = np.mean(np.reshape(np.array(acc), -1))
                print(" Val acc %2.5f | " % acc, end='')
                self.__graphManager.checkpointModel(self.__saver, sess,
                                                    modelPrefix,
                                                    self.__globalStep,
                                                    redirFile=redirFile)
                valAccList.append(acc)
                globalStepList.append((modelPrefix, self.__globalStep))
                self.__globalStep += 1

            # Update y for the current round
            ## Load the best val-acc model
            argAcc = np.argmax(valAccList)
            resPrefix, resStep = globalStepList[argAcc]
            self.__sess.close()
            tf.reset_default_graph()
            sess = tf.Session()
            graph = self.__graphManager.loadCheckpoint(sess, resPrefix, resStep,
                                                       redirFile=redirFile)
            self.__dataPipe.restoreFromGraph(graph)
            self.__emiGraph.restoreFromGraph(graph, None)
            self.__emiTrainer.restoreFromGraph(graph)
            smxOut = self.runOps(sess, [self.__emiTrainer.softmaxPredictions],
                                     x_train, y_train, batchSize, feedDict)
            smxOut= [np.array(smxOut[i][0]) for i in range(len(smxOut))]
            smxOut = np.concatenate(smxOut)[:, :, -1, :]
            newY = updatePolicyFunc(curr_y, smxOut, bag_train,
                                    numClasses, **kwargs)
            currY = newY
        return currY

    def __getLengthScores(self, Y_predicted, val=1):
        '''
        Returns an matrix which contains the length of the longest positive
        subsequence of val ending at that index.
        Y_predicted: [-1, numSubinstance] Is the instance level class
            labels.
        '''
        scores = np.zeros(Y_predicted.shape)
        for i, bag in enumerate(Y_predicted):
            for j, instance in enumerate(bag):
                prev = 0
                if j > 0:
                    prev = scores[i, j-1]
                if instance == val:
                    scores[i, j] = prev + 1
                else:
                    scores[i, j] = 0
        return scores

    def __policyPrune(currentY, softmaxOut, bagLabel, numClases):
        pass

    def __policyTopK(self, currentY, softmaxOut, bagLabel, numClasses, k=1):
        '''
        currentY: [-1, numsubinstance, numClass]
        softmaxOut: [-1, numsubinstance, numClass]
        bagLabel [-1]
        k: minimum length of continuous non-zero examples

        Check which is the longest continuous label for each bag
        If this label is the same as the bagLabel, and if the length is at least k:
            find all the strings with this longest length
            apart from the string having maximum summation of probabilities
            for that class label, label all other instances as 0
        '''
        assert currentY.ndim == 3
        assert k <= currentY.shape[1]
        assert k > 0
        # predicted label for each instance is max of softmax
        predictedLabels = np.argmax(softmaxOut, axis=2)
        scoreList = []
        # classScores[i] is a 2d array where a[j,k] is the longest
        # string of consecutive class labels i in bag j ending at instance k
        classScores = [-1]
        for i in range(1, numClasses):
            scores = self.__getLengthScores(predictedLabels, val=i)
            classScores.append(scores)
            length = np.max(scores, axis=1)
            scoreList.append(length)
        scoreList = np.array(scoreList)
        scoreList = scoreList.T
        # longestContinuousClass[i] is the class label having
        # longest substring in bag i
        longestContinuousClass = np.argmax(scoreList, axis=1) + 1
        # longestContinuousClassLength[i] is length of 
        # longest class substring in bag i
        longestContinuousClassLength = np.max(scoreList, axis=1)
        assert longestContinuousClass.ndim == 1
        assert longestContinuousClass.shape[0] == bagLabel.shape[0]
        assert longestContinuousClassLength.ndim == 1
        assert longestContinuousClassLength.shape[0] == bagLabel.shape[0]
        newY = np.array(currentY)
        index = (bagLabel != 0)
        indexList = np.where(index)[0]
        # iterate through all non-zero bags
        for i in indexList:
            # longest continuous class for this bag
            lcc = longestContinuousClass[i]
            # length of longest continuous class for this bag
            lccl = int(longestContinuousClassLength[i])
            # if bagLabel is not the same as longest continuous
            # class, don't update
            if lcc != bagLabel[i]:
                continue
            # we check for longest string to be at least k
            if lccl < k:
                continue
            lengths = classScores[lcc][i]
            assert np.max(lengths) == lccl
            possibleCandidates = np.where(lengths == lccl)[0]
            # stores (candidateIndex, sum of probabilities
            # over window for this index) pairs
            sumProbsAcrossLongest = {}
            for candidate in possibleCandidates:
                sumProbsAcrossLongest[candidate] = 0.0
                # sum the probabilities over the continuous substring
                for j in range(0, lccl):
                    sumProbsAcrossLongest[candidate] += softmaxOut[i, candidate-j, lcc]
            # we want only the one with maximum sum of
            # probabilities; sort dict by value
            sortedProbs = sorted(sumProbsAcrossLongest.items(),key=lambda x: x[1], reverse=True)
            bestCandidate = sortedProbs[0][0]
            # apart from (bestCanditate-lcc,bestCandidate] label
            # everything else as 0
            newY[i, :, :] = 0
            newY[i, :, 0] = 1
        newY[i, bestCandidate-lccl+1:bestCandidate+1, 0] = 0
        newY[i, bestCandidate-lccl+1:bestCandidate+1, lcc] = 1
        return newY
