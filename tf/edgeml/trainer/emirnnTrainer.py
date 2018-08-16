# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import edgeml.utils as utils
import pandas as pd

class EMI_Trainer:
    def __init__(self, numTimeSteps, numOutput, graph=None,
                 stepSize=0.001, lossType='l2', optimizer='Adam',
                 automode=True):
        '''
        The EMI-RNN trainer. This classes attaches loss functions and training
        operations to the forward EMI-RNN graph. Currently, simple softmax loss
        and l2 loss are supported on the outputs. For optimizers, only ADAM
        optimizer is available.

        numTimesteps: Number of time steps of the RNN model
        numOutput: Number of output classes
        graph: This module supports restoring from a meta graph. Provide the
            meta graph as an argument to enable this behaviour.
        lossType: A valid loss type string in ['l2', 'xentropy'].
        optimizer: A valid optimizer string in ['Adam'].
        automode: Disable or enable the automode behaviour.
        This module takes care of all of the training procedure automatically,
        and the default behaviour is suitable for most cases. In certain cases
        though, the user would want to change certain aspects of the graph;
        specifically, he would want to change to loss operation by, say, adding
        regularization terms for the model matrices. To enable this behaviour,
        the user can perform the following steps:
            1. Disable automode. That is, when initializing, set automode=False
            2. After the __call__ method has been invoked to create the loss
            operation, the user can access the self.lossOp attribute and modify
            it by adding regularization or other terms.
            3. After the modification has been performed, the user needs to
            call the `createOpCollections()` method so that the newly edited
            operations can be added to Tensorflow collections. This helps in

        HELP_WANTED: Automode is more of a hack than a systematic way of
        supporting multiple loss functions/ optimizers. One way of
        accomplishing this would be to make __createTrainOp and __createLossOp
        methods protected or public, and having users override these.
        Alternatively, we can change the structure to incorporate the
        _createExtendedGraph and _restoreExtendedGraph operations used in
        EMI-LSTM and so forth.
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
        self.softmaxPredictions = None
        self.accTilda = None
        self.equalTilda = None
        self.lossIndicatorTensor = None
        self.lossIndicatorPlaceholder = None
        self.lossIndicatorAssignOp = None
        # Input validation
        self.supportedLosses = ['xentropy', 'l2']
        self.supportedOptimizers = ['Adam']
        assert lossType in self.supportedLosses
        assert optimizer in self.supportedOptimizers
        # Internal
        self.scope = 'EMI/Trainer/'

    def __validateInit(self, predicted, target):
        msg = 'Predicted/Target tensors have incorrect dimension'
        assert len(predicted.shape) == 4, msg
        assert predicted.shape[3] == self.numOutput, msg
        assert predicted.shape[2] == self.numTimeSteps, msg
        assert predicted.shape[1] == target.shape[1], msg
        assert len(target.shape) == 3
        assert target.shape[2] == self.numOutput
        self.__validInit = True

    def __call__(self, predicted, target):
        '''
        Constructs the loss and train operations. If already created, returns
        the created operators.

        predicted: The prediction scores outputed from the forward computation
            graph. Expects a 4 dimensional tensor with shape [-1,
            numSubinstance, numTimeSteps, numClass].
        target: The target labels in one hot-encoding. Expects [-1,
            numSubinstance, numClass]
        '''
        if self.graphCreated is True:
            # TODO: These statements are redundant after self.validInit call
            # A simple check to self.__validInit should suffice. Test this.
            assert self.lossOp is not None
            assert self.trainOp is not None
            return self.lossOp, self.trainOp
        self.__validateInit(predicted, target)
        assert self.__validInit is True
        if self.graph is None:
            self._createGraph(predicted, target)
        else:
            self._restoreGraph(predicted, target)
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

    def __createLossOp(self, predicted, target):
        assert self.__validInit is True, 'Initialization failure'
        with tf.name_scope(self.scope):
            # Loss indicator tensor
            li = np.zeros([self.numTimeSteps, self.numOutput])
            li[-1, :] = 1
            liTensor = tf.Variable(li.astype('float32'),
                                   name='loss-indicator',
                                   trainable=False)
            name='loss-indicator-placeholder'
            liPlaceholder = tf.placeholder(tf.float32,
                                           name=name)
            liAssignOp = tf.assign(liTensor, liPlaceholder,
                                   name='loss-indicator-assign-op')
            self.lossIndicatorTensor = liTensor
            self.lossIndicatorPlaceholder = liPlaceholder
            self.lossIndicatorAssignOp = liAssignOp
            # predicted of dim [-1, numSubinstance, numTimeSteps, numOutput]
            dims = [-1, self.numTimeSteps, self.numOutput]
            logits__ = tf.reshape(predicted, dims)
            labels__ = tf.reshape(target, dims)
            diff = (logits__ - labels__)
            diff = tf.multiply(self.lossIndicatorTensor, diff)
            # take loss only for the timesteps indicated by lossIndicator for softmax
            logits__ = tf.multiply(self.lossIndicatorTensor, logits__)
            labels__ = tf.multiply(self.lossIndicatorTensor, labels__)
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

    def _createGraph(self, predicted, target):
        target = self.__transformY(target)
        assert self.__validInit is True
        with tf.name_scope(self.scope):
            self.softmaxPredictions = tf.nn.softmax(predicted, axis=3,
                                                    name='softmaxed-prediction')
            pred = self.softmaxPredictions[:, :, -1, :]
            actu = target[:, :, -1, :]
            resPred = tf.reshape(pred, [-1, self.numOutput])
            resActu = tf.reshape(actu, [-1, self.numOutput])
            maxPred = tf.argmax(resPred, axis=1)
            maxActu = tf.argmax(resActu, axis=1)
            equal = tf.equal(maxPred, maxActu)
            self.equalTilda = tf.cast(equal, tf.float32, name='equal-tilda')
            self.accTilda = tf.reduce_mean(self.equalTilda, name='acc-tilda')

        self.lossOp = self.__createLossOp(predicted, target)
        self.trainOp = self.__createTrainOp()
        if self.automode:
            self.createOpCollections()
        self.graphCreated = True

    def _restoreGraph(self, predicted, target):
        assert self.graphCreated is False
        scope = self.scope
        graph = self.graph
        self.trainOp = tf.get_collection('EMI-train-op')
        self.lossOp = tf.get_collection('EMI-loss-op')
        assert len(self.trainOp) == 1, msg
        assert len(self.lossOp) == 1, msg
        self.trainOp = self.trainOp[0]
        self.lossOp = self.lossOp[0]
        self.lossIndicatorTensor = graph.get_tensor_by_name(scope +
                                                            'loss-indicator:0')
        name = 'loss-indicator-placeholder:0'
        self.lossIndicatorPlaceholder = graph.get_tensor_by_name(scope + name)
        name = 'loss-indicator-assign-op:0'
        self.lossIndicatorAssignOp = graph.get_tensor_by_name(scope + name)
        msg = 'Multiple tensors with the same name in the graph. Are you not'
        msg +=' resetting your graph?'
        name = scope + 'softmaxed-prediction:0'
        self.softmaxPredictions = graph.get_tensor_by_name(name)
        name = scope + 'acc-tilda:0'
        self.accTilda = graph.get_tensor_by_name(name)
        name = scope + 'equal-tilda:0'
        self.equalTilda = graph.get_tensor_by_name(name)
        self.graphCreated = True
        self.__validInit = True

    def createOpCollections(self):
        '''
        Adds the trainOp and lossOp to Tensorflow collections. This enables us
        to restore these operations from saved metagraphs.
        '''
        tf.add_to_collection('EMI-train-op', self.trainOp)
        tf.add_to_collection('EMI-loss-op', self.lossOp)

    def __echoCB(self, sess, feedDict, currentBatch, redirFile, **kwargs):
        _, loss = sess.run([self.trainOp, self.lossOp],
                                feed_dict=feedDict)
        print("\rBatch %5d Loss %2.5f" % (currentBatch, loss),
              end='', file=redirFile)

    def trainModel(self, sess, redirFile=None, echoInterval=15,
                   echoCB=None, feedDict=None, **kwargs):
        '''
        The training routine.

        sess: The Tensorflow session associated with the computation graph.
        redirFile: Output from the training routine can be redirected to a file
            on the disk. Please provide the file pointer to said file to enable
            this behaviour. Defaults to STDOUT. To disable outputs all
            together, please pass a file pointer to DEVNULL or equivalent as an
            argument.
        echoInterval: The number of batch updates between calls to echoCB.
        echoCB: This call back method is used for printing intermittent
            training stats such as validation accuracy or loss value. By default,
            it defaults to self.__echoCB. The signature of the method is,

            echoCB(self, session, feedDict, currentBatch, redirFile, **kwargs)

            Please refer to the __echoCB implementation for a simple example.
            A more complex example can be found in the EMI_Driver.
        feedDict: feedDict, that is required for the session.run() calls. Will
            be directly passed to the sess.run() calls.
        **kwargs: Additional args to echoCB.
        '''
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
        '''
        This method provides an alternate way of restoring
        from a saved meta graph - without having to provide the restored meta
        graph as a parameter to __init__. This is useful when, in between
        training, you want to reset the entire computation graph and reload a
        new meta graph from disk. This method allows you to attach to this
        newly loaded meta graph without having to create a new EMI_DataPipeline
        object. Use this method only when you want to clear/reset the existing
        computational graph.
        '''
        self.graphCreated = False
        self.lossOp = None
        self.trainOp = None
        self.lossIndicatorTensor = None
        self.softmaxPredictions = None
        self.accTilda = None
        self.graph = graph
        self.__validInit = True
        assert self.graphCreated is False
        self._restoreGraph(None, None)
        assert self.graphCreated is True


class EMI_Driver:
    def __init__(self, emiDataPipeline, emiGraph, emiTrainer,
                 max_to_keep=1000, globalStepStart=1000):
        '''
        The driver class that takes care of training an EMI RNN graph. The EMI
        RNN graph consists of three parts - a data input pipeline
        (EMI_DataPipeline), the forward computation graph (EMI-RNN) and the
        loss graph (EMI_Trainer). After the three parts of been created and
        connected, they should be passed as arguments to this module.

        emiDataPipeline: An EMI_DataPipeline object.
        emiGraph: An EMI_RNN object.
        emiTrainer: An EMI_Trainer object.
        max_to_keep: Maximum number of model checkpoints to keep. Make sure
            that this is more than [number of iterations] * [number of rounds].
        globalStepStart: The global step  value is used as a key for naming
        saved meta graphs. Meta graphs and checkpoints will be named from
        globalStepStart through globalStepStart  + max_to_keep.
        '''
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

    def runOps(self, opList, X, Y, batchSize, feedDict=None):
        sess = self.__sess
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
            numIter, numRounds, batchSize, numEpochs, feedDict=None,
            infFeedDict=None, choCB=None, redirFile=None,
            modelPrefix='/tmp/model', updatePolicy='top-k', fracEMI=0.3,
            lossIndicator=None, *args, **kwargs):
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
            print("Using top-k")
            updatePolicyFunc = self.__policyTopK
        else:
            updatePolicyFunc = self.__policyPrune

        if infFeedDict is None:
            infFeedDict = feedDict
        curr_y = np.array(y_train)
        assert fracEMI >= 0
        assert fracEMI <= 1
        emiSteps = int(fracEMI * numRounds)
        emiStep = numRounds - emiSteps
        for cround in range(numRounds):
            print("Round: %d" % cround, file=redirFile)
            if cround == emiStep:
                print("Switching to EMI-Loss function", file=redirFile)
                if lossIndicator is not None:
                    raise NotImplementedError('TODO')
                else:
                    nTs = self.__emiTrainer.numTimeSteps
                    nOut = self.__emiTrainer.numOutput
                    lossIndicator = np.ones([nTs, nOut])
                    sess.run(self.__emiTrainer.lossIndicatorAssignOp,
                         feed_dict={self.__emiTrainer.lossIndicatorPlaceholder:
                                    lossIndicator})
            valAccList, globalStepList = [], []
            # Train the best model for the current round
            for citer in range(numIter):
                self.__dataPipe.runInitializer(sess, x_train, curr_y,
                                               batchSize, numEpochs)
                numBatches = int(np.ceil(len(x_train) / batchSize))
                self.__emiTrainer.trainModel(sess, echoCB=self.fancyEcho,
                                             numBatches=numBatches,
                                             feedDict=feedDict)
                acc = self.runOps([self.__emiTrainer.accTilda],
                                  x_val, y_val, batchSize, feedDict)
                acc = np.mean(np.reshape(np.array(acc), -1))
                print(" Val acc %2.5f | " % acc, end='', file=redirFile)
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
            self.__sess = sess
            smxOut = self.runOps([self.__emiTrainer.softmaxPredictions],
                                     x_train, y_train, batchSize, feedDict)
            smxOut= [np.array(smxOut[i][0]) for i in range(len(smxOut))]
            smxOut = np.concatenate(smxOut)[:, :, -1, :]
            newY = updatePolicyFunc(curr_y, smxOut, bag_train,
                                    numClasses, **kwargs)
            currY = newY
        return currY

    def updateLabel(self, Y, policy, softmaxOut, bagLabel, numClasses, **kwargs):
        assert policy in ['prune-ends', 'top-k']
        if policy == 'top-k':
            updatePolicyFunc = self.__policyTopK
        else:
            updatePolicyFunc = self.__policyPrune
        Y_ = np.array(Y)
        newY = updatePolicyFunc(Y_, softmaxOut, bagLabel, numClasses, **kwargs)
        return newY

    def analyseModel(self, predictions, Y_bag, numSubinstance, numClass,
                     redirFile=None, verbose=False, silent=False):
        '''
        some basic analysis on predictions and true labels
        This is the multiclass version
        predictions [-1, numsubinstance] is the instance level prediction

        verbose: Prints verbose data frame. Includes additionally, precision
            and recall information.

        In the 2 class setting, precision, recall and f-score for
        class 1 is also printed.
        '''
        assert (predictions.ndim == 2)
        assert (predictions.shape[1] == numSubinstance)
        assert (Y_bag.ndim == 1)
        assert (len(Y_bag) == len(predictions))
        pholder = [0.0] * numSubinstance
        df = pd.DataFrame()
        df['len'] = np.arange(1, numSubinstance + 1)
        df['acc'] = pholder
        df['macro-fsc'] = pholder
        df['macro-pre'] = pholder
        df['macro-rec'] = pholder

        df['micro-fsc'] = pholder
        df['micro-pre'] = pholder
        df['micro-rec'] = pholder
        colList = []
        colList.append('acc')
        colList.append('macro-fsc')
        colList.append('macro-pre')
        colList.append('macro-rec')

        colList.append('micro-fsc')
        colList.append('micro-pre')
        colList.append('micro-rec')
        for i in range(0, numClass):
            pre = 'pre_%02d' % i
            rec = 'rec_%02d' % i
            df[pre] = pholder
            df[rec] = pholder
            colList.append(pre)
            colList.append(rec)

        for i in range(1, numSubinstance + 1):
            pred_ = self.getBagPredictions(predictions, numClass=numClass,
                                           minSubsequenceLen=i,
                                           redirFile = redirFile)
            correct = (pred_ == Y_bag).astype('int')
            trueAcc = np.mean(correct)
            cmatrix = utils.getConfusionMatrix(pred_, Y_bag, numClass)
            df.iloc[i-1, df.columns.get_loc('acc')] = trueAcc

            macro, micro = utils.getMacroMicroFScore(cmatrix)
            df.iloc[i-1, df.columns.get_loc('macro-fsc')] = macro
            df.iloc[i-1, df.columns.get_loc('micro-fsc')] = micro

            pre, rec = utils.getMacroPrecisionRecall(cmatrix)
            df.iloc[i-1, df.columns.get_loc('macro-pre')] = pre
            df.iloc[i-1, df.columns.get_loc('macro-rec')] = rec

            pre, rec = utils.getMicroPrecisionRecall(cmatrix)
            df.iloc[i-1, df.columns.get_loc('micro-pre')] = pre
            df.iloc[i-1, df.columns.get_loc('micro-rec')] = rec
            for j in range(numClass):
                pre, rec = utils.getPrecisionRecall(cmatrix, label=j)
                pre_ = df.columns.get_loc('pre_%02d' % j)
                rec_ = df.columns.get_loc('rec_%02d' % j)
                df.iloc[i-1, pre_ ] = pre
                df.iloc[i-1, rec_ ] = rec

        df.set_index('len')
        # Comment this line to include all columns
        colList = ['len', 'acc', 'macro-fsc', 'macro-pre', 'macro-rec']
        colList += ['micro-fsc', 'micro-pre', 'micro-rec']
        if verbose:
            for col in df.columns:
                if col not in colList:
                    colList.append(col)
        if numClass == 2:
            precisionList = df['pre_01'].values
            recallList = df['rec_01'].values
            denom = precisionList + recallList
            denom[denom == 0] = 1
            numer = 2 * precisionList * recallList
            f_ = numer / denom
            df['fscore_01'] = f_
            colList.append('fscore_01')

        df = df[colList]
        if silent is True:
            return df

        with pd.option_context('display.max_rows', 100,
                               'display.max_columns', 100,
                               'expand_frame_repr', True):
            print(df, file=redirFile)

        idx = np.argmax(df['acc'].values)
        val = np.max(df['acc'].values)
        print("Max accuracy %f at subsequencelength %d" % (val, idx + 1),
              file=redirFile)
        val = np.max(df['micro-fsc'].values)
        idx = np.argmax(df['micro-fsc'].values)
        print("Max micro-f %f at subsequencelength %d" % (val, idx + 1),
              file=redirFile)
        val = df['micro-pre'].values[idx]
        print("Micro-precision %f at subsequencelength %d" % (val, idx + 1),
              file=redirFile)
        val = df['micro-rec'].values[idx]
        print("Micro-recall %f at subsequencelength %d" % (val, idx + 1),
              file=redirFile)

        idx = np.argmax(df['macro-fsc'].values)
        val = np.max(df['macro-fsc'].values)
        print("Max macro-f %f at subsequencelength %d" % (val, idx + 1),
              file=redirFile)
        val = df['macro-pre'].values[idx]
        print("macro-precision %f at subsequencelength %d" % (val, idx + 1),
              file=redirFile)
        val = df['macro-rec'].values[idx]
        print("macro-recall %f at subsequencelength %d" % (val, idx + 1),
              file=redirFile)
        if numClass == 2 and verbose:
            idx = np.argmax(df['fscore_01'].values)
            val = np.max(df['fscore_01'].values)
            print('Max fscore %f at subsequencelength %d' % (val, idx + 1),
                  file=redirFile)
            print('Precision %f at subsequencelength %d' %
                  (df['pre_01'].values[idx], idx + 1), file=redirFile)
            print('Recall %f at subsequencelength %d' %
                  (df['rec_01'].values[idx], idx + 1), file=redirFile)
        return df

    def getInstancePredictions(self, x, y, earlyPolicy, batchSize=1024, **kwargs):

        '''
        Takes the softmax outputs from the joint trained model and, applies
        earlyPolicy() on each instance and returns the instance level
        prediction as well as the step at which this prediction was made.

        softmaxOut: [-1, numSubinstance, numTimeSteps, numClass]

        earlyPolicy: callable,
            def earlyPolicy(subinstacePrediction):
                subinstacePrediction: [numTimeSteps, numClass]
                ...
                return predictedClass, predictedStep

        returns: predictions, predictionStep

        predictions: [-1, numSubinstance]
        predictionStep: [-1, numSubinstance]
        '''
        opList = self.__emiTrainer.softmaxPredictions
        smxOut = self.runOps(opList, x, y, batchSize)
        softmaxOut = np.concatenate(smxOut, axis=0)
        assert softmaxOut.ndim == 4
        numSubinstance, numTimeSteps, numClass = softmaxOut.shape[1:]
        softmaxOutFlat = np.reshape(softmaxOut, [-1, numTimeSteps, numClass])
        flatLen = len(softmaxOutFlat)
        predictions = np.zeros(flatLen)
        predictionStep = np.zeros(flatLen)
        for i, instance in enumerate(softmaxOutFlat):
            # instance is [numTimeSteps, numClass]
            assert instance.ndim == 2
            assert instance.shape[0] == numTimeSteps
            assert instance.shape[1] == numClass
            predictedClass, predictedStep = earlyPolicy(instance, **kwargs)
            predictions[i] = predictedClass
            predictionStep[i] = predictedStep
        predictions = np.reshape(predictions, [-1, numSubinstance])
        predictionStep = np.reshape(predictionStep, [-1, numSubinstance])
        return predictions, predictionStep

    def getBagPredictions(self, Y_predicted, minSubsequenceLen = 4,
                          numClass=2, redirFile = None):
        '''
        Returns bag level predictions given instance level predictions

        A bag is considered to belong to a non-zero class if
        minSubsequenceLen is satisfied. Otherwise, it is assumed
        to belong to class 0. class 0 is negative by default. If
        minSubsequenceLen is satisfied by multiple classes, the smaller of the
        two is returned

        Y_predicted is the predicted instance level results
        [-1, numsubinstance]
        Y True is the correct instance level label
        [-1, numsubinstance]
        '''
        assert(Y_predicted.ndim == 2)
        scoreList = []
        for x in range(1, numClass):
            scores = self.__getLengthScores(Y_predicted, val=x)
            length = np.max(scores, axis=1)
            scoreList.append(length)
        scoreList = np.array(scoreList)
        scoreList = scoreList.T
        assert(scoreList.ndim == 2)
        assert(scoreList.shape[0] == Y_predicted.shape[0])
        assert(scoreList.shape[1] == numClass - 1)
        length = np.max(scoreList, axis=1)
        assert(length.ndim == 1)
        assert(length.shape[0] == Y_predicted.shape[0])
        predictionIndex = (length >= minSubsequenceLen)
        prediction = np.zeros((Y_predicted.shape[0]))
        labels = np.argmax(scoreList, axis=1) + 1
        prediction[predictionIndex] = labels[predictionIndex]
        return prediction.astype(int)


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

    def __policyPrune(self, currentY, softmaxOut, bagLabel, numClases, **kwargs):
        pass

    def __policyTopK(self, currentY, softmaxOut, bagLabel, numClasses, k=1,
                     **kwargs):
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
