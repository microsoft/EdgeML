import numpy as np
import tensorflow as tf
import pandas as pd
import shutil
import time
import sys
import os
from edgeml.utils import getConfusionMatrix, printFormattedConfusionMatrix
from edgeml.utils import getPrecisionRecall
from edgeml.utils import getMacroMicroFScore
from edgeml.utils import getMacroPrecisionRecall
from edgeml.utils import getMicroPrecisionRecall

def getEarlySaving(predictionStep, numTimeSteps, returnTotal=False):
    predictionStep = predictionStep + 1
    predictionStep = np.reshape(predictionStep, -1)
    totalSteps = np.sum(predictionStep)
    maxSteps = len(predictionStep) * numTimeSteps
    savings = 1.0 - (totalSteps / maxSteps)
    if returnTotal:
        return savings, totalSteps
    return savings

def earlyPolicy_baseCase(instanceOut):
    '''
    returns the prediction based on the last class
    '''
    assert instanceOut.ndim == 2
    return np.argmax(instanceOut[-1, :]), len(instanceOut)

def earlyPolicy_minProb(instanceOut, minProb):
    assert instanceOut.ndim == 2
    classes = np.argmax(instanceOut, axis=1)
    prob = np.max(instanceOut, axis=1)
    index = np.where(prob >= minProb)[0]
    if len(index) == 0:
        assert (len(instanceOut) - 1) == (len(classes) - 1)
        return classes[-1], len(instanceOut) - 1
    index = index[0]
    return classes[index], index


def earlyPolicy_classSensitive(instanceOut, minProb, classList):
    '''
    Only makes an early prediction if the class with max
    probability belongs to classList
    '''
    assert instanceOut.ndim == 2
    classes = np.argmax(instanceOut, axis=1)
    prob = np.max(instanceOut, axis=1)
    index = np.where(prob >= minProb)[0]
    if len(index) != 0 and classes[index[0]] in classList:
        return classes[index[0]], index[0]
    return classes[-1], len(classes) - 1

def getJointPredictions(softmaxOut, earlyPolicy, **kwargs):
    '''
    Takes the softmax outputs from the joint trained model as input, applies
    earlyPolicy() on each instance and returns the instance level prediction.

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


class NetworkJoint:
    class SupportedCellTypes:
        LSTM, GRU, FastGRNN, FastRNN = 'lstm', 'gru', 'fastgrnn', 'fastrnn'
        allCells = [LSTM, GRU, FastGRNN, FastRNN]

    def __init__(self, numSubinstance, numFeats, numTimeSteps,
                 numHidden, numFC, numOutput, prefetchNum=10,
                 cellType='lstm', useDropout=False,
                 useEmbeddings=False):
        assert(numOutput >= 2)
        ## Parameters
        self.numSubinstance = numSubinstance
        self.numOutput = numOutput
        self.numFeats = numFeats
        self.numTimeSteps = numTimeSteps
        self.numHidden = numHidden
        self.numFC = numFC
        self.prefetchNum = prefetchNum
        msg = '%s not supported cell type' %  cellType
        assert cellType in NetworkJoint.SupportedCellTypes.allCells, msg
        if cellType == 'lstm':
            self.cellType = NetworkJoint.SupportedCellTypes.LSTM
        elif cellType == 'gru':
            self.cellType = NetworkJoint.SupportedCellTypes.GRU
        elif cellType == 'fastgrnn':
            self.cellType = NetworkJoint.SupportedCellTypes.FastGRNN
        elif cellType == 'fastrnn':
            self.cellType = NetworkJoint.SupportedCellTypes.FastRNN

        self.useDropout = useDropout
        self.useEmbeddings = useEmbeddings
        self.lossList = None
        ## Operations
        # Note that operations only execute once per batch.
        # Better to use train/inferrence methods
        # Raw outputs
        self.output = None
        self.pred = None
        self.l2Loss = None
        self.softmaxLoss = None
        self.lossOp = None
        self.predictionClass = None
        self.dataset_init = None
        self.embedded_word_ids = None
        # Accuracy with respect to belief label (not true label)
        self.accTilda = None
        ## Placeholders
        # X is a bag and Y is label on instances in bag
        self.X = None
        self.Y = None
        self.batchSize = None
        self.numEpochs = None
        self.keep_prob = None
        ## Network
        self.B1 = None
        self.W1 = None
        self.B2 = None
        self.W2 = None
        self.cell = None
        self.LSTMVars = None
        self.varList = None
        self.word_embeddings = None
        self.initVarList = None
        self.lossIndicator = None
        self.lossIndicatorTensor = None
        # Private variables
        self.train_step_model = None
        self.sess = None
        self.__saver = None
        self.__dataset_next = None
        # Validity flags
        self.__graphCreated = False

    def __createInputPipeline(self):
        '''
        The painful process of figuring out who this worked without any documentation.
        https://groups.google.com/a/tensorflow.org/forum/#!msg/discuss/SXWDjrz5kZw/Oj1PO_RnBQAJ
        https://stackoverflow.com/questions/47064693/tensorflow-data-api-prefetch
        https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
        https://stackoverflow.com/questions/47403407/is-tensorflow-dataset-api-slower-than-queues
        https://stackoverflow.com/questions/48777889/tf-data-api-how-to-efficiently-sample-small-patches-from-images
        https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
        Don't forget to use prefetch operation.
        Apparently this is how savable iterators are to be used
        Can't figure out how
        https://www.tensorflow.org/api_docs/python/tf/contrib/data/make_saveable_from_iterator
        Stackoverflow
        https://stackoverflow.com/questions/46917588/restoring-a-tensorflow-model-that-uses-iterators/49236050#49236050
        '''
        assert self.__graphCreated is False
        with tf.name_scope('data_input'):
            dim = [None, self.numSubinstance, self.numTimeSteps, self.numFeats]
            if self.useEmbeddings:
                dim = [None, self.numSubinstance, self.numTimeSteps]
            self.X = tf.placeholder(tf.float32, dim,  name='inpX')
            self.Y = tf.placeholder(tf.float32,
                                    [None, self.numSubinstance, self.numOutput], name='inpY')
            self.batchSize = tf.placeholder(tf.int64, name='batchSize')
            self.numEpochs= tf.placeholder(tf.int64, name='numEpochs')
            dataset_x_target = tf.data.Dataset.from_tensor_slices(self.X)
            dataset_y_target = tf.data.Dataset.from_tensor_slices(self.Y)
            dataset_target = tf.data.Dataset.zip((dataset_x_target, dataset_y_target)).repeat(self.numEpochs)
            dataset_target = dataset_target.batch(self.batchSize).prefetch(self.prefetchNum)
            dataset_iterator_target = tf.data.Iterator.from_structure(dataset_target.output_types,
                                                                    dataset_target.output_shapes)
            dataset_next_target = dataset_iterator_target.get_next()
            dataset_init_target = dataset_iterator_target.make_initializer(dataset_target, name='dataset_init')
            self.dataset_init = dataset_init_target
            self.__dataset_next = dataset_next_target

    def __getRNNOut(self, x, zeroStateShape):
        '''
        x: [num_timestep, batch_size, num_feats]
        '''
        # Does not support forget bias due to sum bug
        # assert self.useCudnn is False, 'Cudnn support not complete. This argument is depricated and will be removed'
        # x = tf.convert_to_tensor(x, dtype=tf.float32)
        # self.cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=self.numHidden, name='cudnnCell')
        # outputs, states = self.cell(x)
        '''
        TODO:
            [X] Change names of cell so that people know what cell you are using
                from the graph.
            [X] Make sure forward pass works  without dropout
            [X] Make sure restoring works without dropout
            [X] Make sure forawrd pass works with dropout
            [X] Make sure restoring works with dropout
            [ ] Make sure initialization works with dropout
            [ ] Make sure initialization works without dropout
            [ ] Make suer exportNPY works properly. After restoring, the numpy
                matrices and the tensorflow outputs should match.
        '''

        if self.cellType is NetworkJoint.SupportedCellTypes.LSTM:
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.numHidden,
                                                     forget_bias=1.0,
                                                     name='LSTMcell')
            print('LSTM')
        elif self.cellType is NetworkJoint.SupportedCellTypes.GRU:
            self.cell = tf.nn.rnn_cell.GRUCell(self.numHidden, name='GRUcell')
            print('GRU')
        elif self.cellType is NetworkJoint.SupportedCellTypes.FastGRNN:
            self.cell = tf.nn.rnn_cell.FastGRNNCell(self.numHidden, name='FastGRNNCell')
            print('FastGRNN')
        elif self.cellType is NetworkJoint.SupportedCellTypes.FastRNNCell:
            self.cell = tf.nn.rnn_cell.FastRNNCell(self.numHidden, name='FastRNNCell')
            print('FastRNN')
        else:
            pass

        wrapped_cell = self.cell
        if self.useDropout is True:
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
            wrapped_cell = tf.contrib.rnn.DropoutWrapper(self.cell,
                                                         input_keep_prob=self.keep_prob,
                                                         output_keep_prob=self.keep_prob)

        outputs, states = tf.nn.static_rnn(wrapped_cell, x, dtype=tf.float32)
        return outputs, states

    def __createForwardGraph(self, X, initVarList):
        assert self.__graphCreated is False
        if initVarList is None:
            self.W1 = tf.Variable(tf.random_normal([self.numHidden, self.numFC]), name='W1')
            self.B1 = tf.Variable(tf.random_normal([self.numFC]), name='B1')
            self.W2 = tf.Variable(tf.random_normal([self.numFC, self.numOutput]), name="W2")
            self.B2 = tf.Variable(tf.random_normal([self.numOutput]), name='B2')
        else:
            W1, B1, W2, B2, kernel, bias = initVarList
            assert W1.shape[0] == self.numHidden
            assert W1.shape[1] == self.numFC
            assert B1.shape[0] == self.numFC
            assert W2.shape[0] == self.numFC
            assert W2.shape[1] == self.numOutput
            assert B2.shape[0] == self.numOutput
            self.W1 = tf.Variable(W1, name='W1')
            self.B1 = tf.Variable(B1, name='B1')
            self.W2 = tf.Variable(W2, name='W2')
            self.B2 = tf.Variable(B2, name='B2')

        zeroStateShape = tf.shape(X)[0]
        # Reshape into 3D such that the first dimension is -1 * numSubinstance
        # where each numSubinstance segment corresponds to one bag
        # then shape it back in into 4D
        x = tf.reshape(X, [-1, self.numTimeSteps, self.numFeats])
        x = tf.unstack(x, num=self.numTimeSteps, axis=1)
        # Get the LSTM output
        outputs__, states = self.__getRNNOut(x, zeroStateShape)
        outputs = []
        for output in outputs__:
            outputs.append(tf.expand_dims(output, axis=1))
        with tf.name_scope("linear_layer"):
            outputs = tf.concat(outputs, axis=1, name='concatenated')
            prod = tf.tensordot(outputs, self.W1, axes=1)
            ret = tf.add(prod, self.B1)
            prod2 = tf.tensordot(ret, self.W2, axes=1)
            ret = tf.add(prod2, self.B2)
        # Convert back to bag form
        with tf.name_scope("final_output"):
            self.output = tf.reshape(ret, [-1, self.numSubinstance, self.numTimeSteps, self.numOutput], name='bag_output')
            self.pred = tf.nn.softmax(self.output, axis=3, name='softmax_pred')
            self.predictionClass = tf.argmax(self.pred, axis=3, name='predicted_classes')

        varList = [self.W1, self.B1, self.W2, self.B2]
        self.LSTMVars = self.cell.variables
        varList.extend(self.LSTMVars)
        self.varList = varList

    def __findIndicatorStart(self, lossIndicator):
        assert lossIndicator.ndim == 2
        for i in range(lossIndicator.shape[0]):
            if lossIndicator[i, 0] == 1:
                return i
        return -1

    def __verifyLossIndicator(self, lossIndicator, idx):
        numOutput = lossIndicator.shape[1]
        assert lossIndicator.ndim == 2
        assert np.sum(lossIndicator[:idx]) == 0
        assert np.sum(lossIndicator[idx:]) == (len(lossIndicator) - idx) * numOutput

    def __createLossGraph(self, X, Y, alpha, beta):
        assert self.__graphCreated is False
        assert alpha == 0.0, 'Asymetric loss not supported'
        self.lossIndicatorTensor = tf.Variable(self.lossIndicator, name='lossIndicator', trainable=False)

        idx = self.__findIndicatorStart(self.lossIndicator)
        assert idx >= 0, "invalid lossIndicator passed"
        self.__verifyLossIndicator(self.lossIndicator, idx)

        # pred of dim [-1, numSubinstance, numTimeSteps, numOutput]
        logits__ = tf.reshape(self.output, [-1, self.numTimeSteps, self.numOutput])
        labels__ = tf.reshape(Y, [-1, self.numTimeSteps, self.numOutput])
        diff = (logits__ - labels__)
        diff = tf.multiply(self.lossIndicatorTensor, diff)

        # take loss only for the timesteps indicated by lossIndicator for softmax
        logits__ = logits__[:, idx:, :]
        labels__ = labels__[:, idx:, :]
        logits__ = tf.reshape(logits__, [-1, self.numOutput])
        labels__ = tf.reshape(labels__, [-1, self.numOutput])
        # Regular softmax
        softmax1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels__, logits=logits__))
        # A mask that selects only the negative sets
        # negInstanceMask = tf.reshape(tf.cast(tf.argmin(Y, axis=2), dtype=tf.float32), [-1])
        # Additional penalty for misprediction on negative set
        # softmax2 = tf.reduce_mean(negInstanceMask *  tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
        l2Loss = tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2)
        #self.softmaxLoss = tf.add(softmax1, alpha * softmax2)
        self.softmaxLoss = tf.add(softmax1, beta * l2Loss, name='xentropy-loss')
        self.l2Loss = tf.add(tf.nn.l2_loss(diff), beta * l2Loss, name='l2Loss')

        pred = self.pred[:, :, -1, :]
        targets = Y[:, :, -1, :]
        reshapedPred = tf.reshape(pred, [-1, self.numOutput])
        targets = tf.reshape(targets, [-1, self.numOutput])
        equal = tf.equal(tf.argmax(reshapedPred, axis=1), tf.argmax(targets, axis=1))
        self.accTilda = tf.reduce_mean(tf.cast(equal, tf.float32), name='acc_tilda')


    def __createTrainGraph(self, stepSize, loss, redirFile):
        with tf.name_scope("gradient"):
            lossOp = self.l2Loss
            if loss != 'l2':
                print("Using softmax loss", file=redirFile)
                lossOp = self.softmaxLoss
            else:
                print("Usign L2 loss", file=redirFile)
                lossOp = self.l2Loss

            assert self.train_step_model is None
            assert lossOp is not None
            tst = tf.train.AdamOptimizer(stepSize).minimize(lossOp)
            self.train_step_model = tst
            self.lossOp = lossOp
            tf.add_to_collection("train_step", self.train_step_model)
            tf.add_to_collection("loss_op", self.lossOp)

    def __retrieveEmbeddings(self, X, embeddings_init, trainable):
        w2v = tf.constant(embeddings_init, dtype=tf.float32)
        vocabulary_size, embedding_size = embeddings_init.shape[0], embeddings_init.shape[1]
        assert embedding_size == self.numFeats
        self.word_embeddings = tf.get_variable("word_embeddings", initializer=w2v, trainable=trainable)
        self.embedded_word_ids = tf.nn.embedding_lookup(self.word_embeddings, X, name='embedding_lookup_op')
        return self.embedded_word_ids

    def __createTransformedY(self, Y):
        A_ = tf.expand_dims(Y, axis=2)
        A__ = tf.tile(A_, [1, 1, self.numTimeSteps, 1])
        return A__

    def createGraph(self, stepSize, alpha=0.0, beta=0.0, loss='smx',
                    trainEmbeddings=False, embeddings_init=None, redirFile=None,
                    initVarList=None, lossIndicator=None):
        if lossIndicator is None:
            lossIndicator = np.ones([self.numTimeSteps, self.numOutput]).astype('float32')

        assert loss == 'smx' or loss == 'l2'
        assert lossIndicator.shape[0] == self.numTimeSteps
        assert lossIndicator.shape[1] == self.numOutput

        self.lossIndicator = lossIndicator.astype('float32')

        assert self.__graphCreated is False
        self.initVarList = initVarList
        self.__createInputPipeline()
        X, Y = self.__dataset_next
        if self.useEmbeddings is True:
            assert embeddings_init is not None
            assert initVarList is None, "Embeddings + restore is not tested"
            X = tf.cast(X, dtype=tf.int32)
            X = self.__retrieveEmbeddings(X, embeddings_init, trainEmbeddings)

        Y = self.__createTransformedY(Y)
        self.__createForwardGraph(X, initVarList)
        self.__createLossGraph(X, Y, alpha, beta)
        self.__createTrainGraph(stepSize, loss, redirFile)
        self.__graphCreated = True
        if self.initVarList is not None:
            self.sess = tf.Session()
            print("Running kernel and bias assignment operations", file=redirFile)
            graph = tf.get_default_graph()
            # TODO: This has to be fixed for Non LSTM cells as Parameters are not the same
            if self.cellType is NetworkJoint.SupportedCellTypes.LSTM:
                k_ = graph.get_tensor_by_name('rnn/LSTMcell/kernel:0')
                b_ = graph.get_tensor_by_name('rnn/LSTMcell/bias:0')
            elif self.cellType is NetworkJoint.SupportedCellTypes.GRU:
                k_ = graph.get_tensor_by_name('rnn/GRUcell/gates/kernel:0')
                b_ = graph.get_tensor_by_name('rnn/GRUcell/gates/bias:0')
            kernel, bias = self.initVarList[-2], self.initVarList[-1]
            k_op = tf.assign(k_, kernel)
            b_op = tf.assign(b_, bias)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.sess.run([k_op, b_op])
            print("Done.", file=redirFile)

    def inference(self, x, batch_size):
        '''
        returns raw out and softmax out
        '''
        if self.useEmbeddings is False:
            assert (x.ndim == 4)
            assert (x.shape[1] == self.numSubinstance)
            assert (x.shape[2] == self.numTimeSteps)
            assert (x.shape[3] == self.numFeats)
        else:
            assert x.ndim == 3
            assert x.shape[1] == self.numSubinstance
            assert x.shape[2] == self.numTimeSteps

        assert (self.sess != None)
        # TODO: Figure out a better way of doing this. With two iterators?
        y = np.zeros([x.shape[0], self.numSubinstance, self.numOutput])
        _feed_dict = {self.X: x, self.Y: y, self.batchSize: batch_size, self.numEpochs: 1}
        outputList = []
        predictionList = []
        predictedClassList = []
        self.sess.run(self.dataset_init, feed_dict = _feed_dict)
        while True:
            try:
                if self.useDropout is False:
                    out, pred, pclass = self.sess.run([self.output, self.pred, self.predictionClass])
                else:
                    out, pred, pclass = self.sess.run([self.output, self.pred, self.predictionClass],
                                                      feed_dict={self.keep_prob: 1.0})
                outputList.extend(out)
                predictionList.extend(pred)
                predictedClassList.extend(pclass)
            except tf.errors.OutOfRangeError:
                break
        return np.array(outputList), np.array(predictionList), np.array(predictedClassList)

    def runOpList(self, opList, x, y = None, batch_size = 1000):
        if self.useEmbeddings is False:
            assert (x.ndim == 4)
            assert (x.shape[1] == self.numSubinstance)
            assert (x.shape[2] == self.numTimeSteps)
            assert (x.shape[3] == self.numFeats)
        else:
            assert x.ndim == 3
            assert x.shape[1] == self.numSubinstance
            assert x.shape[2] == self.numTimeSteps

        assert (self.sess != None)
        # TODO: Figure out a better way of doing this. With two iterators?
        if y is None:
            y = np.zeros([x.shape[0], self.numSubinstance, self.numOutput])
        _feed_dict = {self.X: x, self.Y: y, self.batchSize: batch_size, self.numEpochs: 1}
        outputList = []
        self.sess.run(self.dataset_init, feed_dict = _feed_dict)
        while True:
            try:
                if self.useDropout is False:
                    out = self.sess.run(opList)
                else:
                    out = self.sess.run(opList, feed_dict={self.keep_prob: 1.0})
                outputList.append(out)
            except tf.errors.OutOfRangeError:
                break
        return outputList

    def checkpointModel(self, modelPrefix, max_to_keep=5, global_step=1000, redirFile=None):
        saver = self.__saver
        if self.__saver is None:
            saver = tf.train.Saver(max_to_keep=max_to_keep, save_relative_paths=True)
            self.__saver = saver
        sess = self.sess
        assert(sess is not None)
        saver.save(sess, modelPrefix, global_step=global_step)
        print('Model saved to %s, global_step %d' % (modelPrefix, global_step), file=redirFile)

    def importModelTF(self, modelPrefix, global_step=1000, redirFile=None):
        assert self.__saver is None
        if self.sess is None:
            self.sess = tf.Session()
        metaname = modelPrefix + '-%d.meta' % global_step
        basename = os.path.basename(metaname)
        fileList = os.listdir(os.path.dirname(modelPrefix))
        fileList = [x for x in fileList if x.startswith(basename)]
        assert len(fileList) is 1, "%r \n %s" % (fileList, os.path.dirname(modelPrefix))
        chkpt = basename + '/' + fileList[0]
        saver = tf.train.import_meta_graph(metaname)
        metaname = metaname[:-5]
        saver.restore(self.sess, metaname)
        print('Restoring %s' % metaname, file=redirFile)
        graph = tf.get_default_graph()
        # Restore placeholders
        self.X = graph.get_tensor_by_name("data_input/inpX:0")
        self.Y = graph.get_tensor_by_name("data_input/inpY:0")
        self.batchSize = graph.get_tensor_by_name("data_input/batchSize:0")
        self.numEpochs = graph.get_tensor_by_name("data_input/numEpochs:0")
        if self.useDropout:
            self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
        if self.useEmbeddings is True:
            self.word_embeddings = graph.get_tensor_by_name("word_embeddings:0")

        # Restore operations
        self.output = graph.get_tensor_by_name("final_output/bag_output:0")
        self.pred = graph.get_tensor_by_name("final_output/softmax_pred:0")
        self.predictionClass = graph.get_tensor_by_name("final_output/predicted_classes:0")
        self.train_step_model = tf.get_collection("train_step")[0]
        self.lossOp = tf.get_collection("loss_op")[0]
        self.accTilda = graph.get_tensor_by_name("acc_tilda:0")
        if self.useEmbeddings is True:
            self.embedded_word_ids = graph.get_operation_by_name('embedding_lookup_op')

        # Creating datset
        assert self.dataset_init is None
        self.dataset_init = graph.get_operation_by_name('data_input/dataset_init')
        assert self.dataset_init is not None

        # Restore model parameters
        self.B1 = graph.get_tensor_by_name('B1:0')
        self.W1 = graph.get_tensor_by_name('W1:0')
        self.W2 = graph.get_tensor_by_name('W2:0')
        self.B2 = graph.get_tensor_by_name('B2:0')
        # TODO: This has to be fixed for non LSTM Cells as the params are not the same
        if self.cellType == NetworkJoint.SupportedCellTypes.LSTM:
            kernel = graph.get_tensor_by_name("rnn/LSTMcell/kernel:0")
            bias = graph.get_tensor_by_name("rnn/LSTMcell/bias:0")
        elif self.cellType == NetworkJoint.SupportedCellTypes.GRU:
            kernel = graph.get_tensor_by_name("rnn/GRUcell/gates/kernel:0")
            bias = graph.get_tensor_by_name("rnn/GRUcell/gates/bias:0")
        self.LSTMVars = [kernel, bias]
        self.varList = [self.W1, self.B1, self.W2, self.B2].extend(self.LSTMVars)

        self.__graphCreated = True
        return graph

    # TODO: Has to be fixed for non LSTM Cells
    def exportNPY(self, outFolder=None):
        W1, B1, W2, B2 = self.W1, self.B1, self.W2, self.B2
        lstmKernel, lstmBias = self.LSTMVars
        W1, B1, W2, B2, lstmKernel, lstmBias = self.sess.run([W1, B1, W2, B2, lstmKernel, lstmBias])
        FCW = np.matmul(W2.T, W1.T)
        FCB = np.matmul(W2.T, B1) + B2
        lstmKernel = lstmKernel.T
        lstmBias = lstmBias.T
        if outFolder is None:
            return lstmKernel, lstmBias, FCW, FCB
        lstmKernel_f = outFolder + '/' + 'lstmKernel.npy'
        lstmBias_f = outFolder + '/' + 'lstmBias.npy'
        FCB_f = outFolder + '/' + 'fcb.npy'
        FCW_f = outFolder + '/' + 'fcw.npy'
        assert os.path.isdir(outFolder)
        assert os.path.isfile(lstmKernel_f) is False
        assert os.path.isfile(lstmBias_f) is False
        assert os.path.isfile(FCB_f) is False
        assert os.path.isfile(FCW_f) is False
        np.save(lstmKernel_f, lstmKernel)
        np.save(lstmBias_f, lstmBias)
        np.save(FCW_f, FCW)
        np.save(FCB_f, FCB)
        return lstmKernel, lstmBias, FCW, FCB

    def __trainingSetup(self, reuse, gpufrac, redirFile):
        if self.sess is None:
            assert gpufrac is None
            if gpufrac is not None:
                assert (gpufrac >= 0)
                assert (gpufrac <= 1)
                print('GPU Fraction: %f' % gpufrac, file = redirFile)
                print("GPU Fraction is not entirely supported. Make sure its 1.0", file=redirFile)
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpufrac)
                self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            else:
                print('GPU Fraction: 1.0', file=redirFile)
                self.sess = tf.Session()
        else:
            print("Reusing previous session", file=redirFile)

        if reuse is False and self.initVarList is None:
            init = tf.global_variables_initializer()
            self.sess.run(init)
        else:
            print("Reusing previous init", file=redirFile)


    def trainModel(self, x_train, y_train, x_test, y_test,
                   trainingParams, redirFile=None,
                   reuse=False, gpufrac=None):

        assert self.__graphCreated is True
        self.__trainingSetup(reuse, gpufrac, redirFile)
        batch_size= trainingParams['batch_size']
        if self.useDropout:
            keep_prob = trainingParams['keep_prob']
        lossOp = self.lossOp

        max_epochs = trainingParams['max_epochs']
        num_batches = int(np.ceil(len(x_train) / batch_size))
        if self.lossList is None:
            self.lossList = []
        train_step = self.train_step_model

        currentBatch = 0
        print("Executing %d epochs" % max_epochs, file=redirFile)
        self.sess.run(self.dataset_init,
                      feed_dict={self.X: x_train, self.Y: y_train, self.batchSize: batch_size,
                                 self.numEpochs: max_epochs})
        while True:
            try:
                if currentBatch % 15 == 0:
                    if self.useDropout is True:
                        _, acc, loss = self.sess.run([train_step, self.accTilda, lossOp],
                                                    feed_dict = {self.keep_prob: keep_prob})
                    else:
                        _, acc, loss = self.sess.run([train_step, self.accTilda, lossOp])
                    self.lossList.append(loss)
                    epoch = int(currentBatch / num_batches)
                    tmp = int(currentBatch % max(num_batches, 1))
                    print("\rEpoch %3d Batch %5d (%5d) Loss %2.5f Accuracy %2.5f " %
                          (epoch, tmp, currentBatch, loss, acc), end='', file=redirFile)
                else:
                    if self.useDropout is True:
                        self.sess.run(train_step, feed_dict = {self.keep_prob: keep_prob})
                    else:
                        self.sess.run(train_step)
                    ed = time.time()
                currentBatch += 1
            except tf.errors.OutOfRangeError:
                break
        print(file=redirFile)


