# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import tensorflow as tf

class EMI_DataPipeline():
    '''The datainput block for EMI-RNN training
    '''
    def __init__(self, numSubinstance, numTimesteps, numFeats, numOutput,
                 graph=None, prefetchNum =5):

        self.numSubinstance = numSubinstance
        self.numTimesteps = numTimesteps
        self.numFeats = numFeats
        self.graph = graph
        self.prefetchNum = prefetchNum
        self.numOutput = numOutput
        self.graphCreated = False
        # Either restore or create the following
        self.X = None
        self.Y = None
        self.batchSize = None
        self.numEpochs = None
        self.dataset_init = None
        self.dataset_next = None

    def __createGraph(self):
        assert self.graphCreated is False
        dim = [None, self.numSubinstance, self.numTimesteps, self.numFeats]
        X = tf.placeholder(tf.float32, dim, name='EMI/inpX')
        Y = tf.placeholder(tf.float32, [None, self.numSubinstance,
                                       self.numOutput], name='EMI/inpY')
        batchSize = tf.placeholder(tf.int64, name='EMI/batch-size')
        numEpochs = tf.placeholder(tf.int64, name='EMI/num-epochs')

        dataset_x_target = tf.data.Dataset.from_tensor_slices(X)
        dataset_y_target = tf.data.Dataset.from_tensor_slices(Y)
        couple = (dataset_x_target, dataset_y_target)
        ds_target = tf.data.Dataset.zip(couple).repeat(numEpochs)
        ds_target = ds_target.batch(batchSize)
        ds_target = ds_target.prefetch(self.prefetchNum)
        ds_iterator_target = tf.data.Iterator.from_structure(ds_target.output_types,
                                                    ds_target.output_shapes)
        ds_next_target = ds_iterator_target.get_next()
        ds_init_target = ds_iterator_target.make_initializer(ds_target,
                                                             name='EMI/dataset-init')
        self.X = X
        self.Y = Y
        self.batchSize = batchSize
        self.numEpochs = numEpochs
        self.dataset_init = ds_init_target
        self.dataset_next = ds_next_target
        self.graphCreated = True

    def __call__(self):
        if self.graphCreated is True:
            return self.datset_next
        if self.graph is None:
            self.__createGraph()
        else:
            self.__restoreGraph()
        return self.dataset_next

    def runInitializer(self, sess, x_data, y_data, batchSize, numEpochs):
        assert self.graphCreated is True
        assert x_data.ndim == 4
        assert x_data.shape[1] == self.numSubinstance
        assert x_data.shape[2] == self.numTimesteps
        assert x_data.shape[2] == self.numFeats
        assert y_data.shape[0] == x_data.shape[0]
        assert y_data.shape[1] == self.numSubinstance
        assert y_data.shape[2] == self.numOutput
        feed_dict = {
            self.X: x_data,
            self.Y: y_data,
            self.batchSize: batchSize,
            self.numEpochs: numEpochs
        }
        assert self.dataset_init is not None, 'Internal error!'
        sess.run(self.dataset_init, feed_dict=feed_dict)


class EMI_RNN():
    """Abstract base class for RNN architectures compatible with EMI-RNN. This
    class is extended by specific architectures like LSTM/GRU/FastGRNN etc.

    Note: We are not using the PEP recommended abc module since it is difficult
    to support in both python 2 and 3
    """
    def __init__(self, *args, **kwargs):
        # Set to true on first call to __call__
        self.graphCreated = False
        # Model specific matrices, parameter should be saved
        self.varList = []
        self.lossOp = None
        self.trainOp = None
        raise NotImplementedError("This is intended to act similar to an " +
                                  "abstract class. Instantiating is not " +
                                  "allowed.")
    def __call__(self, X, *args, **kwargs):
        raise NotImplementedError("Subclass does not implement this method")

    def getHyperParams(self):
        raise NotImplementedError("Subclass does not implement this method")

    def restoreModel(self, graph, *args, **kwargs):
        '''
        TODO: Note that this is slightly different from the original
        importModelTF. Here you take a graph that the user has already created
        potentially with other models (protoNN etc) and you only restore the
        EMI_RNN part of the graph and ignore everything else
        '''
        raise NotImplementedError("Subclass does not implement this method")

    def initModel(self, initVarList, sess, *args, **kwargs):
        '''
        Initializes model from corresponding matrices
        '''
        pass


class EMI_BasicLSTM(EMI_RNN):
    """EMI-RNN/MI-RNN model using LSTM.
    """

    def __init__(self, numSubinstance, numHidden, numTimeSteps,
                 numFeats, forgetBias=1.0, useDropout=False):
        self.numHidden = numHidden
        self.numTimeSteps = numTimeSteps
        self.numFeats = numFeats
        self.useDropout = useDropout
        self.forgetBias = forgetBias
        self.numSubinstance = numSubinstance
        self.graphCreated = False

        self.cell = None
        self.keep_prob = None
        self.varList = []
        self.output = None

    def __call__(self, X):
        msg = 'EMI_BasicLSTM already part of existing graph.'
        assert self.graphCreated is False, msg
        msg = 'X should be of form [-1, numSubinstance, numTimeSteps, numFeatures]'
        assert X.get_shape().ndims == 4, msg
        # Reshape into 3D such that the first dimension is -1 * numSubinstance
        # where each numSubinstance segment corresponds to one bag
        # then shape it back in into 4D
        x = tf.reshape(X, [-1, self.numTimeSteps, self.numFeats])
        x = tf.unstack(x, num=self.numTimeSteps, axis=1)
        # Get the LSTM output
        with tf.name_scope('EMI-BasicLSTM'):
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.numHidden,
                                                     forget_bias=self.forgetBias,
                                                     name='EMI-BasicLSTM')
            wrapped_cell = self.cell
            if self.useDropout is True:
                keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
                wrapped_cell = tf.contrib.rnn.DropoutWrapper(self.cell,
                                                             input_keep_prob=keep_prob,
                                                             output_keep_prob=keep_prob)
                self.keep_prob = keep_prob
            outputs__, states = tf.nn.static_rnn(wrapped_cell, x, dtype=tf.float32)
        outputs = []
        for output in outputs__:
            outputs.append(tf.expand_dims(output, axis=1))
        # Convert back to bag form
        with tf.name_scope("final_output"):
            outputs = tf.concat(outputs, axis=1, name='concat-output')
            dims = [-1, self.numSubinstance, self.numTimeSteps, self.numHidden]
            self.output = tf.reshape(outputs, dims, name='bag_output')

        LSTMVars = self.cell.variables
        self.varList.extend(LSTMVars)
        self.graphCreated = True
        return self.output

    def getHyperParams(self):
        assert self.graphCreated is True, "Graph is not created"
        assert len(self.varList) == 2
        return self.varList
