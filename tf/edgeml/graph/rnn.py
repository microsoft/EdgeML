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
        self.x_batch = None
        self.y_batch = None

        # Internal
        self.scope = 'EMI/'

    def __createGraph(self):
        assert self.graphCreated is False
        dim = [None, self.numSubinstance, self.numTimesteps, self.numFeats]
        scope = self.scope + 'input-pipeline/'
        with tf.name_scope(scope):
            X = tf.placeholder(tf.float32, dim, name='inpX')
            Y = tf.placeholder(tf.float32, [None, self.numSubinstance,
                                           self.numOutput], name='inpY')
            batchSize = tf.placeholder(tf.int64, name='batch-size')
            numEpochs = tf.placeholder(tf.int64, name='num-epochs')

            dataset_x_target = tf.data.Dataset.from_tensor_slices(X)
            dataset_y_target = tf.data.Dataset.from_tensor_slices(Y)
            couple = (dataset_x_target, dataset_y_target)
            ds_target = tf.data.Dataset.zip(couple).repeat(numEpochs)
            ds_target = ds_target.batch(batchSize)
            ds_target = ds_target.prefetch(self.prefetchNum)
            ds_iterator_target = tf.data.Iterator.from_structure(ds_target.output_types,
                                                        ds_target.output_shapes)
            ds_next_target = ds_iterator_target
            ds_init_target = ds_iterator_target.make_initializer(ds_target,
                                                                 name='dataset-init')
            x_batch, y_batch = ds_iterator_target.get_next()
            tf.add_to_collection('next-x-batch', x_batch)
            tf.add_to_collection('next-y-batch', y_batch)
        self.X = X
        self.Y = Y
        self.batchSize = batchSize
        self.numEpochs = numEpochs
        self.dataset_init = ds_init_target
        self.x_batch, self.y_batch = x_batch, y_batch
        self.graphCreated = True

    def __restoreGraph(self):
        assert self.graphCreated is False
        graph = self.graph
        scope = 'EMI/input-pipeline/'
        self.X = graph.get_tensor_by_name(scope + "inpX:0")
        self.Y = graph.get_tensor_by_name(scope + "inpY:0")
        self.batchSize = graph.get_tensor_by_name(scope + "batch-size:0")
        self.numEpochs = graph.get_tensor_by_name(scope + "num-epochs:0")
        self.dataset_init = graph.get_operation_by_name(scope + "dataset-init")
        self.x_batch = graph.get_collection('next-x-batch')
        self.y_batch = graph.get_collection(scope + 'next-y-batch')
        self.graphCreated = True

    def __call__(self):
        if self.graphCreated is True:
            return self.x_batch, self.y_batch
        if self.graph is None:
            self.__createGraph()
        else:
            self.__restoreGraph()
        return self.x_batch, self.y_batch

    def runInitializer(self, sess, x_data, y_data, batchSize, numEpochs):
        assert self.graphCreated is True
        msg = 'X shape should be [-1, numSubinstance, numTimesteps, numFeats]'
        assert x_data.ndim == 4, msg
        assert x_data.shape[1] == self.numSubinstance, msg
        assert x_data.shape[2] == self.numTimesteps, msg
        assert x_data.shape[3] == self.numFeats, msg
        msg = 'X and Y sould have same first dimension'
        assert y_data.shape[0] == x_data.shape[0], msg
        msg ='Y shape should be [-1, numSubinstance, numOutput]'
        assert y_data.shape[1] == self.numSubinstance, msg
        assert y_data.shape[2] == self.numOutput, msg
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
                 numFeats, graph=None, forgetBias=1.0, useDropout=False):
        self.numHidden = numHidden
        self.numTimeSteps = numTimeSteps
        self.numFeats = numFeats
        self.useDropout = useDropout
        self.forgetBias = forgetBias
        self.numSubinstance = numSubinstance
        self.graph = graph
        self.graphCreated = False

        self.cell = None
        self.keep_prob = None
        self.varList = []
        self.output = None

    def __createGraph(self, X):
        assert self.graphCreated is False
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

    def __call__(self, X):
        if self.graphCreated is True:
            assert self.output is not None
            return self.output
        if self.graph is None:
            self.__createGraph()
        else:
            raise NotImplementedError()
        assert self.graphCreated is True
        return self.output


    def getHyperParams(self):
        assert self.graphCreated is True, "Graph is not created"
        assert len(self.varList) == 2
        return self.varList
