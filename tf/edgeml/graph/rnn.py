# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import tensorflow as tf

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

    def __init__(self, numHidden, numTimeSteps, numFeats, forgetBias=1.0,
                 useDropout=False):
        self.numHidden = numHidden
        self.numTimeSteps = numTimeSteps
        self.numFeats = numFeats
        self.userDropout = userDropout
        self.forgetBias = forgetBias
        self.graphCreated = False

        self.cell = None
        self.keep_prob = None
        self.varList = []

    def __call__(self, X):
        msg = 'EMI_BasicLSTM already part of existing graph.'
        assert self.graphCreated is False, msg
        msg = 'X should be of form [-1, numSubinstance, numTimeSteps, numFeatures]'
        assert X.ndim == 4, msg
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
                self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
                wrapped_cell = tf.contrib.rnn.DropoutWrapper(self.cell,
                                                             input_keep_prob=self.keep_prob,
                                                             output_keep_prob=self.keep_prob)
            outputs__, states = tf.nn.static_rnn(wrapped_cell, x, dtype=tf.float32)
        outputs = []
        for output in outputs__:
            outputs.append(tf.expand_dims(output, axis=1))
        # Convert back to bag form
        with tf.name_scope("final_output"):
            outputs = tf.concat(outputs, axis=1, name='concat-output')
            dims = [-1, self.numSubinstance, self.numTimeSteps, self.numHidden]
            self.output = tf.reshape(output, dims, name='bag_output')

        LSTMVars = self.cell.variables
        varList.extend(LSTMVars)
        self.varList = varList
        self.graphCreated = True

    def getHyperParams(self):
        assert self.graphCreated is True, "Graph is not created"
        assert len(self.varList) == 2
        return self.varList
