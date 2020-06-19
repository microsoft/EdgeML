# import torch
# import torch.nn as nn
# import numpy as np
# from edgeml_pytorch.graph.rnn import *

import tensorflow as tf
#from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import conv_utils
import tf_rnn as rnn

class RNNPoolOP( tf.keras.layers.Layer ):
    def __init__( self, out_channel, nRows, nCols, units=8, trainable=True, name='RNNPoolOP' ):
        super(RNNPoolOP, self).__init__(name=name)
        assert (out_channel % 4) == 0
        self.nRows = nRows
        self.nCols = nCols
        self.nHiddenDims = units
        self.nHiddenDimsOut = out_channel // 4
        self.trainable = trainable
        #self.name = 'RNNPoolOP' if self.name == None else self.name + '/RNNPoolOP'
        #self.outp1 = tf.Variable( [-1, nRows, nHiddenDims], trainable=False )

        # def build(self, _):
        #with tf.variable_scope(self.name):
        self.rnn1 = rnn.FastGRNNCell( self.nHiddenDims,
                                      gate_non_linearity="sigmoid", update_non_linearity="tanh",
                                      zetaInit=100.0, nuInit=-100.0, name=self.name + '/rnn1' )

        self.rnn2 = rnn.FastGRNNCell( self.nHiddenDimsOut,
                                      gate_non_linearity="sigmoid", update_non_linearity="tanh",
                                      zetaInit=100.0, nuInit=-100.0, name=self.name + '/rnn2' )

    def build( self, inputShape ):
        self.inCh = inputShape[-1]

    def call( self, inputs ):
        #with tf.variable_scope(self.name):
        ## row-wise, time-axis: col
        print( 'RNNPool/inputs: ', inputs.shape )                     # DEBUG
        x = tf.reshape( inputs, [-1, self.nCols, self.inCh] )   # [batch, time, ch]
        print( '\tRNNPool/row-wise RNN1-x: ', x.shape )                     # DEBUG
        _, outp1 = tf.nn.dynamic_rnn( cell=self.rnn1, inputs=x, dtype=tf.float32 )
        print( '\tRNNPool/row-wise RNN1-out: ', outp1.shape )                     # DEBUG

        x = tf.reshape( outp1, [-1, self.nRows, self.nHiddenDims] )
        print( '\tRNNPool/row-wise RNN2-x: ', x.shape )                     # DEBUG
        _, outp21 = tf.nn.dynamic_rnn( cell=self.rnn2, inputs=x, dtype=tf.float32)
        _, outp22 = tf.nn.dynamic_rnn( cell=self.rnn2, inputs=tf.reverse( x, [1] ), dtype=tf.float32)
        print( '\tRNNPool/row-wise RNN1-out: ', outp21.shape, outp22.shape )                     # DEBUG
        # debugPrint = tf.compat.v1.concat( [outp21, outp22], 1 )                                 # DEBUG

        ## col-wise, time-axis: row
        x = tf.transpose( inputs, [0, 2, 1, 3] )                # NWHC
        x = tf.reshape( x, [-1, self.nRows, self.inCh] )        # [batch, time, ch]
        print( '\tRNNPool/col-wise RNN1-x: ', x.shape )                     # DEBUG
        _, outp1 = tf.nn.dynamic_rnn( cell=self.rnn1, inputs=x, dtype=tf.float32)
        print( '\tRNNPool/col-wise RNN1-out: ', outp1.shape )                     # DEBUG

        x = tf.reshape( outp1, [-1, self.nCols, self.nHiddenDims] )
        print( '\tRNNPool/col-wise RNN2-x: ', x.shape )                     # DEBUG
        _, outp23 = tf.nn.dynamic_rnn( cell=self.rnn2, inputs=x, dtype=tf.float32)
        _, outp24 = tf.nn.dynamic_rnn( cell=self.rnn2, inputs=tf.reverse( x, [1] ), dtype=tf.float32)
        print( '\tRNNPool/col-wise RNN2-out: ', outp23.shape, outp24.shape )                     # DEBUG
        # debugPrint = tf.compat.v1.concat( [outp23, outp24], 1 )                                 # DEBUG

        #return tf.concat( [outp23, outp24, outp21, outp22], 1 )#, debugPrint
        return tf.concat( [outp21, outp22, outp23, outp24], 1 )


class RNNPool( tf.keras.layers.Layer ):
    def __init__( self, out_channel, kernel_size, strides=(1, 1), units=8, trainable=True, name='RNNPool' ):
        super( RNNPool, self ).__init__(name=name)
        assert out_channel % 4 == 0
        rank = 2
        self.kernel_size = conv_utils.normalize_tuple( kernel_size, rank, 'kernel_size' )
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.nHiddenDims = units
        self.out_channel = out_channel
        self.trainable = trainable
        #self.name = 'RNNPool' if self.name == None else self.name + '/RNNPool'

    def build( self, inputShape ):
        #self.inCh = inputShape.as_list[-1]
        #with tf.name_scope( self.name ):
        self.rnnPoolOP = RNNPoolOP( self.out_channel, self.kernel_size[0], self.kernel_size[1],
                                    self.nHiddenDims, self.trainable, name=self.name + '/RNNPool' )

    def call( self, inputs ):
        #with tf.name_scope( self.name ):
        inCh = inputs.shape.as_list()[-1]
        patches = tf.image.extract_patches( inputs,
                                            sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
                                            strides=[1, self.strides[0], self.strides[1], 1],
                                            rates=[1, 1, 1, 1],
                                            padding='VALID' )
        print( 'RNNPool/patches: ', patches.shape )                     # DEBUG
        outShape = patches.shape.as_list()
        patches = tf.reshape( patches, [-1, self.kernel_size[0], self.kernel_size[1], inCh] )
        print( 'RNNPool/patches: ', patches.shape )                     # DEBUG
        outp = self.rnnPoolOP( patches )
        print( 'RNNPool/out: ', outp.shape )                     # DEBUG
        outp = tf.reshape( outp, [-1, outShape[1], outShape[2], outp.shape[-1]] )
        print( 'RNNPool/out: ', outp.shape )                     # DEBUG

        return outp
