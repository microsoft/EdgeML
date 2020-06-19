## This code is built on https://github.com/yxlijun/S3FD.pytorch
#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
# import torch
# import torch.nn as nn
# import torch.nn.init as init
# import torch.nn.functional as F
# from torch.autograd import Variable

# from layers import *
# from data.config import cfg

# from edgeml_pytorch.graph.rnnpool import *

import sys

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
#from tensorflow.python.keras.engine.base_layer import Layer
import tf_keras_rnnpool as rnnpool
import tf_keras_blocks as blk
from tf_keras_utils import L2Norm, detect_function
from tf_keras_prior_box import priorBox

class S3FD( tf.keras.Model ):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__( self, phase, cfg, data_format='channels_last' ):
        super(S3FD, self).__init__(name='S3FD_Net')
        self.phase = phase
        self.cfg = cfg
        self.num_classes = self.cfg.NUM_CLASSES
        self.data_format = data_format
        '''
        #self.priorbox = PriorBox(size,cfg)
        #self.priors = Variable(self.priorbox.forward(), volatile=True)
        '''

        #def build( self, inputShape ):
        # feature extraction layers

        self.conv = blk.ConvBNReLU(8, strides=2)

        self.rnn_model = rnnpool.RNNPool( 64, kernel_size=8, strides=4, units=16, name=self.name )

        self.bottleneck = MobileNetV2().layers
        self.nBottleneck = len( self.bottleneck )

        # prediction layers
        # Layer learns to scale the l2 normalized features from conv4_3
        l2normTrainable = True
        self.L2Norm3_5 = [ L2Norm(32, 10, l2normTrainable, name=self.name + '/L2Norm3_3'),
                           L2Norm(64, 8, l2normTrainable, name=self.name + '/L2Norm4_3'),
                           L2Norm(96, 5, l2normTrainable, name=self.name + '/L2Norm5_3') ]
        self.loc, self.conf = multibox( self.nBottleneck, self.num_classes )

        if self.phase == 'test':
            self.softmax = tf.compat.v1.keras.layers.Softmax(axis=-1)
            # self.detect = Detect(cfg)

    def call( self, x ):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        batch_size = x.shape.as_list()[0]
        if self.data_format == 'channels_first'  or  self.data_format == 'NCHW':
            x = tf.transpose( x, perm=[0, 2, 3, 1] )

        imgSize = x.shape.as_list()[1:3]    # NHWC

        ### Feature extraction line
        # inputs_placeholder = tf.compat.v1.placeholder( tf.float32, shape=(batch_size, 240, 320, 1) )
        # y = self.conv(inputs_placeholder)
        # import pdb;pdb.set_trace()
        # print(x)
        # y_ = sess.run( [y], feed_dict={inputs_placeholder: x} )
        # print(y_)

        x = self.conv(x)
        
        # sess = tf.compat.v1.Session()
        # print(sess.run(x))
        x = self.rnn_model( x )             # RNNPool
        debugPrint = x
        ### New code: PyTorch RNNPool uses "VALID" padding mode which results output shape [-1, 59, 79, 64]
        ###				To make it [-1, 60, 80, 64], duplicate one col to the right and one row at the bottom.
        x = tf.compat.v1.pad( x, [[0,0], [0,1], [0,1], [0,0]], mode="SYMMETRIC" )
        # debugPrint = x

        # apply bottlenecks and prepare features
        featX = []

        ########
        ### bottleneck block:       self.bottleneck[0]
        ### bottleneck layer:       self.bottleneck[0].layers[0]
        ### PW, DW, PW:             self.bottleneck[0].layers[0].model.layers[0]
        ### Conv2d/dwConv, BN, ReLU:self.bottleneck[0].layers[0].model.layers[0].model.layers[0]
        ### depthwise_kernel        self.bottleneck[2].layers[0].model.layers[1].model.layers[0].depthwise_kernel
        ### kernel_size             self.bottleneck[2].layers[0].model.layers[1].model.layers[0].kernel_size
        ### strides                 self.bottleneck[2].layers[0].model.layers[1].model.layers[0].strides

        # DEBUG: bottleneck[0]
        # self.bottleneck[0]( x )     # build model
        # print( 'bottleneck[0].layers[0].model.layers: ', self.bottleneck[0].layers[0].model.layers )

        ### individual layer execution in a bottleneck
        ## ConvBNReLU( hidden_dim, kernel_size=1, momentum=self.momentum )     # pw
        # debugPrint = self.bottleneck[0].layers[0].model.layers[0]( x )
        # debugPrint = self.bottleneck[0].layers[0].model.layers[0].model.layer[0]( x )     # Conv2d in PW

        ## dwConvBNReLU( strides=self.strides, momentum=self.momentum )        # dw
        # debugPrint = self.bottleneck[0].layers[0].model.layers[1]( debugPrint )

        # # ConvBN( self.out_channel, kernel_size=1, momentum=self.momentum )   # pw-linear
        # debugPrint = self.bottleneck[0].layers[0].model.layers[2]( debugPrint )

        ### individual bottleneck execution
        # debugPrint = self.bottleneck[0].layers[0]( debugPrint )
        # debugPrint = self.bottleneck[0].layers[1]( debugPrint )
        # debugPrint = self.bottleneck[0].layers[2]( debugPrint )

        ### bottleneck block execution
        # debugPrint = self.bottleneck[0]( x )
        ########

        for k in range( self.nBottleneck ):
            x = self.bottleneck[k]( x )
            if k < len(self.L2Norm3_5):
            #     if k == 2:
            #         debugPrint = self.L2Norm3_5[k]( x )
                featX.append( self.L2Norm3_5[k]( x ) )  # apply featrue normalization
            else:
                featX.append( x )

        # debugPrint = x

        ### Prediction line
        # apply multibox head to source layers
        #with tf.name_scope(name=self.name):
        loc  = [ self.loc[0]( featX[0] ) ]
        # Max-out BG Label
        confX = self.conf[0]( featX[0] )
        nBGgroups = confX.shape.as_list()[-1] - self.num_classes + 1
        max_conf = tf.compat.v1.reduce_max( confX[:, :, :, 0:nBGgroups], axis=-1, keepdims=True )
        conf = [ tf.compat.v1.concat( [max_conf, confX[:, :, :, nBGgroups:]], axis=-1 ) ]
        features_maps = [ loc[0].shape.as_list()[1:3] ]

        # debugPrint = features_maps[-1]     # DEBUG

        # process featres from bottleneck 2nd:last
        for i in range( 1, self.nBottleneck ):
            loc.append( self.loc[i]( featX[i] ) )
            conf.append( self.conf[i]( featX[i] ) )
            features_maps += [loc[i].shape.as_list()[1:3]]     # HW
            # if i == 3:
            #     debugPrint = loc[-1]     # DEBUG
            #     # debugPrint = conf[-1]     # DEBUG

        # import pdb;pdb.set_trace()
        priors = priorBox( imgSize, features_maps, self.cfg )
        #priors = tf.expand_dims( priors, [0] )
        # debugPrint = priors     # DEBUG

        loc  = tf.compat.v1.concat( [tf.reshape( o, [batch_size, -1] ) for o in loc], 1 )
        conf = tf.compat.v1.concat( [tf.reshape( o, [batch_size, -1] ) for o in conf], 1 )

        # return loc,conf

        if self.phase == 'test':
            output = detect_function( self.cfg,
                tf.reshape( loc, [loc.shape.as_list()[0], -1, 4 ] ),                   # loc preds
                self.softmax( tf.reshape( conf, [conf.shape.as_list()[0], -1, self.num_classes] ) ),    # conf preds
                priors  #.type( type(x.data) )                  # default boxes
            )

        else:
            output = (
                tf.reshape( loc, [loc.shape.as_list()[0], -1, 4 ] ),
                tf.reshape( conf, [conf.shape.as_list()[0], -1, self.num_classes] ),
                priors
            )

        return output, debugPrint


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v



class MobileNetV2:
    def __init__( self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8 ):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        # super(MobileNetV2, self).__init__()
        block = blk.InvertedResidual
        #input_channel = 32
        #last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [2, 32, 4, 1],
                [2, 64, 4, 1],
                [2, 96, 3, 2],
                [2, 128, 3, 1]
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        self.layers = []
        subBlocks = []
        # building inverted residual blocks
        for j, (t, c, n, s) in enumerate( inverted_residual_setting ):
            output_channel = _make_divisible( c * width_mult, round_nearest )
            
            for i in range(n):
                stride = s if i == 0 else 1
                subBlocks.append( block( output_channel, strides=stride, expand_ratio=t, name=str(i) ) )
            # import pdb;pdb.set_trace()
            self.layers.append( tf.keras.Sequential( subBlocks[j*3:j*3+3], name='bottleneck_'+str(j) ) )



def multibox( nBottlenecks, num_classes ):
    loc_layers  = []
    conf_layers = []
    for ii in range( nBottlenecks ):    #[16, 24, 32, 64, 96]:
        loc_layers  += [tf.keras.layers.Conv2D( 4, kernel_size=3, padding='SAME',
                                                name='loc_'+str(ii), use_bias=True )]
        nClass = 3 + (num_classes - 1) if ii == 0 else num_classes
        conf_layers += [tf.keras.layers.Conv2D( nClass, kernel_size=3, padding='SAME',
                                                name='conf_'+str(ii), use_bias=True )]

    return loc_layers, conf_layers


def build_s3fd( phase, cfg, data_format='channels_last' ):
    return S3FD( phase, cfg, data_format )


if __name__ == '__main__':
    os.environ['IS_QVGA_MONO'] = '1'
    from data.choose_config import cfg
    cfg = cfg.cfg

    inputs = tf.compat.v1.random.normal([1, 320, 240, 1])
    # with tf.compat.v1.variable_scope( 'model_scope', default_name=None, values=[inputs], reuse=tf.compat.v1.AUTO_REUSE):
    net = S3FD( 'test', cfg )

    output = net( inputs )
    print( type( output ) )
    print( output )

    net.summary()
    #tf.keras.utils.plot_model(net, 'tf_keras_model.png')

    """
    print( "TF parameters" )
    with open("paramNames_tf.txt", "w") as f:
        # for v in tf.compat.v1.trainable_variables():
        # totList = []
        for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES): #GLOBAL_VARIABLES, TRAINABLE_VARIABLES
            print( v.name, '\t', v.shape.as_list() )
            f.write( '{:90s}\t{}\n'.format(v.name, v.shape.as_list()) )
            if 'moving_variance' in v.name:
                f.write( '\n' )
            # if 'global_step' not in v.name:
            #     totList.append( [v.name, v.shape.as_list()] )
    """
