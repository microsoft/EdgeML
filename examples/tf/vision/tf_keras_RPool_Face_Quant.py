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

import numpy as np
import tensorflow as tf
#from tensorflow.python.keras.engine.base_layer import Layer
import tf_keras_rnnpool as rnnpool
import tf_keras_blocks as blk
from tf_keras_utils import L2Norm, Detect
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
        self.num_classes = cfg.NUM_CLASSES
        self.data_format = data_format
        '''
        #self.priorbox = PriorBox(size,cfg)
        #self.priors = Variable(self.priorbox.forward(), volatile=True)
        '''

        #def build( self, inputShape ):
        # feature extraction layers
        self.conv_top = tf.keras.Sequential( [
                            blk.ConvBNReLU( 4, kernel_size=3, strides=2, name='0' ),
                            blk.ConvBNReLU( 4, kernel_size=3, name='1' )
                            ], name='conv_top' )

        self.rnn_model = rnnpool.RNNPool( 32, kernel_size=8, strides=4, units=8, name=self.name )

        self.bottleneck = MobileNetV2().layers
        self.nBottleneck = len( self.bottleneck )

        # prediction layers
        # Layer learns to scale the l2 normalized features from conv4_3
        l2normTrainable = True
        self.L2Norm3   = L2Norm(4, 10, l2normTrainable, name=self.name + '/L2Norm3_3')
        self.L2Norm4_5 = [ L2Norm(16, 8, l2normTrainable, name=self.name + '/L2Norm4_3'),
                           L2Norm(24, 5, l2normTrainable, name=self.name + '/L2Norm5_3') ]
        self.loc, self.conf = multibox( self.nBottleneck, self.num_classes )

        if self.phase == 'test':
            self.softmax = tf.compat.v1.keras.layers.Softmax(axis=-1)
            self.detect = Detect(cfg)

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

        imgSize = x.shape.as_list()[1:3]     # NWHC

        x = self.conv_top( x )          # the firs two conv2d: [ConvBNReLU, ConvBNReLU]

        featX = [ self.L2Norm3( x ) ]                       # featX = [Conv3_3]

        x = self.rnn_model( x )         # RNNPool

        # apply bottlenecks: [16, 24, 32, 64, 96]
        for k in range( self.nBottleneck ):
            x = self.bottleneck[k]( x )
            if k < 2:
                featX.append( self.L2Norm4_5[k]( x ) )      # featX = [Conv3_3, bottleneck 1, bottleneck 2]
            else:
                featX.append( x )                           # featX = [Conv3_3, bottleneck 1, bottleneck 2, ...]

        # apply multibox head to source layers
        # featX = [Conv3_3, bottleneck 1, bottleneck 2, bottleneck 3, bottleneck 4, bottleneck 5]
        # Conv3_3: loc[0]=[ConvBNReLU(4,8), Conv2d(8,x)]
        #with tf.name_scope(name=self.name):
        loc  = [ self.loc[0]( featX[0] ) ]
        confX = self.conf[0]( featX[0] )
        # Max-out BG Label
        max_conf = tf.compat.v1.reduce_max( confX[:, :, :, 0:3], axis=-1, keepdims=True )
        conf = [ tf.compat.v1.concat( [max_conf, confX[:, :, :, 3:]], axis=-1 ) ]
        features_maps = [ [loc[0].shape.as_list()[1], loc[0].shape.as_list()[2]] ]

        # process Conv4_3, ..., Conv7_2
        for i in range( 1, self.nBottleneck + 1 ):
            loc.append( self.loc[i]( featX[i] ) )
            conf.append( self.conf[i]( featX[i] ) )
            features_maps += [[loc[i].shape.as_list()[1], loc[i].shape.as_list()[2]]]     # HW

        priors = priorBox( imgSize, features_maps, cfg )

        loc  = tf.compat.v1.concat( [tf.reshape( o, [batch_size, -1] ) for o in loc], 1 )
        conf = tf.compat.v1.concat( [tf.reshape( o, [batch_size, -1] ) for o in conf], 1 )

        if self.phase == 'test':
            output = self.Detect(
                tf.reshape( loc, [loc.shape.as_list()[0], -1, 4 ] ),                   # loc preds
                self.softmax( tf.reshape( conf, [conf.shape.as_list()[0], -1, self.num_classes] ) ),    # conf preds
                priors.type( type(x.data) )                  # default boxes
            )

        else:
            output = (
                tf.reshape( loc, [loc.shape.as_list()[0], -1, 4 ] ),
                tf.reshape( conf, [conf.shape.as_list()[0], -1, self.num_classes] ),
                priors
            )

        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            mdata = tf.load(base_file, map_location=lambda storage, loc: storage)
            weights = mdata['weight']
            epoch = mdata['epoch']
            self.load_state_dict(weights)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
        return epoch

    def xavier(self, param):
        tf.keras.initializers.GlorotUniform()

    def weights_init(self, m):
        if isinstance(m, tf.keras.layers.Conv2D):
            self.xavier(m.weight.data)
            #m.bias.data.zero_()




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
                # [1, 16, 1, 1],
                # [1, 24, 1, 1],
                [2, 16, 4, 1],
                [2, 24, 4, 2],
                [2, 32, 2, 2],
                [2, 64, 1, 2],
                [2, 96, 1, 2],
                # [2, 320, 1, 2],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        self.layers = []
        # building inverted residual blocks
        for j, (t, c, n, s) in enumerate( inverted_residual_setting ):
            output_channel = _make_divisible( c * width_mult, round_nearest )
            subBlocks = []
            for i in range(n):
                stride = s if i == 0 else 1
                subBlocks.append( block( output_channel, stride, expand_ratio=t, name=str(i) ) )
            self.layers.append( tf.keras.Sequential( subBlocks, name='bottleneck_'+str(j) ) )


def multibox( nBottlenecks, num_classes ):
    loc_layers  = [tf.keras.Sequential([ blk.ConvBNReLU( 8, kernel_size=3, strides=2 ),
                                         tf.keras.layers.Conv2D( 4,  kernel_size=3, padding='SAME', use_bias=False ) ],
                                       name='loc_0') ]
    conf_layers = [tf.keras.Sequential([ blk.ConvBNReLU( 8, kernel_size=3, strides=2),
                                         tf.keras.layers.Conv2D( 3 + (num_classes - 1), kernel_size=3, padding='SAME', use_bias=False ) ],
                                       name='conf_0') ]

    for ii in range( nBottlenecks ):    #[16, 24, 32, 64, 96]:
        loc_layers  += [tf.keras.layers.Conv2D( 4, kernel_size=3, padding='SAME',
                                                name='loc_'+str(ii + 1), use_bias=False )]
        conf_layers += [tf.keras.layers.Conv2D( num_classes, kernel_size=3, padding='SAME',
                                                name='conf_'+str(ii + 1), use_bias=False )]

    return loc_layers, conf_layers


def build_s3fd( phase, cfg ):
    return S3FD( phase, cfg )


if __name__ == '__main__':
    from data.config import cfg
    phase = 'train'
    num_classes = 2
    inputs = tf.compat.v1.random.normal([1, 640, 640, 3])
    with tf.compat.v1.variable_scope( 'model_scope', default_name=None, values=[inputs], reuse=tf.compat.v1.AUTO_REUSE):
        net = S3FD( phase, cfg )

        output = net( inputs )
        print( type( output ) )

        net.summary()
        #tf.keras.utils.plot_model(net, 'tf_keras_model.png')

    print( "TF parameters" )
    for v in tf.compat.v1.trainable_variables():
    # totList = []
    # for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES): #GLOBAL_VARIABLES, TRAINABLE_VARIABLES
        print( v.name, '\t', v.shape.as_list() )
        # if 'global_step' not in v.name:
        #     totList.append( [v.name, v.shape.as_list()] )
