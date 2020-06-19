## This code is built on https://github.com/yxlijun/S3FD.pytorch

#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# import torch
# import torch.nn as nn
# import torch.nn.init as init
# from torch.autograd import Function
# from torch.autograd import Variable

import tensorflow as tf

#from data.config import cfg
from tf_keras_bbox_utils import decode, nms
# from torch.autograd import Function


class L2Norm( tf.keras.layers.Layer ):
    def __init__( self, n_channels, scale, trainable=True, data_format='channels_last', name='L2Norm' ):
        super(L2Norm, self).__init__()
        #with tf.variable_scope(name, "l2_normalize", [inputs]) as name:
        gamma = scale or None
        self.eps = 1e-10
        self.axis = -1 if data_format == 'channels_last' else 1
        self.weight = tf.compat.v1.get_variable(name=name + '/weight', shape=(n_channels,), trainable=trainable, initializer=tf.constant_initializer([gamma] * n_channels))
        if data_format == 'channels_last':
            self.weight = tf.reshape(self.weight, [1, 1, 1, -1], name='reshape')
        else:
            self.weight = tf.reshape(self.weight, [1, -1, 1, 1], name='reshape')


    def call( self, x ):
        #with tf.variable_scope(name, "l2_normalize", [inputs]) as name:
        square_sum = tf.reduce_sum( tf.square( x ), self.axis, keepdims=True )
        inputs_inv_norm = tf.math.rsqrt( tf.maximum( square_sum, self.eps ) )

        return tf.multiply( tf.multiply( x, inputs_inv_norm, 'normalize' ), self.weight, 'rescale')


def detect_function(cfg, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            conf_data: (tensor) Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
        """
        # print( 'cfg: ', cfg )
        # print( 'loc: ', loc_data )
        # print( 'conf: ', conf_data )
        # print( 'priors: ', prior_data )

        num = loc_data.shape[0]
        num_priors = prior_data.shape[0]

        conf_preds = tf.transpose( tf.reshape( conf_data, [num, num_priors, cfg.NUM_CLASSES] ), perm=[0, 2, 1] )
        batch_priors = tf.broadcast_to( tf.reshape( prior_data, [-1, num_priors, 4] ), [num, num_priors, 4] )
        batch_priors = tf.reshape( batch_priors, [-1, 4] )

        decoded_boxes = decode( tf.reshape( loc_data, [-1, 4] ), batch_priors, cfg.VARIANCE)
        decoded_boxes = tf.reshape( decoded_boxes, [num, num_priors, 4] )

        output = tf.compat.v1.zeros( [num, cfg.NUM_CLASSES, cfg.TOP_K, 5] )

        # print( 'conf_preds: ', conf_preds )
        # print( 'batch_priors: ', batch_priors )
        # print( 'decoded_boxes: ', decoded_boxes )
        # print( 'output: ', output.shape )

        output = []
        for i in range(num):
            boxes = decoded_boxes[i]
            conf_scores = conf_preds[i]

            output_img = [tf.compat.v1.zeros( [cfg.TOP_K, 5] )]
            for cl in range(1, cfg.NUM_CLASSES):
                c_mask = tf.compat.v1.math.greater( conf_scores[cl], cfg.CONF_THRESH )

                if tf.compat.v1.count_nonzero( c_mask ) <= 0:
                    continue

                scores = conf_scores[cl][c_mask]
                # l_mask = tf.broadcast_to( tf.expand_dims( c_mask, 1 ), boxes.shape.as_list() )
                # boxes_ = tf.reshape( boxes[l_mask], [-1, 4] )
                boxes_ = boxes[c_mask]
                ids, count = nms( boxes_, scores, cfg.NMS_THRESH, cfg.NMS_TOP_K)
                # print( 'ids: ', ids )
                # print( 'count: ', count )
                count = count if count < cfg.TOP_K else cfg.TOP_K
                # print( 'ids[:count]: ', ids[:count] )

                # output[i, cl, :count] = tf.compat.v1.concat( [tf.expand_dims( scores[ids[:count]], 1 ),
                #                                               boxes_[ids[:count]]], 1 )
                idx = ids[:count]
                pad = cfg.TOP_K - count
                pad_zeros = tf.compat.v1.zeros( [pad, 5] )
                output_temp = tf.compat.v1.concat( [tf.expand_dims( tf.compat.v1.gather( scores, idx ), 1 ),
                                                         tf.compat.v1.gather( boxes_, idx, axis=0 )], 1 )
                output_img.append(tf.compat.v1.concat([output_temp,pad_zeros], 0))
            # print(i)
            # import pdb;pdb.set_trace()
            output.append( tf.compat.v1.stack(output_img) )
        output = tf.compat.v1.stack( output )

        return output

