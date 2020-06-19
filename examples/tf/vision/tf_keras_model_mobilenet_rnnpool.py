# import os
# import re
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import torch.utils.checkpoint as cp
# from collections import OrderedDict
# from torchvision.models.utils import load_state_dict_from_url
# from edgeml_pytorch.graph.rnnpool import *

import tensorflow as tf
#from tensorflow.python.keras.engine.base_layer import Layer
import tf_keras_rnnpool as rnnpool

# import time

__all__ = ['MobileNetV2', 'mobilenetv2_rnnpool']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


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



def _dropout( x, keep_drop, is_train, name='dp' ):
    with tf.variable_scope(name):
        dp = tf.cond( is_train, lambda: tf.nn.dropout(x, keep_drop, name=name), lambda: x )
    return dp


class ConvBN( tf.keras.layers.Layer ):
    def __init__( self, out_channel, kernel_size=3, stride=1, padding='SAME', name='ConvBN',
                  momentum=0.01, trainable=True, reuse=None ):
        super( ConvBN, self ).__init__(name = name)
        self.reuse = reuse

        # def build( self ):
        #with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
        self.model = tf.keras.Sequential( [
            tf.keras.layers.Conv2D( out_channel, kernel_size, stride, padding, use_bias=False ),
            tf.compat.v1.keras.layers.BatchNormalization( momentum=momentum ) #, trainable=trainable )
            ] )

    def call( self, inp ):
        #with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
        return self.model( inp )


class ConvBNReLU( tf.keras.layers.Layer ):
    def __init__( self, out_channel, kernel_size=3, stride=1, padding='SAME', name='ConvBNReLU',
                  momentum=0.01, trainable=True, reuse=None ):
        super( ConvBNReLU, self ).__init__(name = name)
        self.reuse = reuse

        # def build( self ):
        #with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
        self.model = tf.keras.Sequential( [
            tf.keras.layers.Conv2D( out_channel, kernel_size, stride, padding, use_bias=False ),
            tf.compat.v1.keras.layers.BatchNormalization( momentum=momentum ),#, trainable=trainable ),
            tf.compat.v1.keras.layers.ReLU( max_value=6 )
            ] )

    def call( self, inp ):
        #with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
        return self.model( inp )


class dwConvBNReLU( tf.keras.layers.Layer ):
    def __init__( self, kernel_size=3, stride=1, padding='SAME', name='dwConvBNReLU',
                  momentum=0.01, trainable=False, reuse=None ):
        super( dwConvBNReLU, self ).__init__(name = name)
        self.reuse = reuse

        # def build( self ):
        #with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
        self.model = tf.keras.Sequential( [
            tf.compat.v1.keras.layers.DepthwiseConv2D( kernel_size, stride, padding, use_bias=False ),
            tf.compat.v1.keras.layers.BatchNormalization( momentum=momentum ),#, trainable=trainable ),
            tf.compat.v1.keras.layers.ReLU( max_value=6 )
            ] )

    def call( self, inp ):
        #with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
        return self.model( inp )


class InvertedResidual( tf.keras.layers.Layer ):
    def __init__( self, out_channel, kernel_size=3, stride=1, padding='SAME', expand_ratio=1,
                  name='InvertedResidual', momentum=0.01, trainable=True, reuse=None ):
        super( InvertedResidual, self ).__init__(name = name)
        assert stride in [1, 2]
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.expand_ratio = expand_ratio
        self.momentum = momentum
        #self.name = 'InvertedResidual' if name == None else name + '/InvertedResidual'
        self.reuse = reuse

    def build( self, inputShape ):
        hidden_dim = int(round(inputShape.as_list()[-1] * self.expand_ratio))

        self.dimMatch = self.stride == 1  and  inputShape.as_list()[-1] == self.out_channel
        layers = []
        #with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
        if self.expand_ratio != 1:
            # pw
            layers = [ ConvBNReLU( hidden_dim, kernel_size=1, momentum=self.momentum ) ]
        layers.extend( [
            # dw
            dwConvBNReLU( stride=self.stride, momentum=self.momentum ),
            # pw-linear
            ConvBN( self.out_channel, kernel_size=1, momentum=self.momentum )
            #tf.compat.v1.keras.layers.Conv2D( self.out_channel, 1, 1, self.padding, use_bias=False ),
            #tf.compat.v1.keras.layers.BatchNormalization( momentum=0.01)#, trainable=self.trainable )
            ] )
        self.model = tf.keras.Sequential( layers )

    def call( self, inp ):
        #with tf.variable_scope( self.name ):
        if self.dimMatch:
            return self.model( inp ) + inp
        else:
            return self.model( inp )


class MobileNetV2( tf.keras.Model ):
    def __init__(self, 
                 num_classes=1000, 
                 width_mult=0.5,
                 inverted_residual_setting=None, 
                 round_nearest=8,
                 block=None,
                 last_channel = 1280 ):
                 #trainable=True, name=None ):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.last_channel = last_channel
        self.round_nearest = round_nearest
        #self.trainable = trainable

        # self.train_dir = train_dir
        # train_data = inputs['train_data']
        # train_labels = inputs['train_labels']
        # test_data = inputs['test_data']
        # test_labels = inputs['test_labels']
        # self.hp = hp
        tf.compat.v1.set_random_seed(420)
        
        # if len(train_data) != len(train_labels):
        #     print('Number of training samples is wrong')
        #     return
        # if len(test_data) != len(test_labels):
        #     print('Number of testing samples is wrong')
        #     return
        # input_shape = train_data[0].shape
        # for i in range(1, len(train_data)):
        #     if input_shape != train_data[i].shape:
        #         print('Shapes of training samples are not the same')
        #         return
        # for i in range(len(test_data)):
        #     if input_shape != test_data[i].shape:
        #         print('Shape of testing samples are not the same')
        #         return
        #     if i == len(test_data) - 1:
        #         self.input_shape = list(input_shape)
        # #While getting minimum and maximum for quantization, use the values of training data, not testing data
        # self.minval = np.amin(train_data)
        # self.maxval = np.amax(train_data)
        # self.train_data = Data([train_data, train_labels])
        # self.test_data = Data([test_data, test_labels], shuffle=False)

        if block is None:
            block = InvertedResidual

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                # [1, 16, 1, 1],
                # [6, 24, 2, 2],
                #[3, 64, 1, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        self.block = block
        self.inverted_residual_setting = inverted_residual_setting

    def build( self, inputShape ):
        ### build_model
        # filterSize = self.hp['filterSize']
        # channelSize = self.hp['channelSize']
        # fullconnectSize = self.hp['fullconnectSize']
        # nClass = self.hp['nClass']
        # model_type = self.hp['model_type']
        # self.shapes = {}
        # tf.compat.v1.reset_default_graph()
        # self.graph = tf.Graph()
        # with self.graph.as_default():
        # self.inputs = tf.compat.v1.placeholder(tf.float32, [None] + inputShape, name='inputs')
        # self.labels = tf.compat.v1.placeholder(tf.int32, [None, self.num_classes], name='labels')
        # self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, name='learning_rate')

        feature_channel = 8
        feature_channel = _make_divisible( feature_channel, self.round_nearest )
        layers = [
            # building first layer
            ConvBNReLU( feature_channel, kernel_size=3, stride=2 ),#, name="Layer0/ConvBNReLU" ),
            # building RNNPool layer
            rnnpool.RNNPool( kernel_size=6, stride=4,
                             nHiddenDims=8, nHiddenDimsBiDir=8, inputDims=feature_channel )
            ]

        # building inverted residual blocks
        for j, (t, c, n, s) in enumerate( self.inverted_residual_setting ):
            output_channel = _make_divisible( c * self.width_mult, self.round_nearest )
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append( self.block( output_channel, stride=stride, expand_ratio=t,
                                            name="Bottleneck_" + str(j) + "." + str(i) ) )
                print( j, i, t, c, n, s, output_channel, stride)

        # building last several layers
        last_channel = _make_divisible( self.last_channel * max(1.0, self.width_mult), self.round_nearest )
        layers.extend( [
            ConvBNReLU( last_channel, kernel_size=1, name="Classifier/ConvBNReLU" ),
            tf.compat.v1.keras.layers.AvgPool2D( 7 ),
            # building classifier
            #nn.Dropout(0.2)
            tf.compat.v1.keras.layers.Reshape( (last_channel,) ),
            tf.compat.v1.keras.layers.Dense( self.num_classes, name='logits' )
            ] )

        self.model = tf.keras.models.Sequential( layers )

    def call( self, inp ):
        return self.model( inp )


        """
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        """
    """
    def train(self, x):
        batch_size = x.shape[0]
        num_steps_per_epoch = 1000
        decay = [5, 10, 20, 40]
        
        with tf.Session(graph=self.graph) as sess:
            tf.random.set_random_seed(42)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                init_step = int(ckpt.model_checkpoint_path.split('-')[-1])
                max_acc = int(os.path.split(ckpt.model_checkpoint_path)[-1].split('.')[0]) / 10000
                sess.run(tf.assign(self.global_step, init_step))
            else:
                max_acc = 0
                init_step = 0

            iter_count = init_step
            starting_epoch = init_step // num_steps_per_epoch
            for e in range(starting_epoch, decay[-1]):
                print('Epoch %d'%(e+1))
                for i in range(len(decay)):
                    if e < decay[i]:
                        _lr = lr[i]
                        break
                for _ in range(num_steps_per_epoch):
                    start_time = time.time()
                    batch_x, batch_y = self.train_data.next_batch(batch_size)
                    fd_train = {self.inputs: batch_x, self.labels: batch_y, self.learning_rate: _lr}
                    _, train_acc, train_loss = sess.run([self.train_op, self.accuracy, self.loss], feed_dict=fd_train)
                    iter_count += 1
                    duration = time.time() - start_time
                    if iter_count % 100 == 0:
                        print('iter_count %d, training_acc: %.4f, training_loss, %.4f, duration: %.3f'%(iter_count, train_acc, train_loss, duration))
                test_acc, test_loss, test_cm = self.test(sess, is_print=False)
                if test_acc > max_acc:
                    max_acc = test_acc
                    checkpoint_path = os.path.join(self.train_dir, str(int(max_acc*10000))+'.ckpt')
                    saver.save(sess, checkpoint_path, global_step=iter_count)
                print('Epoch %d, testing_acc: %.4f, testing_loss, %.4f'\
                        %(e+1, test_acc, test_loss))
                print('Testing Confusion Matrix: \n', test_cm)
            print('Finished training!')
            test_acc, _, _ = self.test(sess, is_print=False)
            if test_acc > max_acc:
                max_acc = test_acc
                checkpoint = os.path.join(self.train_dir, str(int(max_acc*10000))+'.ckpt')
                saver.save(sess, checkpoint, global_step=iter_count)

    def test(self, sess, is_print=True):
        batch_size = self.hp['batch_size']
        nClass = self.hp['nClass']
        num_steps_per_testing = int(np.ceil(self.test_data.num_samples/batch_size))
        test_corr = 0
        test_loss = 0.0
        test_cm = np.zeros([nClass, nClass], dtype=int)
        for _ in range(num_steps_per_testing):
            batch_x, batch_y = self.test_data.next_batch(batch_size)
            actual_batch_size = len(batch_x)
            fd_test = {self.inputs: batch_x, self.labels: batch_y}
            _corr, _loss, _cm = sess.run([self.correct, self.loss, self.confusion], feed_dict=fd_test)
            test_corr += np.sum(_corr)
            test_loss += _loss * actual_batch_size
            test_cm += _cm
        test_acc = test_corr / self.test_data.num_samples
        test_loss /= self.test_data.num_samples

        if is_print:
            print('Test acc: %.4f, test loss: %.4f'%(test_acc, test_loss))
            print('Testing Confusion Matrix: \n', test_cm)

        return test_acc, test_loss, test_cm
    """


def mobilenetv2_rnnpool( pretrained=False, progress=True, **kwargs ):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    """
    if pretrained:
        state_dict = load_state_dict_from_url( model_urls['mobilenet_v2'], progress=progress )
        model.load_state_dict( state_dict )
    """

    return model

