
import tensorflow as tf
#from tensorflow.python.keras.engine.base_layer import Layer

# import time

DEBUG_Printed = 0


def _dropout( x, keep_drop, is_train, name='dp' ):
    with tf.variable_scope(name):
        dp = tf.cond( is_train, lambda: tf.nn.dropout(x, keep_drop, name=name), lambda: x )
    return dp


class ConvBN( tf.keras.layers.Layer ):
    def __init__( self, out_channel, kernel_size=3, strides=1, padding='SAME', name='ConvBN',
                  epsilon=1e-05, momentum=0.1, trainable=True, reuse=None ):
        super( ConvBN, self ).__init__(name = name)
        self.reuse = reuse
        self.kernel_size = kernel_size
        self.strides = strides
        self.padSame = padding == 'SAME'  or  padding == 'same'

        # def build( self ):
        #with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
        self.model = tf.keras.Sequential( [
            tf.keras.layers.Conv2D( out_channel, kernel_size, strides, 'VALID', use_bias=False ),
            tf.compat.v1.keras.layers.BatchNormalization( epsilon=epsilon, momentum=momentum ) #, trainable=trainable )
            ], name=self.name )

    def call( self, inp ):
        #with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
        if self.padSame:
            _, h, w, _ = inp.shape.as_list()
            t = h % self.strides
            pad_height = self.kernel_size - ( self.strides if t == 0 else (h % self.strides) )
            t = w % self.strides
            pad_width  = self.kernel_size - ( self.strides if t == 0 else (w % self.strides) )
            if pad_height > 0  or  pad_width > 0:
                pad_top   = (pad_height + 1) // 2 # divied by 2
                # pad_bottom = pad_height - pad_top
                pad_left  = (pad_width + 1) // 2
                # pad_right = pad_width - pad_left
                inp = tf.compat.v1.pad( inp, [[0, 0], [pad_top, pad_top], [pad_left, pad_left], [0, 0]])

        return self.model( inp )


class ConvBNReLU( tf.keras.layers.Layer ):
    def __init__( self, out_channel, kernel_size=3, strides=1, padding='SAME', name='ConvBNReLU',
                  epsilon=1e-05, momentum=0.1, trainable=True, reuse=None ):
        # epsilon=1e-05, momentum=0.1
        super( ConvBNReLU, self ).__init__(name = name)
        self.reuse = reuse
        self.kernel_size = kernel_size
        self.strides = strides
        self.padSame = padding == 'SAME'  or  padding == 'same'

        # def build( self ):
        #with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
        self.model = tf.keras.Sequential( [
            tf.keras.layers.Conv2D( out_channel, kernel_size, strides, 'VALID', use_bias=False ),
            tf.compat.v1.keras.layers.BatchNormalization( epsilon=epsilon, momentum=momentum ),#, trainable=trainable ),
            tf.compat.v1.keras.layers.ReLU( max_value=6 )
            ], name=self.name )

    def call( self, inp ):
        if self.padSame:
            _, h, w, _ = inp.shape.as_list()
            t = h % self.strides
            pad_height = self.kernel_size - ( self.strides if t == 0 else (h % self.strides) )
            t = w % self.strides
            pad_width  = self.kernel_size - ( self.strides if t == 0 else (w % self.strides) )
            if pad_height > 0  or  pad_width > 0:
                pad_top   = (pad_height + 1) // 2 # divied by 2
                # pad_bottom = pad_height - pad_top
                pad_left  = (pad_width + 1) // 2
                # pad_right = pad_width - pad_left
                inp = tf.compat.v1.pad( inp, [[0, 0], [pad_top, pad_top], [pad_left, pad_left], [0, 0]])

        return self.model( inp )


class dwConvBNReLU( tf.keras.layers.Layer ):
    def __init__( self, kernel_size=3, strides=1, padding='SAME', name='dwConvBNReLU',
                  epsilon=1e-05, momentum=0.1, trainable=True, reuse=None ):
        super( dwConvBNReLU, self ).__init__(name = name)
        self.reuse = reuse
        self.kernel_size = kernel_size
        self.strides = strides
        self.padSame = padding == 'SAME'  or  padding == 'same'

        #def build( self, inputShape ):
        #with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
        self.model = tf.keras.Sequential( [
            tf.compat.v1.keras.layers.DepthwiseConv2D( kernel_size, strides, 'VALID', use_bias=False ),
            tf.compat.v1.keras.layers.BatchNormalization( epsilon=epsilon, momentum=momentum ),#, trainable=trainable ),
            tf.compat.v1.keras.layers.ReLU( max_value=6 )
            ], name=self.name )

    def call( self, inp ):
        #with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
        if self.padSame:
            _, h, w, _ = inp.shape.as_list()
            t = h % self.strides
            pad_height = self.kernel_size - ( self.strides if t == 0 else (h % self.strides) )
            t = w % self.strides
            pad_width  = self.kernel_size - ( self.strides if t == 0 else (w % self.strides) )
            if pad_height > 0  or  pad_width > 0:
                pad_top   = (pad_height + 1) // 2 # divied by 2
                # pad_bottom = pad_height - pad_top
                pad_left  = (pad_width + 1) // 2
                # pad_right = pad_width - pad_left
                inp = tf.compat.v1.pad( inp, [[0, 0], [pad_top, pad_top], [pad_left, pad_left], [0, 0]])

        return self.model( inp )


class InvertedResidual( tf.keras.layers.Layer ):
    def __init__( self, out_channel, kernel_size=3, strides=1, padding='SAME', expand_ratio=1,
                  name='InvertedResidual', epsilon=1e-05, momentum=0.1, trainable=True, reuse=None ):
        super( InvertedResidual, self ).__init__(name = name)
        assert strides in [1, 2]
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.expand_ratio = expand_ratio
        self.epsilon = epsilon
        self.momentum = momentum
        #self.name = 'InvertedResidual' if name == None else name + '/InvertedResidual'
        self.reuse = reuse

    def build( self, inputShape ):
        hidden_dim = int(round(inputShape.as_list()[-1] * self.expand_ratio))

        self.dimMatch = self.strides == 1  and  inputShape.as_list()[-1] == self.out_channel
        layers = []
        #with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
        if self.expand_ratio != 1:
            # pw
            layers = [ ConvBNReLU( hidden_dim, kernel_size=1, padding=self.padding, epsilon=self.epsilon, momentum=self.momentum ) ]
        layers.extend( [
            # dw
            dwConvBNReLU( strides=self.strides, padding=self.padding, epsilon=self.epsilon, momentum=self.momentum ),
            # pw-linear
            ConvBN( self.out_channel, kernel_size=1, padding=self.padding, epsilon=self.epsilon, momentum=self.momentum )
            #tf.compat.v1.keras.layers.Conv2D( self.out_channel, 1, 1, self.padding, use_bias=False ),
            #tf.compat.v1.keras.layers.BatchNormalization( momentum=0.01)#, trainable=self.trainable )
            ] )
        self.model = tf.keras.Sequential( layers, name=str(self.out_channel) )

    def call( self, inp ):
        #with tf.variable_scope( self.name ):
        if self.dimMatch:
            return self.model( inp ) + inp
        else:
            return self.model( inp )
