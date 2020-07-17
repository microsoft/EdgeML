# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import print_function
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from edgeml_tf.tflite.protoNNLayer import ProtoNNLayer
import helpermethods as helper

def main():
    assert tf.__version__.startswith('2')==True, 'Only Tensorflow-2.X API is supported.'

    config = helper.getProtoNNArgs()

    # Get hyper parameters
    DATA_DIR = config.data_dir
    OUT_DIR = config.output_dir

    # Load data
    train = np.load(DATA_DIR + '/train.npy')
    test = np.load(DATA_DIR + '/test.npy')
    x_train, y_train = train[:, 1:], train[:, 0]
    x_test, y_test = test[:, 1:], test[:, 0]
    # Convert y to one-hot
    minval = min(min(y_train), min(y_test))
    numClasses = max(y_train) - min(y_train) + 1
    numClasses = max(numClasses, max(y_test) - min(y_test) + 1)
    numClasses = int(numClasses)
    y_train = helper.to_onehot(y_train, numClasses, minlabel=minval)
    y_test = helper.to_onehot(y_test, numClasses, minlabel=minval)
    dataDimension = x_train.shape[1]

    W = np.load(OUT_DIR + '/W.npy')
    B = np.load(OUT_DIR + '/B.npy')
    Z = np.load(OUT_DIR + '/Z.npy')
    gamma = np.load(OUT_DIR + '/gamma.npy')

    n_dim = inputDimension = W.shape[0]
    projectionDimension = W.shape[1]
    numPrototypes = B.shape[1]
    numOutputLabels = Z.shape[0]

    errmsg = 'Dimensions mismatch! Should be W[d, d_cap], B[d_cap, m] and Z[L,m]'
    assert B.shape[0] == projectionDimension, errmsg 
    assert Z.shape[1] == numPrototypes, errmsg 

    dense = ProtoNNLayer( inputDimension, projectionDimension, numPrototypes, numOutputLabels, gamma )

    model = keras.Sequential([
        keras.Input(shape=(n_dim)),
        dense
    ])

    dummy_tensor = tf.convert_to_tensor( np.zeros((1,n_dim), np.float32) )
    out_tensor = model( dummy_tensor )

    model.summary()

    dense.set_weights( [W, B, Z] )

    print( 'gamma = ', gamma )
    print( 'inputDim = ', inputDimension )
    print( 'projectionDim = ', projectionDimension )
    print( 'numPrototypes = ', numPrototypes )
    print( 'numOutputLabels = ', numOutputLabels )

    # Save the Keras model in tflite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the TF Lite model as file
    out_tflite_model_file = OUT_DIR + '/protoNN_model.tflite'
    f = open(out_tflite_model_file, "wb")
    f.write(tflite_model)
    f.close()

    # Delete any reference to existing models in order to avoid conflicts
    del model 
    del tflite_model

    # Prediction on an example input using tflite model we just saved
    x, y = x_train[0], y_train[0]
    x = x.astype(np.float32)
    x = np.expand_dims( x, 0 )

    # Run inference with TensorFlow Lite
    interpreter = tf.lite.Interpreter(model_path=out_tflite_model_file)
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], x)
    interpreter.invoke()

    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])()[0]
    print('true y = ', np.argmax(y))
    print('predicted y = ', np.argmax(output))


if __name__ == '__main__':
    main()
