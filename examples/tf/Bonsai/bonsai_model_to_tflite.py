# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import helpermethods
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
import argparse
from edgeml_tf.tflite.bonsaiLayer import BonsaiLayer
from helpermethods import str2bool

def main():
    assert tf.__version__.startswith('2')==True, 'Only Tensorflow-2.X API is supported.'

    # Hyper Param pre-processing
    parser = argparse.ArgumentParser(
        description='HyperParams for Bonsai Algorithm')
    parser.add_argument('-dir', '--data-dir', required=True,
                        help='Data directory containing' +
                        'train.npy and test.npy')

    parser.add_argument('-model', '--model-dir', required=True,
                        help='Model directory containing' +
                        'model parameter matrices and hyper-parameters')

    parser.add_argument('-regression', type=str2bool, default=False,
                        help='boolean argument which controls whether to perform ' +
                        'regression or classification.' +
                        'default : False (Classification) values: [True, False]')

    args = parser.parse_args()

    # Set 'isRegression' to be True, for regression. Default is 'False'.
    isRegression = args.regression
    assert isRegression==False, 'Currently tflite is not supported for regression tasks.'

    dataDir = args.data_dir
    model_dir = args.model_dir

    (dataDimension, numClasses, Xtrain, Ytrain, Xtest, Ytest,
     mean, std) = helpermethods.preProcessData(dataDir, isRegression)

    if numClasses == 2:
        numClasses = 1

    print('Model dir = ', model_dir)

    Z = np.load( model_dir + 'Z.npy', allow_pickle=True )
    W = np.load( model_dir + 'W.npy', allow_pickle=True )
    V = np.load( model_dir + 'V.npy', allow_pickle=True )
    T = np.load( model_dir + 'T.npy', allow_pickle=True )
    hyperparams = np.load( model_dir + 'hyperParam.npy', allow_pickle=True ).item()
    
    n_dim = dataDimension = hyperparams['dataDim']
    projectionDimension = hyperparams['projDim']
    numClasses = hyperparams['numClasses']
    depth = hyperparams['depth']
    sigma = hyperparams['sigma']

    print( 'dataDim = ', dataDimension )
    print( 'projectionDim = ', projectionDimension )
    print( 'numClasses = ', numClasses )
    print( 'depth = ', depth )
    print( 'sigma = ', sigma )

    dense = BonsaiLayer( numClasses, dataDimension, projectionDimension, depth, sigma )

    model = keras.Sequential([
        keras.Input(shape=(n_dim)),
        dense
    ])

    dummy_tensor = tf.convert_to_tensor( np.zeros((1,n_dim), np.float32) )
    out_tensor = model( dummy_tensor )

    model.summary()
    
    dense.set_weights( [Z, W, V, T] )
    
    # Save the Keras model in tflite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the TF Lite model as file
    out_tflite_model_file = model_dir + '/bonsai_model.tflite'
    f = open(out_tflite_model_file, "wb")
    f.write(tflite_model)
    f.close()

    # Delete any reference to existing models in order to avoid conflicts
    del model 
    del tflite_model

    # Prediction on an example input using tflite model we just saved
    x, y = Xtrain[0], Ytrain[0]
    x = x.astype(np.float32)
    x = np.expand_dims( x, 0 )

    # Run inference with TensorFlow Lite
    interpreter = tf.lite.Interpreter(model_path=out_tflite_model_file)
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], x)
    interpreter.invoke()

    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])()[0]
    print('true y = ', np.argmax(y))
    print('predicted y = ', output)

if __name__ == '__main__':
    main()


