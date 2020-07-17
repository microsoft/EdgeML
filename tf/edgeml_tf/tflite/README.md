## Edge Machine Learning: Tensorflow Library : Keras ProtoNN and Bonsai Layers

This directory includes Keras layer implementations for [Bonsai](/docs/publications/Bonsai.pdf) 
and [ProtoNN](/docs/publications/ProtoNN.pdf). Currently, the code based does not use 
these layers for training purpose, hence its advisable to use these strictly for inference.

The [examples/tf directory](/examples/tf) provide a utility that shows how to use these layers
for inference. Essentially it creates a sequential Keras model, loads the pre-trained models,
converts this model into tflite format and invoke the interpreter for inference.

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT license.
