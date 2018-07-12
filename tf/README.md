# EdgeML: Tensorflow Implementations 

This directory includes, Tensorflow implementations of various techniques and algorithms developed as
part of EdgeML. Currently, the following algorithms are available in
Tensorflow:

1. [ProtoNN](https://github.com/Microsoft/EdgeML/blob/master/publications/ProtoNN.pdf)
2. [Bonsai](https://github.com/Microsoft/EdgeML/blob/master/publications/Bonsai.pdf)

## ProtoNN
`edgeml.graph.protoNN` implements the ProtoNN prediction graph in Tensorflow.
The training routine for ProtoNN is decoupled from the forward graph to
facilitate a plug and play behaviour wherein ProtoNN can be combined with or
used as a final layer classifier for other architectures (RNNs, CNNs).

For training vanilla ProtoNN, `edgeml.trainer.protoNNTrainer` implements the
ProtoNN training routine in Tensorflow. A simple example,
`examples/protoNN_example.py` is provided to illustrate its usage.

For detailed usage instructions of the example, in the `examples` directory, please use

    python protoNN_example.py -h

Note that, `protoNN_example.py` assumes that data is in a specific format.
It is assumed that train and test data is contained in two files,
`train.npy` and `test.npy`. Each containing a 2D numpy array of dimension
`[numberOfExamples, numberOfFeatures + 1]`. The first column of each
matrix is assumed to contain label information.  For an N-Class problem,
we assume the labels are integers from 0 through N-1.


## Bonsai
`edgeml.graph.Bonsai` implements the Bonsai prediction graph in tensorflow.
Similar to ProtoNN, the three-phase training routine for Bonsai is decoupled from
the forward graph to facilitate a plug and play behaviour wherein Bonsai can be
combined with or used as a final layer classifier for other architectures (RNNs, CNNs).

For training vanilla Bonsai, `edgeml.trainer.BonsaiTrainer` implements the
Bonsai training routine in Tensorflow. A simple example,
`examples/bonsai_example.py` is provided to illustrate its usage.

For detailed usage instructions of the example, in the `examples` directory, please use

    python bonsai_example.py -h


Note that, similar to `protoNN_example.py`, `bonsai_example.py` 
assumes that data is in a specific format.
It is assumed that train and test data is contained in two files,
`train.npy` and `test.npy`. Each containing a 2D numpy array of dimension
`[numberOfExamples, numberOfFeatures + 1]`. The first column of each
matrix is assumed to contain label information.  For an N-Class problem,
we assume the labels are integers from 0 through N-1.

Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT license.
