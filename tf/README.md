# EdgeML: Tensorflow Implementations 

This directory includes, Tensorflow implementations of various techniques and algorithms developed as
part of EdgeML. Currently, the following algorithms are available in
Tensorflow:

1. [ProtoNN](https://github.com/Microsoft/EdgeML/blob/master/publications/ProtoNN.pdf)
2. [Bonsai](https://github.com/Microsoft/EdgeML/blob/master/publications/Bonsai.pdf)

## ProtoNN
`edgeml.graph.protoNN` implements the ProtoNN prediction graph in tensorflow.
The training routine for ProtoNN is decoupled from the forward graph to
facilitate a plug and play behaviour wherein ProtoNN can be combined with or
used as a final layer classifier for other architectures (RNNs, CNNs).

For training vanilla ProtoNN, `edgeml.trainer.protoNNTrainer` implements the
ProtoNN training routine in Tensorflow. A simple example,
`examples/protoNN_example.py` is provided to illustrate its usage.

For detailed usage instructions of the example, in `examples` directory, please use

    python protoNN_example.py -h

## Bonsai
`edgeml.graph.Bonsai` implements the Bonsai prediction graph in tensorflow.
The three phase training routine for Bonsai is decoupled from the forward graph to
facilitate a plug and play behaviour wherein Bonsai can be combined with or
used as a final layer classifier for other architectures (RNNs, CNNs).

For training vanilla Bonsai, `edgeml.trainer.BonsaiTrainer` implements the
Bonsai training routine in Tensorflow. A simple example,
`examples/bonsai_example.py` is provided to illustrate its usage.

For detailed usage instructions of the example, in `examples` directory, please use

    python bonsai_example.py -h

Note that bonsai_example.py expects the data to be in specific format:
Expects .npy files having [lbl feats] form for each datapoint and is a part of the bigger matrix (train or test) in a row-wise fashion
`train.npy` and `test.npy` should be present in the data directory being passed as an argument


