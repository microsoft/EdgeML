# EdgeML: Tensorflow Implementations 

This directory includes, Tensorflow implementations of various techniques and algorithms developed as
part of EdgeML. Currently, the following algorithms are available in
Tensorflow:

1. ProtoNN
2. Bonsai

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


