# FastRNN and FastGRNN - FastCells

This document aims to explain and elaborate on specific details of FastCells 
present as part of `tf/edgeml/graph/rnn/.py`. The endpoint usecase scripts with 
3 phase training along with an example notebook are present in `tf/examples/FastCells/`.
One can use the endpoint script to test out the RNN architectures on any dataset 
while specifying budget constraints as part of hyper-parameters in terms of sparsity and rank 
of weight matrices.

# FastRNN
![FastRNN](img/FastRNN.png)

# FastGRNN
![FastGRNN Base Architecture](img/FastGRNN.png)

# Plug and Play Cells

`FastGRNNCell` and `FastRNNCell` present in `edgeml.graph.rnn` are very similar to 
Tensorflow's inbuilt GRUCell, BasicLSTMCell and UGRNNCell. Thereby allowing us to 
replace any of the standard RNN Cell in our architecture and replace it with FastCells. 
One can see the plug and play nature at the endpoint script for FastCells, where the graph 
building is very similar to LSTM/GRU in Tensorflow

You can find [FastRNNCell](../edgeml/graph/rnn.py#L198) and [FastGRNNCell](../edgeml/graph/rnn.py#L31).

# 3 phase Fast Training