# EdgeML FastCells on a sample public dataset

This directory includes example notebook and general execution script of
FastCells (FastRNN & FastGRNN) developed as part of EdgeML along with modified
UGRNN, GRU and LSTM to support the LSQ training routine. 
Also, we include a sample cleanup and use-case on the USPS10 public dataset.

`edgeml.graph.rnn` implements the custom RNN cells of **FastRNN** ([`FastRNNCell`](../../edgeml/graph/rnn.py#L215)) and **FastGRNN** ([`FastGRNNCell`](../../edgeml/graph/rnn.py#L40)) with
multiple additional features like Low-Rank parameterisation, custom
non-linearities etc., Similar to Bonsai and ProtoNN, the three-phase training
routine for FastRNN and FastGRNN is decoupled from the custom cells to
facilitate a plug and play behaviour of the custom RNN cells in other
architectures (NMT, Encoder-Decoder etc.,) in place of the inbuilt `RNNCell`, `GRUCell`, `BasicLSTMCell` etc., 
`edgeml.graph.rnn` also contains modified RNN cells of **UGRNN** ([`UGRNNLRCell`](../../edgeml/graph/rnn.py#L862)), 
**GRU** ([`GRULRCell`](../../edgeml/graph/rnn.py#L635)) and **LSTM** ([`LSTMLRCell`](../../edgeml/graph/rnn.py#L376)). These cells also can be substituted for FastCells where ever feasible. 

For training FastCells, `edgeml.trainer.fastTrainer` implements the three-phase
FastCell training routine in Tensorflow. A simple example,
`examples/fastcell_example.py` is provided to illustrate its usage.

Note that `fastcell_example.py` assumes that data is in a specific format.  It
is assumed that train and test data is contained in two files, `train.npy` and
`test.npy`. Each containing a 2D numpy array of dimension `[numberOfExamples,
numberOfFeatures]`. numberOfFeatures is `timesteps x inputDims`, flattened
across timestep dimension. So the input of 1st timestep followed by second and
so on.  For an N-Class problem, we assume the labels are integers from 0
through N-1. Lastly, the training data, `train.npy`, is assumed to well shuffled 
as the training routine doesn't shuffle internally.

**Tested With:** Tensorflow >1.6 with Python 2 and Python 3

## Download and clean up sample dataset

We will be testing out the validation of the code by using the USPS dataset.
The download and cleanup of the dataset to match the above-mentioned format is
done by the script [fetch_usps.py](fetch_usps.py) and
[process_usps.py](process_usps.py)

```
python fetch_usps.py
python process_usps.py
```


## Sample command for FastCells on USPS10
The following sample run on usps10 should validate your library:

Note: Even though usps10 is not a time-series dataset, it can be assumed as, a time-series where each row is coming in at one single time.
So the number of timesteps = 16 and inputDims = 16

```bash
python fastcell_example.py -dir usps10/ -id 16 -hd 32
```
This command should give you a final output screen which reads roughly similar to (might not be exact numbers due to various version mismatches):

```
Maximum Test accuracy at compressed model size(including early stopping): 0.9407075 at Epoch: 262
Final Test Accuracy: 0.93721974

Non-Zeros: 1932 Model Size: 7.546875 KB hasSparse: False
```
`usps10/` directory will now have a consolidated results file called `FastRNNResults.txt` or `FastGRNNResults.txt` depending on the choice of the RNN cell.
A directory `FastRNNResults` or `FastGRNNResults` with the corresponding models with each run of the code on the `usps10` dataset

## Byte Quantization(Q) for model compression
If you wish to quantize the generated model to use byte quantized integers use `quantizeFastModels.py`. Usage Instructions:

```
python quantizeFastModels.py -h
```

This will generate quantized models with a suffix of `q` before every param stored in a new directory `QuantizedFastModel` inside the model directory.
One can use this model further on edge devices.

Copyright (c) Microsoft Corporation. All rights reserved. 

Licensed under the MIT license.
