# EdgeML FastCells on a sample public dataset

This directory includes example notebooks and scripts for training
FastCells (FastRNN & FastGRNN) along with modified
UGRNN, GRU and LSTM to support the LSQ training routine. 
There is also a sample cleanup and train/test script for the USPS10 public dataset.
The subfolder [`KWS-training`](KWS-training) contains code
for training a keyword spotting model using a single- or multi-layer RNN.

Please follow the instructions in [`EdgeML/pytorch`](../../../pytorch) before using this example.

[`edgeml_pytorch.graph.rnn`](../../../pytorch/pytorch_edgeml/graph/rnn.py) 
provides two RNN cells **FastRNNCell**  and **FastGRNNCell** with additional
features like low-rank parameterisation and custom non-linearities. Akin to
Bonsai and ProtoNN, the three-phase training routine for FastRNN and FastGRNN
is decoupled from the custom cells to facilitate a plug and play behaviour of 
the custom RNN cells in other architectures (NMT, Encoder-Decoder etc.).
Additionally, numerically equivalent CUDA-based implementations **FastRNNCUDA**
and **FastGRNNCUDA** are provided for faster training. 
`edgeml_pytorch.graph.rnn` also contains modified RNN cells of **UGRNNCell**, 
**GRUCell**, and **LSTMCell**, which can be substituted for Fast(G)RNN,
as well as untrolled RNNs which are equivalent to `nn.LSTM` and `nn.GRU`. 

Note that all the cells and wrappers have `batch_first` argument set to False by default i.e. 
data is asumed to be in the format [timeSteps, batchSize, inputDims] by default, but can also 
support [batchSize, timeSteps, inputDims] format if batch_first argument is set to True. 

For training FastCells, `edgeml_pytorch.trainer.fastTrainer` implements the three-phase
FastCell training routine in PyTorch. A simple example `fastcell_example.py` is provided
to illustrate its usage. Note that `fastcell_example.py` assumes that data is in a specific format.
It is assumed that train and test data is contained in two files, `train.npy` and
`test.npy`, each containing a 2D numpy array of dimension `[numberOfExamples,
numberOfFeatures]`. numberOfFeatures is `timesteps x inputDims`, flattened
across timestep dimension with the input of the first time step followed by the second
and so on.  For an N-Class problem, we assume the labels are integers from 0
through N-1. Lastly, the training data, `train.npy`, is assumed to well shuffled 
as the training routine doesn't shuffle internally.

**Tested With:** PyTorch = 1.1 with Python 3.6

## Download and clean up sample dataset

To validate the code with USPS dataset, first download and format the dataset to match
the required format using the script [fetch_usps.py](fetch_usps.py) and
[process_usps.py](process_usps.py)

```
python fetch_usps.py
python process_usps.py
```

Note: Even though usps10 is not a time-series dataset, it can be regarding as a time-series
dataset where time step sees a new row. So the number of timesteps = 16 and inputDims = 16.

## Sample command for FastCells on USPS10
The following is a sample run on usps10 :

```bash
python fastcell_example.py -dir usps10/ -id 16 -hd 32
```
This command should give you a final output that reads roughly similar to
(might not be exact numbers due to various version mismatches):

```
Maximum Test accuracy at compressed model size(including early stopping): 0.9422 at Epoch: 100
Final Test Accuracy: 0.9347

Non-Zeros: 1932 Model Size: 7.546875 KB hasSparse: False
```
`usps10/` directory will now have a consolidated results file called `FastRNNResults.txt`, 
`FastGRNNResults.txt` or `FastGRNNCUDAResults.txt` depending on the choice of the RNN cell. A directory `FastRNNResults`, 
`FastGRNNResults` or `FastGRNNCUDAResults` with the corresponding models with each run of the code on the `usps10` dataset.

Note that the scalars like `alpha`, `beta`, `zeta` and `nu` correspond to the values before
the application of the sigmoid function.

## Byte Quantization(Q) for model compression
If you wish to quantize the generated model, use `quantizeFastModels.py`. Usage Instructions:

```
python quantizeFastModels.py -h
```

This will generate quantized models with a suffix of `q` before every param stored in a
new directory `QuantizedFastModel` inside the model directory.

Note that the scalars like `qalpha`, `qbeta`, `qzeta` and `qnu` correspond to values 
after the application of the sigmoid function over them post quantization;
they can be directly plugged into the inference pipleines.

Copyright (c) Microsoft Corporation. All rights reserved. 
Licensed under the MIT license.
