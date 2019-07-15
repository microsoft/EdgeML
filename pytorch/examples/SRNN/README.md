# Pytorch Shallow RNN Examples

This directory includes an example [notebook](SRNN_Example.ipynb) of how to use
SRNN on the [Google Speech Commands
Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html).

`pytorch_edgeml.graph.rnn.SRNN2` implements a 2 layer SRNN network. We will use
this with an LSTM cell on this dataset. The training routine for SRNN is
implemented in `pytorch_edgeml.trainer.srnnTrainer` and will be used as part of
this example.

**Tested With:** pytorch > 1.1.0 with Python 2 and Python 3

## Fetching Data

The script - [fetch_google.sh](fetch_google.py), can be used to  automatically
download the data. You can also manually download and extract the data.
[process_google.py](process_google.py), will perform feature extraction on this
dataset and write numpy files that confirm to the required format.

 To run this script, please use:

    ./fetch_google.py
    python process_google.py

With the provided configuration, you can expect a validation accuracy of about
92%.

Copyright (c) Microsoft Corporation. All rights reserved. 
Licensed under the MIT license.
