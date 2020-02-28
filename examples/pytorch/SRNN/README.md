# Pytorch Shallow RNN Examples

This directory includes an example [notebook](SRNN_Example.ipynb)  and a
[python script](SRNN_Example.py) that explains the basic setup of SRNN by
training a simple model on the [Google Speech Commands
Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html).

Please follow the instructions in [`EdgeML/pytorch`](../../../pytorch) before using this example.

`edgeml_pytorch.graph.rnn.SRNN2` implements a 2 layer SRNN network. We will use
this with an LSTM cell on this dataset. The training routine for SRNN is
implemented in `edgeml_pytorch.trainer.srnnTrainer` and will be used as part of
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

## Training the Model

A sample [notebook](SRNN_Example.ipynb) and a corresponding command line script
is provided for training. To run the command line script, please use:
  
```
python SRNN_Example.py --data-dir ./GoogleSpeech/Extracted/ --brick-size 11
```

With the provided default configuration, you can expect a validation accuracy
of about 92%.

Copyright (c) Microsoft Corporation. All rights reserved. 
Licensed under the MIT license.
