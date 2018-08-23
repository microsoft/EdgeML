# Tensorflow EMI-RNN Examples

This directory includes example notebooks EMI-RNN developed as part of EdgeML.
The example is based on the UCI Human Activity Recognition dataset.

Please refer to `tf/docs/EMI-RNN.md` for detailed documentation of EMI-RNN.

Please refer to `00_emi_lstm_example.ipynb` for a quick and dirty getting
started guide.

Note that, EMI-RNN currently natively supports the following cells:
- LSTM
- GRU
- FastGRNN
- FastRNN
- UGRNN

**Tested With:** Tensorflow >1.6 with Python 3

## Fetching Data

The script - [fetch_har.py](fetch_har.py), can be used to  automatically
download and [process_har.py](process_har.py) can be used to processes the
dataset. 
 To run this script, please use

    python fetch_har.py
    python process_har.py

If `bash` is not available, you will have to manually download the data files. 
Instructions can be found in [fetch_har.py](fetch_har.py).

Copyright (c) Microsoft Corporation. All rights reserved. 
Licensed under the MIT license.
