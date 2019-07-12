# Tensorflow ProtoNN Examples

This directory includes an example [notebook](protoNN_example.ipynb)  and a
command line execution script of ProtoNN developed as part of EdgeML. The
example is based on the USPS dataset.

`edgeml.graph.protoNN` implements the ProtoNN prediction graph in Tensorflow.
The training routine for ProtoNN is decoupled from the forward graph to
facilitate a plug and play behaviour wherein ProtoNN can be combined with or
used as a final layer classifier for other architectures (RNNs, CNNs). The
training routine is implemented in `edgeml.trainer.protoNNTrainer`.

Note that, `protoNN_example.py` assumes the data to be in a specific format.  It
is assumed that train and test data is contained in two files, `train.npy` and
`test.npy`. Each containing a 2D numpy array of dimension `[numberOfExamples,
numberOfFeatures + 1]`. The first column of each matrix is assumed to contain
label information. For an N-Class problem, we assume the labels are integers
from 0 through N-1. 

**Tested With:** Tensorflow >1.6 with Python 2 and Python 3

## Fetching Data

The script - [fetch_usps.py](fetch_usps.py), can be used to  automatically
download and [process_usps.py](process_usps.py), can be used to process the
data into the required format.
 To run this script, please use:

    python fetch_usps.py
    python process_usps.py


## Running the ProtoNN execution script

Along with the example notebook, a command line execution script for ProtoNN is
provided in `protoNN_example.py`. After the USPS data has been setup, this
script can be used with the following command:

```
python protoNN_example.py \
      --data-dir ./usps10 \
      --projection-dim 60 \
      --num-prototypes 80 \
      --gamma 0.0015 \
      --learning-rate 0.1 \
      --epochs 200 \
      --val-step 10 \
      --output-dir ./
```

You can expect a test set accuracy of about 92.5%.

Copyright (c) Microsoft Corporation. All rights reserved. 
Licensed under the MIT license.
