# EdgeML ProtoNN on a sample public dataset

This directory includes, example notebook and general execution script of
ProtoNN developed as part of EdgeML. The example is based on the USPS dataset.

**Tested With:** Tensorflow >1.6 with Python 2 and Python 3

## Fetching Data

The script [fetch_usps.py](fetch_usps.py) automatically downloads and processes
the dataset if `bash` is available. To run this script, please use:

    python fetch_usps.py

If `bash` is not available, manual download of the data is required.
Instructions can be found in [fetch_usps.py](fetch_usps.py).

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
      --epochs 200
```

You can expect a test set accuracy of about 91.8%.

Copyright (c) Microsoft Corporation. All rights reserved. 
Licensed under the MIT license.
