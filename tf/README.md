## Edge Machine Learning: Tensorflow Library 

This directory includes Tensorflow implementations of various techniques and
algorithms developed as part of EdgeML. Currently, the following algorithms are
available in Tensorflow:

1. [Bonsai](/docs/publications/Bonsai.pdf)
2. [EMI-RNN](/docs/publications/emi-rnn-nips18.pdf)
3. [FastRNN & FastGRNN](/docs/publications/FastGRNN.pdf)
4. [ProtoNN](/docs/publications/ProtoNN.pdf)

The TensorFlow compute graphs for these algoriths are packaged as `edgeml_tf.graph`
and trainers are in `edgeml_tf.trainer`. Usage directions and example notebook for
these algorithms are provided in the [examples/tf directory](/examples/tf). 


## Installation

It is highly recommended that EdgeML be installed in a virtual environment. 
Please create a new virtual environment using your environment manager
 ([virtualenv](https://virtualenv.pypa.io/en/stable/userguide/#usage) or 
 [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)).
Make sure the new environment is active before running the below mentioned commands.

Use pip to install the requirements before installing the `edgeml_tf` library. 
Details for cpu based installation and gpu based installation provided below.

### CPU

``` 
pip install -r requirements-cpu.txt
pip install -e .
```

Tested on Python3.5 and python 2.7 with >= Tensorflow 1.6.0.

### GPU

Install appropriate CUDA and cuDNN [Tested with >= CUDA 8.1 and cuDNN >= 6.1]

```
pip install -r requirements-gpu.txt
pip install -e .
```

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT license.
