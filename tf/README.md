## Edge Machine Learning: Tensorflow Library 

This directory includes, Tensorflow implementations of various techniques and
algorithms developed as part of EdgeML. Currently, the following algorithms are
available in Tensorflow:

1. [Bonsai](../docs/publications/Bonsai.pdf)
2. [EMI-RNN](../docs/publications/emi-rnn-nips18.pdf)
3. [FastRNN & FastGRNN](../docs/publications/FastGRNN.pdf)
4. [ProtoNN](../docs/publications/ProtoNN.pdf)

The TensorFlow compute graphs for these algoriths are packaged as
`edgeml.graph`. Trainers for these algorithms are in `edgeml.trainer`. Usage
directions and examples for these algorithms are provided in `examples`
directory. To get started with any of the provided algorithms, please follow
the notebooks in the the `examples` directory.

## Installation

Use pip and the provided requirements file to first install required
dependencies before installing the `edgeml` library. Details for cpu based
installation and gpu based installation provided below.

It is highly recommended that EdgeML be installed in a virtual environment. Please create
a new virtual environment using your environment manager ([virtualenv](https://virtualenv.pypa.io/en/stable/userguide/#usage) or [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)).
Make sure the new environment is active before running the below mentioned commands.

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
