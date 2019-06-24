## Edge Machine Learning: PyTorch Library 

This directory includes, PyTorch implementations of various techniques and
algorithms developed as part of EdgeML. Currently, the following algorithms are
available in PyTorch:

1. [Bonsai](../docs/publications/Bonsai.pdf)
2. [FastRNN & FastGRNN](../docs/publications/FastGRNN.pdf)

The PyTorch compute graphs for these algoriths are packaged as
`pytorch_edgeml.graph`. Trainers for these algorithms are in `pytorch_edgeml.trainer`. Usage
directions and examples for these algorithms are provided in `examples`
directory. To get started with any of the provided algorithms, please follow
the notebooks in the the `examples` directory.

## Installation

Use pip and the provided requirements file to first install required
dependencies before installing the `pytorch_edgeml` library. Details for installation provided below.

It is highly recommended that EdgeML be installed in a virtual environment. Please create
a new virtual environment using your environment manager ([virtualenv](https://virtualenv.pypa.io/en/stable/userguide/#usage) or [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)).
Make sure the new environment is active before running the below mentioned commands.

### CPU

``` 
pip install -r requirements-cpu.txt
pip install -e .
```

Tested on Python 3.6 with PyTorch 1.1.

### GPU

Install appropriate CUDA and cuDNN [Tested with >= CUDA 9.0 and cuDNN >= 7.0]

```
pip install -r requirements-gpu.txt
pip install -e .
```

Note: If the above commands don't go through for PyTorch installation on CPU and GPU, please follow this [link](https://pytorch.org/get-started/locally/).

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT license.
