# EdgeML: Tensorflow Implementations 

This directory includes, Tensorflow implementations of various techniques and
algorithms developed as part of EdgeML. Currently, the following algorithms are
available in Tensorflow:

1. [Bonsai](../docs/publications/Bonsai.pdf)
2. [EMI-RNN](../docs/publications/EMI-RNN.pdf)
3. [Fast(G)RNN](../docs/publications/FastGRNN.pdf)
4. [ProtoNN](../docs/publications/ProtoNN.pdf)

Usage directions and examples of all of the above algorithms are provided in
`examples` subdirectory. 

## Dependencies

Tested on both Python2.7 and >= Python3.5 with >= Tensorflow 1.6.0.

### CPU
``` 
pip install -r requirements-cpu.txt
```
### GPU

Install appropriate CUDA and cuDNN [Tested with >= CUDA 8.1 and cuDNN >= 6.1]
```
pip install -r requirements-gpu.txt
```

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT license.
