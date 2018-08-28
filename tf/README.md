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

Tested on both Python2 and Python3

pip install
``` 
1. numpy
2. scikit-learn
3. scipy
4. tensorflow [Choose gpu/cpu appropriately after installing CUDA and cuDNN for GPU. Tested till Cuda 9.0 and cuDNN 7.1]
5. jupyter
6. pandas
```

**TODO**: TF library requirements.

Copyright (c) Microsoft Corporation. All rights reserved. 
Licensed under the MIT license.
