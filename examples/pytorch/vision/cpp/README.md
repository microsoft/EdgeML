# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license

The `rnnpool_quantized.cpp` code takes the activations preceding the RNNpool layer
and produces the output of a quantized RNN pool layer. The input numpy file consists 
of all activation patches corresponding to a single image. In the `trace_0_input.npy`
there are 6241 patches of dimensions 6x6 with 8 channels to which RNNPool is applied.
The output is of size 4*8. This can be compared to the floatin point output stored in
`trace_0_output.npy`

```shell
g++ -o rnnpool_quantized rnnpool_quantized.cpp
./rnnpool_quantized <#patches> <input.npy> <output.npy>
```

