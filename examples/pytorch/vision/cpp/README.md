# RNNPool quantized sample code

The `rnnpool_quantized.cpp` code takes the activations preceding the RNNpool layer
and produces the output of a quantized RNN pool layer. The input numpy file consists 
of all activation patches corresponding to a single image. In `trace_0_input.npy`,
there are 6241 patches of dimensions 8x8 with 4 channels to which RNNPool is applied.
The output is of size 6241*4*8. This can be compared to the floatin point output stored in
`trace_0_output.npy`

```shell
g++ -o rnnpool_quantized rnnpool_quantized.cpp

# Usage: ./rnnpool_quantized <#patches> <input.npy> <output.npy>
./rnnpool_quantized 6241 trace_0_input.npy trace_0_output_quantized.npy
```

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT license
