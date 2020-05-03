# Code for Converting Numpy Model Weights/Traces to Header Files


## RNNPool

Specify the path to model weights saved in numpy format in <your-numpy-model-weights-dir> and path to input output traces saved in numpy format in <your-numpy-traces-dir>. Specify folder to save converted header files for first rnn (which does the row-wise/column-wise traversal) of RNNPool in <your-path-to-rnn1-header-output-file> and that of second rnn (which does the bidirectional summarization of outputs of first rnn pass) in <your-path-to-rnn2-header-output-file>.

```shell

python3 ConvertNumpytoHeaderFilesRNNPool.py --model-dir <your-numpy-model-weights-dir> -tidir <your-numpy-traces-dir> -todir <your-traces-header-outputs-dir> -rnn1oF <your-path-to-rnn1-header-output-file> -rnn2oF <your-path-to-rnn2-header-output-file>                                 

```

eg.
```shell

python3 ConvertNumpytoHeaderFilesRNNPool.py --model-dir ./model_weights_face -tidir ./model_traces_face -todir ./traces_headers -rnn1oF rnn1.h -rnn2oF rnn2.h                                 

```