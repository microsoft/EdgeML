# EdgeML FastCells on a sample public dataset

This directory includes, example notebook and general execution script of FastCells (FastGRNN & FastRNN) 
developed as part of EdgeML. Also, we inlude a sample cleanup and use-case on USPS10 public dataset.

Note FAST_EXP directory is '.'

Note that `fastcell_example.py` assumes that data is in a specific format.
It is assumed that train and test data is contained in two files,
`train.npy` and `test.npy`. Each containing a 2D numpy array of dimension
`[numberOfExamples, numberOfFeatures]`. numberOfFeatures is `timesteps x inputDims`,
flattened across timestep dimension. So input of 1st timestep followed by second and so on.
For an N-Class problem, we assume the labels are integers from 0 through N-1.

**Tested With:** Tensorflow >1.6 with Python 2 and Python 3

## Download a sample dataset
Follow the bash commands given below to download a sample dataset, USPS10, to the root current directory. FastCells comes with sample script to run on the usps10 dataset. FAST_EXP is defined in the previous section.

```bash
cd <FAST_EXP>
mkdir usps10
cd usps10
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2
bzip2 -d usps.bz2
bzip2 -d usps.t.bz2
mv usps train.txt
mv usps.t test.txt
cd ..
python dataCleanup.py usps10
```

## Sample command for FastCells on USPS10
The following sample run on usps10 should validate your library:

Note: Even though usps10 is not a time-series dataset, it can be assumed as, a time-series where each row is coming in at one single time.
So the number of timesteps = 16 and inputDims = 16

```bash
python fastcell_example.py -dir usps10/ -id 16 -hd 32
```
This command should give you a final output screen which reads roughly similar to(might not be exact numbers due to various version mismatches):

```
Maximum Test accuracy at compressed model size(including early stopping): 0.937718 at Epoch: 80
Final Test Accuracy: 0.92077726

Non-Zeros: 1932 Model Size: 7.546875 KB hasSparse: False
```
usps10 directory will now have a consilidated results file called `FastGRNNResults.txt` or `FastRNNResults.txt` depending on the choice of the RNN cell.
A directory `FastGRNNResults` or `FastRNNResults` with the corresponding models with each run of the code on the usps10 dataset

If you wish to quantise the generated model to use byte quantized integers use `quantizeFastModels.py`. Usage Instructions:

```
python quantizeFastModels.py -h
```

This will generate quantised models with a suffix of `q` before every param stored in a new directory `QuantizedFastModel` inside the model directory.
One can use this model further on edge devices.

Copyright (c) Microsoft Corporation. All rights reserved. 
Licensed under the MIT license.
