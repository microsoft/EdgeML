# EdgeML TF Library on a sample public dataset

This directory includes, example implementations of various techniques and algorithms developed as
part of EdgeML. Also, we inlude a sample cleanup and use-case on USPS10 public dataset

Note EDGEML_TF_EXP directory is '.'

## Download a sample dataset
Follow the bash commands given below to download a sample dataset, USPS10, to the root of the repository. Bonsai and ProtoNN come with sample scripts to run on the usps10 dataset. EDGEML_TF_EXP is defined in the previous section.

```bash
cd <EDGEML_TF_EXP>
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

## Sample command for Bonsai on USPS10
The following sample run on usps10 should validate your library:

```bash
python bonsai_example.py -dir usps10/ -d 3 -p 28 -rW 0.001 -rZ 0.0001 -rV 0.001 -rT 0.001 -sZ 0.2 -sW 0.3 -sV 0.3 -sT 0.62 -e 100 -s 1
```
This command should give you a final output screen which reads roughly similar to(might not be exact numbers due to various version mismatches):
```
Maximum Test accuracy at compressed model size(including early stopping): 0.94369704 at Epoch: 66
Final Test Accuracy: 0.93024415

Non-Zeros: 4156.0 Model Size: 31.703125 KB hasSparse: True
```

usps10 directory will now have a consilidated results file called `TFBonsaiResults.txt` and a directory `TFBonsaiResults` with the corresponding models with each run of the code on the usps10 dataset


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
