# EdgeML Bonsai on a sample public dataset

This directory includes, example notebook and general execution script of Bonsai developed as
part of EdgeML. Also, we inlude a sample cleanup and use-case on USPS10 public dataset.

Note BONSAI_EXP directory is '.'

Note that `bonsai_example.py` assumes that data is in a specific format.
It is assumed that train and test data is contained in two files,
`train.npy` and `test.npy`. Each containing a 2D numpy array of dimension
`[numberOfExamples, numberOfFeatures + 1]`. The first column of each
matrix is assumed to contain label information.  For an N-Class problem,
we assume the labels are integers from 0 through N-1.

**Tested With:** Tensorflow >1.6 with Python 2 and Python 3

## Download a sample dataset
Follow the bash commands given below to download a sample dataset, USPS10, to the root current directory. Bonsai and ProtoNN come with sample scripts to run on the usps10 dataset. BONSAI_EXP is defined in the previous section.

```bash
cd <BONSAI_EXP>
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


Copyright (c) Microsoft Corporation. All rights reserved. 
Licensed under the MIT license.