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
cd <EDGEML_TF_EXP>
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
