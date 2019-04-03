# EdgeML Bonsai on a sample public dataset

This directory includes, example notebook and general execution script of
Bonsai developed as part of EdgeML. Also, we include a sample cleanup and
use-case on the USPS10 public dataset.

`edgeml.graph.bonsai` implements the Bonsai prediction graph in tensorflow.
The three-phase training routine for Bonsai is decoupled from the forward graph
to facilitate a plug and play behaviour wherein Bonsai can be combined with or
used as a final layer classifier for other architectures (RNNs, CNNs).

Note that `bonsai_example.py` assumes that data is in a specific format.  It is
assumed that train and test data is contained in two files, `train.npy` and
`test.npy`. Each containing a 2D numpy array of dimension `[numberOfExamples,
numberOfFeatures + 1]`. The first column of each matrix is assumed to contain
label information.  For an N-Class problem, we assume the labels are integers
from 0 through N-1. `bonsai_example.py` also supports univariate regression 
and can be accessed using the help options of the script. Multivariate regression 
requires restructuring of the input data format and can further help in extending 
bonsai to multi-label classification and multi-variate regression. Lastly, 
the training data, `train.npy`, is assumed to well shuffled 
as the training routine doesn't shuffle internally.

**Tested With:** Tensorflow >1.6 with Python 2 and Python 3

## Download and clean up sample dataset

We will be testing out the validation of the code by using the USPS dataset.
The download and cleanup of the dataset to match the above-mentioned format is
done by the script [fetch_usps.py](fetch_usps.py) and
[process_usps.py](process_usps.py)

```
python fetch_usps.py
python process_usps.py
```

## Sample command for Bonsai on USPS10
The following sample run on usps10 should validate your library:

```bash
python bonsai_example.py -dir usps10/ -d 3 -p 28 -rW 0.001 -rZ 0.0001 -rV 0.001 -rT 0.001 -sZ 0.2 -sW 0.3 -sV 0.3 -sT 0.62 -e 100 -s 1
```
This command should give you a final output screen which reads roughly similar to (might not be exact numbers due to various version mismatches):
```
Maximum Test accuracy at compressed model size(including early stopping): 0.94369704 at Epoch: 66
Final Test Accuracy: 0.93024415

Non-Zeros: 4156.0 Model Size: 31.703125 KB hasSparse: True
```

usps10 directory will now have a consolidated results file called `TFBonsaiResults.txt` and a directory `TFBonsaiResults` with the corresponding models with each run of the code on the usps10 dataset

## Byte Quantization (Q) for model compression
If you wish to quantize the generated model to use byte quantized integers use `quantizeBonsaiModels.py`. Usage Instructions:

```
python quantizeBonsaiModels.py -h
```

This will generate quantized models with a suffix of `q` before every param stored in a new directory `QuantizedTFBonsaiModel` inside the model directory.
One can use this model further on edge devices.


Copyright (c) Microsoft Corporation. All rights reserved. 

Licensed under the MIT license.
