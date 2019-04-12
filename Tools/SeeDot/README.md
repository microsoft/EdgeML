# SeeDot

SeeDot is an automatic quantization tool that generates efficient ML inference code for IoT devices.

Most ML models are expressed in floating-point and IoT devices typically lack hardware support for floating-point. SeeDot bridges this gap. SeeDot takes as input trained floating-point models (Bonsai or ProtoNN) and generates efficient fixed-point code to run on Arduino microcontrollers.

To know more about SeeDot, please refer our paper [here](https://www.microsoft.com/en-us/research/publication/compiling-kb-sized-machine-learning-models-to-constrained-hardware/).

1. 

**Software requirements**

1. [**Python 3**](https://www.python.org/) with following packages:
   - **[Antrl4](http://www.antlr.org/)** (antlr4-python3-runtime; tested with version 4.7.2)
   - **[Numpy](http://www.numpy.org/)** (tested with version 1.16.2)
   - **[Scikit-learn](https://scikit-learn.org/stable/)** (tested with version 0.20.3)
2. Linux packages:
   - gcc (tested with version 7.3.0)
   - make (tested with version 4.1)

**Usage**

SeeDot can be invoked using the SeeDot.py file. The arguments are supplied as follows:

```
usage: SeeDot.py [-h] [-a] --train  --test  --model  [--tempdir] [-o]

optional arguments:
  -h, --help      show this help message and exit
  -a , --algo     Algorithm to run ('bonsai' or 'protonn')
  --train         Training set file as .npy
  --test          Testing set file as .npy
  --model         Directory containing trained model (output from
                  Bonsai/ProtoNN trainer)
  --tempdir       Scratch directory for intermediate files
  -o , --outdir   Directory to output the generated Arduino sketch
```

An example invocation is as follows:
`python SeeDot.py -a bonsai --train train.npy --test test.npy --model Bonsai/model`

> SeeDot expects `train.npy` and `test.npy` in a specific format. The shape of each data file should be `[numberOfDataPoints, numberOfFeatures + 1]`, where the class label is in the first column.
>
> The `model` directory contains the output of Bonsai/ProtoNN trainer. 

**Output**

After the execution, the compiler generates two files in the `OUTPUT` directory:

1. **`model.h`** containing the training model along with the Arduino pragmas.
2. **`predict.cpp`** containing the fixed-point prediction code.

**Prediction on device**

