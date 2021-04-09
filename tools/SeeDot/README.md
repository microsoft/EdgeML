# SeeDot

SeeDot is an automatic quantization tool that generates efficient machine learning (ML) inference code for IoT devices.

### **Overview**

ML models are usually expressed in floating-point, and IoT devices typically lack hardware support for floating-point arithmetic. Hence, running such ML models on IoT devices involves simulating floating-point arithmetic in software, which is very inefficient. SeeDot addresses this issue by generating fixed-point code with only integer operations. To enable this, SeeDot takes as input trained floating-point models (like [Bonsai](https://github.com/microsoft/EdgeML/blob/master/docs/publications/Bonsai.pdf) or [ProtoNN](https://github.com/microsoft/EdgeML/blob/master/docs/publications/ProtoNN.pdf) or [FastGRNN](https://github.com/microsoft/EdgeML/blob/master/docs/publications/FastGRNN.pdf)) and generates efficient fixed-point code that can run on micro-controllers. The SeeDot compiler uses novel compilation techniques to automatically infer certain parameters used in the fixed-point code, optimized exponentiation computation, etc. With these techniques, the generated fixed-point code has comparable classification accuracy and performs significantly faster than the floating-point code.

To know more about SeeDot, please refer to our publications [here](https://www.microsoft.com/en-us/research/publication/compiling-kb-sized-machine-learning-models-to-constrained-hardware/) and [here](https://www.microsoft.com/en-us/research/publication/shiftry-rnn-inference-in-2kb-of-ram/).

This document describes the tool's usage with an example.

### **Software requirements**

1. [**Python 3**](https://www.python.org/) with following packages:
   - **[Antrl4](http://www.antlr.org/)** (antlr4-python3-runtime; tested with version 4.7.2)
   - **[Numpy](http://www.numpy.org/)** (tested with version 1.16.4)
   - **[Scikit-learn](https://scikit-learn.org/)** (tested with version 0.21.2)
   - **[Bokeh](https://bokeh.org/)** (tested with version 2.1.1)
   - **[ONNX](https://onnx.ai/)** (tested with version 1.8.0)
2. Linux packages:
   - **[gcc](https://www.gnu.org/software/gcc/)** (tested with version 9.3.0)
   - **[make](https://www.gnu.org/software/make/)** (tested with version 4.2.1)

### **Usage**

SeeDot can be invoked using **`SeeDot-dev.py`** file. The arguments for the script are supplied as follows:

```
usage: SeeDot-dev.py [-h] [-a] [-e] [-d] [-m] [-n] [-dt] [-t] [-s] [-sf] [-l] [-lsf] [-tdr] [-o]

optional arguments:
  -h,   --help             Show this help message and exit
  -a,   --algo             Algorithm to run ['bonsai' or 'protonn' or 'fastgrnn' or 'rnnpool'] 
                           (Default: 'fastgrnn')

  -e,   --encoding         Floating-point ['float'] or Fixed-point ['fixed'] 
                           (Default: 'fixed')

  -d,   --dataset          Dataset to use 
                           (Default: 'usps10')

  -m,   --metric           Select the metric that will be used to measure the correctness of an inference, to obtain the 
                           best quantization of variables.
                              1) Accuracy ('acc'):                The accuracy of prediction will be used as a metric for 
                                                                  correctness. (A maximising metric).

                              2) Disagreement Count ('disagree'): The correctness will be measured against the
                                                                  floating-point code's output. (A minimising metric).
                              3) Reduced Disagreement Count 
                                                ('red_disagree'): The correctness will be measured against the
                                                                  floating-point code's output only when the output matches the correct label. (A minimising metric).
                           (Default: 'red_disagree')

  -n,   --numOutputs       Number of outputs (e.g., classification problems have only 1 output, i.e., the class label)
                           (Default: 1)

  -dt,  --datasetType      Dataset type being used ['training', 'testing']
                           (Default: 'testing')

  -t,   --target           Target device ['x86', 'arduino', 'm3']
                           (Default: 'x86')

  -s,   --source           Model source type ['seedot', 'onnx', 'tf']
                           (Default: 'seedot')
  
  -sf,  --max-scale-factor Max scaling factor for code generation (If not specified then it will be inferred from data)
  
  -l,   --log              Logging level (in increasing order) ['error', 'critical', 'warning', 'info', 'debug']
                           (Default: 'error')

  -tdr, --tempdir          Scratch directory for intermediate files
                           (Default: 'temp/')

  -o,   --outdir           Directory to output the generated targetdevice sketch
                           (Default: 'arduinodump/' for Arduino, 'temp/' for x86 and, 'm3dump/' for M3)
```

An example invocation is as follows:
```
python SeeDot-dev.py -a fastgrnn -e fixed -d usps10 -n 1 -t arduino -m red_disagree -l info
```

SeeDot expects the `train` and the `test` data files in a specific format. Each data file should be of the shape `[numberOfDataPoints, n + numberOfFeatures]`, where the ground truth/output is in the first `n` columns. The tool currently supports numpy arrays (.npy) for inputting model parameters.
The data files must be present in the directory `datasets/<algo>/<dataset>`.

After training, the learned parameters are stored in this directory in a specific format. For FastGRNN, the learned parameters are `W`, `U`, `Bg`, `Bh`, `FC`, `FCBias`, `zeta` and `nu`. These parameters are numpy arrays (.npy). The model files must be present in the directory `model/<algo>/<dataset>`.

The compiler output is present in the `temp` directory for x86, the `arduinodump` directory for arduino, and the `m3dump` directory for m3.

## Getting started: Quantizing FastGRNN on usps10

To help get started with SeeDot, we provide 1) a pre-loaded fixed-point model, and 2) instructions to generate fixed-point code for the FastGRNN predictor on the **[usps10](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/)** dataset. The process for generating fixed-point code for the Bonsai or ProtoNN predictor is similar.

### Generating fixed-point code

This process consists of four steps: 1) installing EdgeML TensorFlow library, 2) training FastGRNN on usps10, 3) quantizing the trained model with SeeDot, and 4) performing prediction on the device.

#### **Step 1: Installing EdgeML TensorFlow library**

1. Clone the EdgeML repository and navigate to the right directory.
     ```
     git clone https://github.com/Microsoft/EdgeML
     cd EdgeML/tf/
     ```

2. Install the EdgeML library.
     ```
     pip install -r requirements-cpu.txt
     pip install -e .
     ```

#### **Step 2: Training FastGRNN on usps10**

1. Navigate to the FastGRNN examples directory.
     ```
     cd ../examples/tf/FastCells
     ```
     
2. Fetch and process usps10 data and create an output directory.
     ```
     python fetch_usps.py
     python process_usps.py
     ```

3. Invoke FastGRNN trainer using the following command.
      ```
      python fastcell_example.py --data-dir ./usps10 --input-dim 16 --hidden-dim 32
      ```
  This would give around 93-95% classification accuracy. The trained model is stored in the `usps10/FastGRNNResults/<timestamp>` directory.

More information on using the FastGRNN trainer can be found [here](https://github.com/microsoft/EdgeML/tree/master/examples/tf/FastCells).

#### **Step 3: Quantizing with SeeDot**

1. Copy the dataset and model files into the correct directory.
     ```
     cd ../../../tools/SeeDot/
     mkdir -p datasets/fastgrnn/usps10
     mkdir -p model/fastgrnn/usps10
     cp ../../examples/tf/FastCells/usps10/*.npy ./datasets/fastgrnn/usps10/
     cp ../../examples/tf/FastCells/usps10/FastGRNNResults/<timestamp>/* ./model/fastgrnn/usps10/
     ```
2. Copy the example code for FastGRNN in the SeeDot language:
     ```
     cp seedot/compiler/input/fastgrnn.sd model/fastgrnn/usps10/input.sd
     ```

3. Invoke SeeDot compiler using the following command.
      ```
      python SeeDot-dev.py -a fastgrnn -e fixed -t arduino -m red_disagree -n 1 -d usps10
      ```

   The SeeDot-generated code would give around 92-95% classification accuracy. The difference in classification accuracy is 0-3% compared to the floating-point code. The generated code is stored in the `arduinodump` folder which contains the sketch along with two files: model.h and predict.cpp. `model.h` contains the quantized model and `predict.cpp` contains the inference code.

#### **Step 4: Prediction on the device**

Follow the below steps to perform prediction on the device, where the SeeDot-generated code is run on a single data-point stored on the device's flash memory.

1. The model files are generated within `arduinodump/arduino/16/fastgrnn/usps10`. Copy all the files to `arduinodump/arduino`.
2. Open the Arduino sketch file located at `arduinodump/arduino/arduino.ino` in the [Arduino IDE](https://www.arduino.cc/en/main/software).
3. Connect the Arduino micro-controller to the computer and choose the correct board configuration.
4. Upload the sketch to the device.
5. Open the Serial Monitor and select baud rate specified in the sketch (default is 115200) to monitor the output.
6. The average prediction time is computed for every iteration. On an Arduino Uno, the average prediction time is around 280000 micro seconds.

More device-specific details on extending the Arduino sketch for other use cases can be found in [`arduino/README.md`](https://github.com/microsoft/EdgeML/blob/Feature/SeeDot/Tools/SeeDot/seedot/arduino/README.md).


The above workflow has been tested on Arduino Uno. It is expected to work on other Arduino devices as well.


Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT license.
