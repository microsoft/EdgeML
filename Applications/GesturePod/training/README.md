Training ML model for gesture recognition
=========================

## Table of Contents

- [Introduction](#intro)
- [Data Collection](#data-dollection)
- [Data Labelling](#data-labelling)
- [Feature Extraction](#feature-extraction)
- [Training and Deploying](#train-deploy)
- [Dependencies](#dependencies)


## Introduction

Training a model to recognize gesture on the GesturePod consists of 5 parts:

1. Data collection
2. Data labelling
3. Extracting the features from the labelled data
4. Training a ML model
5. Deploying the model on the GesturePod to recognize gestures in real life.


## Data Collection

Accelorometer and gyroscope values from the MPU6050 sensor is collected. 
Connect the GesturePod to computer over Serial COM Port.

1. Refer
   [here](https://github.com/microsoft/EdgeML/blob/master/Applications/GesturePod/onMKR1000/README.md#quick-start)
to set up the required dependencies for the MKR1000 platform.
		
2. Compile, Build and Upload
	Navigate to `./serialDataCollection` upload `serialDataCollection.ino` onto the GesturePod.
	Make sure you select the ```COM``` port to which MKR1000 is connected.
	
3. Launch the ```COM``` port and verify that data is being received. It should be of the format 
[time, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z].

4. On a python3 environment run the following command to save the raw data: 
```
python csvFromSerial.py [outputFileName] [COMPort]
``` 
The raw data file is stored at `./data/raw_data/`.


## Data Labelling

1. The raw data in `./data/raw_data/foo.csv` must be labelled with the following command:
```
python labelData.py foo.csv
```
This command will launch an interactive GUI. Sensor values will keep rolling
acorss the screen. Use `spacebar` to pause the action. Center the signature
using the arrow keys, and key in the corresponding "label" using number keys.
Press spacebar to continue. For example, on seeing a signature of Double Tap -
hit spacebar to freeze the values. Then after centering the signature using
right/left arrow keys, hit 3 (DTAP). Press spacebar to continue.

The labelled data file `foo_labelled.csv` will be written at `./data/labelled_data/`.  


## Feature Extraction

1. Enlist the labelled files in *labelledFileList* List in
   `generateFeatures.py`. Files that do not contain any gestures - for example,
data of climbing stairs, walking in park, etc should be listed in
*allNoiseFileList*.

2. Generate the set of features for labelled data. 
```
python generateFeatures.py
```
This will generate a `train.csv` and `test.csv` files that should be used to generate a ML model.

## Training and Deploying
Using the TensorFlow / PyTorch / cpp implementation of the ProtoNN algorithm from the EdgeML repository, train a
model on the ```train.csv``` file generated above.  Extract W, B, Z, and gamma
values from the trained ProtoNN model. Update these values in
```EdgeML/Applications/GesturePod/onMKR1000/src/data.h``` to deploy the model on
the GesturePod. Alternately, update
```EdgeML/Applications/GesturePod/onComputer/src/data.h``` to simulate inference
of the new model on your computer.


## Dependencies
To communicate with the MPU6050, we use [jrowberg's](https://github.com/jrowberg/i2cdevlib) ```i2cdevlib``` library.  Last tested with commit [900b8f9](https://github.com/jrowberg/i2cdevlib/tree/900b8f959e9fa5c3126e0301f8a61d45a4ea99cc).
