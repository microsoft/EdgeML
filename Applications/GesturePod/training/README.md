Data Collection *(for MKR1000)*
=========================

## Table of Contents

- [Introduction](#intro)
- [Data Collection](#data-dollection)
- [Data Labelling](#data-labelling)
- [Feature Extraction](#feature-extraction)
- [Also](#also)
- [Dependencies](#dependencies)


## Introduction

Training a model to recognize gesture on the GesturePod consists of 3 parts:

1. Data collection
2. Data labelling
3. Extracting the features from the labelled data
4. Training a ML model
5. Deploying the model on the GesturePod to recognize gestures in real life.


## Data Collection

Accelorometer and gyroscope values from the MPU6050 sensor is collected. 
Connect the GesturePod to computer over Serial COM Port.

1. Refer [here](https://github.com/microsoft/EdgeML/blob/master/Applications/GesturePod/onMKR1000/README.md#quick-start) to set up the required dependencies for the MKR1000 platform.
		
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

*Coming Soon..*


## Feature Extraction

1. Enlist the labelled files in *labelledFileList* List in `featureExtraction.py`. Files that do not contain any gestures - for example, data of climbing stairs, walking in park, etc should be listed in *allNoiseFileList*.

2. Generate the set of features for labelled data. 
```
python generateFeatures.py
```
This will generate a `train.csv` and `test.csv` files that should be used to generate a ML model.


## Dependencies
To communicate with the MPU6050, we use [jrowberg's](https://github.com/jrowberg/i2cdevlib) ```i2cdevlib``` library.  Last tested with commit [900b8f9](https://github.com/jrowberg/i2cdevlib/tree/900b8f959e9fa5c3126e0301f8a61d45a4ea99cc).
