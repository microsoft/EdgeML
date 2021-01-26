# Face Detection using SeeDot

Face Detection using SeeDot can be performed on the `face-4` dataset, which is a processed subset of the [SCUT Heads Part B](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release) dataset.

Face Detection is a regression problem that involves an image input with (or without) faces and bounding boxes around the faces as output. 
Face Detection is supported for x86 and M3 target devices.  

### Obtaining face-4 dataset
To obtain the dataset for the face detection problem, run `fetchFDDataset.py` in `seedot/compiler/input/` folder.
Or simply run the commands from SeeDot's home directory (`EdgeML/tools/SeeDot/`):
```
    cd seedot/compiler/input/
    python fetchFDDataset.py
```

This command will place the model files and datasets in `model/rnnpool/face-4/` and `datasets/rnnpool/face-4/` directories respectively.

### Run Face Detection for x86
To run face detection using the SeeDot quantizer for x86 devices, run the command. 

```
    python SeeDot-dev.py -a rnnpool -e fixed -d face-4 -m disagree -t x86 -n 18000 
```
The output will be stored in the `temp/` folder by default. Use the `-o <destination folder>` flag to store the output to an already existing location. 

### Run Face Detection for M3
To run face detection using the SeeDot quantizer for M3 device, run the command. 

```
    python SeeDot-dev.py -a rnnpool -e fixed -d face-4 -m disagree -t m3 -n 18000 
```

The output will be stored to the `m3dump/` by default. Use the `-o <destination folder>` flag to store the output to 
an already existing location. 

Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT license.

