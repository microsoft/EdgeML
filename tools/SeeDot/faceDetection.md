# Face Detection using SeeDot

Face Detection using SeeDot can be performed on the `face-4` dataset, which is a processed subset of the [SCUT Heads Part B](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release) dataset.

Face Detection is a regression problem that involves an image input with (or without) faces and bounding boxes around the faces as output. 
Face Detection is supported for x86 and M3 target devices.  

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


