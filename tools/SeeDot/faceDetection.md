# Quantized Face Detection using SeeDot

Face detection using SeeDot can be performed on the `face-2` and `face-4` datasets, which are subsets of the [SCUT Head Part B](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release) dataset.

Face detection is a regression problem that involves an image input with (or without) faces and bounding boxes superimposed around the faces as output.
Face detection is supported for x86 and ARM Cortex-M3 target devices.  

Note: 
1. This README has been tested with **Python 3.7.9**. Quantization with SeeDot requires **GCC** *version 8 or higher*.
2. The dataset `face-2` corresponds to the model **RPool_Face_QVGA_monochrome** and; `face-4` corresponds to the model **RPool_Face_M4**.

## Training Face Detection 

Run the following commands to download the training data for face detection and train the model. This trained model would be used for quantized face detection.

### Setup and Downloading the Dataset

Assuming that the current directory is `EdgeML/tools/seedot`, please run the following commands. 

Create directories for SeeDot's input and install python libraries used by SeeDot (which will be generated at the end of this section):
```
    cd ../../tf/
    
    # For system with CPU only
    pip install -r requirements-cpu.txt

    # For systems with CPU+GPU
    pip install -r requirements-gpu.txt

    # For both
    pip install -e .
    cd ../tools/SeeDot/

    mkdir -p model/rnnpool/face-2/
    mkdir -p model/rnnpool/face-4/

    mkdir -p datasets/rnnpool/face-2/
    mkdir -p datasets/rnnpool/face-4/
```

Install the python libraries: 
```
    cd ../../pytorch

    # For a CPU only system
    pip install -r requirements-cpu.txt

    # For a CPU+GPU system
    pip install -r requirements-gpu.txt

    #For Both
    pip install -e .
```

Next, please visit the [PyTorch website](https://pytorch.org/get-started/locally/) and install PyTorch version 1.7.1 as per your system requirements.
Here, we are using PyTorch version 1.7.1 with CUDA 11.

```
    pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

```
Then, continue running the following commands:
```
    # For a CPU+GPU system
    pip install -e edgeml_pytorch/cuda/

    # For both
    cd ../examples/pytorch/vision/Face_Detection/
    pip install -r requirements.txt
```

Steps to download the dataset: 

Note: The datasets can be stored in any location outside the repository as well. 
Here, let's assume that the dataset will be downloaded to `/mnt/` and the current directory be represented by the environment variable *CUR_DIR*. 

```
    export CUR_DIR=$(pwd)
    cd /mnt/
    mkdir -p WIDER_FACE/
```

Download WIDER face dataset images and annotations from [http://shuoyang1213.me/WIDERFACE/](http://shuoyang1213.me/WIDERFACE/) and place it in the `WIDER_FACE` folder created above. Now the `WIDER_FACE` folder must contain `WIDER_train.zip`, `WIDER_test.zip`, `WIDER_val.zip`, `wider_face_split.zip`.

Download the `SCUT Head Part B` dataset from the [drive link](https://drive.google.com/file/d/1LZ_KlTPStDEcqycfqUkDkqQ-aNMMC3cl/view) and place it in `/mnt` folder (the dataset directory). So the dataset directory should contain `SCUT_HEAD_Part_B.zip`.

Now, decompress the datasets and add the `DATA_HOME` environment variable (for the training script to find the dataset) using the following commands:
```
    cd WIDER_FACE/
    unzip WIDER_train.zip
    unzip WIDER_val.zip
    unzip WIDER_test.zip
    unzip wider_face_split.zip
    cd ../

    unzip SCUT_HEAD_Part_B.zip
    export DATA_HOME=$(pwd) # For the scripts to find the datasets

    cd $CUR_DIR # To go back to Face_Detection directory
    
    # Data pre-processing
    IS_QVGA_MONO=1 python prepare_wider_data.py 
```

### Training

From here, we have two model options. For `face-2`, we use the model, **RPool_Face_QVGA_monochrome** and for `face-4`, we use the model **RPool_Face_M4**.

Note: The training script has the arguments `--cuda` and `--multigpu` which have to be set `True` or `False` based on the system configuration. (In this README, both have been set to `True` and; `cuda` and `multigpu` arguments always have to be set to `True` or `False` together). 

To start training:
1.  For `face-2`:
    ```
        IS_QVGA_MONO=1 python train.py --batch_size 64 --model_arch RPool_Face_QVGA_monochrome --cuda True --multigpu True --save_folder weights/ --epochs 300 --save_frequency 5000

        # In case the training has to be stopped prematurely, then it can be resumed using the following command
        # IS_QVGA_MONO=1 python train.py --batch_size 64 --model_arch RPool_Face_QVGA_monochrome --cuda True --multigpu True --save_folder weights/ --epochs 300 --save_frequency 5000 --resume weights/RPool_Face_QVGA_monochrome_best_state.pth
    ```
2. For `face-4`: 
    ```
        IS_QVGA_MONO=1 python train.py --batch_size 64 --model_arch RPool_Face_M4 --cuda True --multigpu True --save_folder weights/ --epochs 300 --save_frequency 5000

        # In case the training has to be stopped prematurely, then it can be resumed using the following command
        # IS_QVGA_MONO=1 python train.py --batch_size 64 --model_arch RPool_Face_M4 --cuda True --multigpu True --save_folder weights/ --epochs 300 --save_frequency 5000 --resume weights/RPool_Face_M4_best_state.pth
    ```

This will train the model on the **WIDER face** dataset. Now, to fine-tune the model on **SCUT Head Part B** dataset, run the following commands. 

1.  For `face-2`:
    ```
        IS_QVGA_MONO=1 python train.py --batch_size 64 --model_arch RPool_Face_QVGA_monochrome --cuda True --multigpu True --save_folder weights/ --epochs 300 --save_frequency 5000 --resume weights/RPool_Face_QVGA_monochrome_best_state.pth --finetune True

    ```
2. For `face-4`: 
    ```
        IS_QVGA_MONO=1 python train.py --batch_size 64 --model_arch RPool_Face_M4 --cuda True --multigpu True --save_folder weights/ --epochs 300 --save_frequency 5000 --resume weights/RPool_Face_M4_best_state.pth --finetune True
    ```

### Generating SeeDot's Input

Now that the model is trained, we need to generate the model traces and convert them to SeeDot format. For that, we have to create a subset of the SCUT Head B dataset which will be used for quantization. 

Run the following commands:
1. Creating a subset of `SCUT_HEAD_Part_B`:
    ```
        mkdir images/
        cp ${DATA_HOME}/SCUT_HEAD_Part_B/JPEGImages/PartB_00009.jpg images/ && \
        cp ${DATA_HOME}/SCUT_HEAD_Part_B/JPEGImages/PartB_00052.jpg images/ && \
        cp ${DATA_HOME}/SCUT_HEAD_Part_B/JPEGImages/PartB_00082.jpg images/ && \
        cp ${DATA_HOME}/SCUT_HEAD_Part_B/JPEGImages/PartB_00101.jpg images/ && \
        cp ${DATA_HOME}/SCUT_HEAD_Part_B/JPEGImages/PartB_00112.jpg images/ && \
        cp ${DATA_HOME}/SCUT_HEAD_Part_B/JPEGImages/PartB_00170.jpg images/ && \
        cp ${DATA_HOME}/SCUT_HEAD_Part_B/JPEGImages/PartB_00195.jpg images/ && \
        cp ${DATA_HOME}/SCUT_HEAD_Part_B/JPEGImages/PartB_00376.jpg images/ && \
        cp ${DATA_HOME}/SCUT_HEAD_Part_B/JPEGImages/PartB_00398.jpg images/ && \
        cp ${DATA_HOME}/SCUT_HEAD_Part_B/JPEGImages/PartB_00601.jpg images/ && \
        cp ${DATA_HOME}/SCUT_HEAD_Part_B/JPEGImages/PartB_00675.jpg images/ && \
        cp ${DATA_HOME}/SCUT_HEAD_Part_B/JPEGImages/PartB_00735.jpg images/ && \
        cp ${DATA_HOME}/SCUT_HEAD_Part_B/JPEGImages/PartB_00973.jpg images/ && \
        cp ${DATA_HOME}/SCUT_HEAD_Part_B/JPEGImages/PartB_02378.jpg images/ && \
        cp ${DATA_HOME}/SCUT_HEAD_Part_B/JPEGImages/PartB_02396.jpg images/
    ```
2. Obtaining traces:
    1. For `face-2`:
        ```
            IS_QVGA_MONO=1 python eval.py --model_arch RPool_Face_QVGA_monochrome --model ./weights/RPool_Face_QVGA_monochrome_best_state.pth --image_folder images/ --save_dir results/ --thresh 0.5 --multigpu True --save_traces True
        ```
    2. For `face-4`:
        ```
            IS_QVGA_MONO=1 python eval.py --model_arch RPool_Face_M4 --model ./weights/RPool_Face_M4_best_state.pth --image_folder images/ --save_dir results/ --thresh 0.5 --multigpu True --save_traces True
        ```
3. Converting the traces to SeeDot format:
    1. For `face-2`:
        ```
            IS_QVGA_MONO=1 python convert_RPool_Face_to_SeeDot.py --model_arch RPool_Face_QVGA_monochrome --model ./weights/RPool_Face_QVGA_monochrome_best_state.pth
        ```
    2. For `face-4`:
        ```
            IS_QVGA_MONO=1 python convert_RPool_Face_to_SeeDot.py  --model_arch RPool_Face_M4 --model ./weights/RPool_Face_M4_best_state.pth
        ```
This will store SeeDot's input to `EdgeML/tools/SeeDot/model/rnnpool/face-2/`, `EdgeML/tools/SeeDot/datasets/rnnpool/face-2/` or; `EdgeML/tools/SeeDot/model/rnnpool/face-4/`, `EdgeML/tools/SeeDot/datasets/rnnpool/face-4/` respectively.

### Setting up SeeDot

To finish setting up SeeDot, run the following commands:

```
    cd ../../../../tools/SeeDot

    # For face-2
    python fixSeeDotInput.py --seedot_file seedot/compiler/input/rnnpool-face-2.sd --model_dir model/rnnpool/face-2/ --dataset_dir datasets/rnnpool/face-2/ -n 18000
    
    # For face-4
    python fixSeeDotInput.py --seedot_file seedot/compiler/input/rnnpool-face-4.sd --model_dir model/rnnpool/face-4/ --dataset_dir datasets/rnnpool/face-4/ -n 18000
```
 

## Running SeeDot

### Run Face Detection for x86
To run face detection using the SeeDot quantizer on x86 devices, run the following command: 

1. For `face-2`:
    ```
        python SeeDot-dev.py -a rnnpool -e fixed -m disagree -d face-2 -dt testing -t x86 -n 18000 
    ```

2. For `face-4`:
    ```
        python SeeDot-dev.py -a rnnpool -e fixed -m disagree -d face-4 -dt testing -t x86 -n 18000 
    ```


The non-optional arguments used in the above commands are:
```
    Argument            Value       Description
    
    -a, --algo          rnnpool     Face detection problem uses the 'rnnpool' machine learning
                                    algorithm.

    -e, --encoding      fixed/      The encoding to use for face detection. 
                        float       The options are 'float' and 'fixed'.

    -m, --metric        disagree    The metric used to measure correctness of prediction.
                                    This is used for quantization. 

    -d, --dataset       face-2/     The dataset to use for face detection. 
                        face-4      The options are 'face-2' and 'face-4'.
    
    -dt, --datesetType  training/   The type of the dataset to use for quantization. 
                        testing     The options are 'training' and 'testing'.
                                    Default is 'testing'.

    -t, --target        x86         The target device for which the code has to be generated. 
                                    'x86' in this case.

    -n, --numOutputs    18000       The size of the output vector of rnnpool face detection
                                    algorithm.
                      
```
The output will be stored in the `temp/` directory by default. Use the `-o <destination folder>` flag to store the output to an already existing location. 

### Run Face Detection for M3
To run face detection using the SeeDot quantizer for M3 device, run the command: 

```
    python SeeDot-dev.py -a rnnpool -e fixed -m disagree -d face-2 -dt testing -t m3 -n 18000 
```
for the `face-2` dataset, and:
```
    python SeeDot-dev.py -a rnnpool -e fixed -m disagree -d face-4 -dt testing -t m3 -n 18000 
```
for the `face-4` dataset.

Note: The target device specified using the `-t` argument has the value `m3` in this case. See above for a detailed argument description.

The output will be stored in the `m3dump/` directory by default. Use the `-o <destination folder>` flag to store the output to an already existing location. 

## Run SeeDot's Output - Quantized Prediction Code

This section discusses the execution of the quantized fixed-point model generated by SeeDot on 
random images from the `SCUT Head Part B` dataset.

### Library Installation
This section requires installing `pytorch`, `cv2`, `easydict`, and `six` python libraries. Run the following commands for installing them:
```
    pip install opencv-python
    pip install easydict
    pip install six
```

### Running Face Detection on Image Input

By default, the quantized `x86` codes are stored in the `temp/` directory. 

To run the model, we use the `Predictor` executable in the `temp/Predictor`directory.

First, we need to download an image to test the quantized `Predictor`.

For that, we follow the below steps:
```
    cd faceDetection/
    mkdir -p images/ && cd images/
    cp ${DATA_HOME}/SCUT_HEAD_Part_B/JPEGImages/PartB_00007.jpg ./
    cd ..
    cp -r ../../../examples/pytorch/vision/Face_Detection/layers/ .
    cp -r ../../../examples/pytorch/vision/Face_Detection/utils/ .
    mkdir -p data/
    cp -r ../../../examples/pytorch/vision/Face_Detection/data/choose_config.py data/
    cp -r ../../../examples/pytorch/vision/Face_Detection/data/config_qvga.py data/
```

Note: The last 5 commands copy scripts from another location of this repository, to help in the conversion of images to text files and vice-versa. 

This will copy `PartB_00007.jpg` to the `images/` directory. Users can use their own images as well instead of the above image.
Multiple images can be added to the `images/` directory, which will be processed in a single execution.  

Note that we run all of the following python scripts in the `SeeDot/faceDetection` directory. 

To create the processed file used by `Predictor` from the set of images, we run the following command:

```
    python scale_image.py --image_dir images/ --out_dir input/
```

The script `scale_image.py` reads all the images in the `images/` directory (which is the default for the `image_dir` field). 
And outputs the files `X.csv` and `Y.csv` in the `input/` directory (which is the default for the `out_dir` field). 

Now, we copy the executable to this directory and run it using the below commands:
```
    cp ../temp/Predictor/Predictor .
    mkdir -p input/
    mkdir -p output/
    ./Predictor fixed testing regression 18000 False
```

For running the executable, specifying all arguments is necessary. The arguments' descriptions are:
```
    Argument        Description

    fixed           The encoding. This is dependent on the 'encoding' argument 
                    given to SeeDot. 
    
    testing         This is the argument in SeeDot specified by 'datasetType' field.
    
    regression      This is the type of problem ('classification' and 'regression'). 
                    This argument is inferred by SeeDot automatically.
    
    18000           This is equal to the argument specified in SeeDot's execution using
                    the 'numOutputs' field.
```

The executable is copied from `temp/Predictor` because that is the default output directory of SeeDot for target **x86**.

This executable takes its input from the `input/` directory (hence the use of `input` as the default for `out_dir` argument of `scale_image.py`). 

`X.csv` contains the floating-point input values; and `Y.csv`, consists of the correct integer labels (floating-point outputs) in case of classification (regression). However, since we are only concerned with the predicted output here, 
the contents of `Y.csv` are irrelevant. 

In the case of `face-2` and `face-4`, the input layer size is *76800* and the output 
has *18000* values (the number of columns in `X.csv` and `Y.csv` respectively). 

The output of `Predictor` is stored to `trace.txt`.

The executable dumps the execution stats and accuracy results in the `output/` directory. While it must exist, this directory is irrelevant to our discussion.



Now we construct the bounding boxes from `trace.txt`.

For that we run the below commands:
```
    python eval_fromquant.py --save_dir results/ --image_dir images/ --trace_file trace.txt
```

The images on which the prediction was carried out are read from the `images/` directory (which is the default for the `image_dir` field). The `image_dir` argument must be the same for both `scale_image.py` and `eval_fromquant.py`.

The images with bounding boxes drawn are stored in the `results/` directory (which is the default for the `save_dir` field).

`trace_file` field takes the location of `trace.txt` as the argument (which is `./trace.txt` by default).





Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT license.
