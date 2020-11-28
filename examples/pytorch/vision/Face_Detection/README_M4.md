# Code for Face Detection Experiments with RNNPool
## Requirements
1. Follow instructions to install EdgeML operators and their pre-requisites [here](https://github.com/microsoft/EdgeML/blob/master/pytorch/README.md).
2. Install requirements for face detection model using
``` pip install -r requirements.txt ``` 
We have tested the installation and the code on Ubuntu 18.04 with Python 3.6, Cuda 10.2 and CuDNN 7.6

## Dataset - WIDER Face
1. Download WIDER face dataset images and annotations from http://shuoyang1213.me/WIDERFACE/ and place them all in a folder with name 'WIDER_FACE'. That is, download WIDER_train.zip, WIDER_test.zip, WIDER_val.zip, wider_face_split.zip and place it in WIDER_FACE folder, and unzip files using: 

```shell
cd WIDER_FACE
unzip WIDER_train.zip
unzip WIDER_test.zip
unzip WIDER_val.zip
unzip wider_face_split.zip
cd ..

```

2. In `data/config_qvga.py` , set _C.HOME to the parent directory of the above folder, and set the _C.FACE.WIDER_DIR to the folder path. 
That is, if the WIDER_FACE folder is created in /mnt folder, then _C.HOME='/mnt'
_C.FACE.WIDER_DIR='/mnt/WIDER_FACE'.

3. Run
``` python prepare_wider_data.py ```

## Dataset - SCUT Head B
1. Download SCUT Head Part B dataset images and annotations from https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release. Unzipping will create a folder by the name 'SCUT_HEAD_Part_B'.

2. In `data/config_qvga.py` set the _C.FACE.SCUT_DIR to the folder path. 
That is, if the SCUT_HEAD_Part_B folder is created in /mnt folder, then _C.FACE.SCUT_DIR='/mnt/SCUT_HEAD_Part_B'.

Note that for Windows '/' should be replaced by '\' for each path in the config files.

# Usage

## Training

```shell

IS_QVGA_MONO=1 python train.py --batch_size 128 --model_arch RPool_Face_M4 --cuda True --multigpu True --save_folder weights/ --epochs 300 --save_frequency 5000 

```
This will save checkpoints after every '--save_frequency' number of iterations in a weight file with 'checkpoint.pth' at the end and weights for the best state in a file with 'best_state.pth' at the end. These will be saved in '--save_folder'. For resuming training from a checkpoint, use '--resume <checkpoint_name>.pth' with the above command. For example, 


```shell

IS_QVGA_MONO=1 python train.py --batch_size 128 --model_arch RPool_Face_M4 --cuda True --multigpu True --save_folder weights/ --epochs 300 --save_frequency 5000 --resume <checkpoint_name>.pth

```

If IS_QVGA_MONO is 0 then training input images will be 640x640 and RGB. 
If IS_QVGA_MONO is 1 then training input images will be 320x320 and converted to monochrome. 

Input images for training models are cropped and reshaped to square to maintain consistency with [S3FD](https://arxiv.org/abs/1708.05237). However testing can be done on any size of images, thus we resize testing input image size to have area equal to VGA (640x480)/QVGA (320x240), so that aspect ratio is not changed.

The architecture RPool_Face_QVGA_monochrome and RPool_Face_M4 is for QVGA monochrome format while RPool_Face_C and RPool_Face_Quant are for VGA RGB format.

## Finetuning

To obtain a model better suited for conference room scenarios we finetune our model on the SCUT Head B dataset. Set --finetune as True and pass the model pretrained on WIDER_FACE in --resume as follows:

```shell

IS_QVGA_MONO=1 python train.py --batch_size 64 --model_arch RPool_Face_M4 --cuda True --multigpu True --save_folder weights/ --epochs 300 --save_frequency 5000 --resume ./weights/RPool_Face_M4_best_state.pth --finetune True

```


## Test
There are two modes of testing the trained model -- the evaluation mode to generate bounding boxes for a set of sample images, and the test mode to compute statistics like mAP scores.

#### Evaluation Mode

Given a set of images in <your_image_folder>, `eval/py` generates bounding boxes around faces (where the confidence is higher than certain threshold - 0.5 in this case) and write the images in <your_save_folder>. To evaluate the `rpool_face_best_state.pth` model (stored in ./weights), execute the following command: 

```shell
IS_QVGA_MONO=1 python eval.py --model_arch RPool_Face_M4 --model ./weights/RPool_Face_M4_best_state.pth --image_folder <your_image_folder> --save_dir <your_save_folder> --thresh 0.5
```

This will save images in <your_save_folder> with bounding boxes around faces, where the confidence is high. It is recommended to use the model finetuned on SCUT Head for evaluation.

If IS_QVGA_MONO=0 the evaluation code accepts an image of any size and resizes it to 640x480x3 while preserving original image aspect ratio.

If IS_QVGA_MONO=1 the evaluation code accepts an image of any size and resizes and converts it to monochrome to make it 320x240x1 while preserving original image aspect ratio.

#### SCUT Head Validation Set Test
In this mode, we test the generated model against the provided SCUT Head Part B validation dataset. Use the SCUT Head finetuned model for this step.

For this, first run the following to generate predictions of the model and store output in the '--save_folder' folder. 

```shell
IS_QVGA_MONO=1 python scut_test.py --model_arch RPool_Face_M4 --model ./weights/RPool_Face_M4_best_state.pth --save_folder rpool_face_m4_val --subset val
```

The above command generates predictions for each image in the "validation" dataset. For each image, a separate prediction file is provided (image_name.txt file in appropriate folder). The first line of the prediction file contains the total number of boxes identified. 
Then each line in the file corresponds to an identified box. For each box, five numbers are generated: length of the box, height of the box, x-axis offset, y-axis offset, confidence value for presence of a face in the box. 

If IS_QVGA_MONO=1 then testing is done by converting images to monochrome and QVGA, else if IS_QVGA_MONO=0 then testing is done on VGA RGB images.

###### For calculating mAP scores:
Now using these boxes, we can compute the standard mAP score that is widely used in this literature (see [here](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173) for more details).

In this directory run:
``` 
git clone https://github.com/wondervictor/WiderFace-Evaluation.git
cd WiderFace-Evaluation 
python3 setup.py build_ext --inplace
mv ../scut_evaluation.py ./
```

Run ```IS_QVGA_MONO=1 python3 scut_evaluation.py -p <your_save_folder> ``` in WiderFace-Evaluation folder.

where `prediction_dir` is the '--save_folder' used for `scut_test.py` above. 

This script should output the mAP on SCUT Head Part B Validation set. Our best performance using RPool_Face_M4 model is: 0.61.


##### Dump RNNPool Input Output Traces and Weights

For saving model weights and/or input output pairs for each patch through RNNPool in numpy format use the command below. Put images which you want to save traces for in <your_image_folder> . Specify output folder for saving model weights in numpy format in <your_save_model_numpy_folder>. Specify output folder for saving input output traces of RNNPool in numpy format in <your_save_traces_numpy_folder>. Note that input traces will be saved in a folder named 'inputs' and output traces in a folder named 'outputs' inside <your_save_traces_numpy_folder>.

```shell
python3 dump_model.py --model ./weights/RPool_Face_M4_best_state.pth --model_arch RPool_Face_M4 --image_folder <your_image_folder> --save_model_npy_dir <your_save_model_numpy_folder> --save_traces_npy_dir <your_save_traces_numpy_folder>
```
If you wish to save only model weights, do not specify --save_traces_npy_dir. If you wish to save only traces do not specify --save_model_npy_dir.

Code has been built upon https://github.com/yxlijun/S3FD.pytorch