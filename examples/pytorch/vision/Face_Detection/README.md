# Code for Face Detection experiments with RNNPool
## Requirements
1. Follow instructions to install requirements for EdgeML operators and the EdgeML operators [here](pytorch/README.md).
2. Install requirements for face detection model in this directory using
``` pip install -r requirements.txt ``` 

## Dataset
1. Download WIDER face dataset images and annotations from http://shuoyang1213.me/WIDERFACE/ and place them all in a folder with name 'WIDER_FACE'
2. In `data/config.py`, set _C.HOME to the parent directory of the above folder, and set the _C.FACE.WIDER_DIR to the folder path 
3. Run
``` python prepare_wider_data.py ```


# Usage
## Training

```shell

python train.py --batch_size 32 --model_arch RPool_Face_Quant --cuda True --multigpu True --save_folder weights/ --epochs 300 --save_frequency 5000 

```
This will save checkpoints after every '--save_frequency' number of iterations in a weight file with 'checkpoint.pth' at the end and weights for the best state in a file with 'best_state.pth' at the end. These will be saved in '--save_folder'. For resuming training from a checkpoint, use '--resume <checkpoint_name>.pth' with the above command.


## Test
Run the following to generate predictions of the model and store output in the '--save_folder' folder.
```shell
python wider_test.py --model_arch RPool_Face_Quant --model ./weights/rpool_face_best_state.pth --save_folder rpool_face_quant_val --subset val
```
 
For calculating MAP scores:

1. Download eval_tools.zip from http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip and unzip in a folder of same name in this directory.

2. Set up scripts to use the Matlab '.mat' data files in eval_tools/ground_truth folder for MAP calculation: The following installs python files that provide the same functionality as the '.m' matlab scripts in eval_tools folder:
``` 
git clone https://github.com/wondervictor/WiderFace-Evaluation.git
cd WiderFace-Evaluation 
python3 setup.py build_ext --inplace
```

3. Run ```python3 evaluation.py -p <prediction_dir> -g <groud truth dir>``` in WiderFace-Evaluation folder

where `prediction_dir` is the '--save_folder' used for `wider_test.py` above and <groud truth dir> is the subfolder `eval_tools/ground_truth`


## Evaluation
Place images you wish to evaluate in a folder and run the following script inside Face_Detection directory:
```shell
python eval.py --model_arch RPool_Face_Quant --model ./weights/rpool_face_best_state.pth --image_folder <your_image_folder>
```
The evaluation code accepts an image of any size and resizes it to 640x480 while preserving original image aspect ratio. The output consists of images with bounding boxes around faces.

Code has been built upon https://github.com/yxlijun/S3FD.pytorch
