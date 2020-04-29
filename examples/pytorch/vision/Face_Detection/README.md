# Codes for Face Detection experiments with RNNPool
## Requirements
1. Follow instructions for installation in EdgeML/pytorch/README.md
1. ``` pip install -r requirements.txt ``` in EdgeML/examples/pytorch/vision/Face_Detection/

## Dataset
1. Download WIDER face dataset images and annotations from http://shuoyang1213.me/WIDERFACE/ and place them all in one folder with name 'WIDER_FACE'
2. In data/config.py modify _C.HOME to the parent directory of above folder and subsequently _C.FACE.WIDER_DIR as the folder path 
3. ``` python prepare_wider_data.py ```



# Usage
## Training

```shell

python train.py --batch_size 32 --model_arch RPool_Face_Quant --cuda True --multigpu True --save_folder weights/ --epochs 300 --save_frequency 5000 

```
This will save the checkpoints after every '--save_frequency' number of iterations in a weight file with 'checkpoint.pth' at the end and the best state weight in a file with 'best_state.pth' at the end. These will be saved in '--save_folder'. For resuming training from a checkpoint, use '--resume <checkpoint_name>.pth' with the above command.


## Test

```shell
python wider_test.py --model_arch RPool_Face_Quant --model ./weights/rpool_face_best_state.pth --save_folder rpool_face_quant_val --subset val
```
This will save test predictions of the model in eval_tools folder with name '--save_folder'. 

For calculating MAP scores:

1. Download eval_tools.zip from http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip and unzip in a folder of same name inside EdgeML/examples/pytorch/vision/Face_Detection/

2. ``` git clone https://github.com/wondervictor/WiderFace-Evaluation.git && cd WiderFace-Evaluation && python3 setup.py build_ext --inplace```

3. ```python3 evaluation.py -p <your prediction dir> -g <groud truth dir>``` in WiderFace-Evaluation folder

where <your prediction dir> is the path to the '--save_folder' in the command above this and <groud truth dir> is EdgeML/examples/pytorch/vision/Face_Detection/eval_tools/ground_truth


## Evaluation

```shell
python eval.py --model_arch RPool_Face_Quant --model ./weights/rpool_face_best_state.pth --image_folder ./Himax_images
```
The evaluation code accepts any size of image and resizes it to have area = 640x480 while preserving original image aspect ratio.

Code has been built upon https://github.com/yxlijun/S3FD.pytorch