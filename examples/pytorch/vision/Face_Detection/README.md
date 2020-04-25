# Codes for Face Detection experiments with RNNPool
## Requirements
1. ``` pip3 install -r requirements.txt ```

## Dataset
1. Download WIDER face dataset images and annotations from http://shuoyang1213.me/WIDERFACE/ and place them all in one folder with name 'WIDER_FACE'
2. In data/config.py modify _C.HOME to the parent directory of above folder and subsequently _C.FACE.WIDER_DIR as the folder path 
3. ``` python prepare_wider_data.py ```



# Usage
## Training

```shell

python train.py --batch_size 32 --model_arch RPool_Face_Quant --cuda True --multigpu True --save_folder weights/ --epochs 300 --save_frequency 5000 

```
This will save the checkpoints after every '--save_frequency' number of iterations in a weight file with 'checkpoint.pth' at the end and the best state weight in a file with 'best_state.pth' at the end. These will be saved in '--save_folder'. For resuming training from a checkpoint, use '--resume checkpoint_name.pth' with the above command.


## Test
Download eval_tools.zip from http://shuoyang1213.me/WIDERFACE/ and unzip in a folder of same name inside this folder

```shell
python wider_test.py --model_arch RPool_Face_Quant --model ./weights/rpool_face_quant.pth --save_folder rpool_face_quant_val --subset val --checkpoint_type old
```
This will save test predictions of the model in eval_tools folder with name '--save_folder'. Specify whether you are using a checkpoint already provided or if you trained your own in --checkpoint_type.

## Evaluation

```shell
python eval.py --model_arch RPool_Face_Quant --model ./weights/rpool_face_quant.pth --image_folder ./Himax_images  --checkpoint_type old
```
Specify whether you are using a checkpoint already provided or if you trained your own in '--checkpoint_type'


Code has been built upon https://github.com/yxlijun/S3FD.pytorch
