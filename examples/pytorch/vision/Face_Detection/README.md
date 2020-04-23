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

python train.py --batch_size 32 --model_arch RPool_Face_Quant --cuda True --multigpu True

```

## Test
```shell
python wider_test.py --model_arch RPool_Face_Quant --model ./weights/rpool_face_quant.pth
```

## Evaluation
1. Download eval_tools.zip from http://shuoyang1213.me/WIDERFACE/ and unzip in a folder of same name inside this folder

```shell
python eval.py --model_arch RPool_Face_Quant --model ./weights/rpool_face_quant.pth --image_folder ./Himax_images
```

Code has been built upon https://github.com/yxlijun/S3FD.pytorch