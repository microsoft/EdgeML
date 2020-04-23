# Codes for Face Detection experiments with RNNPool


## Dataset
1. download WIDER face dataset images and annotations from http://shuoyang1213.me/WIDERFACE/ and place them all in one folder with name 'WIDER_FACE'
2. in data/config.py modify HOME to where the above folder is placed to and subsequently FACE.WIDER_DIR as the folder path 

``` python prepare_wider_data.py ```



# Usage
## Training

```shell

python train.py --batch_size 32 --model_arch RPool_Face_C --cuda True --multigpu True

```


## Test
```shell
python wider_test.py --model_arch RPool_Face_C --model ./weights/rpool_face_c.pth
```

## Evaluation
1. download eval_tools.zip from http://shuoyang1213.me/WIDERFACE/ and unzip in a folder of same name inside this folder

```shell
python eval.py --model_arch RPool_Face_C --model ./weights/rpool_face_c.pth --image_folder ./Himax_images
```

Code has been built upon https://github.com/yxlijun/S3FD.pytorch