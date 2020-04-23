# Codes for Face Detection experiments with RNNPool


## Dataset
1. download WIDER face dataset from http://shuoyang1213.me/WIDERFACE/
2. modify data/config.py 
3. ``` python prepare_wider_data.py ```



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
```shell
python eval.py --model_arch RPool_Face_C --model ./weights/rpool_face_c.pth --image_folder ./Himax_images
```

Code has been built upon https://github.com/yxlijun/S3FD.pytorch