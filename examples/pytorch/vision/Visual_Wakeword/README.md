# Code for Visual Wake Words experiments with RNNPool

The Visual Wake Word challenge is a binary classification problem of detecting whether a person is present in 
an image or not, as introduced by [Chowdhery et. al](https://arxiv.org/abs/1906.05721).

## Dataset
The Visual Wake Words Dataset is derived from the publicly available [COCO](cocodataset.org/#/home) dataset. The Visual Wake Words Challenge evaluates accuracy on the [minival image ids](https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_minival_ids.txt),
and for training uses the remaining 115k images of the COCO training/validation dataset. The process of creating the Visual Wake Words dataset from COCO dataset is as follows.
Each image is assigned a label 1 or 0. 
The label 1 is assigned as long as it has at least one bounding box corresponding 
to the object of interest (e.g. person) with the box area greater than a certain threshold 
(e.g. 0.5% of the image area).

To download the COCO dataset use the script `download_coco.sh`
```bash
bash scripts/download_mscoco.sh path-to-mscoco-dataset
```

To create COCO annotation files that converts to the minival split use:
`scripts/create_coco_train_minival_split.py`

```bash
TRAIN_ANNOTATIONS_FILE="path-to-mscoco-dataset/annotations/instances_train2014.json"
VAL_ANNOTATIONS_FILE="path-to-mscoco-dataset/annotations/instances_val2014.json"
DIR="path-to-mscoco-dataset/annotations/"
python scripts/create_coco_train_minival_split.py \
  --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
  --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
  --output_dir="${DIR}"
```


To generate the new annotations, use the script `scripts/create_visualwakewords_annotations.py`.
```bash
MAXITRAIN_ANNOTATIONS_FILE="path-to-mscoco-dataset/annotations/instances_maxitrain.json"
MINIVAL_ANNOTATIONS_FILE="path-to-mscoco-dataset/annotations/instances_minival.json"
VWW_OUTPUT_DIR="new-path-to-visualwakewords-dataset/annotations/"
python scripts/create_visualwakewords_annotations.py \
  --train_annotations_file="${MAXITRAIN_ANNOTATIONS_FILE}" \
  --val_annotations_file="${MINIVAL_ANNOTATIONS_FILE}" \
  --output_dir="${VWW_OUTPUT_DIR}" \
  --threshold=0.005 \
  --foreground_class='person'
```


# Training

```bash
python train_visualwakewords.py \
    --model_arch model_mobilenet_rnnpool \
    --lr 0.05 \
    --epochs 900 \
    --data "path-to-mscoco-dataset" \
    --ann "new-path-to-visualwakewords-dataset"
```
Specify the paths used for storing MS COCO dataset and the Visual Wakeword dataset as used in dataset creation steps in --data and --ann respectively. This script should reach a validation accuracy of about 89.57 upon completion.

# Evaluation

```bash
python eval.py \
    --weights vww_rnnpool.pth \
    --model_arch model_mobilenet_rnnpool \
    --image_folder images \
```

The weights argument is the saved checkpoint of the model trained with architecture which is passed in model_arch argument. The folder with images for evaluation has to be passed in image_folder argument. This script will print 'Person present' or 'No person present' for each image in the folder specified.


Dataset creation code is from https://github.com/Mxbonn/visualwakewords/
