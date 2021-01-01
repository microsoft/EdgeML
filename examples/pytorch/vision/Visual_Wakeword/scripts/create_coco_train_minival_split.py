## Code from https://github.com/Mxbonn/visualwakewords


"""Create maxitrain and minival annotations.
    This script generates a new train validation split with 115k training and 8k validation images.
    Based on the split used by Google
    (https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_minival_ids.txt).

    Usage:
    From this folder, run the following commands: (2014 can be replaced by 2017 if you downloaded the 2017 dataset)
        TRAIN_ANNOTATIONS_FILE="path-to-mscoco-dataset/annotations/instances_train2014.json"
        VAL_ANNOTATIONS_FILE="path-to-mscoco-dataset/annotations/instances_val2014.json"
        OUTPUT_DIR="path-to-mscoco-dataset/annotations/"
        python create_coco_train_minival_split.py \
          --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
          --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
          --output_dir="${OUTPUT_DIR}"
"""
import json
import os
from argparse import ArgumentParser


def create_maxitrain_minival(train_file, val_file, output_dir):
    """ Generate maxitrain and minival annotations files.
    Loads COCO 2014/2017 train and validation json files and creates a new split with
    115k training images and 8k validation images.
    Based on the split used by Google
    (https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_minival_ids.txt).
    Args:
        train_file: JSON file containing COCO 2014 or 2017 train annotations
        val_file: JSON file containing COCO 2014 or 2017 validation annotations
        output_dir: Directory where the new annotation files will be stored.
    """
    maxitrain_path = os.path.join(
        output_dir, 'instances_maxitrain.json')
    minival_path = os.path.join(
        output_dir, 'instances_minival.json')
    train_json = json.load(open(train_file, 'r'))
    val_json = json.load(open(val_file, 'r'))

    info = train_json['info']
    categories = train_json['categories']
    licenses = train_json['licenses']

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, 'mscoco_minival_ids.txt')
    minival_ids_f = open(file_path, 'r')
    minival_ids = minival_ids_f.readlines()
    minival_ids = [int(i) for i in minival_ids]

    train_images = train_json['images']
    val_images = val_json['images']
    train_annotations = train_json['annotations']
    val_annotations = val_json['annotations']

    maxitrain_images = []
    minival_images = []
    maxitrain_annotations = []
    minival_annotations = []

    for _images in [train_images, val_images]:
        for img in _images:
            img_id = img['id']
            if img_id in minival_ids:
                minival_images.append(img)
            else:
                maxitrain_images.append(img)

    for _annotations in [train_annotations, val_annotations]:
        for ann in _annotations:
            img_id = ann['image_id']
            if img_id in minival_ids:
                minival_annotations.append(ann)
            else:
                maxitrain_annotations.append(ann)

    with open(maxitrain_path, 'w') as fp:
        json.dump(
            {
                "info": info,
                "licenses": licenses,
                'images': maxitrain_images,
                'annotations': maxitrain_annotations,
                'categories': categories,
            }, fp)

    with open(minival_path, 'w') as fp:
        json.dump(
            {
                "info": info,
                "licenses": licenses,
                'images': minival_images,
                'annotations': minival_annotations,
                'categories': categories,
            }, fp)


def main(args):
    output_dir = os.path.realpath(os.path.expanduser(args.output_dir))
    train_annotations_file = os.path.realpath(os.path.expanduser(args.train_annotations_file))
    val_annotations_file = os.path.realpath(os.path.expanduser(args.val_annotations_file))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    create_maxitrain_minival(train_annotations_file, val_annotations_file, output_dir)


if __name__ == '__main__':
    parser = ArgumentParser(description="Script that takes the 2014/2017 training and validation annotations and"
                                        "creates a train split of 115k images and a minival of 8k.")
    parser.add_argument('--train_annotations_file', type=str, required=True,
                        help='COCO2014/2017 Training annotations JSON file')
    parser.add_argument('--val_annotations_file', type=str, required=True,
                        help='COCO2014/2017 Validation annotations JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory where the maxitrain and minival annotations files will be stored')

    args = parser.parse_args()
    main(args)
