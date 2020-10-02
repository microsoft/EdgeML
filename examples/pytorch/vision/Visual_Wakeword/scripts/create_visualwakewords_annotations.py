## Code from https://github.com/Mxbonn/visualwakewords


"""Create Visual Wakewords annotations.
    This script generates the Visual WakeWords dataset annotations from the raw COCO dataset.
    The resulting annotations can then be used with `pyvww.utils.VisualWakeWords` and
    `pyvww.pytorch.VisualWakeWordsClassification`.

    Visual WakeWords Dataset is derived from the COCO dataset to design tiny models
    classifying two classes, such as person/not-person. The COCO annotations
    are filtered to two classes: foreground_class and background
    (for e.g. person and not-person). Bounding boxes for small objects
    with area less than 5% of the image area are filtered out.
    The resulting annotations file follows the COCO data format.
    {
      "info" : info,
      "images" : [image],
      "annotations" : [annotation],
      "licenses" : [license],
    }

    info{
      "year" : int,
      "version" : str,
      "description" : str,
      "url" : str,
    }

    image{
      "id" : int,
      "width" : int,
      "height" : int,
      "file_name" : str,
      "license" : int,
      "flickr_url" : str,
      "coco_url" : str,
      "date_captured" : datetime,
    }

    license{
      "id" : int,
      "name" : str,
      "url" : str,
    }

    annotation{
      "id" : int,
      "image_id" : int,
      "category_id" : int,
      "area" : float,
      "bbox" : [x,y,width,height],
      "iscrowd" : 0 or 1,
    }

    Example usage:
    From this folder, run the following commands:
        bash download_mscoco.sh path-to-mscoco-dataset
        TRAIN_ANNOTATIONS_FILE="path-to-mscoco-dataset/annotations/instances_train2014.json"
        VAL_ANNOTATIONS_FILE="path-to-mscoco-dataset/annotations/instances_val2014.json"
        DIR="path-to-mscoco-dataset/annotations/"
        python create_coco_train_minival_split.py \
          --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
          --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
          --output_dir="${DIR}"
        MAXITRAIN_ANNOTATIONS_FILE="path-to-mscoco-dataset/annotations/instances_maxitrain.json"
        MINIVAL_ANNOTATIONS_FILE="path-to-mscoco-dataset/annotations/instances_minival.json"
        VWW_OUTPUT_DIR="new-path-to-visualwakewords-dataset/annotations/"
        python create_visualwakewords_annotations.py \
          --train_annotations_file="${MAXITRAIN_ANNOTATIONS_FILE}" \
          --val_annotations_file="${MINIVAL_ANNOTATIONS_FILE}" \
          --output_dir="${VWW_OUTPUT_DIR}" \
          --threshold=0.005 \
          --foreground_class='person'
"""

import json
import os
from argparse import ArgumentParser

from pycocotools.coco import COCO


def create_visual_wakeword_annotations(annotations_file,
                                       visualwakewords_annotations_path,
                                       object_area_threshold,
                                       foreground_class_name):
    """Generate visual wake words annotations file.
    Loads COCO annotation json files and filters to foreground_class_name/not-foreground_class_name
    (by default it will be person/not-person) to generate visual wake words annotations file.
    Each image is assigned a label 1 or 0. The label 1 is assigned as long
    as it has at least one foreground_class_name (e.g. person)
    bounding box greater than object_area_threshold (e.g. 5% of the image area).
    Args:
      annotations_file: JSON file containing COCO bounding box annotations
      visualwakewords_annotations_path: output path to annotations file
      object_area_threshold: threshold on fraction of image area below which
        small object bounding boxes are filtered
      foreground_class_name: category from COCO dataset that is filtered by
        the visual wakewords dataset
    """
    print('Processing {}...'.format(annotations_file))
    coco = COCO(annotations_file)

    info = {"description": "Visual Wake Words Dataset",
            "url": "https://arxiv.org/abs/1906.05721",
            "version": "1.0",
            "year": 2019,
            }

    # default object of interest is person
    foreground_class_id = 1
    dataset = coco.dataset
    licenses = dataset['licenses']

    images = dataset['images']
    # Create category index
    foreground_category = None
    background_category = {'supercategory': 'background', 'id': 0, 'name': 'background'}
    for category in dataset['categories']:
        if category['name'] == foreground_class_name:
            foreground_class_id = category['id']
            foreground_category = category
    foreground_category['id'] = 1
    background_category['name'] = "not-{}".format(foreground_category['name'])
    categories = [background_category, foreground_category]

    if not 'annotations' in dataset:
        raise KeyError('Need annotations in json file to build the dataset.')
    new_ann_id = 0
    annotations = []
    positive_img_ids = set()
    foreground_imgs_ids = coco.getImgIds(catIds=foreground_class_id)
    for img_id in foreground_imgs_ids:
        img = coco.imgs[img_id]
        img_area = img['height'] * img['width']
        for ann_id in coco.getAnnIds(imgIds=img_id, catIds=foreground_class_id):
            ann = coco.anns[ann_id]
            if 'area' in ann:
                normalized_ann_area = ann['area'] / img_area
                if normalized_ann_area > object_area_threshold:
                    new_ann = {
                        "id": new_ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "area": ann["area"],
                        "bbox": ann["bbox"],
                        "iscrowd": ann["iscrowd"],
                    }
                    annotations.append(new_ann)
                    positive_img_ids.add(img_id)
                    new_ann_id += 1
    print("There are {} images that now have label {}, of the {} images in total.".format(len(positive_img_ids),
                                                                                          foreground_class_name,
                                                                                          len(coco.imgs)))
    negative_img_ids = list(set(coco.imgs.keys()) - positive_img_ids)
    for img_id in negative_img_ids:
        new_ann = {
            "id": new_ann_id,
            "image_id": img_id,
            "category_id": 0,
            "area": 0.0,
            "bbox": [],
            "iscrowd": 0,
        }
        annotations.append(new_ann)
        new_ann_id += 1

    # Output Visual WakeWords annotations and labels
    with open(visualwakewords_annotations_path, 'w') as fp:
        json.dump(
            {
                "info": info,
                "licenses": licenses,
                'images': images,
                'annotations': annotations,
                'categories': categories,
            }, fp)


def main(args):
    output_dir = os.path.realpath(os.path.expanduser(args.output_dir))
    train_annotations_file = os.path.realpath(os.path.expanduser(args.train_annotations_file))
    val_annotations_file = os.path.realpath(os.path.expanduser(args.val_annotations_file))
    visualwakewords_annotations_train = os.path.join(
        output_dir, 'instances_train.json')
    visualwakewords_annotations_val = os.path.join(
        output_dir, 'instances_val.json')
    small_object_area_threshold = args.threshold
    foreground_class_of_interest = args.foreground_class

    # Create the Visual WakeWords annotations from COCO annotations
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    create_visual_wakeword_annotations(
        train_annotations_file, visualwakewords_annotations_train,
        small_object_area_threshold, foreground_class_of_interest)
    create_visual_wakeword_annotations(
        val_annotations_file, visualwakewords_annotations_val,
        small_object_area_threshold, foreground_class_of_interest)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_annotations_file', type=str, required=True,
                        help='(COCO) Training annotations JSON file')
    parser.add_argument('--val_annotations_file', type=str, required=True,
                        help='(COCO) Validation annotations JSON file')
    parser.add_argument('--output_dir', type=str, default='/tmp/visualwakewords/',
                        help='Output directory where the Visual WakeWords annotations files be stored')
    parser.add_argument('--threshold', type=float, default=0.005,
                        help='Threshold of fraction of image area below which small objects are filtered.')
    parser.add_argument('--foreground_class', type=str, default='person',
                        help='Annotations will have a label indicating if this object is present or absent'
                             'in the scene (default is person/not-person).')

    args = parser.parse_args()
    main(args)
