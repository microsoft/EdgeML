## This code is built on https://github.com/yxlijun/S3FD.pytorch
#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import torch
from PIL import Image, ImageDraw
# import torch.utils.data as data
import numpy as np
import random
import sys; sys.path.append('../')
from tf_keras_augmentations import preprocess

import tensorflow as tf


class WIDERDetection():
    """docstring for WIDERDetection"""

    def __init__(self, list_file, mode='train'):
        super(WIDERDetection, self).__init__()
        self.mode = mode
        self.fnames = []
        self.boxes = []
        self.labels = []

        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            num_faces = int(line[1])
            box = []
            label = []
            for i in range(num_faces):
                x = float(line[2 + 5 * i])
                y = float(line[3 + 5 * i])
                w = float(line[4 + 5 * i])
                h = float(line[5 + 5 * i])
                c = int(line[6 + 5 * i])
                if w <= 0 or h <= 0:
                    continue
                box.append([x, y, x + w, y + h])
                label.append(c)
            if len(box) > 0:
                self.fnames.append(line[0])
                self.boxes.append(box)
                self.labels.append(label)

        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img, target, h, w = self.pull_item(index)
        return img, target

    def pull_item(self, index):
        while True:
            image_path = self.fnames[index]
            img = Image.open(image_path)
            # img.show()
            # print( img.size )
            if img.mode == 'L':
                img = img.convert('RGB')

            im_width, im_height = img.size
            boxes = self.annotransform( np.array(self.boxes[index]), im_width, im_height)
            label = np.array(self.labels[index])
            bbox_labels = np.hstack((label[:, np.newaxis], boxes)).tolist()
            img, sample_labels = preprocess( img, bbox_labels, self.mode, image_path)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) > 0:
                target = np.hstack( (sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis]))
                assert (target[:, 2] > target[:, 0]).any()
                assert (target[:, 3] > target[:, 1]).any()
                break
            else:
                index = random.randrange(0, self.num_samples)

        # # CHW -> HWC
        # #from data.config import cfg
        # img2 = np.round(img + cfg.img_mean)
        # img2 = img2.astype('uint8')
        # img2 = np.swapaxes(img2, 1, 0)
        # img2 = np.swapaxes(img2, 1, 2)
        # print( img2.shape )
        # w,h = img2.shape[:2]
        # imgDraw = Image.fromarray(img2, 'RGB')
        # imgDraw.show()
        # draw = ImageDraw.Draw(imgDraw)
        # for bbox in sample_labels:
        #     bbox = (bbox[1:] * np.array([w, h, w, h])).tolist()
        #     draw.rectangle(bbox, outline='red')
        # imgDraw.save('image.jpg')
        # imgDraw.show()

        return tf.convert_to_tensor(img, np.float32), target, im_height, im_width


    def annotransform(self, boxes, im_width, im_height):
        boxes[:, 0] /= im_width
        boxes[:, 1] /= im_height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(tf.Float32(sample[1]))

    return tf.stack(imgs, 0), targets


if __name__ == '__main__':
    from data.config import cfg
    dataset = WIDERDetection(cfg.FACE.TRAIN_FILE)
    #for i in range(len(dataset)):
    for i in range( 10, 20 ):
        d = dataset.pull_item(i)
        print( d )
        input('Hit Enter')
