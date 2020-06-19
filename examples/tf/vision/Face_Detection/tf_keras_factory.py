## This code is built on https://github.com/yxlijun/S3FD.pytorch

#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


from tf_keras_widerface import WIDERDetection
from data.config import cfg

# import torch
import tensorflow as tf


def dataset_factory(dataset):
    if dataset == 'face':
        train_dataset = WIDERDetection(cfg.FACE.TRAIN_FILE, mode='train')
        val_dataset = WIDERDetection(cfg.FACE.VAL_FILE, mode='val')
    # if dataset == 'hand':
    #     train_dataset = WIDERDetection(cfg.HAND.TRAIN_FILE, mode='train')
    #     val_dataset = WIDERDetection(cfg.HAND.VAL_FILE, mode='val')
    # if dataset == 'head':
    #     train_dataset = VOCDetection(cfg.HEAD.DIR, image_sets=[
    #                                  ('PartA', 'trainval'), ('PartB', 'trainval')],
    #                                  target_transform=VOCAnnotationTransform(),
    #                                  mode='train',
    #                                  dataset_name='VOCPartAB')
    #     val_dataset = VOCDetection(cfg.HEAD.DIR, image_sets=[('PartA', 'test'), ('PartB', 'test')],
    #                                target_transform=VOCAnnotationTransform(),
    #                                mode='test',
    #                                dataset_name='VOCPartAB')
    return train_dataset, val_dataset


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
        targets.append(tf.Tensor(sample[1], dtype=tf.float32))

    return tf.stack(imgs, 0), targets


class dataLoader():
    def __init__(self, dataset, batch_size=1 ):
        self.dataset = dataset
        self.batchSize = batch_size
        self.nextIndex = 0

    def reset( self ):
        self.nextIndex = 0

    def call( self ):
        if self.nextIndex >= self.dataset.num_samples:
            self.reset()
        endIdx = self.nextIndex + self.batchSize
        if endIdx > self.dataset.num_samples:
            endIdx = self.dataset.num_samples
        batch = []
        while self.nextIndex < endIdx:
            batch += [ [self.dataset.__getitem__( self.nextIndex )] ]
            self.nextIndex += 1

        return detection_collate( batch )
