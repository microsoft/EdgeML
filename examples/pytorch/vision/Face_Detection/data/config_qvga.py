#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
from easydict import EasyDict
import numpy as np


_C = EasyDict()
cfg = _C
# data augument config
_C.expand_prob = 0.5
_C.expand_max_ratio = 2
_C.hue_prob = 0.5
_C.hue_delta = 18
_C.contrast_prob = 0.5
_C.contrast_delta = 0.5
_C.saturation_prob = 0.5
_C.saturation_delta = 0.5
_C.brightness_prob = 0.5
_C.brightness_delta = 0.125
_C.data_anchor_sampling_prob = 0.5
_C.min_face_size = 1.0
_C.apply_distort = True
_C.apply_expand = False
_C.img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype(
    'float32')
_C.resize_width = 320
_C.resize_height = 320
_C.scale = 1 / 127.0
_C.anchor_sampling = True
_C.filter_min_face = True

# train config
#_C.LR_STEPS = (120, 198, 250)
_C.MAX_STEPS = 200000
_C.LR_STEPS = (80000,100000,120000)
_C.EPOCHES = 900
_C.IS_MONOCHROME = True

# anchor config
_C.FEATURE_MAPS = [80, 80, 40, 40]
_C.INPUT_SIZE = 320
_C.STEPS = [4, 4, 8, 8]
_C.ANCHOR_SIZES = [8, 16, 32, 48]
_C.CLIP = False
_C.VARIANCE = [0.1, 0.2]

# detection config
_C.NMS_THRESH = 0.3
_C.NMS_TOP_K = 5000
_C.TOP_K = 750
_C.CONF_THRESH = 0.05

# loss config
_C.NEG_POS_RATIOS = 3
_C.NUM_CLASSES = 2
_C.USE_NMS = True

# dataset config
_C.HOME = '/mnt/'

# hand config
_C.HAND = EasyDict()
_C.HAND.TRAIN_FILE = './data/hand_train.txt'
_C.HAND.VAL_FILE = './data/hand_val.txt'
_C.HAND.DIR = '/home/data/lj/egohands/'
_C.HAND.OVERLAP_THRESH = 0.35

# face config
_C.FACE = EasyDict()
_C.FACE.TRAIN_FILE = './data/face_train.txt'
_C.FACE.VAL_FILE = './data/face_val.txt'
_C.FACE.FDDB_DIR = '/home/data/lj/FDDB'
_C.FACE.WIDER_DIR = '/mnt/WIDER_FACE'
_C.FACE.AFW_DIR = '/home/data/lj/AFW'
_C.FACE.PASCAL_DIR = '/home/data/lj/PASCAL_FACE'
_C.FACE.OVERLAP_THRESH = [0.1, 0.35, 0.5]

# head config
_C.HEAD = EasyDict()
_C.HEAD.DIR = '/home/data/lj/VOCHead/'
_C.HEAD.OVERLAP_THRESH = [0.1, 0.35, 0.5]