# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

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


_C.IS_MONOCHROME = True

# anchor config
_C.FEATURE_MAPS = [40, 40, 20, 20]
_C.INPUT_SIZE = 320
_C.STEPS = [8, 8, 16, 16]
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
_C.HOME = '/mnt'

# face config
_C.FACE = EasyDict()
_C.FACE.TRAIN_FILE = './data/face_train.txt'
_C.FACE.VAL_FILE = './data/face_val.txt'
_C.FACE.WIDER_DIR = '/mnt/WIDER_FACE'
_C.FACE.SCUT_DIR = '/mnt/SCUT_HEAD_Part_B'
_C.FACE.OVERLAP_THRESH = [0.1, 0.35, 0.5]
