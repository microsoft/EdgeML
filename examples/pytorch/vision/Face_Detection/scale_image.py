# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import argparse
import cv2
import numpy as np
from PIL import Image
from data.choose_config import cfg
cfg = cfg.cfg

parser = argparse.ArgumentParser(description='Image Scaling Script')
parser.add_argument('--scale', type=str,
                    default='1', help='Scale for the input image (use scaleForX variable value from the generated m3dump/scales.h file here)')
parser.add_argument('--image_path', default=None, type=str, help='Path to the image to be scaled')

args = parser.parse_args()

img = Image.open(args.image_path)
img = img.convert('RGB')
img = np.array(img)
scale = 2 ** int(args.scale)

max_im_shrink_x = 320 / (img.shape[1])
max_im_shrink_y = 240 / (img.shape[0])

image = cv2.resize(img, None, None, fx=max_im_shrink_x,
                  fy=max_im_shrink_y, interpolation=cv2.INTER_LINEAR)

if len(image.shape) == 3:
    image = np.swapaxes(image, 1, 2)
    image = np.swapaxes(image, 1, 0)
# RBG to BGR
x = image[[2, 1, 0], :, :]

x = x.astype('float32')
x -= cfg.img_mean
x = x[[2, 1, 0], :, :]
x = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
x /= scale
x = np.rint(x).astype(int)

print('static const Q7_T input_image[76800] = {', end='')
for i in range(240):
    for j in range(320):
        print(str(x[i, j]), end=', ')
print('};')
