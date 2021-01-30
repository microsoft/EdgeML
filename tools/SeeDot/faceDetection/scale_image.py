# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import argparse
import cv2
import numpy as np
from PIL import Image
import os 

os.environ['IS_QVGA_MONO'] = '1'
from data.choose_config import cfg


cfg = cfg.cfg

parser = argparse.ArgumentParser(description='Generating input to quantized face detection code')
parser.add_argument('--image_dir', default="images", type=str, help='Folder containing image(s)')
parser.add_argument('--out_dir', default="input", type=str, help='Folder containing the CSV files')
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)


img_list = [os.path.join(args.image_dir, x)
                for x in os.listdir(args.image_dir)]

xoutfile = open(os.path.join(args.out_dir, "X.csv"), "w")

for image_path in sorted(img_list): 
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = np.array(img)
    scale = 1

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

    for i in range(240):
        for j in range(320):
            if i == 239 and j == 319:
                xoutfile.write(str(x[i, j]) + "\n")
            else:
                xoutfile.write(str(x[i, j]) + ', ')

youtfile = open(os.path.join(args.out_dir, "Y.csv"), "w")
for _ in range(len(img_list)):
    for i in range(18000):
        if i == 17999:
            youtfile.write("0\n")
        else:
            youtfile.write("0, ")
