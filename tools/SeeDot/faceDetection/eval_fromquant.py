# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import os
import time
import argparse
import numpy as np
from PIL import Image
import cv2

os.environ['IS_QVGA_MONO'] = '1'
from data.choose_config import cfg
cfg = cfg.cfg

from utils.augmentations import to_chw_bgr
from layers import *


parser = argparse.ArgumentParser(description='Face Detection from quantized model\'s output.')
parser.add_argument('--save_dir', type=str, default='results/',
                    help='Directory for detect result')

parser.add_argument('--thresh', default=0.45, type=float,
                    help='Final confidence threshold')

parser.add_argument('--image_dir', default="images/", type=str, help='Folder containing image(s)')
parser.add_argument('--trace_file', default="trace.txt", type=str, help='File containing output traces')
parser.add_argument('--scale', type=str, default='0', help='Scale for the output image (use scaleForY variable value from the generated m3dump/scales.h file here)')


args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


def detect(loc, conf, img_path, thresh):
    features_maps = []
    for i in range(len(loc)):
        feat = []
        feat += [loc[i].size(1), loc[i].size(2)]
        features_maps += [feat]

    priorbox = PriorBox(torch.Size([240, 320]), features_maps, cfg)
    
    priors = priorbox.forward()
    softmax = nn.Softmax(dim=-1)

    loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
    conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
   
    output = detect_function(cfg,
            loc.view(loc.size(0), -1, 4),                   
            softmax(conf.view(conf.size(0), -1,
                                   2)),     
            priors               
        )
    detections = output.data
    # import pdb;pdb.set_trace()
    

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (1280, 960))

    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
            j += 1
            cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 5)
            conf = "{:.3f}".format(score)
            point = (int(left_up[0]), int(left_up[1] - 5))
            cv2.putText(img, conf, point, cv2.FONT_HERSHEY_COMPLEX,
                       1, (0, 255, 0), 1)

    # t2 = time.time()
    # print('detect:{} timer:{}'.format(img_path, t2 - t1))
    cv2.imwrite(os.path.join(args.save_dir, os.path.basename(img_path)), img)
    

if __name__ == '__main__':
    img_path = args.image_dir
    img_list = [os.path.join(img_path, x)
                for x in os.listdir(img_path)]
    a = open(args.trace_file)
    l = a.readlines()
    preds = np.zeros((len(img_list), 18000))

    j = -1
    for i in l:
        j += 1
        nums = i.strip().split(' ')
        c = []
        for n in nums:
            c.append(float(n) * 2 ** (int(args.scale)))
        preds[j,:] = np.array(c)

    countr=0
    for path in sorted(img_list):
        temp_preds = preds[countr]
        conf = [torch.Tensor(temp_preds[0:2400].reshape(1,30,40,2)),
        torch.Tensor(temp_preds[2400:4800].reshape(1,30,40,2)),
        torch.Tensor(temp_preds[4800:5400].reshape(1,15,20,2)),
        torch.Tensor(temp_preds[5400:6000].reshape(1,15,20,2))]

        loc = [torch.Tensor(temp_preds[6000:10800].reshape(1,30,40,4)),
        torch.Tensor(temp_preds[10800:15600].reshape(1,30,40,4)),
        torch.Tensor(temp_preds[15600:16800].reshape(1,15,20,4)),
        torch.Tensor(temp_preds[16800:18000].reshape(1,15,20,4))]
    # print(sorted(img_list))
        detect(loc, conf, path, args.thresh)
        countr+=1

