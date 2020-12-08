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

from data.choose_config import cfg
cfg = cfg.cfg

from utils.augmentations import to_chw_bgr

from importlib import import_module

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='face detection demo')
parser.add_argument('--save_dir', type=str, default='results/',
                    help='Directory for detect result')
parser.add_argument('--model', type=str,
                    default='weights/rpool_face_c.pth', help='trained model')
parser.add_argument('--thresh', default=0.17, type=float,
                    help='Final confidence threshold')
parser.add_argument('--multigpu',
                    default=False, type=str2bool,
                    help='Specify whether model was trained with multigpu')
parser.add_argument('--model_arch',
                    default='RPool_Face_C', type=str,
                    choices=['RPool_Face_C', 'RPool_Face_Quant', 'RPool_Face_QVGA_monochrome', 'RPool_Face_M4'],
                    help='choose architecture among rpool variants')
parser.add_argument('--image_folder', default=None, type=str, help='folder containing images')


args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def detect(net, img_path, thresh):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = np.array(img)
    height, width, _ = img.shape

    if os.environ['IS_QVGA_MONO'] == '1':
        max_im_shrink = np.sqrt(
            320 * 240 / (img.shape[0] * img.shape[1]))
    else:
        max_im_shrink = np.sqrt(
            640 * 480 / (img.shape[0] * img.shape[1]))

    image = cv2.resize(img, None, None, fx=max_im_shrink,
                      fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
    # img = cv2.resize(img, (640, 640))
    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]


    if cfg.IS_MONOCHROME == True:
        x = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    else:
        x = torch.from_numpy(x).unsqueeze(0)
    if use_cuda:
        x = x.cuda()
    t1 = time.time()
    y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
            j += 1
            cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
            conf = "{:.3f}".format(score)
            point = (int(left_up[0]), int(left_up[1] - 5))
            cv2.putText(img, conf, point, cv2.FONT_HERSHEY_COMPLEX,
                       0.6, (0, 255, 0), 1)

    t2 = time.time()
    print('detect:{} timer:{}'.format(img_path, t2 - t1))

    cv2.imwrite(os.path.join(args.save_dir, os.path.basename(img_path)), img)


if __name__ == '__main__':

    module = import_module('models.' + args.model_arch)
    net = module.build_s3fd('test', cfg.NUM_CLASSES)

    if args.multigpu == True:
        net = torch.nn.DataParallel(net)

    checkpoint_dict = torch.load(args.model)

    model_dict = net.state_dict()


    model_dict.update(checkpoint_dict) 
    net.load_state_dict(model_dict)

    net.eval()

    if use_cuda:
        net.cuda()
        cudnn.benckmark = True

    img_path = args.image_folder
    img_list = [os.path.join(img_path, x)
                for x in os.listdir(img_path)]
    for path in img_list:
        detect(net, path, args.thresh)
        