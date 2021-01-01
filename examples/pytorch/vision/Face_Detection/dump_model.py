# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
import numpy as np
from PIL import Image, ImageFilter

from data.config import cfg
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr

from importlib import import_module

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='face detection dump')

parser.add_argument('--model', type=str,
                    default='weights/rpool_face_c.pth', help='trained model')
                    #small_fgrnn_smallram_sd.pth', help='trained model')
parser.add_argument('--model_arch',
                    default='RPool_Face_C', type=str,
                    choices=['RPool_Face_C', 'RPool_Face_B', 'RPool_Face_A', 'RPool_Face_Quant'],
                    help='choose architecture among rpool variants')
parser.add_argument('--image_folder', default=None, type=str, help='folder containing images')
parser.add_argument('--save_model_npy_dir', default=None, type=str, help='Directory for saving model in numpy array format')
parser.add_argument('--save_traces_npy_dir', default=None, type=str, help='Directory for saving RNNPool input and output traces in numpy array format')


args = parser.parse_args()


use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')



def saveModelNpy(net):
    if os.path.isdir(args.save_model_npy_dir) is False:
        try:
            os.mkdir(args.save_model_npy_dir)
        except OSError:
            print("Creation of the directory %s failed" % args.save_model_npy_dir)
            return

    np.save(args.save_model_npy_dir+'/W1.npy', net.rnn_model.cell_rnn.cell.W.cpu().detach().numpy())
    np.save(args.save_model_npy_dir+'/W2.npy', net.rnn_model.cell_bidirrnn.cell.W.cpu().detach().numpy())
    np.save(args.save_model_npy_dir+'/U1.npy', net.rnn_model.cell_rnn.cell.U.cpu().detach().numpy())
    np.save(args.save_model_npy_dir+'/U2.npy', net.rnn_model.cell_bidirrnn.cell.U.cpu().detach().numpy())
    np.save(args.save_model_npy_dir+'/Bg1.npy', net.rnn_model.cell_rnn.cell.bias_gate.cpu().detach().numpy())
    np.save(args.save_model_npy_dir+'/Bg2.npy', net.rnn_model.cell_bidirrnn.cell.bias_gate.cpu().detach().numpy())
    np.save(args.save_model_npy_dir+'/Bh1.npy', net.rnn_model.cell_rnn.cell.bias_update.cpu().detach().numpy())
    np.save(args.save_model_npy_dir+'/Bh2.npy', net.rnn_model.cell_bidirrnn.cell.bias_update.cpu().detach().numpy())
    np.save(args.save_model_npy_dir+'/nu1.npy', net.rnn_model.cell_rnn.cell.nu.cpu().detach().numpy())
    np.save(args.save_model_npy_dir+'/nu2.npy', net.rnn_model.cell_bidirrnn.cell.nu.cpu().detach().numpy())
    np.save(args.save_model_npy_dir+'/zeta1.npy', net.rnn_model.cell_rnn.cell.zeta.cpu().detach().numpy())
    np.save(args.save_model_npy_dir+'/zeta2.npy', net.rnn_model.cell_bidirrnn.cell.zeta.cpu().detach().numpy())



activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def saveTracesNpy(net, img_list):
    if os.path.isdir(args.save_traces_npy_dir) is False:
        try:
            os.mkdir(args.save_traces_npy_dir)
        except OSError:
            print("Creation of the directory %s failed" % args.save_traces_npy_dir)
            return

    if os.path.isdir(os.path.join(args.save_traces_npy_dir,'inputs')) is False:
        try:
            os.mkdir(os.path.join(args.save_traces_npy_dir,'inputs'))
        except OSError:
            print("Creation of the directory %s failed" % os.path.join(args.save_traces_npy_dir,'inputs'))
            return

    if os.path.isdir(os.path.join(args.save_traces_npy_dir,'outputs')) is False:
        try:
            os.mkdir(os.path.join(args.save_traces_npy_dir,'outputs'))
        except OSError:
            print("Creation of the directory %s failed" % os.path.join(args.save_traces_npy_dir,'outputs'))
            return

    inputDims = net.rnn_model.inputDims
    nRows = net.rnn_model.nRows
    nCols = net.rnn_model.nCols
    count=0
    for img_path in img_list:
        img = Image.open(os.path.join(args.image_folder, img_path))
        
        img = img.convert('RGB')

        img = np.array(img)
        max_im_shrink = np.sqrt(
            640 * 480 / (img.shape[0] * img.shape[1]))
        image = cv2.resize(img, None, None, fx=max_im_shrink,
                          fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)

        x = to_chw_bgr(image)
        x = x.astype('float32')
        x -= cfg.img_mean
        x = x[[2, 1, 0], :, :]

        x = Variable(torch.from_numpy(x).unsqueeze(0))
        if use_cuda:
            x = x.cuda()
        t1 = time.time()
        y = net(x)


        patches = activation['prepatch']
        patches = torch.cat(torch.unbind(patches,dim=2),dim=0)
        patches = torch.reshape(patches,(-1,inputDims,nRows,nCols))

        rnnX = activation['rnn_model']

        patches_all = torch.stack(torch.split(patches, split_size_or_sections=1, dim=0),dim=-1)
        rnnX_all = torch.stack(torch.split(rnnX, split_size_or_sections=1, dim=0),dim=-1)

        for k in range(patches_all.shape[-1]):
            patches_tosave = patches_all[0,:,:,:,k].cpu().numpy().transpose(1,2,0)
            rnnX_tosave = rnnX_all[0,:,k].cpu().numpy()
            np.save(args.save_traces_npy_dir+'/inputs/trace_'+str(count)+'_'+str(k)+'.npy', patches_tosave)
            np.save(args.save_traces_npy_dir+'/outputs/trace_'+str(count)+'_'+str(k)+'.npy', rnnX_tosave)

        count+=1



    


if __name__ == '__main__':

    module = import_module('models.' + args.model_arch)
    net = module.build_s3fd('test', cfg.NUM_CLASSES)

    # net = torch.nn.DataParallel(net)

    checkpoint_dict = torch.load(args.model)

    model_dict = net.state_dict()


    model_dict.update(checkpoint_dict) 
    net.load_state_dict(model_dict)



    net.eval()

    if use_cuda:
        net.cuda()
        cudnn.benckmark = True



    if args.save_model_npy_dir is not None:
        saveModelNpy(net)

    if args.save_traces_npy_dir is not None:
        net.unfold.register_forward_hook(get_activation('prepatch'))     
        net.rnn_model.register_forward_hook(get_activation('rnn_model'))  
        img_path = args.image_folder
        img_list = [os.path.join(img_path, x)
                for x in os.listdir(img_path)]
        saveTracesNpy(net, img_list)
