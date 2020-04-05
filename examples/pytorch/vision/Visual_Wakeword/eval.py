import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import random
from PIL import Image
import numpy as np
from importlib import import_module
import skimage
from skimage import filters





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


#Arg parser
parser = argparse.ArgumentParser(description='PyTorch VisualWakeWords evaluation')
parser.add_argument('--weights', default=None, type=str, help='load from checkpoint')
parser.add_argument('--model_arch',
                    default='model_mobilenet_rnnpool', type=str,
                    choices=['model_mobilenet_rnnpool', 'model_mobilenet_2rnnpool'],
                    help='choose architecture among rpool variants')
parser.add_argument('--image_folder', default=None, type=str, help='folder containing images')

args = parser.parse_args()


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
]) 
 

    

if __name__ == '__main__':

    module = import_module(args.model_arch)
    model = module.mobilenetv2_rnnpool(num_classes=2, width_mult=0.35, last_channel=320)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    

    checkpoint = torch.load(args.weights)
    checkpoint_dict = checkpoint['model']
    model_dict = model.state_dict()
    model_dict.update(checkpoint_dict) 
    model.load_state_dict(model_dict)

    # import pdb;pdb.set_trace()

    # model.module.rnn_model.cell_rnn.unrollRNN.RNNCell.W = torch.nn.Parameter(torch.transpose(model.module.rnn_model.cell_rnn.unrollRNN.RNNCell.W, 0, 1))
    # model.module.rnn_model.cell_rnn.unrollRNN.RNNCell.U = torch.nn.Parameter(torch.transpose(model.module.rnn_model.cell_rnn.unrollRNN.RNNCell.U, 0, 1))


    # model.module.rnn_model.cell_bidirrnn.unrollRNN.RNNCell_reverse.W = torch.nn.Parameter(torch.transpose(model.module.rnn_model.cell_bidirrnn.unrollRNN.RNNCell_reverse.W, 0, 1))
    # model.module.rnn_model.cell_bidirrnn.unrollRNN.RNNCell_reverse.U = torch.nn.Parameter(torch.transpose(model.module.rnn_model.cell_bidirrnn.unrollRNN.RNNCell_reverse.U, 0, 1))
    # model.module.rnn_model.cell_bidirrnn.unrollRNN.RNNCell.W = torch.nn.Parameter(torch.transpose(model.module.rnn_model.cell_bidirrnn.unrollRNN.RNNCell.W, 0, 1))
    # model.module.rnn_model.cell_bidirrnn.unrollRNN.RNNCell.U = torch.nn.Parameter(torch.transpose(model.module.rnn_model.cell_bidirrnn.unrollRNN.RNNCell.U, 0, 1))



    

    model.eval()
    img_path = args.image_folder
    img_list = [os.path.join(img_path, x)
                for x in os.listdir(img_path) if x.endswith('bmp')]
    
    for path in sorted(img_list):
        # img = skimage.io.imread(path)
        # img = skimage.transform.rescale(img, scale=0.5)
        # img = skimage.transform.rescale(img, scale=2.0)
        # img = filters.unsharp_mask(img,amount=5.0, radius=2.0, multichannel=True)
        # img  = Image.fromarray(img.astype('uint8'), mode='RGB')
        img = Image.open(path).convert('RGB')
        img = transform_test(img)
        img = (img.cuda())
        img = img.unsqueeze(0)
       
        out = model(img)
    
        print(path)
        print(out)
        if out[0][0]>0.15:
            print('No person present')
        else:
            print('Person present')



