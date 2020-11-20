# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


from layers import *
from data.config_qvga import cfg
import numpy as np

from edgeml_pytorch.graph.rnnpool import *

class S3FD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, head, num_classes):
        super(S3FD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        '''
        self.priorbox = PriorBox(size,cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        '''
        # SSD network
        self.conv = ConvBNReLU(1, 4, stride=2)

        self.unfold = nn.Unfold(kernel_size=(8,8),stride=(4,4))

        self.rnn_model = RNNPool(8, 8, 16, 16, 4, 
                                w1Sparsity=0.5, u1Sparsity=0.3, w2Sparsity=0.3, u2Sparsity=0.3)#num_init_features)
        
        self.mob = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm3_3 = L2Norm(32, 10)
        self.L2Norm4_3 = L2Norm(32, 8)
        self.L2Norm5_3 = L2Norm(64, 5)


        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

                
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1) 


    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        size = x.size()[2:]
        batch_size = x.shape[0]
        sources = list()
        loc = list()
        conf = list()

        x = self.conv(x)

        patches = self.unfold(x)
        patches = torch.cat(torch.unbind(patches,dim=2),dim=0)
        patches = torch.reshape(patches,(-1,4,8,8))

        output_x = int((x.shape[2]-8)/4 + 1)
        output_y = int((x.shape[3]-8)/4 + 1)

        rnnX = self.rnn_model(patches, int(batch_size)*output_x*output_y)

        x = torch.stack(torch.split(rnnX, split_size_or_sections=int(batch_size), dim=0),dim=2)

        x = F.fold(x, kernel_size=(1,1), output_size=(output_x,output_y))

        x = F.pad(x, (0,1,0,1), mode='replicate')

        for k in range(1):
            x = self.mob[k](x)

        s = self.L2Norm3_3(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(1, 2):
            x = self.mob[k](x)

        s = self.L2Norm4_3(x)
        sources.append(s)

        for k in range(2, 3):
            x = self.mob[k](x)

        s = self.L2Norm5_3(x)
        sources.append(s)

        for k in range(3, 4):
            x = self.mob[k](x)
        sources.append(x)

      
        # apply multibox head to source layers

        loc_x = self.loc[0](sources[0])
        conf_x = self.conf[0](sources[0])

        max_conf, _ = torch.max(conf_x[:, 0:3, :, :], dim=1, keepdim=True)
        conf_x = torch.cat((max_conf, conf_x[:, 3:, :, :]), dim=1)

        loc.append(loc_x.permute(0, 2, 3, 1).contiguous())
        conf.append(conf_x.permute(0, 2, 3, 1).contiguous())

        for i in range(1, len(sources)):
            x = sources[i]
            conf.append(self.conf[i](x).permute(0, 2, 3, 1).contiguous())
            loc.append(self.loc[i](x).permute(0, 2, 3, 1).contiguous())


        features_maps = []
        for i in range(len(loc)):
            feat = []
            feat += [loc[i].size(1), loc[i].size(2)]
            features_maps += [feat]

        self.priorbox = PriorBox(size, features_maps, cfg)
        
        self.priors = self.priorbox.forward()

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

       
        if self.phase == 'test':
            output = detect_function(cfg,
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),      # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )

        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            mdata = torch.load(base_file,
                               map_location=lambda storage, loc: storage)
            weights = mdata['weight']
            epoch = mdata['epoch']
            self.load_state_dict(weights)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
        return epoch

    def xavier(self, param):
        init.xavier_uniform(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            m.bias.data.zero_()




def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 64

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s               
                [2, 32, 1, 1],
                [2, 32, 1, 1],
                [2, 64, 1, 2],
                [2, 64, 1, 1],              
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.layers = []
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                self.layers.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)




def multibox(mobilenet, num_classes):
    loc_layers = []
    conf_layers = []

    loc_layers += [nn.Conv2d(32, 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(32, 3 + (num_classes-1), kernel_size=3, padding=1)]

    loc_layers += [nn.Conv2d(32, 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(32, num_classes, kernel_size=3, padding=1)]

    loc_layers += [nn.Conv2d(64, 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(64, num_classes, kernel_size=3, padding=1)]

    loc_layers += [nn.Conv2d(64, 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(64, num_classes, kernel_size=3, padding=1)]

      
    return mobilenet, (loc_layers, conf_layers)


def build_s3fd(phase, num_classes=2):
    base_, head_ = multibox(
        MobileNetV2().layers, num_classes)
    
    return S3FD(phase, base_, head_, num_classes)


if __name__ == '__main__':
    net = build_s3fd('train', num_classes=2)
    inputs = Variable(torch.randn(4, 1, 320, 320))
    output = net(inputs)
