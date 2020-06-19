## This code is built on https://github.com/yxlijun/S3FD.pytorch

#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# import torch
import tensorflow as tf
from itertools import product as product
import math


def priorBox( imgSize, feature_maps, cfg ):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    imh = imgSize[0]
    imw = imgSize[1]

    # number of priors for feature map location (either 4 or 6)
    # variance = cfg.VARIANCE or [0.1]
    # for v in variance:
    #     if v <= 0:
    #         raise ValueError('Variances must be greater than 0')
    mean = []
    for k in range( len(feature_maps) ):
        feath = feature_maps[k][0]
        featw = feature_maps[k][1]
        f_kw = cfg.STEPS[k] / imw
        f_kh = cfg.STEPS[k] / imh
        s_kw = cfg.ANCHOR_SIZES[k] / imw
        s_kh = cfg.ANCHOR_SIZES[k] / imh
        for i, j in product(range(feath), range(featw)):
            cx = (j + 0.5) * f_kw
            cy = (i + 0.5) * f_kh
            mean += [[cx, cy, s_kw, s_kh]]

    mean = tf.compat.v1.convert_to_tensor(mean, dtype=tf.float32)
    if cfg.CLIP:
        mean = tf.compat.v1.clip_by_value( mean, clip_value_min=0, clip_value_max=1 )

    return mean


class PriorBoxReal(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, input_size, feature_maps, cfg):
        super(PriorBoxReal, self).__init__()
        self.imh = input_size[0]
        self.imw = input_size[1]

        # number of priors for feature map location (either 4 or 6)
        self.variance = cfg.VARIANCE or [0.1]
        #self.feature_maps = cfg.FEATURE_MAPS
        self.min_sizes = cfg.ANCHOR_SIZES
        self.steps = cfg.STEPS
        self.clip = cfg.CLIP
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')
        self.feature_maps = feature_maps


    def forward(self):
        mean = []
        # import pdb; pdb.set_trace()
        for k in range(len(self.feature_maps)):
            feath = self.feature_maps[k][0]
            featw = self.feature_maps[k][1]
            for i, j in product(range(feath), range(featw)):
                f_kw = self.imw / self.steps[k]
                f_kh = self.imh / self.steps[k]

                cx = (j + 0.5) / f_kw
                cy = (i + 0.5) / f_kh

                s_kw_1 = self.min_sizes[k] / self.imw
                s_kh_1 = self.min_sizes[k] / self.imh

                # s_kw_2 = self.min_sizes[2*k+1] / self.imw
                # s_kh_2 = self.min_sizes[2*k+1] / self.imh

                mean += [cx, cy, s_kw_1, s_kh_1]
                # mean += [cx, cy, s_kw_2, s_kh_2]

        output = tf.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


if __name__ == '__main__':
    from data.config import cfg
    p = priorBox([640, 640], cfg)
    out = p.forward()
    print(out.size())
