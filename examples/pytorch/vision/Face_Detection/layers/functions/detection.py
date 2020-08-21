# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch

from ..bbox_utils import decode, nms



def detect_function(cfg, loc_data, conf_data, prior_data):
    """
    Args:
        loc_data: (tensor) Loc preds from loc layers
            Shape: [batch,num_priors*4]
        conf_data: (tensor) Shape: Conf preds from conf layers
            Shape: [batch*num_priors,num_classes]
        prior_data: (tensor) Prior boxes and variances from priorbox layers
            Shape: [1,num_priors,4] 
    """
    with torch.no_grad():
        num = loc_data.size(0)
        num_priors = prior_data.size(0)

        conf_preds = conf_data.view(
            num, num_priors, cfg.NUM_CLASSES).transpose(2, 1)
        batch_priors = prior_data.view(-1, num_priors,
                                       4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)

        decoded_boxes = decode(loc_data.view(-1, 4),
                               batch_priors, cfg.VARIANCE)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)

        output = torch.zeros(num, cfg.NUM_CLASSES, cfg.TOP_K, 5)

        for i in range(num):
            boxes = decoded_boxes[i].clone()
            conf_scores = conf_preds[i].clone()

            for cl in range(1, cfg.NUM_CLASSES):
                c_mask = conf_scores[cl].gt(cfg.CONF_THRESH)
                scores = conf_scores[cl][c_mask]
                
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes_ = boxes[l_mask].view(-1, 4)
                ids, count = nms(
                    boxes_, scores, cfg.NMS_THRESH, cfg.NMS_TOP_K)
                count = count if count < cfg.TOP_K else cfg.TOP_K

                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                   boxes_[ids[:count]]), 1)

    return output
