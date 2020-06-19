#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import logging
import random

# import torch
# import torch.nn as nn
# import torch.utils.data as data
# import torch.backends.cudnn as cudnn
# import torchvision.transforms as transforms

import tensorflow as tf

import cv2
import time
import numpy as np
from PIL import Image, ImageFilter

os.environ['IS_QVGA_MONO'] = '1'
from data.choose_config import cfg
cfg = cfg.cfg

# from torch.autograd import Variable
from utils.augmentations import to_chw_bgr

from importlib import import_module

# import tf_keras_RPool_Face_QVGA_monochrome as models


# python eval.py --model_arch RPool_Face_Quant  --model ./weights/rpool_face_best_state.pth --image_folder <your_image_folder> --save_dir <your_save_folder>

parser = argparse.ArgumentParser(description='face detection demo')
#parser.add_argument('--save_dir', type=str, default='results/',
# parser.add_argument('--save_dir', type=str, default='results-eta/',
parser.add_argument('--save_dir', type=str, default='results-debug-tf/',
                    help='Directory for detect result')
parser.add_argument('--model', type=str,
                    default='../RNNPool-FaceDetection/zFromMSR/rpool_face_qvgamono_withscut_trainaugfacele48.pth', help='trained model')
                    #default='weights/rpool_face_c.pth', help='trained model')
                    #small_fgrnn_smallram_sd.pth', help='trained model')
parser.add_argument('--thresh', default=0.5, type=float,
                    help='Final confidence threshold')
parser.add_argument('--model_arch',
                    default='RPool_Face_QVGA_monochrome', type=str,
                    choices=['RPool_Face_C', 'RPool_Face_Quant', 'RPool_Face_QVGA_monochrome'],
                    help='choose architecture among rpool variants')
#parser.add_argument('--image_folder', default=None, type=str, help='folder containing images')
# parser.add_argument('--image_folder', default='data-eta', type=str, help='folder containing images')
parser.add_argument('--image_folder', default='data-debug', type=str, help='folder containing images')

args = parser.parse_args()
print( args )

def detect(sess, detectOp, img_path, thresh, debugPrint):
    #img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = Image.open(img_path)

    #if img.mode == 'L':
    img = img.convert('RGB')

    # img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    img = np.array(img)
    print( img_path, ', original: ', img.shape )    # DEBUG
    height, width, _ = img.shape
    if os.environ['IS_QVGA_MONO'] == '1':
        # max_im_shrink = np.sqrt(320 * 240 / (img.shape[0] * img.shape[1]))
        image = cv2.resize(img, (320, 240))     # (width, height)
    else:
        max_im_shrink = np.sqrt(640 * 480 / (img.shape[0] * img.shape[1]))
        image = cv2.resize(img, None, None, fx=max_im_shrink, fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
    # img = cv2.resize(img, (640, 640))
    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]      # CHW, BGR=>RGB

    if cfg.IS_MONOCHROME == True:
        x = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
        x = np.expand_dims( np.expand_dims( x, axis=0 ), axis=0 )
    else:
        x = np.expand_dims( x, axis=0 )

    print( "\tbefore transpose: ", x.shape )
    x = np.transpose( x, (0, 2, 3, 1) )     # NCHW => NHWC
    print( "\t after transpose: ", x.shape )

    # if use_cuda:
    #     x = x.cuda()
    t1 = time.time()
    y, xxx = sess.run( [detectOp, debugPrint], feed_dict={inputs_placeholder: x} )
    
    # print( np.shape(xxx) )  # DEBUG
    # print( xxx.flatten() )  # DEBUG

    print( type(y), y.shape )
    detections = y
    # scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
    scale = [width, height, width, height]

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    print( "Detection: ", np.shape( detections ) )

    for i in range(1, detections.shape[1]):
        j = 0
        while j < detections.shape[2]  and  (detections[0, i, j, 0] >= thresh ):# or  j < 10):
            # print( detections[0, i, j, :] )
            score = detections[0, i, j, 0]
            pt = ( np.multiply( detections[0, i, j, 1:], scale ) ).astype(int)
            # print( "\tscore =", score, "  pts[", pt[0], pt[1], pt[2], pt[3], "]")
            left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
            j += 1
            cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
            conf = "{:.3f}".format(score)
            point = (int(left_up[0]), int(left_up[1] - 5))
            cv2.putText(img, conf, point, cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)

    # t2 = time.time()
    # print('detect:{} timer:{}'.format(img_path, t2 - t1))

    # print( np.shape( img ) )
    # cv2.imshow('Result', img)
    cv2.imwrite(os.path.join(args.save_dir, os.path.basename(img_path)), img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.compat.v1.set_random_seed(230)

    args = parser.parse_args()
    print( '\nParams: ', args )


    #device='cpu'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    module = import_module('tf_keras_' + args.model_arch)
    net = module.build_s3fd( 'test', cfg )
    # import pdb;pdb.set_trace()

    #net = torch.nn.DataParallel(net)

    batch_size = 1
    if os.environ['IS_QVGA_MONO'] == '1':
        inputs_placeholder = tf.compat.v1.placeholder( tf.float32, shape=(batch_size, 240, 320, 1) )
    else:
        inputs_placeholder = tf.compat.v1.placeholder( tf.float32, shape=(batch_size, 640, 480, 3) )
    detectOp, debugPrint = net( inputs_placeholder )

    img_path = args.image_folder
    img_list = [os.path.join(img_path, f) for f in os.listdir(img_path)]

    checkpoint_name = 'weights/pt_tf_weights_smaller'
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        # net.load_state_dict(torch.load(args.model))
        sess.run( tf.compat.v1.global_variables_initializer() )

        if os.path.isfile( checkpoint_name + '.meta' ):
            print( '\nLode weights: ', checkpoint_name )
            saver.restore(sess, checkpoint_name)
            # net.load_weights(checkpoint_name)
        else:
            import dictParamName_Map_tf_pt as tf2ptMap
            import dictParamNameVal_pt as pt_dict

            for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES): #GLOBAL_VARIABLES, TRAINABLE_VARIABLES
                print( v.name, '\t', v.shape.as_list() )
                if v.name not in tf2ptMap.dictParamName_Map_tf_pt:
                    continue

                pt_name  = tf2ptMap.dictParamName_Map_tf_pt[v.name]
                pt_shape = pt_dict.dictParamNameVal_pt[pt_name][0]
                pt_value = pt_dict.dictParamNameVal_pt[pt_name][1]
                if 'depthwise_kernel:' in v.name:
                    tf_val = np.transpose( pt_value, (2, 3, 0, 1) )
                    _ = sess.run( tf.compat.v1.assign(v, tf_val) )
                elif 'kernel:' in v.name:
                    tf_val = np.transpose( pt_value, (2, 3, 1, 0) )
                    _ = sess.run( tf.compat.v1.assign(v, tf_val) )
                else:
                    _ = sess.run( tf.compat.v1.assign(v, pt_value) )
            print( '\nSave weights: ', checkpoint_name )
            saver.save(sess, checkpoint_name)
            # net.save_weights(checkpoint_name)

        for img_filename in img_list:
            detect( sess, detectOp, img_filename, args.thresh, debugPrint )

    # net.save( checkpoint_name + '_pb', save_format="tf" )
    # tf.saved_model.save( net, checkpoint_name + '_pb' )
