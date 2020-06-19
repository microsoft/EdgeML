## This code is built on https://github.com/yxlijun/S3FD.pytorch
#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import time
# import torch
import argparse
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
# import torchvision
# import torchvision.models as models
# import torchvision.transforms as transforms

# import random
from PIL import Image

#from torchvision.datasets.vision import VisionDataset

from data.config import cfg

from importlib import import_module

import tf_keras_multibox_loss as MultiBoxLoss
from tf_keras_factory import dataset_factory, dataLoader

#from pyvww.utils import VisualWakeWords

import numpy as np
import tensorflow as tf
# import tf_keras_model_mobilenet_rnnpool as mobilenet
import tf_keras_rnnpool as rnnpool

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

#Arg parser
parser = argparse.ArgumentParser(description='S3FD face Detector Training')
parser.add_argument('--dataset',
                    default='face',
                    choices=['hand', 'face', 'head'],
                    help='Train target')
parser.add_argument('--batch_size',
                    default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--model_arch',
                    default='RPool_Face_Quant', type=str,
                    choices=['RPool_Face_C', 'RPool_Face_B', 'RPool_Face_A', 'RPool_Face_Quant'],
                    help='choose architecture among rpool variants')
parser.add_argument('--num_workers',
                    #default=128, type=int,
                    default=1, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda',
                    default=False, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multigpu',
                    default=False, type=str2bool,
                    help='Use mutil Gpu training')
parser.add_argument('--save_folder',
                    default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


#device = tf.device("cuda" if tf.test.is_gpu_available() else "cpu")
device = tf.device("cpu")

#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.enabled = True

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# Data

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

train_dataset, val_dataset = dataset_factory(args.dataset)

train_loader = dataLoader( train_dataset, args.batch_size )

val_batchsize = 1   #args.batch_size // 2
val_loader = dataLoader(val_dataset, val_batchsize )


# Model

module = import_module( 'tf_keras_RPool_Face_Quant' )
net = module.build_s3fd('train', cfg.NUM_CLASSES)

#tf.compat.v1.reset_default_graph()
#with tf.Graph().as_default():
#    with tf.Session() as sess:

"""
inputs = train_dataset.pull_item(20)
print( inputs[0].shape )
y = net( tf.expand_dims( tf.transpose( inputs[0], [1, 2, 0] ), 0 ) )
print( y[0].eval(session=tf.compat.v1.Session()) )

#net.summary()
#tf.keras.utils.plot_model(model.model, 'tf_keras_model.png')

print( "TF parameters" )
#for v in tf.compat.v1.trainable_variables():
totList = []
for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES): #GLOBAL_VARIABLES, TRAINABLE_VARIABLES
    #print( v.name, '\t', v.shape.as_list() )
    #if 'global_step' not in v.name:
    totList.append( [v.name, v.shape.as_list()] )

import pickle
with open('param_Name_Size_List.bin', 'wb') as f:
    pickle.dump(totList, f)
"""

# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# optimizer = tf.keras.optimizers.Adam()
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
# model.compile( optimizer=optimizer, loss=loss_fn, metrics=['accuracy'] )
# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test,  y_test, verbose=2)
# probability_model = tf.keras.Sequential([
#                                            model,
#                                           tf.keras.layers.Softmax()
#                                            ])

#model.build_train_op()
#model = model.to(device)
"""
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('./checkpoints/'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoints/' + args.resume)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = tf.nn.softmax_cross_entropy_with_logits().cuda()
#criterion = nn.BCEWithLogitsLoss()
#criterion = LabelSmoothingLoss(2,0.1)
#optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=4e-5)#, alpha=0.9)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9)

# Training
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
"""

# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9)
criterion = MultiBoxLoss(cfg, args.dataset, args.cuda)
print('Loading wider dataset...')
print('Using the specified args:')
print(args)


def time2str( t0, t1 ):
    dt = t1 - t0
    m = dt // 60
    s = dt - m * 60
    if m < 60:
        return '{:.0f}m {:.3f}s'.format(m, s)
    else:
        h = m // 60
        m -= h * 60
        return '{:.0f}h {:.0f}m {:.3f}s'.format(h, m, s)


def train():
    step_index = 0
    iteration = 0
    startTime = time.time()

    for epoch in range(start_epoch, cfg.EPOCHES):
        startTimeEpoch = time.time()
        net.train()
        losses = 0
        train_loader_len = len(train_loader)
        for batch_idx, (images, targets) in enumerate(train_loader):
            adjust_learning_rate(optimizer, epoch, batch_idx, train_loader_len)

            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda())
                           for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann) for ann in targets]

            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            losses += loss.item()

            if iteration % 10 == 0:
                tloss = losses / (batch_idx + 1)
                #print('Timer: %.4f' % (t1 - t0))
                print( 'Elapse time: %s per batch, %s from epoch, %s total' % \
                      ( time2str(t0, t1), time2str(startTimeEpoch, t1), time2str(startTime, t1) ) )
                print('epoch:' + repr(epoch) + ' || iter:' +
                      repr(iteration) + ' || Loss:%.4f' % (tloss))
                print('->> conf loss:{:.4f} || loc loss:{:.4f}'.format(
                    loss_c.item(), loss_l.item()))
                print('->>lr:{:.6f}'.format(optimizer.param_groups[0]['lr']))

            if iteration != 0 and iteration % 500 == 0:
                print('Saving state, iter:', iteration)
                file = 'sfd_' + args.dataset + '_' + repr(iteration) + '.pth'
                torch.save(net.state_dict(), os.path.join(args.save_folder, file))
                torch.save(net, os.path.join( args.save_folder, 'net_' + file))
            iteration += 1
        print( 'Elapse time: %s per batch, %s per epoch, %s total' % \
              ( time2str(t0, t1), time2str(startTimeEpoch, t1), time2str(startTime, t1) ) )

        val(epoch)
        if iteration == cfg.MAX_STEPS:
            break


def val(epoch):
    net.eval()
    loc_loss = 0
    conf_loss = 0
    step = 0
    t1 = time.time()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda())
                           for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann) for ann in targets]

            out = net(images)
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            step += 1

    tloss = (loc_loss + conf_loss) / step
    t2 = time.time()
    #print('Timer: %.4f' % (t2 - t1))
    dTime = t2 - t1
    print( 'Elapse time: %dm %ds for validation' % ( int(dTime / 60), dTime - int(dTime / 60) * 60 ) )
    print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))

    global min_loss
    if tloss < min_loss:
        print('Saving best state, epoch', epoch)
        file = 'rpool_{}_best.pth'.format(args.dataset)
        torch.save(net.state_dict(), os.path.join( args.save_folder, file))
        min_loss = tloss

    states = {
        'epoch': epoch,
        'weight': net.state_dict(),
    }
    file = 'rpool_{}_checkpoint.pth'.format(args.dataset)
    torch.save(states, os.path.join( args.save_folder, file))



from math import cos, pi
def adjust_learning_rate(optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = 300 * num_iter

    lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2

    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
