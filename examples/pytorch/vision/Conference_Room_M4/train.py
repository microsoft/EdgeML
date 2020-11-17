# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from data import *
from layers.modules import MultiBoxLoss
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import torch.backends.cudnn as cudnn

from data.choose_config import cfg
cfg = cfg.cfg
from importlib import import_module


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='S3FD face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
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
                    default='RPool_Face_M4', type=str,
                    choices=['RPool_Face_M4'],
                    help='choose architecture among rpool variants')
parser.add_argument('--num_workers',
                    default=128, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda',
                    default=True, type=str2bool,
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
parser.add_argument('--epochs',
                    default=300, type=int,
                    help='total epochs')
parser.add_argument('--save_frequency',
                    default=5000, type=int,
                    help='iterations interval after which checkpoint is saved')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)


train_dataset = WIDERDetection(cfg.FACE.TRAIN_FILE, mode='train', mono_mode=cfg.IS_MONOCHROME)
val_dataset = WIDERDetection(cfg.FACE.VAL_FILE, mode='val', mono_mode=cfg.IS_MONOCHROME)

train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               collate_fn=detection_collate,
                               pin_memory=True)

val_batchsize = args.batch_size // 2
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=args.num_workers,
                             shuffle=False,
                             collate_fn=detection_collate,
                             pin_memory=True)

min_loss = np.inf
start_epoch = 0

module = import_module('models.' + args.model_arch)
net = module.build_s3fd('train', cfg.NUM_CLASSES)



if args.cuda:
    if args.multigpu:
        net = torch.nn.DataParallel(net)
    net = net.cuda()
    cudnn.benckmark = True

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    net.load_state_dict(torch.load(args.resume))

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)

criterion = MultiBoxLoss(cfg, args.dataset, args.cuda)
print('Loading wider dataset...')
print('Using the specified args:')
print(args)


def train():
    step_index = 0
    iteration = 0
    
    for epoch in range(start_epoch, args.epochs):
        net.train()
        losses = 0
        train_loader_len = len(train_loader)
        for batch_idx, (images, targets) in enumerate(train_loader):
            adjust_learning_rate(optimizer, epoch, batch_idx, train_loader_len)

            if args.cuda:
                images = images.cuda()
                targets = [ann.cuda()
                           for ann in targets]
            else:
                images = images
                targets = [ann for ann in targets]

              
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
                print('Timer: %.4f' % (t1 - t0))
                print('epoch:' + repr(epoch) + ' || iter:' +
                      repr(iteration) + ' || Loss:%.4f' % (tloss))
                print('->> conf loss:{:.4f} || loc loss:{:.4f}'.format(
                    loss_c.item(), loss_l.item()))
                print('->>lr:{:.6f}'.format(optimizer.param_groups[0]['lr']))

            if iteration != 0 and iteration % args.save_frequency == 0:
                print('Saving state, iter:', iteration)
                file = 'rpool_' + args.dataset + '_' + repr(iteration) + '_checkpoint.pth'
                torch.save(net.state_dict(),
                           os.path.join(args.save_folder, file))
            iteration += 1

            net.module.rnn_model.cell_rnn.cell.sparsify()
            net.module.rnn_model.cell_bidirrnn.cell.sparsify()
            net.to('cuda')

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
                images = images.cuda()
                targets = [ann.cuda()
                           for ann in targets]
            else:
                images = images
                targets = [ann for ann in targets]

            out = net(images)
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            step += 1

    tloss = (loc_loss + conf_loss) / step
    t2 = time.time()
    print('Timer: %.4f' % (t2 - t1))
    print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))

    global min_loss
    if tloss < min_loss:
        print('Saving best state,epoch', epoch)
        file = '{}_best_state.pth'.format(args.model_arch)
        torch.save(net.state_dict(), os.path.join(
            args.save_folder, file))
        min_loss = tloss



from math import cos, pi
def adjust_learning_rate(optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2

    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()