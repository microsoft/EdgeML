# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
import argparse
import random
from PIL import Image
import numpy as np
from torchvision.datasets.vision import VisionDataset
from importlib import import_module
from pyvww.utils import VisualWakeWords





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

best_acc = 0  # best test accuracy
start_epoch = 0 

#Arg parser
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--epochs', default=900, type=int, help='total epochs')
parser.add_argument('--resume', default=None, type=str, help='load from checkpoint')
parser.add_argument('--model_arch',
                    default='model_mobilenet_rnnpool', type=str,
                    choices=['model_mobilenet_rnnpool', 'model_mobilenet_2rnnpool'],
                    help='choose architecture among rpool variants')
parser.add_argument('--ann', default=None, type=str, 
    help='specify new-path-to-visualwakewords-dataset used in dataset creation step')
parser.add_argument('--data', default=None, type=str, 
    help='specify path-to-mscoco-dataset used in dataset creation step')
args = parser.parse_args()


# Data

class VisualWakeWordsClassification(VisionDataset):
    """`Visual Wake Words <https://arxiv.org/abs/1906.05721>`_ Dataset.
    Args:
        root (string): Root directory where COCO images are downloaded to.
        annFile (string): Path to json visual wake words annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, root, annFile, transform=None, target_transform=None, split='val'):
        # super(VisualWakeWordsClassification, self).__init__(root, annFile, transform, target_transform, split)
        self.vww = VisualWakeWords(annFile)
        self.ids = list(sorted(self.vww.imgs.keys()))
        self.split = split

        self.transform = transform
        self.target_transform = target_transform
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the index of the target class.
        """
        vww = self.vww
        img_id = self.ids[index]
        ann_ids = vww.getAnnIds(imgIds=img_id)
        target = vww.loadAnns(ann_ids)[0]['category_id']

        path = vww.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')


        if self.transform is not None:
            img = self.transform(img)
           

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    # transforms.RandomAffine(10, translate=None, shear=(5,5,5,5), resample=False, fillcolor=0),
    transforms.RandomResizedCrop(size=(224,224), scale=(0.2,1.0)),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomAffine(10, translate=None, shear=(5,5,5,5), resample=False, fillcolor=0),
    # transforms.ColorJitter(brightness=(0.6,1.4), saturation=(0.9,1.1), hue=(-0.1,0.1)),
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
]) 
 




trainset = VisualWakeWordsClassification(root=os.path.join(args.data,'all2014'), 
                    annFile=os.path.join(args.ann, 'annotations/instances_train.json'), 
                    transform=transform_train, split='train')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, 
                                                num_workers=32)

testset = VisualWakeWordsClassification(root=os.path.join(args.data,'all2014'), 
                    annFile=os.path.join(args.ann, 'annotations/instances_val.json'), 
                    transform=transform_test, split='val')

testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, 
                                                num_workers=32)

 
# Model

module = import_module(args.model_arch)
model = module.mobilenetv2_rnnpool(num_classes=2, width_mult=0.35, last_channel=320)
model = model.to(device)
model = torch.nn.DataParallel(model)



if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('./checkpoints/'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoints/' + args.resume)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss().cuda()

optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=4e-5)#, alpha=0.9)
  

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0 
    total = 0
    train_loader_len = len(trainloader)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        adjust_learning_rate(optimizer, epoch, batch_idx, train_loader_len)
        
        batch_size = inputs.shape[0]
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
       
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('train_loss: ',train_loss/total, ' acc: ', correct/total)
    print('->>lr:{:.6f}'.format(optimizer.param_groups[0]['lr']))

def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch_size = inputs.shape[0]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('test_loss: ',test_loss/total, ' test_acc: ', correct/total)

    # Save checkpoint.
    print('best acc: ', best_acc)
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('./checkpoints/'):
            os.mkdir('./checkpoints/')
        torch.save(state, './checkpoints/model_mobilenet_rnnpool.pth')
        best_acc = acc


from math import cos, pi
def adjust_learning_rate(optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = 150 * num_iter


    lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2

    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)    
    test(epoch)
