from __future__ import print_function
import sys
import os
import numpy as np
import torch

from pytorch_edgeml.graph.rnn import SRNN2
from pytorch_edgeml.trainer.srnnTrainer import SRNNTrainer
import pytorch_edgeml.utils as utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_DIR = '/datadrive/data/SRNN/GoogleSpeech/Extracted/'
x_train_, y_train = np.squeeze(np.load(DATA_DIR + 'x_train.npy')), np.squeeze(np.load(DATA_DIR + 'y_train.npy'))
x_val_, y_val = np.squeeze(np.load(DATA_DIR + 'x_val.npy')), np.squeeze(np.load(DATA_DIR + 'y_val.npy'))
x_test_, y_test = np.squeeze(np.load(DATA_DIR + 'x_test.npy')), np.squeeze(np.load(DATA_DIR + 'y_test.npy'))
# Mean-var normalize
mean = np.mean(np.reshape(x_train_, [-1, x_train_.shape[-1]]), axis=0)
std = np.std(np.reshape(x_train_, [-1, x_train_.shape[-1]]), axis=0)
std[std[:] < 0.000001] = 1
x_train_ = (x_train_ - mean) / std
x_val_ = (x_val_ - mean) / std
x_test_ = (x_test_ - mean) / std

x_train = np.swapaxes(x_train_, 0, 1)
x_val = np.swapaxes(x_val_, 0, 1)
x_test = np.swapaxes(x_test_, 0, 1)
print("Train shape", x_train.shape, y_train.shape)
print("Val shape", x_val.shape, y_val.shape)
print("Test shape", x_test.shape, y_test.shape)

numTimeSteps = x_train.shape[0]
numInput = x_train.shape[-1]
brickSize = 11
numClasses = y_train.shape[1]

hiddenDim0 = 64
hiddenDim1 = 32
cellType = 'LSTM'
learningRate = 0.01
batchSize = 128
epochs = 10

srnn2 = SRNN2(numInput, numClasses, hiddenDim0, hiddenDim1, cellType).to(device) 
trainer = SRNNTrainer(srnn2, learningRate, lossType='xentropy', device=device)

trainer.train(brickSize, batchSize, epochs, x_train, x_val, y_train, y_val, printStep=200, valStep=5)