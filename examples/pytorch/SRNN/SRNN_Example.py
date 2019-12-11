# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import print_function
import sys
import os
import numpy as np
import torch

from edgeml_pytorch.graph.rnn import SRNN2
from edgeml_pytorch.trainer.srnnTrainer import SRNNTrainer
import edgeml_pytorch.utils as utils
import helpermethods as helper

config = helper.getSRNN2Args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_DIR = config.data_dir
x_train_ = np.load(DATA_DIR + 'x_train.npy')
y_train = np.load(DATA_DIR + 'y_train.npy')
x_val_ = np.load(DATA_DIR + 'x_val.npy')
y_val = np.load(DATA_DIR + 'y_val.npy')
x_test_ = np.load(DATA_DIR + 'x_test.npy')
y_test = np.load(DATA_DIR + 'y_test.npy')

# Mean-var normalize
mean = np.mean(np.reshape(x_train_, [-1, x_train_.shape[-1]]), axis=0)
std = np.std(np.reshape(x_train_, [-1, x_train_.shape[-1]]), axis=0)
std[std[:] < 0.000001] = 1
x_train_ = (x_train_ - mean) / std
x_val_ = (x_val_ - mean) / std
x_test_ = (x_test_ - mean) / std
np.save('mean.npy', mean)
np.save('std.npy', std)

x_train = np.swapaxes(x_train_, 0, 1)
x_val = np.swapaxes(x_val_, 0, 1)
x_test = np.swapaxes(x_test_, 0, 1)
print("Train shape", x_train.shape, y_train.shape)
print("Val shape", x_val.shape, y_val.shape)
print("Test shape", x_test.shape, y_test.shape)

numTimeSteps = x_train.shape[0]
numInput = x_train.shape[-1]
numClasses = y_train.shape[1]

hiddenDim0 = config.hidden_dim0
brickSize = config.brick_size
hiddenDim1 = config.hidden_dim1
cellType = config.cell_type
learningRate = config.learning_rate
batchSize = config.batch_size
epochs = config.epochs
printStep = config.print_step
valStep = config.val_step
dropoutProbability_l0 = 0.2
dropoutProbability_l1 = 0.2

print(cellType)
'''
cellArgs (optional) will be passed to the respective cell

Example OPTIONAL args for FastGRNNCell
cellArgs = {'gate_non_linearity':"sigmoid",'update_non_linearity':"tanh",
				'wRank':None, 'uRank':None,'zetaInit':1.0, 'nuInit':-4.0, 
				'batch_first':False}

'''
cellArgs = {}

srnn2 = SRNN2(numInput, numClasses, hiddenDim0, hiddenDim1, cellType,
			 dropoutProbability_l0, dropoutProbability_l1,
			 **cellArgs).to(device)  
trainer = SRNNTrainer(srnn2, learningRate, lossType='xentropy', device=device)

trainer.train(brickSize, batchSize, epochs, x_train, x_val, y_train, y_val,
              printStep=printStep, valStep=valStep)

print('Saving trained model:')
torch.save(srnn2.state_dict(), 'model_srnn.pt')
