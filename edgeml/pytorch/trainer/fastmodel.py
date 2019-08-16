# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from edgeml.pytorch.graph.rnn import *

def get_model_class(inheritance_class=nn.Module):
    class RNNClassifierModel(inheritance_class):
        """This class is a PyTorch Module that implements a 1, 2 or 3 layer
           RNN-based classifier
        """

        def __init__(self, input_dim, num_layers, hidden_units_list,
                     wRank_list, uRank_list, wSparsity_list, uSparsity_list,
                     gate_nonlinearity, update_nonlinearity,
                     num_classes=None, linear=True, batch_first=False, apply_softmax=True):
            """
            Initialize the KeywordSpotter with the following parameters:
            input_dim - the size of the input audio frame in # samples.
            hidden_units - the size of the hidden state of the FastGrnn nodes
            num_keywords - the number of predictions to come out of the model.
            num_layers - the number of FastGrnn layers to use (1, 2 or 3)
            """
            self.input_dim = input_dim
            self.hidden_units_list = hidden_units_list
            self.num_layers = num_layers
            self.num_classes = num_classes
            self.wRank_list = wRank_list
            self.uRank_list = uRank_list
            self.wSparsity_list = wSparsity_list
            self.uSparsity_list = uSparsity_list
            self.gate_nonlinearity = gate_nonlinearity
            self.update_nonlinearity = update_nonlinearity
            self.linear = linear
            self.batch_first = batch_first
            self.apply_softmax = apply_softmax
            self.rnn_list = []

            if self.linear:
                if not self.num_classes:
                    raise Exception("num_classes need to be specified if linear is True")

            super(RNNClassifierModel, self).__init__()

            # The FastGRNN takes audio sequences as input, and outputs hidden states
            # with dimensionality hidden_units.
            self.rnn1 = FastGRNN(self.input_dim, self.hidden_units_list[0],
                                      gate_nonlinearity=self.gate_nonlinearity,
                                      update_nonlinearity=self.update_nonlinearity,
                                      wRank=self.wRank_list[0], uRank=self.uRank_list[0],
                                      wSparsity=self.wSparsity_list[0], uSparsity=self.uSparsity_list[0],
                                      batch_first = self.batch_first)
            self.rnn2 = None
            self.rnn_list.append(self.rnn1)
            last_output_size = self.hidden_units_list[0]
            if self.num_layers > 1:
                self.rnn2 = FastGRNN(self.hidden_units_list[0], self.hidden_units_list[1],
                                          gate_nonlinearity=self.gate_nonlinearity,
                                          update_nonlinearity=self.update_nonlinearity,
                                          wRank=self.wRank_list[1], uRank=self.uRank_list[1],
                                          wSparsity=self.wSparsity_list[1], uSparsity=self.uSparsity_list[1],
                                          batch_first = self.batch_first)
                last_output_size = self.hidden_units_list[1]
                self.rnn_list.append(self.rnn2)
            self.rnn3 = None
            if self.num_layers > 2:
                self.rnn3 = FastGRNN(self.hidden_units_list[1], self.hidden_units_list[2],
                                          gate_nonlinearity=self.gate_nonlinearity,
                                          update_nonlinearity=self.update_nonlinearity,
                                          wRank=self.wRank_list[2], uRank=self.uRank_list[2],
                                          wSparsity=self.wSparsity_list[2], uSparsity=self.uSparsity_list[2],
                                          batch_first = self.batch_first)
                last_output_size = self.hidden_units_list[2]
                self.rnn_list.append(self.rnn3)

            # The linear layer is a fully connected layer that maps from hidden state space
            # to number of expected keywords
            if self.linear:
                self.hidden2keyword = nn.Linear(last_output_size, num_classes)
            self.init_hidden()

        def sparsify(self):
            for rnn in self.rnn_list:
                rnn.cell.sparsify()

        def sparsifyWithSupport(self):
            for rnn in self.rnn_list:
                rnn.cell.sparsifyWithSupport()

        def get_model_size(self):
            total_size = 4 * self.hidden_units_list[self.num_layers-1] * self.num_classes
            for rnn in self.rnn_list:
                total_size += rnn.cell.get_model_size()
            return total_size

        def normalize(self, mean, std):
            self.mean = mean
            self.std = std
        
        def name(self):
            return "{} layer FastGRNN".format(self.num_layers)

        def move_to(self, device):
            for rnn in self.rnn_list:
                rnn.to(device)
            if hasattr(self, 'hidden2keyword'):
                self.hidden2keyword.to(device)

        def init_hidden_bag(self, hidden_bag_size, device):
            self.hidden_bag_size = hidden_bag_size
            self.hidden1_bag = torch.from_numpy(np.zeros([self.hidden_bag_size, self.hidden_units_list[0]],
                                                    dtype=np.float32)).to(device)
            if self.num_layers >= 2:
                self.hidden2_bag = torch.from_numpy(np.zeros([self.hidden_bag_size, self.hidden_units_list[1]],
                                                        dtype=np.float32)).to(device)
            if self.num_layers == 3:
                self.hidden3_bag = torch.from_numpy(np.zeros([self.hidden_bag_size, self.hidden_units_list[2]],
                                                        dtype=np.float32)).to(device)

        def rolling_step(self):
            shuffled_indices = list(range(self.hidden_bag_size))
            np.random.shuffle(shuffled_indices)
            if self.hidden1 is not None:
                batch_size = self.hidden1.shape[0]
                temp_indices = shuffled_indices[:batch_size]
                self.hidden1_bag[temp_indices, :] = self.hidden1
                self.hidden1 = self.hidden1_bag[0:batch_size, :]
                if self.num_layers >= 2:
                    self.hidden2_bag[temp_indices, :] = self.hidden2
                    self.hidden2 = self.hidden2_bag[0:batch_size, :]
                if self.num_layers == 3:
                    self.hidden3_bag[temp_indices, :] = self.hidden3
                    self.hidden3 = self.hidden3_bag[0:batch_size, :]

        def init_hidden(self):
            """ Clear the hidden state for the GRU nodes """
            self.hidden1 = None
            self.hidden2 = None
            self.hidden3 = None

        def forward(self, input):
            """ Perform the forward processing of the given input and return the prediction """
            # input is shape: [seq,batch,feature]
            if self.mean is not None:
                input = (input - self.mean) / self.std
            rnn_out1 = self.rnn1(input, hiddenState=self.hidden1)
            model_output = rnn_out1
            # we have to detach the hidden states because we may keep them longer than 1 iteration.
            self.hidden1 = rnn_out1.detach()[-1, :, :]
            if self.tracking:
                weights = self.rnn1.getVars()
                rnn_out1 = onnx_exportable_rnn(input, weights, self.rnn1.cell, output=rnn_out1)
                model_output = rnn_out1
            if self.rnn2 is not None:
                rnn_out2 = self.rnn2(rnn_out1, hiddenState=self.hidden2)
                self.hidden2 = rnn_out2.detach()[-1, :, :]
                if self.tracking:
                    weights = self.rnn2.getVars()
                    rnn_out2 = onnx_exportable_rnn(rnn_out1, weights, self.rnn2.cell, output=rnn_out2)
                model_output = rnn_out2
            if self.rnn3 is not None:
                rnn_out3 = self.rnn3(rnn_out2, hiddenState=self.hidden3)
                self.hidden3 = rnn_out3.detach()[-1, :, :]
                if self.tracking:
                    weights = self.rnn3.getVars()
                    rnn_out3 = onnx_exportable_rnn(rnn_out2, weights, self.rnn3.cell, output=rnn_out3)
                model_output = rnn_out3
            if self.linear:
                model_output = self.hidden2keyword(model_output[-1, :, :])
            if self.apply_softmax:
                model_output = F.log_softmax(model_output, dim=1)
            return model_output
    return RNNClassifierModel