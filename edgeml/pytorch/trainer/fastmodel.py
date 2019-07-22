# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from edgeml.pytorch.graph.rnn import *

def fastgrnnmodel(inheritance_class=nn.Module):
    class FastGRNNModel(inheritance_class):
        """This class is a PyTorch Module that implements a 1, 2 or 3 layer GRU based audio classifier"""

        def __init__(self, input_dim, num_layers, hidden_units_list, wRank_list, uRank_list, gate_nonlinearity, update_nonlinearity, num_classes=None, linear=True, batch_first=False, apply_softmax=True):
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
            self.gate_nonlinearity = gate_nonlinearity
            self.update_nonlinearity = update_nonlinearity
            self.linear = linear
            self.batch_first = batch_first
            self.apply_softmax = apply_softmax
            if self.linear:
                if not self.num_classes:
                    raise Exception("num_classes need to be specified if linear is True")

            super(FastGRNNModel, self).__init__()

            # The FastGRNN takes audio sequences as input, and outputs hidden states
            # with dimensionality hidden_units.
            self.fastgrnn1 = FastGRNN(self.input_dim, self.hidden_units_list[0],
                                      gate_nonlinearity=self.gate_nonlinearity,
                                      update_nonlinearity=self.update_nonlinearity,
                                      wRank=self.wRank_list[0], uRank=self.uRank_list[0],
                                      batch_first = self.batch_first)
            self.fastgrnn2 = None
            last_output_size = self.hidden_units_list[0]
            if self.num_layers > 1:
                self.fastgrnn2 = FastGRNN(self.hidden_units_list[0], self.hidden_units_list[1],
                                          gate_nonlinearity=self.gate_nonlinearity,
                                          update_nonlinearity=self.update_nonlinearity,
                                          wRank=self.wRank_list[1], uRank=self.uRank_list[1],
                                          batch_first = self.batch_first)
                last_output_size = self.hidden_units_list[1]
            self.fastgrnn3 = None
            if self.num_layers > 2:
                self.fastgrnn3 = FastGRNN(self.hidden_units_list[1], self.hidden_units_list[2],
                                          gate_nonlinearity=self.gate_nonlinearity,
                                          update_nonlinearity=self.update_nonlinearity,
                                          wRank=self.wRank_list[2], uRank=self.uRank_list[2],
                                          batch_first = self.batch_first)
                last_output_size = self.hidden_units_list[2]

            # The linear layer is a fully connected layer that maps from hidden state space
            # to number of expected keywords
            if self.linear:
                self.hidden2keyword = nn.Linear(last_output_size, num_classes)
            self.init_hidden()

        def normalize(self, mean, std):
            self.mean = mean
            self.std = std
        
        def name(self):
            return "{} layer FastGRNN".format(self.num_layers)

        def init_hidden_bag(self, hidden_bag_size, device):
            self.hidden_bag_size = hidden_bag_size
            self.device = device
            self.hidden1_bag = torch.from_numpy(np.zeros([self.hidden_bag_size, self.hidden_units_list[0]],
                                                    dtype=np.float32)).to(self.device)
            if self.num_layers >= 2:
                self.hidden2_bag = torch.from_numpy(np.zeros([self.hidden_bag_size, self.hidden_units_list[1]],
                                                        dtype=np.float32)).to(self.device)
            if self.num_layers == 3:
                self.hidden3_bag = torch.from_numpy(np.zeros([self.hidden_bag_size, self.hidden_units_list[2]],
                                                        dtype=np.float32)).to(self.device)

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
            fastgrnn_out1 = self.fastgrnn1(input, hiddenState=self.hidden1)
            fastgrnn_output = fastgrnn_out1
            # we have to detach the hidden states because we may keep them longer than 1 iteration.
            self.hidden1 = fastgrnn_out1.detach()[-1, :, :]
            if self.tracking:
                weights = self.fastgrnn1.getVars()
                fastgrnn_out1 = onnx_exportable_fastgrnn(input, weights,
                                                         output=fastgrnn_out1, hidden_size=self.hidden_units_list[0],
                                                         wRank=self.wRank_list[0], uRank=self.uRank_list[0],
                                                         gate_nonlinearity=self.gate_nonlinearity,
                                                         update_nonlinearity=self.update_nonlinearity)
                fastgrnn_output = fastgrnn_out1
            if self.fastgrnn2 is not None:
                fastgrnn_out2 = self.fastgrnn2(fastgrnn_out1, hiddenState=self.hidden2)
                self.hidden2 = fastgrnn_out2.detach()[-1, :, :]
                if self.tracking:
                    weights = self.fastgrnn2.getVars()
                    fastgrnn_out2 = onnx_exportable_fastgrnn(fastgrnn_out1, weights,
                                                            output=fastgrnn_out2, hidden_size=self.hidden_units_list[1],
                                                            wRank=self.wRank_list[1], uRank=self.uRank_list[1],
                                                            gate_nonlinearity=self.gate_nonlinearity,
                                                            update_nonlinearity=self.update_nonlinearity)
                fastgrnn_output = fastgrnn_out2
            if self.fastgrnn3 is not None:
                fastgrnn_out3 = self.fastgrnn3(fastgrnn_out2, hiddenState=self.hidden3)
                self.hidden3 = fastgrnn_out3.detach()[-1, :, :]
                if self.tracking:
                    weights = self.fastgrnn3.getVars()
                    fastgrnn_out3 = onnx_exportable_fastgrnn(fastgrnn_out2, weights,
                                                            output=fastgrnn_out3, hidden_size=self.hidden_units_list[2],
                                                            wRank=self.wRank_list[2], uRank=self.uRank_list[2],
                                                            gate_nonlinearity=self.gate_nonlinearity,
                                                            update_nonlinearity=self.update_nonlinearity)
                fastgrnn_output = fastgrnn_out3
            if self.linear:
                fastgrnn_output = self.hidden2keyword(fastgrnn_output[-1, :, :])
            if self.apply_softmax:
                fastgrnn_output = F.log_softmax(fastgrnn_output, dim=1)
            return fastgrnn_output
    return FastGRNNModel