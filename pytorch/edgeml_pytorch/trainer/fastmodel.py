# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from edgeml_pytorch.graph.rnn import *

def get_model_class(inheritance_class=nn.Module):
    class RNNClassifierModel(inheritance_class):
        """This class is a PyTorch Module that implements a 1, 2 or 3 layer
           RNN-based classifier
        """

        def __init__(self, rnn_name, input_dim, num_layers, hidden_units_list,
                     wRank_list, uRank_list, wSparsity_list, uSparsity_list,
                     gate_nonlinearity, update_nonlinearity, num_classes=None,
                     linear=True, batch_first=False, apply_softmax=True):
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
            self.rnn_name = rnn_name

            if self.linear:
                if not self.num_classes:
                    raise Exception("num_classes need to be specified if linear is True")

            super(RNNClassifierModel, self).__init__()

            RNN = getattr(getattr(getattr(__import__('edgeml_pytorch'), 'graph'), 'rnn'), rnn_name)
            self.rnn_list = nn.ModuleList([
                RNN(self.input_dim if l==0 else self.hidden_units_list[l-1], 
                            self.hidden_units_list[l], 
                            gate_nonlinearity=self.gate_nonlinearity,
                            update_nonlinearity=self.update_nonlinearity,
                            wRank=self.wRank_list[l], uRank=self.uRank_list[l],
                            wSparsity=self.wSparsity_list[l],
                            uSparsity=self.uSparsity_list[l],
                            batch_first = self.batch_first)
                for l in range(self.num_layers)])

            if rnn_name == "FastGRNNCUDA":
                RNN_ = getattr(getattr(getattr(__import__('edgeml_pytorch'), 'graph'), 'rnn'), 'FastGRNN')
                self.rnn_list_ = nn.ModuleList([
                    RNN_(self.input_dim if l==0 else self.hidden_units_list[l-1],
                        self.hidden_units_list[l],
                        gate_nonlinearity=self.gate_nonlinearity,
                        update_nonlinearity=self.update_nonlinearity,
                        wRank=self.wRank_list[l], uRank=self.uRank_list[l],
                        wSparsity=self.wSparsity_list[l],
                        uSparsity=self.uSparsity_list[l],
                        batch_first = self.batch_first)
                    for l in range(self.num_layers)])
            # The linear layer is a fully connected layer that maps from hidden state space
            # to number of expected keywords
            if self.linear:
                last_output_size = self.hidden_units_list[self.num_layers-1]
                self.hidden2keyword = nn.Linear(last_output_size, num_classes)
            self.init_hidden()

        def sparsify(self):
            for rnn in self.rnn_list:
                if self.rnn_name is "FastGRNNCUDA":
                    rnn.to(torch.device("cpu"))
                    rnn.sparsify()
                    rnn.to(torch.device("cuda"))
                else:
                    rnn.cell.sparsify()

        def sparsifyWithSupport(self):
            for rnn in self.rnn_list:
                if self.rnn_name is "FastGRNNCUDA":
                    rnn.to(torch.device("cpu"))
                    rnn.sparsifyWithSupport()
                    rnn.to(torch.device("cuda"))
                else:
                    rnn.cell.sparsifyWithSupport()

        def get_model_size(self):
            total_size = 4 * self.hidden_units_list[self.num_layers-1] * self.num_classes
            print(self.rnn_name)
            for rnn in self.rnn_list:
                if self.rnn_name == "FastGRNNCUDA":
                    total_size += rnn.get_model_size()
                else:
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
            self.device = device
            self.hidden_bags_list = []

            for l in range(self.num_layers):
                self.hidden_bags_list.append(
                   torch.from_numpy(np.zeros([self.hidden_bag_size, self.hidden_units_list[l]],
                                              dtype=np.float32)).to(self.device))

        def rolling_step(self):
            shuffled_indices = list(range(self.hidden_bag_size))
            np.random.shuffle(shuffled_indices)
            # if self.hidden1 is not None:
            #     batch_size = self.hidden1.shape[0]
            if self.hidden_states[0] is not None:
                batch_size = self.hidden_states[0].shape[0]
                temp_indices = shuffled_indices[:batch_size]
                for l in range(self.num_layers):
                    bag = self.hidden_bags_list[l]
                    bag[temp_indices, :] = self.hidden_states[l]
                    self.hidden_states[l] = bag[0:batch_size, :]

        def init_hidden(self):
            """ Clear the hidden state for the GRU nodes """
            # self.hidden1 = None
            # self.hidden2 = None
            # self.hidden3 = None
            self.hidden_states = []
            for l in range(self.num_layers):
                self.hidden_states.append(None)

        def forward(self, input):
            """ Perform the forward processing of the given input and return the prediction """
            # input is shape: [seq,batch,feature]
            if self.mean is not None:
                input = (input - self.mean) / self.std

            rnn_in = input
            if self.rnn_name == "FastGRNNCUDA":
                if self.tracking:
                    for l in range(self.num_layers):
                        print("Layer: ", l)
                        rnn_ = self.rnn_list_[l]
                        model_output = rnn_(rnn_in, hiddenState=self.hidden_states[l])
                        self.hidden_states[l] = model_output.detach()[-1, :, :]
                        weights = self.rnn_list[l].getVars()
                        weights = [weight.clone() for weight in weights]
                        model_output = onnx_exportable_rnn(rnn_in, weights, rnn_.cell, output=model_output)
                        rnn_in = model_output
                else:
                    for l in range(self.num_layers):
                        rnn = self.rnn_list[l]
                        model_output = rnn(rnn_in, hiddenState=self.hidden_states[l])
                        self.hidden_states[l] = model_output.detach()[-1, :, :]
                        rnn_in = model_output
            else:
                for l in range(self.num_layers):
                    rnn = self.rnn_list[l]
                    if self.hidden_states[l] is not None:
                        self.hidden_states[l] = self.hidden_states[l].clone().unsqueeze(0)
                    model_output = rnn(rnn_in, hiddenState=self.hidden_states[l])
                    self.hidden_states[l] = model_output.detach()[-1, :, :]
                    if self.tracking:
                        weights = rnn.getVars()
                        model_output = onnx_exportable_rnn(rnn_in, weights, rnn.cell, output=model_output)
                    rnn_in = model_output

            if self.linear:
                model_output = self.hidden2keyword(model_output[-1, :, :])
            if self.apply_softmax:
                model_output = F.log_softmax(model_output, dim=1)
            return model_output
    return RNNClassifierModel
