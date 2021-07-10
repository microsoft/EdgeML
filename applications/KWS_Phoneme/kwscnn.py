# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils import _single
import math
import numpy as np 
from edgeml_pytorch.graph.rnn import *

class _IndexSelect(torch.nn.Module):
    def __init__(self, channels, direction, groups):
        """
        Channel permutation module. The purpose of this is to allow mixing across the CNN groups
        """
        super(_IndexSelect, self).__init__()

        if channels % groups != 0:
            raise ValueError('Channels should be a multiple of the groups')

        self._index = torch.zeros((channels), dtype=torch.int64)
        count = 0

        if direction > 0:
            for gidx in range(groups):
                for nidx in range(gidx, channels, groups):
                    self._index[count] = nidx
                    count += 1
        else:
            for gidx in range(groups):
                for nidx in range(gidx, channels, groups):
                    self._index[nidx] = count
                    count += 1

    def forward(self, value):
        if value.device != self._index.device:
            self._index = self._index.to(value.device)

        return torch.index_select(value, 1, self._index)


class _TanhGate(torch.nn.Module):
    def __init__(self):
        super(_TanhGate, self).__init__()

    def forward(self, value):
        """
        Applies a custom activation function
        The first half of the channels are passed through sigmoid layer and the next half though a tanh
        The outputs are multiplied and returned
        
        Input
            value: A tensor of shape (batch, channels, *)
        
        Output
            activation output of shape (batch, channels/2, *)
        """
        channels = value.shape[1]
        piv = int(channels/2)

        sig_data = value[:, 0:piv, :]
        tanh_data = value[:, piv:, :]

        sig_data = torch.sigmoid(sig_data)
        tanh_data = torch.tanh(tanh_data)
        return sig_data * tanh_data


class LR_conv(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', rank=50):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        """
        A convolution layer with the weight matrix subjected to a low-rank decomposition. Currently for kernel size of 5
        
        Input
            rank            : The rank used for the low-rank decomposition on the weight/kernel tensor
            All other parameters are similar to that of a convolution layer
            Only change is the decomposition of the output channels into low-rank tensors
        """
        self.kernel_size = kernel_size
        self.rank =  rank
        self.W1 = Parameter(torch.Tensor(self.out_channels, rank))
        # As per PyTorch Standard
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        self.W2 = Parameter(torch.Tensor(rank, self.in_channels * self.kernel_size))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        self.weight = None

    def forward(self, input):
        """
        The decomposed weights are multiplied to enforce the low-rank constraint
        The conv1d is performed as usual post multiplication
        
        Input
            input: Input of shape similar to that of which is fed to a conv layer
        
        Output
            convolution output
        """
        lr_weight = torch.matmul(self.W1, self.W2)
        lr_weight = torch.reshape(lr_weight, (self.out_channels, self.in_channels, self.kernel_size))
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            lr_weight, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(input, lr_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class PreRNNConvBlock(torch.nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel,
            stride=1, groups=1, avg_pool=2, dropout=0.1,
            batch_norm=0.1, shuffle=0,
            activation='sigmoid', rank=50):
        super(PreRNNConvBlock, self).__init__()
        """
        A low-rank convolution layer combination with pooling and activation layers. Currently for kernel size of 5

        Input
            in_channels     : number of input channels for the conv layer
            out_channels    : number of output channels for the conv layer
            kernel          : conv kernel size
            stride          : conv stride
            groups          : number of groups for conv layer
            avg_pool        : kernel size for average pooling layer
            dropout         : dropout layer probability
            batch_norm      : momentum for batch norm
            activation      : activation layer
            rank            : rank for low-rank decomposition for conv layer weights
        """
        activators = {
            'sigmoid': torch.nn.Sigmoid(),
            'relu': torch.nn.ReLU(),
            'leakyrelu': torch.nn.LeakyReLU(),
            'tanhgate': _TanhGate(),
            'none': None
        }

        if activation not in activators:
            raise ValueError('Available activations are: %s' % ', '.join(activators.keys()))

        if activation == 'tanhgate':
            in_channels = int(in_channels/2)

        nonlin = activators[activation]

        if batch_norm > 0.0:
            batch_block = torch.nn.BatchNorm1d(in_channels, affine=False, momentum=batch_norm)
        else:
            batch_block = None

        depth_cnn = None
        point_cnn = LR_conv(in_channels, out_channels, kernel_size=kernel, stride=stride, groups=groups, rank=rank, padding=2)

        if shuffle != 0 and groups > 1:
            shuffler = _IndexSelect(in_channels, shuffle, groups)
        else:
            shuffler = None

        if avg_pool > 0:
            pool = torch.nn.AvgPool1d(kernel_size=avg_pool, stride=1)
        else:
            pool = None

        if dropout > 0:
            dropout_block = torch.nn.Dropout(p=dropout)
        else:
            dropout_block = None

        seq1 = [nonlin, batch_block, depth_cnn, shuffler, point_cnn, dropout_block, pool]
        seq_f1 = [item for item in seq1 if item is not None]
        if len(seq_f1) == 1:
            self._op1 = seq_f1[0]
        else:
            self._op1 = torch.nn.Sequential(*seq_f1)

    def forward(self, x):
        """
        Apply the set of layers initialized in __init__
        
        Input
            x: A tensor of shape (batch, channels, length)
        
        Output
            network block output of shape (batch, channels, length)
        """
        x = self._op1(x)
        return x

class DSCNNBlockLR(torch.nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel,
            stride=1, groups=1, avg_pool=2, dropout=0.1,
            batch_norm=0.1, shuffle=0,
            activation='sigmoid', rank=50):
        super(DSCNNBlockLR, self).__init__()
        """
        A depthwise separable low-rank convolution layer combination with pooling and activation layers

        Input
            in_channels     : number of input channels for the pointwise conv layer
            out_channels    : number of output channels for the pointwise conv layer
            kernel          : conv kernel size for depthwise layer
            stride          : conv stride
            groups          : number of groups for conv layer
            avg_pool        : kernel size for average pooling layer
            dropout         : dropout layer probability
            batch_norm      : momentum for batch norm
            activation      : activation layer
            rank            : rank for low-rank decomposition for conv layer weights
        """
        activators = {
            'sigmoid': torch.nn.Sigmoid(),
            'relu': torch.nn.ReLU(),
            'leakyrelu': torch.nn.LeakyReLU(),
            'tanhgate': _TanhGate(),
            'none': None
        }

        if activation not in activators:
            raise ValueError('Available activations are: %s' % ', '.join(activators.keys()))

        if activation == 'tanhgate':
            in_channels = int(in_channels/2)

        nonlin = activators[activation]

        if batch_norm > 0.0:
            batch_block = torch.nn.BatchNorm1d(in_channels, affine=False, momentum=batch_norm)
        else:
            batch_block = None

        depth_cnn = torch.nn.Conv1d(in_channels, in_channels, kernel_size=kernel, stride=1, groups=in_channels, padding=2)
        point_cnn = LR_conv(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, rank=rank)

        if shuffle != 0 and groups > 1:
            shuffler = _IndexSelect(in_channels, shuffle, groups)
        else:
            shuffler = None

        if avg_pool > 0:
            pool = torch.nn.AvgPool1d(kernel_size=avg_pool, stride=1)
        else:
            pool = None

        if dropout > 0:
            dropout_block = torch.nn.Dropout(p=dropout)
        else:
            dropout_block = None

        seq = [nonlin, batch_block, depth_cnn, shuffler, point_cnn, dropout_block, pool]
        seq_f = [item for item in seq if item is not None]
        if len(seq_f) == 1:
            self._op = seq_f[0]
        else:
            self._op = torch.nn.Sequential(*seq_f)

    def forward(self, x):
        """
        Apply the set of layers initialized in __init__
        
        Input
            x: A tensor of shape (batch, channels, length)
        
        Output
            network block output of shape (batch, channels, length)
        """
        x = self._op(x)
        return x

class BiFastGRNN(nn.Module):
    """
    Bi Directional FastGRNN

    Parameters and arguments are similar to the torch RNN counterparts
    """
    def __init__(self, inputDims, hiddenDims, gate_nonlinearity,
                 update_nonlinearity, rank):
        super(BiFastGRNN, self).__init__()

        self.cell_fwd = FastGRNNCUDA(inputDims,
                                     hiddenDims,
                                     gate_nonlinearity,
                                     update_nonlinearity,
                                     batch_first=True,
                                     wRank=rank,
                                     uRank=rank)
        
        self.cell_bwd = FastGRNNCUDA(inputDims,
                                     hiddenDims,
                                     gate_nonlinearity,
                                     update_nonlinearity,
                                     batch_first=True,
                                     wRank=rank,
                                     uRank=rank)

    def forward(self, input_f, input_b):
        """
        Pass the inputs to forward and backward layers
        Please note the backward layer is similar to the forward layer and the input needs to be fed reversed accordingly
        Tensors are of the shape (batch, length, channels)

        Input
            input_f : input to the forward layer
            input_b : input to the backward layer. Input needs to be reversed before passing through this forward method
        
        Output
            output1 : output of the forward layer
            output2 : output of the backward layer
        """
        # Bidirectional FastGRNN
        output1 = self.cell_fwd(input_f)
        output2 = self.cell_bwd(input_b)
        #Returning the flipped output only for the bwd pass
        #Will align it in the post processing
        return output1, output2


def X_preRNN_process(X, fwd_context, bwd_context):
    """
    A depthwise separable low-rank convolution layer combination with pooling and activation layers

    Input
        in_channels     : number of input channels for the pointwise conv layer
        out_channels    : number of output channels for the pointwise conv layer
        kernel          : conv kernel size for depthwise layer
        stride          : conv stride
        groups          : number of groups for conv layer
        avg_pool        : kernel size for average pooling layer
        dropout         : dropout layer probability
        batch_norm      : momentum for batch norm
        activation      : activation layer
        rank            : rank for low-rank decomposition for conv layer weights
    """
    #FWD bricking
    brickLength = fwd_context
    hopLength = 3
    X_bricked_f1 = X.unfold(1, brickLength, hopLength)
    X_bricked_f2 = X_bricked_f1.permute(0, 1, 3, 2)
    #X_bricked_f [batch, num_bricks, brickLen, inpDim]
    oldShape_f = X_bricked_f2.shape
    X_bricked_f = torch.reshape(
        X_bricked_f2, [oldShape_f[0] * oldShape_f[1], oldShape_f[2], -1])
    # X_bricked_f [batch*num_bricks, brickLen, inpDim]

    #BWD bricking
    brickLength = bwd_context
    hopLength = 3
    X_bricked_b = X.unfold(1, brickLength, hopLength)
    X_bricked_b = X_bricked_b.permute(0, 1, 3, 2)
    #X_bricked_f [batch, num_bricks, brickLen, inpDim]
    oldShape_b = X_bricked_b.shape
    X_bricked_b = torch.reshape(
        X_bricked_b, [oldShape_b[0] * oldShape_b[1], oldShape_b[2], -1])
    # X_bricked_f [batch*num_bricks, brickLen, inpDim]
    return X_bricked_f, oldShape_f, X_bricked_b, oldShape_b


def X_postRNN_process(X_f, oldShape_f, X_b, oldShape_b):
    """
    A depthwise separable low-rank convolution layer combination with pooling and activation layers

    Input
        in_channels     : number of input channels for the pointwise conv layer
        out_channels    : number of output channels for the pointwise conv layer
        kernel          : conv kernel size for depthwise layer
        stride          : conv stride
        groups          : number of groups for conv layer
        avg_pool        : kernel size for average pooling layer
        dropout         : dropout layer probability
        batch_norm      : momentum for batch norm
        activation      : activation layer
        rank            : rank for low-rank decomposition for conv layer weights
    """
    #Forward bricks folding
    X_f = torch.reshape(X_f, [oldShape_f[0], oldShape_f[1], oldShape_f[2], -1])
    #X_f [batch, num_bricks, brickLen, hiddenDim]
    X_new_f = X_f[:, 0, ::3, :]  
    #batch,brickLen,hiddenDim
    X_new_f_rest = X_f[:, :, -1, :].squeeze(2)  
    #batch, numBricks-1,hiddenDim
    shape = X_new_f_rest.shape
    X_new_f = torch.cat((X_new_f, X_new_f_rest), dim=1)  
    #batch,seqLen,hiddenDim
    #X_new_f [batch, seqLen, hiddenDim]

    #Backward Bricks folding
    X_b = torch.reshape(X_b, [oldShape_b[0], oldShape_b[1], oldShape_b[2], -1])
    #X_b [batch, num_bricks, brickLen, hiddenDim]
    X_b = torch.flip(X_b, [1])  
    #Reverse the ordering of the bricks (bring last brick to start)

    X_new_b = X_b[:, 0, ::3, :]  
    #batch,brickLen,inpDim
    X_new_b_rest = X_b[:, :, -1, :].squeeze(2)  
    #batch,(seqlen-brickLen),hiddenDim
    X_new_b = torch.cat((X_new_b, X_new_b_rest), dim=1)  
    #batch,seqLen,hiddenDim
    X_new_b = torch.flip(X_new_b, [1])  
    #inverting the flip operation

    X_new = torch.cat((X_new_f, X_new_b), dim=2)  
    #batch,seqLen,2*hiddenDim
    
    return X_new


class DSCNN_RNN_Block(torch.nn.Module):
    def __init__(self, cnn_channels, rnn_hidden_size, rnn_num_layers,
                 device, gate_nonlinearity="sigmoid", update_nonlinearity="tanh",
                 isBi=True, num_labels=41, rank=None, fwd_context=15, bwd_context=9):
        super(DSCNN_RNN_Block, self).__init__()
        """
        A depthwise separable low-rank convolution layer combination with pooling and activation layers

        Input
            cnn_channels        : number of the output channels for the first CNN block
            rnn_hidden_size     : hidden dimensions of the FastGRNN
            rnn_num_layers      : number of FastGRNN layers
            device              : device on which the tensors would placed
            gate_nonlinearity   : activation function for the gating in the FastGRNN
            update_nonlinearity : activation function for the update function in the FastGRNN
            isBi                : boolean flag to use bi-directional FastGRNN
            fwd_context         : window for the forward pass
            bwd_context         : window for the backward pass
        """
        self.cnn_channels = cnn_channels
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.gate_nonlinearity = gate_nonlinearity
        self.update_nonlinearity = update_nonlinearity
        self.num_labels = num_labels
        self.device = device
        self.fwd_context = fwd_context
        self.bwd_context = bwd_context
        self.isBi = isBi
        if self.isBi:
            self.direction_param = 2
        else:
            self.direction_param = 1

        self.declare_network(cnn_channels, rnn_hidden_size, rnn_num_layers,
                             num_labels, rank)

        self.__name__ = 'DSCNN_RNN_Block'

    def declare_network(self, cnn_channels, rnn_hidden_size, rnn_num_layers,
                        num_labels, rank):
        """
        Declare the netwok layers
        Arguments can be inferred from the __init__
        """
        self.CNN1 = torch.nn.Sequential(
            PreRNNConvBlock(80, cnn_channels, 5, 1, 1,
                           0, 0, batch_norm=1e-2,
                           activation='none', rank=rank))

        #torch tanh layer directly in the forward pass 
        self.bnorm_rnn = torch.nn.BatchNorm1d(cnn_channels, affine=False, momentum=1e-2)
        self.RNN0 = BiFastGRNN(cnn_channels, rnn_hidden_size,
                               self.gate_nonlinearity,
                               self.update_nonlinearity, rank)
        
        self.CNN2 = DSCNNBlockLR(2 * rnn_hidden_size,
                    2 * rnn_hidden_size,
                    batch_norm=1e-2,
                    dropout=0,
                    kernel=5,
                    activation='tanhgate', rank=rank)
        
        self.CNN3 = DSCNNBlockLR(2 * rnn_hidden_size,
                    2 * rnn_hidden_size,
                    batch_norm=1e-2,
                    dropout=0,
                    kernel=5,
                    activation='tanhgate', rank=rank)
        
        self.CNN4 = DSCNNBlockLR(2 * rnn_hidden_size,
                    2 * rnn_hidden_size,
                    batch_norm=1e-2,
                    dropout=0,
                    kernel=5,
                    activation='tanhgate', rank=rank)
        
        self.CNN5 = DSCNNBlockLR(2 * rnn_hidden_size,
                           num_labels,
                           batch_norm=1e-2,
                           dropout=0,
                           kernel=5,
                           activation='tanhgate', rank=rank)

    def forward(self, features):
        """
        Apply the set of layers initialized in __init__
        
        Input
            features: A tensor of shape (batch, channels, length)
        
        Output
            network block output in the form (batch, channels, length)
        """
        batch, _, max_seq_len = features.shape
        X = self.CNN1(features)  # Down to 30ms inference / 250ms window
        X = torch.tanh(X)
        X = self.bnorm_rnn(X)
        X = X.permute((0, 2, 1))  #  NCL to NLC

        X = X.contiguous()
        assert X.shape[1] % 3 == 0
        X_f, oldShape_f, X_b, oldShape_b = X_preRNN_process(X, self.fwd_context, self.bwd_context)
        #X [batch * num_bricks, brickLen, inpDim]
        X_b_f = torch.flip(X_b, [1])
                
        X_f, X_b = self.RNN0(X_f, X_b_f)
        
        X = X_postRNN_process(X_f, oldShape_f, X_b, oldShape_b)

        # re-permute to get [batch, channels, max_seq_len/3 ]
        X = X.permute((0, 2, 1))  #  NLC to NCL
        X = self.CNN2(X)
        X = self.CNN3(X)
        X = self.CNN4(X)
        X = self.CNN5(X)
        return X


class Binary_Classification_Block(torch.nn.Module):
    def __init__(self, in_size, rnn_hidden_size, rnn_num_layers,
                 device, islstm=False, isBi=True, momentum=1e-2,
                 num_labels=2, dropout=0, batch_assertion=False):
        super(Binary_Classification_Block, self).__init__()
        """
        A depthwise separable low-rank convolution layer combination with pooling and activation layers

        Input
            in_size         : number of input channels to the layer
            rnn_hidden_size : hidden dimensions of the RNN layer
            rnn_num_layers  : number of layers for the RNN layer
            device          : device on which the tensors would placed
            islstm          : boolean flag to use the LSTM. False would use GRU
            isBi            : boolean flag to use the bi-directional variant of the RNN
            momentum        : momentum for the batch-norm layer
            num_labels      : number of output labels
            dropout         : probability for the dropout layer
        """
        self.in_size = in_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.num_labels = num_labels
        self.device = device
        self.islstm = islstm
        self.isBi = isBi
        self.momentum = momentum
        self.dropout = dropout
        self.batch_assertion = batch_assertion

        if self.isBi:
            self.direction_param = 2
        else:
            self.direction_param = 1

        self.declare_network(in_size, rnn_hidden_size, rnn_num_layers, num_labels)

        self.__name__ = 'Binary_Classification_Block_2lay'

    def declare_network(self, in_size, rnn_hidden_size, rnn_num_layers, num_labels):
        """
        Declare the netwok layers
        Arguments can be inferred from the __init__
        """
        self.CNN1 = torch.nn.Sequential(
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm1d(in_size, affine=False,
                                 momentum=self.momentum),
            torch.nn.Dropout(self.dropout))

        if self.islstm:
            self.RNN = nn.LSTM(input_size=in_size,
                               hidden_size=rnn_hidden_size,
                               num_layers=rnn_num_layers,
                               batch_first=True,
                               bidirectional=self.isBi)
        else:
            self.RNN = nn.GRU(input_size=in_size,
                              hidden_size=rnn_hidden_size,
                              num_layers=rnn_num_layers,
                              batch_first=True,
                              bidirectional=self.isBi)

        self.FCN = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(self.direction_param * self.rnn_hidden_size,
                            num_labels))

    def forward(self, features, seqlen):
        """
        Apply the set of layers initialized in __init__
        
        Input
            features: A tensor of shape (batch, channels, length)
        
        Output
            network block output in the form (batch, length, channels). length will be 1
        """
        batch, _, _ = features.shape

        if self.islstm:
            hidden1 = self.init_hidden(batch, self.rnn_hidden_size,
                                       self.rnn_num_layers)
            hidden2 = self.init_hidden(batch, self.rnn_hidden_size,
                                       self.rnn_num_layers)
        else:
            hidden = self.init_hidden(batch, self.rnn_hidden_size,
                                      self.rnn_num_layers)

        X = self.CNN1(features)  # Down to 30ms inference / 250ms window

        X = X.permute((0, 2, 1))  #  NCL to NLC

        max_seq_len = X.shape[1]

        # modify seqlen
        max_seq_len = min(torch.max(seqlen).item(), max_seq_len)
        seqlen = torch.clamp(seqlen, max=max_seq_len)
        self.seqlen = seqlen

        # pad according to seqlen
        X = torch.nn.utils.rnn.pack_padded_sequence(X,
                                                    seqlen,
                                                    batch_first=True,
                                                    enforce_sorted=False)

        self.RNN.flatten_parameters()
        if self.islstm:
            X, (hh, _) = self.RNN(X, (hidden1, hidden2))
        else:
            X, hh = self.RNN(X, hidden)

        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        X = X.view(batch, max_seq_len,
                   self.direction_param * self.rnn_hidden_size)

        X = X[torch.arange(batch).long(),
              seqlen.long() - 1, :].view(
                  batch, 1, self.direction_param * self.rnn_hidden_size)

        X = self.FCN(X)

        return X

    def init_hidden(self, batch, rnn_hidden_size, rnn_num_layers):
        """
        Used to initialize the first hidden state of the RNN. It is currently zero. THe user is free to edit this function for additional analysis
        """
        # the weights are of the form (batch, num_layers * num_directions , hidden_size)
        if self.batch_assertion:
            hidden = torch.zeros(rnn_num_layers * self.direction_param, batch,
                                 rnn_hidden_size)
        else:
            hidden = torch.zeros(rnn_num_layers * self.direction_param, batch,
                                 rnn_hidden_size)

        hidden = hidden.to(self.device)

        hidden = Variable(hidden)

        return hidden

