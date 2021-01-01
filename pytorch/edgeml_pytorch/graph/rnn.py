# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np

import edgeml_pytorch.utils as utils

try:
    if utils.findCUDA() is not None:
        import fastgrnn_cuda
except:
    print("Running without FastGRNN CUDA")
    pass


# All the matrix vector computations of the form Wx are done 
# in the form of xW (with appropriate changes in shapes) to 
# be consistent with tesnorflow and pytorch internal implementations


def onnx_exportable_rnn(input, fargs, cell, output):
    class RNNSymbolic(Function):
        @staticmethod
        def symbolic(g, *fargs):
            # NOTE: args/kwargs contain RNN parameters
            return g.op(cell.name, *fargs,
                        outputs=1, hidden_size_i=cell.state_size,
                        wRank_i=cell.wRank, uRank_i=cell.uRank,
                        gate_nonlinearity_s=cell.gate_nonlinearity,
                        update_nonlinearity_s=cell.update_nonlinearity)

        @staticmethod
        def forward(ctx, *fargs):
            return output

        @staticmethod
        def backward(ctx, *gargs, **gkwargs):
            raise RuntimeError("FIXME: Traced RNNs don't support backward")

    return RNNSymbolic.apply(input, *fargs)

def gen_nonlinearity(A, nonlinearity):
    '''
    Returns required activation for a tensor based on the inputs

    nonlinearity is either a callable or a value in
        ['tanh', 'sigmoid', 'relu', 'quantTanh', 'quantSigm', 'quantSigm4']
    '''
    if nonlinearity == "tanh":
        return torch.tanh(A)
    elif nonlinearity == "sigmoid":
        return torch.sigmoid(A)
    elif nonlinearity == "relu":
        return torch.relu(A, 0.0)
    elif nonlinearity == "quantTanh":
        return torch.max(torch.min(A, torch.ones_like(A)), -1.0 * torch.ones_like(A))
    elif nonlinearity == "quantSigm":
        A = (A + 1.0) / 2.0
        return torch.max(torch.min(A, torch.ones_like(A)), torch.zeros_like(A))
    elif nonlinearity == "quantSigm4":
        A = (A + 2.0) / 4.0
        return torch.max(torch.min(A, torch.ones_like(A)), torch.zeros_like(A))
    else:
        # nonlinearity is a user specified function
        if not callable(nonlinearity):
            raise ValueError("nonlinearity is either a callable or a value " +
                             "['tanh', 'sigmoid', 'relu', 'quantTanh', " +
                             "'quantSigm'")
        return nonlinearity(A)


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size,
                 gate_nonlinearity, update_nonlinearity,
                 num_W_matrices, num_U_matrices, num_biases,
                 wRank=None, uRank=None,
                 wSparsity=1.0, uSparsity=1.0):
        super(RNNCell, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._gate_nonlinearity = gate_nonlinearity
        self._update_nonlinearity = update_nonlinearity
        self._num_W_matrices = num_W_matrices
        self._num_U_matrices = num_U_matrices
        self._num_biases = num_biases
        self._num_weight_matrices = [self._num_W_matrices, self._num_U_matrices,
                                     self._num_biases]
        self._wRank = wRank
        self._uRank = uRank
        self._wSparsity = wSparsity
        self._uSparsity = uSparsity
        self.oldmats = []


    @property
    def state_size(self):
        return self._hidden_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def gate_nonlinearity(self):
        return self._gate_nonlinearity

    @property
    def update_nonlinearity(self):
        return self._update_nonlinearity

    @property
    def wRank(self):
        return self._wRank

    @property
    def uRank(self):
        return self._uRank

    @property
    def num_W_matrices(self):
        return self._num_W_matrices

    @property
    def num_U_matrices(self):
        return self._num_U_matrices

    @property
    def num_weight_matrices(self):
        return self._num_weight_matrices

    @property
    def name(self):
        raise NotImplementedError()

    def forward(self, input, state):
        raise NotImplementedError()

    def getVars(self):
        raise NotImplementedError()

    def get_model_size(self):
        '''
        Function to get aimed model size
        '''
        mats = self.getVars()
        endW = self._num_W_matrices
        endU = endW + self._num_U_matrices

        totalnnz = 2  # For Zeta and Nu
        for i in range(0, endW):
            device = mats[i].device
            totalnnz += utils.countNNZ(mats[i].cpu(), self._wSparsity)
            mats[i].to(device)
        for i in range(endW, endU):
            device = mats[i].device
            totalnnz += utils.countNNZ(mats[i].cpu(), self._uSparsity)
            mats[i].to(device)
        for i in range(endU, len(mats)):
            device = mats[i].device
            totalnnz += utils.countNNZ(mats[i].cpu(), False)
            mats[i].to(device)
        return totalnnz * 4

    def copy_previous_UW(self):
        mats = self.getVars()
        num_mats = self._num_W_matrices + self._num_U_matrices
        if len(self.oldmats) != num_mats:
            for i in range(num_mats):
                self.oldmats.append(torch.FloatTensor())
        for i in range(num_mats):
            self.oldmats[i] = torch.FloatTensor(mats[i].detach().clone().to(mats[i].device))

    def sparsify(self):
        mats = self.getVars()
        endW = self._num_W_matrices
        endU = endW + self._num_U_matrices
        for i in range(0, endW):
            mats[i] = utils.hardThreshold(mats[i], self._wSparsity)
        for i in range(endW, endU):
            mats[i] = utils.hardThreshold(mats[i], self._uSparsity)
        self.copy_previous_UW()

    def sparsifyWithSupport(self):
        mats = self.getVars()
        endU = self._num_W_matrices + self._num_U_matrices
        for i in range(0, endU):
            mats[i] = utils.supportBasedThreshold(mats[i], self.oldmats[i])

class FastGRNNCell(RNNCell):
    '''
    FastGRNN Cell with Both Full Rank and Low Rank Formulations
    Has multiple activation functions for the gates
    hidden_size = # hidden units

    gate_nonlinearity = nonlinearity for the gate can be chosen from
    [tanh, sigmoid, relu, quantTanh, quantSigm]
    update_nonlinearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    wRank = rank of W matrix (creates two matrices if not None)
    uRank = rank of U matrix (creates two matrices if not None)

    wSparsity = intended sparsity of W matrix(ces)
    uSparsity = intended sparsity of U matrix(ces)
    Warning:
    The Cell will not automatically sparsify.
    The user must invoke .sparsify to hard threshold.

    zetaInit = init for zeta, the scale param
    nuInit = init for nu, the translation param

    FastGRNN architecture and compression techniques are found in
    FastGRNN(LINK) paper

    Basic architecture is like:

    z_t = gate_nl(Wx_t + Uh_{t-1} + B_g)
    h_t^ = update_nl(Wx_t + Uh_{t-1} + B_h)
    h_t = z_t*h_{t-1} + (sigmoid(zeta)(1-z_t) + sigmoid(nu))*h_t^

    W and U can further parameterised into low rank version by
    W = matmul(W_1, W_2) and U = matmul(U_1, U_2)
    '''

    def __init__(self, input_size, hidden_size, gate_nonlinearity="sigmoid",
                 update_nonlinearity="tanh", wRank=None, uRank=None,
                 wSparsity=1.0, uSparsity=1.0, zetaInit=1.0, nuInit=-4.0,
                 name="FastGRNN"):
        super(FastGRNNCell, self).__init__(input_size, hidden_size,
                                          gate_nonlinearity, update_nonlinearity,
                                          1, 1, 2, wRank, uRank, wSparsity,
                                          uSparsity)
        self._zetaInit = zetaInit
        self._nuInit = nuInit
        if wRank is not None:
            self._num_W_matrices += 1
            self._num_weight_matrices[0] = self._num_W_matrices
        if uRank is not None:
            self._num_U_matrices += 1
            self._num_weight_matrices[1] = self._num_U_matrices
        self._name = name

        if wRank is None:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, hidden_size]))
        else:
            self.W1 = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))

        if uRank is None:
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, hidden_size]))
        else:
            self.U1 = nn.Parameter(0.1 * torch.randn([hidden_size, uRank]))
            self.U2 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))

        self.bias_gate = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size]))
        self.zeta = nn.Parameter(self._zetaInit * torch.ones([1, 1]))
        self.nu = nn.Parameter(self._nuInit * torch.ones([1, 1]))

        # self.copy_previous_UW()

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "FastGRNN"

    def forward(self, input, state):
        if self._wRank is None:
            wComp = torch.matmul(input, self.W)
        else:
            wComp = torch.matmul(
                torch.matmul(input, self.W1), self.W2)

        if self._uRank is None:
            uComp = torch.matmul(state, self.U)
        else:
            uComp = torch.matmul(
                torch.matmul(state, self.U1), self.U2)

        pre_comp = wComp + uComp
        z = gen_nonlinearity(pre_comp + self.bias_gate,
                              self._gate_nonlinearity)
        c = gen_nonlinearity(pre_comp + self.bias_update,
                              self._update_nonlinearity)
        new_h = z * state + (torch.sigmoid(self.zeta) *
                             (1.0 - z) + torch.sigmoid(self.nu)) * c

        return new_h

    def getVars(self):
        Vars = []
        if self._num_W_matrices == 1:
            Vars.append(self.W)
        else:
            Vars.extend([self.W1, self.W2])

        if self._num_U_matrices == 1:
            Vars.append(self.U)
        else:
            Vars.extend([self.U1, self.U2])

        Vars.extend([self.bias_gate, self.bias_update])
        Vars.extend([self.zeta, self.nu])
        return Vars

class FastGRNNCUDACell(RNNCell):
    '''
    A CUDA implementation of FastGRNN Cell with Full Rank Support
    hidden_size = # hidden units

    zetaInit = init for zeta, the scale param
    nuInit = init for nu, the translation param

    FastGRNN architecture and compression techniques are found in
    FastGRNN(LINK) paper

    Basic architecture is like:

    z_t = non_linearity(Wx_t + Uh_{t-1} + B_g)
    h_t^ = tanh(Wx_t + Uh_{t-1} + B_h)
    h_t = z_t*h_{t-1} + (sigmoid(zeta)(1-z_t) + sigmoid(nu))*h_t^

    '''
    def __init__(self, input_size, hidden_size, gate_nonlinearity="sigmoid", 
    update_nonlinearity="tanh", wRank=None, uRank=None, zetaInit=1.0, nuInit=-4.0, wSparsity=1.0, uSparsity=1.0, name="FastGRNNCUDACell"):
        super(FastGRNNCUDACell, self).__init__(input_size, hidden_size, gate_nonlinearity, update_nonlinearity, 
                                                1, 1, 2, wRank, uRank, wSparsity, uSparsity)
        if utils.findCUDA() is None:
            raise Exception('FastGRNNCUDA is supported only on GPU devices.')
        NON_LINEARITY = {"sigmoid": 0, "relu": 1, "tanh": 2}
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._zetaInit = zetaInit
        self._nuInit = nuInit
        self._name = name
        self.device = torch.device("cuda")

        if wRank is not None:
            self._num_W_matrices += 1
            self._num_weight_matrices[0] = self._num_W_matrices
        if uRank is not None:
            self._num_U_matrices += 1
            self._num_weight_matrices[1] = self._num_U_matrices
        self._name = name

        if wRank is None:
            self.W = nn.Parameter(0.1 * torch.randn([hidden_size, input_size], device=self.device))
            self.W1 = torch.empty(0)
            self.W2 = torch.empty(0)
        else:
            self.W = torch.empty(0)
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, input_size], device=self.device))
            self.W2 = nn.Parameter(0.1 * torch.randn([hidden_size, wRank], device=self.device))

        if uRank is None:
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, hidden_size], device=self.device))
            self.U1 = torch.empty(0)
            self.U2 = torch.empty(0)
        else:
            self.U = torch.empty(0)
            self.U1 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size], device=self.device))
            self.U2 = nn.Parameter(0.1 * torch.randn([hidden_size, uRank], device=self.device))

        self._gate_non_linearity = NON_LINEARITY[gate_nonlinearity]

        self.bias_gate = nn.Parameter(torch.ones([1, hidden_size], device=self.device))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size], device=self.device))
        self.zeta = nn.Parameter(self._zetaInit * torch.ones([1, 1], device=self.device))
        self.nu = nn.Parameter(self._nuInit * torch.ones([1, 1], device=self.device))

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "FastGRNNCUDACell"

    def forward(self, input, state):
        # Calls the custom autograd function while invokes the CUDA implementation
        if not input.is_cuda:
            input.to(self.device)
        if not state.is_cuda:
            state.to(self.device)
        return FastGRNNFunction.apply(input, self.bias_gate, self.bias_update, self.zeta, self.nu, state,
            self.W, self.U, self.W1, self.W2, self.U1, self.U2, self._gate_non_linearity)

    def getVars(self):
        Vars = []
        if self._num_W_matrices == 1:
            Vars.append(self.W)
        else:
            Vars.extend([self.W1, self.W2])

        if self._num_U_matrices == 1:
            Vars.append(self.U)
        else:
            Vars.extend([self.U1, self.U2])

        Vars.extend([self.bias_gate, self.bias_update, self.zeta, self.nu])
        return Vars

class FastRNNCell(RNNCell):
    '''
    FastRNN Cell with Both Full Rank and Low Rank Formulations
    Has multiple activation functions for the gates
    hidden_size = # hidden units

    update_nonlinearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    wRank = rank of W matrix (creates two matrices if not None)
    uRank = rank of U matrix (creates two matrices if not None)

    wSparsity = intended sparsity of W matrix(ces)
    uSparsity = intended sparsity of U matrix(ces)
    Warning:
    The Cell will not automatically sparsify.
    The user must invoke .sparsify to hard threshold.

    alphaInit = init for alpha, the update scalar
    betaInit = init for beta, the weight for previous state

    FastRNN architecture and compression techniques are found in
    FastGRNN(LINK) paper

    Basic architecture is like:

    h_t^ = update_nl(Wx_t + Uh_{t-1} + B_h)
    h_t = sigmoid(beta)*h_{t-1} + sigmoid(alpha)*h_t^

    W and U can further parameterised into low rank version by
    W = matmul(W_1, W_2) and U = matmul(U_1, U_2)
    '''

    def __init__(self, input_size, hidden_size,
                 update_nonlinearity="tanh", wRank=None, uRank=None,
                 wSparsity=1.0, uSparsity=1.0, alphaInit=-3.0, betaInit=3.0,
                 name="FastRNN"):
        super(FastRNNCell, self).__init__(input_size, hidden_size,
                                           None, update_nonlinearity,
                                           1, 1, 1, wRank, uRank, wSparsity,
                                           uSparsity)

        self._alphaInit = alphaInit
        self._betaInit = betaInit
        if wRank is not None:
            self._num_W_matrices += 1
            self._num_weight_matrices[0] = self._num_W_matrices
        if uRank is not None:
            self._num_U_matrices += 1
            self._num_weight_matrices[1] = self._num_U_matrices
        self._name = name

        if wRank is None:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, hidden_size]))
        else:
            self.W1 = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))

        if uRank is None:
            self.U = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
        else:
            self.U1 = nn.Parameter(0.1 * torch.randn([hidden_size, uRank]))
            self.U2 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))

        self.bias_update = nn.Parameter(torch.ones([1, hidden_size]))
        self.alpha = nn.Parameter(self._alphaInit * torch.ones([1, 1]))
        self.beta = nn.Parameter(self._betaInit * torch.ones([1, 1]))

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "FastRNN"

    def forward(self, input, state):
        if self._wRank is None:
            wComp = torch.matmul(input, self.W)
        else:
            wComp = torch.matmul(
                torch.matmul(input, self.W1), self.W2)

        if self._uRank is None:
            uComp = torch.matmul(state, self.U)
        else:
            uComp = torch.matmul(
                torch.matmul(state, self.U1), self.U2)

        pre_comp = wComp + uComp

        c = gen_nonlinearity(pre_comp + self.bias_update,
                              self._update_nonlinearity)
        new_h = torch.sigmoid(self.beta) * state + \
            torch.sigmoid(self.alpha) * c

        return new_h

    def getVars(self):
        Vars = []
        if self._num_W_matrices == 1:
            Vars.append(self.W)
        else:
            Vars.extend([self.W1, self.W2])

        if self._num_U_matrices == 1:
            Vars.append(self.U)
        else:
            Vars.extend([self.U1, self.U2])

        Vars.extend([self.bias_update])
        Vars.extend([self.alpha, self.beta])

        return Vars


class LSTMLRCell(RNNCell):
    '''
    LR - Low Rank
    LSTM LR Cell with Both Full Rank and Low Rank Formulations
    Has multiple activation functions for the gates
    hidden_size = # hidden units

    gate_nonlinearity = nonlinearity for the gate can be chosen from
    [tanh, sigmoid, relu, quantTanh, quantSigm]
    update_nonlinearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    wRank = rank of all W matrices
    (creates 5 matrices if not None else creates 4 matrices)
    uRank = rank of all U matrices
    (creates 5 matrices if not None else creates 4 matrices)

    LSTM architecture and compression techniques are found in
    LSTM paper

    Basic architecture:

    f_t = gate_nl(W1x_t + U1h_{t-1} + B_f)
    i_t = gate_nl(W2x_t + U2h_{t-1} + B_i)
    C_t^ = update_nl(W3x_t + U3h_{t-1} + B_c)
    o_t = gate_nl(W4x_t + U4h_{t-1} + B_o)
    C_t = f_t*C_{t-1} + i_t*C_t^
    h_t = o_t*update_nl(C_t)

    Wi and Ui can further parameterised into low rank version by
    Wi = matmul(W, W_i) and Ui = matmul(U, U_i)
    '''

    def __init__(self, input_size, hidden_size, gate_nonlinearity="sigmoid",
                 update_nonlinearity="tanh", wRank=None, uRank=None,
                 wSparsity=1.0, uSparsity=1.0, name="LSTMLR"):
        super(LSTMLRCell, self).__init__(input_size, hidden_size,
                                          gate_nonlinearity, update_nonlinearity,
                                          4, 4, 4, wRank, uRank, wSparsity,
                                          uSparsity)

        if wRank is not None:
            self._num_W_matrices += 1
            self._num_weight_matrices[0] = self._num_W_matrices
        if uRank is not None:
            self._num_U_matrices += 1
            self._num_weight_matrices[1] = self._num_U_matrices
        self._name = name

        if wRank is None:
            self.W1 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W2 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W3 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W4 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
        else:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W3 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W4 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))

        if uRank is None:
            self.U1 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U2 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U3 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U4 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
        else:
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, uRank]))
            self.U1 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U2 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U3 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U4 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))

        self.bias_f = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_i = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_c = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_o = nn.Parameter(torch.ones([1, hidden_size]))

    @property
    def gate_nonlinearity(self):
        return self._gate_nonlinearity

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "LSTMLR"

    def forward(self, input, hiddenStates):
        (h, c) = hiddenStates

        if self._wRank is None:
            wComp1 = torch.matmul(input, self.W1)
            wComp2 = torch.matmul(input, self.W2)
            wComp3 = torch.matmul(input, self.W3)
            wComp4 = torch.matmul(input, self.W4)
        else:
            wComp1 = torch.matmul(
                torch.matmul(input, self.W), self.W1)
            wComp2 = torch.matmul(
                torch.matmul(input, self.W), self.W2)
            wComp3 = torch.matmul(
                torch.matmul(input, self.W), self.W3)
            wComp4 = torch.matmul(
                torch.matmul(input, self.W), self.W4)

        if self._uRank is None:
            uComp1 = torch.matmul(h, self.U1)
            uComp2 = torch.matmul(h, self.U2)
            uComp3 = torch.matmul(h, self.U3)
            uComp4 = torch.matmul(h, self.U4)
        else:
            uComp1 = torch.matmul(
                torch.matmul(h, self.U), self.U1)
            uComp2 = torch.matmul(
                torch.matmul(h, self.U), self.U2)
            uComp3 = torch.matmul(
                torch.matmul(h, self.U), self.U3)
            uComp4 = torch.matmul(
                torch.matmul(h, self.U), self.U4)
        pre_comp1 = wComp1 + uComp1
        pre_comp2 = wComp2 + uComp2
        pre_comp3 = wComp3 + uComp3
        pre_comp4 = wComp4 + uComp4

        i = gen_nonlinearity(pre_comp1 + self.bias_i,
                              self._gate_nonlinearity)
        f = gen_nonlinearity(pre_comp2 + self.bias_f,
                              self._gate_nonlinearity)
        o = gen_nonlinearity(pre_comp4 + self.bias_o,
                              self._gate_nonlinearity)

        c_ = gen_nonlinearity(pre_comp3 + self.bias_c,
                               self._update_nonlinearity)

        new_c = f * c + i * c_
        new_h = o * gen_nonlinearity(new_c, self._update_nonlinearity)
        return new_h, new_c

    def getVars(self):
        Vars = []
        if self._num_W_matrices == 4:
            Vars.extend([self.W1, self.W2, self.W3, self.W4])
        else:
            Vars.extend([self.W, self.W1, self.W2, self.W3, self.W4])

        if self._num_U_matrices == 4:
            Vars.extend([self.U1, self.U2, self.U3, self.U4])
        else:
            Vars.extend([self.U, self.U1, self.U2, self.U3, self.U4])

        Vars.extend([self.bias_f, self.bias_i, self.bias_c, self.bias_o])

        return Vars


class GRULRCell(RNNCell):
    '''
    GRU LR Cell with Both Full Rank and Low Rank Formulations
    Has multiple activation functions for the gates
    hidden_size = # hidden units

    gate_nonlinearity = nonlinearity for the gate can be chosen from
    [tanh, sigmoid, relu, quantTanh, quantSigm]
    update_nonlinearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    wRank = rank of W matrix
    (creates 4 matrices if not None else creates 3 matrices)
    uRank = rank of U matrix
    (creates 4 matrices if not None else creates 3 matrices)

    GRU architecture and compression techniques are found in
    GRU(LINK) paper

    Basic architecture is like:

    r_t = gate_nl(W1x_t + U1h_{t-1} + B_r)
    z_t = gate_nl(W2x_t + U2h_{t-1} + B_g)
    h_t^ = update_nl(W3x_t + r_t*U3(h_{t-1}) + B_h)
    h_t = z_t*h_{t-1} + (1-z_t)*h_t^

    Wi and Ui can further parameterised into low rank version by
    Wi = matmul(W, W_i) and Ui = matmul(U, U_i)
    '''

    def __init__(self, input_size, hidden_size, gate_nonlinearity="sigmoid",
                 update_nonlinearity="tanh", wRank=None, uRank=None,
                 wSparsity=1.0, uSparsity=1.0, name="GRULR"):
        super(GRULRCell, self).__init__(input_size, hidden_size,
                                           gate_nonlinearity, update_nonlinearity,
                                           3, 3, 3, wRank, uRank, wSparsity,
                                           uSparsity)

        if wRank is not None:
            self._num_W_matrices += 1
            self._num_weight_matrices[0] = self._num_W_matrices
        if uRank is not None:
            self._num_U_matrices += 1
            self._num_weight_matrices[1] = self._num_U_matrices
        self._name = name

        if wRank is None:
            self.W1 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W2 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W3 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
        else:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W3 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))

        if uRank is None:
            self.U1 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U2 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U3 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
        else:
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, uRank]))
            self.U1 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U2 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U3 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))

        self.bias_r = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_gate = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size]))
        self._device = self.bias_update.device

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "GRULR"

    def forward(self, input, state):
        if self._wRank is None:
            wComp1 = torch.matmul(input, self.W1)
            wComp2 = torch.matmul(input, self.W2)
            wComp3 = torch.matmul(input, self.W3)
        else:
            wComp1 = torch.matmul(
                torch.matmul(input, self.W), self.W1)
            wComp2 = torch.matmul(
                torch.matmul(input, self.W), self.W2)
            wComp3 = torch.matmul(
                torch.matmul(input, self.W), self.W3)

        if self._uRank is None:
            uComp1 = torch.matmul(state, self.U1)
            uComp2 = torch.matmul(state, self.U2)
        else:
            uComp1 = torch.matmul(
                torch.matmul(state, self.U), self.U1)
            uComp2 = torch.matmul(
                torch.matmul(state, self.U), self.U2)

        pre_comp1 = wComp1 + uComp1
        pre_comp2 = wComp2 + uComp2

        r = gen_nonlinearity(pre_comp1 + self.bias_r,
                              self._gate_nonlinearity)
        z = gen_nonlinearity(pre_comp2 + self.bias_gate,
                              self._gate_nonlinearity)

        if self._uRank is None:
            pre_comp3 = wComp3 + torch.matmul(r * state, self.U3)
        else:
            pre_comp3 = wComp3 + \
                torch.matmul(torch.matmul(r * state, self.U), self.U3)

        c = gen_nonlinearity(pre_comp3 + self.bias_update,
                              self._update_nonlinearity)

        new_h = z * state + (1.0 - z) * c
        return new_h

    def getVars(self):
        Vars = []
        if self._num_W_matrices == 3:
            Vars.extend([self.W1, self.W2, self.W3])
        else:
            Vars.extend([self.W, self.W1, self.W2, self.W3])

        if self._num_U_matrices == 3:
            Vars.extend([self.U1, self.U2, self.U3])
        else:
            Vars.extend([self.U, self.U1, self.U2, self.U3])

        Vars.extend([self.bias_r, self.bias_gate, self.bias_update])

        return Vars


class UGRNNLRCell(RNNCell):
    '''
    UGRNN LR Cell with Both Full Rank and Low Rank Formulations
    Has multiple activation functions for the gates
    hidden_size = # hidden units

    gate_nonlinearity = nonlinearity for the gate can be chosen from
    [tanh, sigmoid, relu, quantTanh, quantSigm]
    update_nonlinearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    wRank = rank of W matrix
    (creates 3 matrices if not None else creates 2 matrices)
    uRank = rank of U matrix
    (creates 3 matrices if not None else creates 2 matrices)

    UGRNN architecture and compression techniques are found in
    UGRNN(LINK) paper

    Basic architecture is like:

    z_t = gate_nl(W1x_t + U1h_{t-1} + B_g)
    h_t^ = update_nl(W1x_t + U1h_{t-1} + B_h)
    h_t = z_t*h_{t-1} + (1-z_t)*h_t^

    Wi and Ui can further parameterised into low rank version by
    Wi = matmul(W, W_i) and Ui = matmul(U, U_i)
    '''

    def __init__(self, input_size, hidden_size, gate_nonlinearity="sigmoid",
                 update_nonlinearity="tanh", wRank=None, uRank=None,
                 wSparsity=1.0, uSparsity=1.0, name="UGRNNLR"):
        super(UGRNNLRCell, self).__init__(input_size, hidden_size,
                                          gate_nonlinearity, update_nonlinearity,
                                          2, 2, 2, wRank, uRank, wSparsity, uSparsity)

        if wRank is not None:
            self._num_W_matrices += 1
            self._num_weight_matrices[0] = self._num_W_matrices
        if uRank is not None:
            self._num_U_matrices += 1
            self._num_weight_matrices[1] = self._num_U_matrices
        self._name = name

        if wRank is None:
            self.W1 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W2 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
        else:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))

        if uRank is None:
            self.U1 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U2 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
        else:
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, uRank]))
            self.U1 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U2 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))

        self.bias_gate = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size]))
        self._device = self.bias_update.device

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "UGRNNLR"

    def forward(self, input, state):
        if self._wRank is None:
            wComp1 = torch.matmul(input, self.W1)
            wComp2 = torch.matmul(input, self.W2)
        else:
            wComp1 = torch.matmul(
                torch.matmul(input, self.W), self.W1)
            wComp2 = torch.matmul(
                torch.matmul(input, self.W), self.W2)

        if self._uRank is None:
            uComp1 = torch.matmul(state, self.U1)
            uComp2 = torch.matmul(state, self.U2)
        else:
            uComp1 = torch.matmul(
                torch.matmul(state, self.U), self.U1)
            uComp2 = torch.matmul(
                torch.matmul(state, self.U), self.U2)

        pre_comp1 = wComp1 + uComp1
        pre_comp2 = wComp2 + uComp2

        z = gen_nonlinearity(pre_comp1 + self.bias_gate,
                              self._gate_nonlinearity)
        c = gen_nonlinearity(pre_comp2 + self.bias_update,
                              self._update_nonlinearity)

        new_h = z * state + (1.0 - z) * c
        return new_h

    def getVars(self):
        Vars = []
        if self._num_W_matrices == 2:
            Vars.extend([self.W1, self.W2])
        else:
            Vars.extend([self.W, self.W1, self.W2])

        if self._num_U_matrices == 2:
            Vars.extend([self.U1, self.U2])
        else:
            Vars.extend([self.U, self.U1, self.U2])

        Vars.extend([self.bias_gate, self.bias_update])

        return Vars


class BaseRNN(nn.Module):
    '''
    Generic equivalent of static_rnn in tf
    Used to unroll all the cell written in this file
    We assume batch_first to be False by default 
    (following the convention in pytorch) ie.,
    [timeSteps, batchSize, inputDims] else
    [batchSize, timeSteps, inputDims]
    '''

    def __init__(self, cell: RNNCell, batch_first=False, cell_reverse: RNNCell=None, bidirectional=False):
        super(BaseRNN, self).__init__()
        self.RNNCell = cell 
        self._batch_first = batch_first
        self._bidirectional = bidirectional
        if cell_reverse is not None:
            self.RNNCell_reverse = cell_reverse
        elif self._bidirectional:
            self.RNNCell_reverse = cell

    def getVars(self):
        return self.RNNCell.getVars()

    def forward(self, input, hiddenState=None,
                cellState=None):
        self.device = input.device
        self.num_directions = 2 if self._bidirectional else 1
        # hidden
        # for i in range(num_directions):
        hiddenStates = torch.zeros(
                [input.shape[0], input.shape[1],
                 self.RNNCell.output_size]).to(self.device)

        if self._bidirectional:
                hiddenStates_reverse = torch.zeros(
                    [input.shape[0], input.shape[1],
                     self.RNNCell_reverse.output_size]).to(self.device)

        if hiddenState is None:
                hiddenState = torch.zeros(
                    [self.num_directions, input.shape[0] if self._batch_first else input.shape[1],
                    self.RNNCell.output_size]).to(self.device)

        if self._batch_first is True:
            if self.RNNCell.cellType == "LSTMLR":
                cellStates = torch.zeros(
                    [input.shape[0], input.shape[1],
                     self.RNNCell.output_size]).to(self.device)
                if self._bidirectional:
                    cellStates_reverse = torch.zeros(
                    [input.shape[0], input.shape[1],
                     self.RNNCell_reverse.output_size]).to(self.device)
                if cellState is None:
                    cellState = torch.zeros(
                        [self.num_directions, input.shape[0], self.RNNCell.output_size]).to(self.device)
                for i in range(0, input.shape[1]):
                    hiddenState[0], cellState[0] = self.RNNCell(
                        input[:, i, :], (hiddenState[0].clone(), cellState[0].clone()))
                    hiddenStates[:, i, :] = hiddenState[0]
                    cellStates[:, i, :] = cellState[0]
                    if self._bidirectional:
                        hiddenState[1], cellState[1] = self.RNNCell_reverse(
                            input[:, input.shape[1]-i-1, :], (hiddenState[1].clone(), cellState[1].clone()))
                        hiddenStates_reverse[:, i, :] = hiddenState[1]
                        cellStates_reverse[:, i, :] = cellState[1]
                if not self._bidirectional:
                    return hiddenStates, cellStates
                else:
                    return torch.cat([hiddenStates,hiddenStates_reverse],-1), torch.cat([cellStates,cellStates_reverse],-1)  
            else:
                for i in range(0, input.shape[1]):
                    hiddenState[0] = self.RNNCell(input[:, i, :], hiddenState[0].clone())
                    hiddenStates[:, i, :] = hiddenState[0]
                    if self._bidirectional:
                        hiddenState[1] = self.RNNCell_reverse(
                            input[:, input.shape[1]-i-1, :], hiddenState[1].clone())
                        hiddenStates_reverse[:, i, :] = hiddenState[1]
                if not self._bidirectional:
                    return hiddenStates
                else:
                    return torch.cat([hiddenStates,hiddenStates_reverse],-1)
        else:
            if self.RNNCell.cellType == "LSTMLR":
                cellStates = torch.zeros(
                    [input.shape[0], input.shape[1],
                     self.RNNCell.output_size]).to(self.device)
                if self._bidirectional:
                    cellStates_reverse = torch.zeros(
                    [input.shape[0], input.shape[1],
                     self.RNNCell_reverse.output_size]).to(self.device)
                if cellState is None:
                    cellState = torch.zeros(
                        [self.num_directions, input.shape[1], self.RNNCell.output_size]).to(self.device)
                for i in range(0, input.shape[0]):
                    hiddenState[0], cellState[0] = self.RNNCell(
                        input[i, :, :], (hiddenState[0].clone(), cellState[0].clone()))
                    hiddenStates[i, :, :] = hiddenState[0]
                    cellStates[i, :, :] = cellState[0]
                    if self._bidirectional:
                        hiddenState[1], cellState[1] = self.RNNCell_reverse(
                            input[input.shape[0]-i-1, :, :], (hiddenState[1].clone(), cellState[1].clone()))
                        hiddenStates_reverse[i, :, :] = hiddenState[1]
                        cellStates_reverse[i, :, :] = cellState[1]
                if not self._bidirectional:
                    return hiddenStates, cellStates
                else:
                    return torch.cat([hiddenStates,hiddenStates_reverse],-1), torch.cat([cellStates,cellStates_reverse],-1)
            else:
                for i in range(0, input.shape[0]):
                    hiddenState[0] = self.RNNCell(input[i, :, :], hiddenState[0].clone())
                    hiddenStates[i, :, :] = hiddenState[0]
                    if self._bidirectional:
                        hiddenState[1] = self.RNNCell_reverse(
                            input[input.shape[0]-i-1, :, :], hiddenState[1].clone())
                        hiddenStates_reverse[i, :, :] = hiddenState[1]
                if not self._bidirectional:
                    return hiddenStates
                else:
                    return torch.cat([hiddenStates,hiddenStates_reverse],-1)


class LSTM(nn.Module):
    """Equivalent to nn.LSTM using LSTMLRCell"""

    def __init__(self, input_size, hidden_size, gate_nonlinearity="sigmoid",
                 update_nonlinearity="tanh", wRank=None, uRank=None,
                 wSparsity=1.0, uSparsity=1.0, batch_first=False, 
                 bidirectional=False, is_shared_bidirectional=True):
        super(LSTM, self).__init__()
        self._bidirectional = bidirectional
        self._batch_first = batch_first
        self._is_shared_bidirectional = is_shared_bidirectional
        self.cell = LSTMLRCell(input_size, hidden_size,
                               gate_nonlinearity=gate_nonlinearity,
                               update_nonlinearity=update_nonlinearity,
                               wRank=wRank, uRank=uRank,
                               wSparsity=wSparsity, uSparsity=uSparsity)
        self.unrollRNN = BaseRNN(self.cell, batch_first=self._batch_first, bidirectional=self._bidirectional)

        if self._bidirectional is True and self._is_shared_bidirectional is False:
            self.cell_reverse = LSTMLRCell(input_size, hidden_size,
                               gate_nonlinearity=gate_nonlinearity,
                               update_nonlinearity=update_nonlinearity,
                               wRank=wRank, uRank=uRank,
                               wSparsity=wSparsity, uSparsity=uSparsity)
            self.unrollRNN = BaseRNN(self.cell, self.cell_reverse, batch_first=self._batch_first, bidirectional=self._bidirectional)

    def forward(self, input, hiddenState=None, cellState=None):
        return self.unrollRNN(input, hiddenState, cellState)


class GRU(nn.Module):
    """Equivalent to nn.GRU using GRULRCell"""

    def __init__(self, input_size, hidden_size, gate_nonlinearity="sigmoid",
                 update_nonlinearity="tanh", wRank=None, uRank=None,
                 wSparsity=1.0, uSparsity=1.0, batch_first=False, 
                 bidirectional=False, is_shared_bidirectional=True):
        super(GRU, self).__init__()
        self._bidirectional = bidirectional
        self._batch_first = batch_first
        self._is_shared_bidirectional = is_shared_bidirectional
        self.cell = GRULRCell(input_size, hidden_size,
                              gate_nonlinearity=gate_nonlinearity,
                              update_nonlinearity=update_nonlinearity,
                              wRank=wRank, uRank=uRank,
                              wSparsity=wSparsity, uSparsity=uSparsity)
        self.unrollRNN = BaseRNN(self.cell, batch_first=self._batch_first, bidirectional=self._bidirectional)

        if self._bidirectional is True and self._is_shared_bidirectional is False:
            self.cell_reverse = GRULRCell(input_size, hidden_size,
                              gate_nonlinearity=gate_nonlinearity,
                              update_nonlinearity=update_nonlinearity,
                              wRank=wRank, uRank=uRank,
                              wSparsity=wSparsity, uSparsity=uSparsity)
            self.unrollRNN = BaseRNN(self.cell, self.cell_reverse, batch_first=self._batch_first, bidirectional=self._bidirectional)

    def forward(self, input, hiddenState=None, cellState=None):
        return self.unrollRNN(input, hiddenState, cellState)


class UGRNN(nn.Module):
    """Equivalent to nn.UGRNN using UGRNNLRCell"""

    def __init__(self, input_size, hidden_size, gate_nonlinearity="sigmoid",
                 update_nonlinearity="tanh", wRank=None, uRank=None,
                 wSparsity=1.0, uSparsity=1.0, batch_first=False, 
                 bidirectional=False, is_shared_bidirectional=True):
        super(UGRNN, self).__init__()
        self._bidirectional = bidirectional
        self._batch_first = batch_first
        self._is_shared_bidirectional = is_shared_bidirectional
        self.cell = UGRNNLRCell(input_size, hidden_size,
                                gate_nonlinearity=gate_nonlinearity,
                                update_nonlinearity=update_nonlinearity,
                                wRank=wRank, uRank=uRank,
                                wSparsity=wSparsity, uSparsity=uSparsity)
        self.unrollRNN = BaseRNN(self.cell, batch_first=self._batch_first, bidirectional=self._bidirectional)

        if self._bidirectional is True and self._is_shared_bidirectional is False:
            self.cell_reverse = UGRNNLRCell(input_size, hidden_size,
                                gate_nonlinearity=gate_nonlinearity,
                                update_nonlinearity=update_nonlinearity,
                                wRank=wRank, uRank=uRank,
                                wSparsity=wSparsity, uSparsity=uSparsity)
            self.unrollRNN = BaseRNN(self.cell, self.cell_reverse, batch_first=self._batch_first, bidirectional=self._bidirectional)

    def forward(self, input, hiddenState=None, cellState=None):
        return self.unrollRNN(input, hiddenState, cellState)


class FastRNN(nn.Module):
    """Equivalent to nn.FastRNN using FastRNNCell"""

    def __init__(self, input_size, hidden_size, gate_nonlinearity="sigmoid",
                 update_nonlinearity="tanh", wRank=None, uRank=None,
                 wSparsity=1.0, uSparsity=1.0, alphaInit=-3.0, betaInit=3.0,
                 batch_first=False, bidirectional=False, is_shared_bidirectional=True):
        super(FastRNN, self).__init__()
        self._bidirectional = bidirectional
        self._batch_first = batch_first
        self._is_shared_bidirectional = is_shared_bidirectional
        self.cell = FastRNNCell(input_size, hidden_size,
                                gate_nonlinearity=gate_nonlinearity,
                                update_nonlinearity=update_nonlinearity,
                                wRank=wRank, uRank=uRank,
                                wSparsity=wSparsity, uSparsity=uSparsity,
                                alphaInit=alphaInit, betaInit=betaInit)
        self.unrollRNN = BaseRNN(self.cell, batch_first=self._batch_first, bidirectional=self._bidirectional)

        if self._bidirectional is True and self._is_shared_bidirectional is False:
            self.cell_reverse = FastRNNCell(input_size, hidden_size,
                                gate_nonlinearity=gate_nonlinearity,
                                update_nonlinearity=update_nonlinearity,
                                wRank=wRank, uRank=uRank,
                                wSparsity=wSparsity, uSparsity=uSparsity,
                                alphaInit=alphaInit, betaInit=betaInit)
            self.unrollRNN = BaseRNN(self.cell, self.cell_reverse, batch_first=self._batch_first, bidirectional=self._bidirectional)

    def forward(self, input, hiddenState=None, cellState=None):
        return self.unrollRNN(input, hiddenState, cellState)


class FastGRNN(nn.Module):
    """Equivalent to nn.FastGRNN using FastGRNNCell"""

    def __init__(self, input_size, hidden_size, gate_nonlinearity="sigmoid",
                 update_nonlinearity="tanh", wRank=None, uRank=None,
                 wSparsity=1.0, uSparsity=1.0, zetaInit=1.0, nuInit=-4.0,
                 batch_first=False, bidirectional=False, is_shared_bidirectional=True):
        super(FastGRNN, self).__init__()
        self._bidirectional = bidirectional
        self._batch_first = batch_first
        self._is_shared_bidirectional = is_shared_bidirectional
        self.cell = FastGRNNCell(input_size, hidden_size,
                                 gate_nonlinearity=gate_nonlinearity,
                                 update_nonlinearity=update_nonlinearity,
                                 wRank=wRank, uRank=uRank,
                                 wSparsity=wSparsity, uSparsity=uSparsity,
                                 zetaInit=zetaInit, nuInit=nuInit)
        self.unrollRNN = BaseRNN(self.cell, batch_first=self._batch_first, bidirectional=self._bidirectional)

        if self._bidirectional is True and self._is_shared_bidirectional is False:
            self.cell_reverse = FastGRNNCell(input_size, hidden_size,
                                 gate_nonlinearity=gate_nonlinearity,
                                 update_nonlinearity=update_nonlinearity,
                                 wRank=wRank, uRank=uRank,
                                 wSparsity=wSparsity, uSparsity=uSparsity,
                                 zetaInit=zetaInit, nuInit=nuInit)
            self.unrollRNN = BaseRNN(self.cell, self.cell_reverse, batch_first=self._batch_first, bidirectional=self._bidirectional)

    def getVars(self):
        return self.unrollRNN.getVars()

    def forward(self, input, hiddenState=None, cellState=None):
        return self.unrollRNN(input, hiddenState, cellState)

class FastGRNNCUDA(nn.Module):
    """
        Unrolled implementation of the FastGRNNCUDACell
        Note: update_nonlinearity is fixed to tanh, only gate_nonlinearity
        is configurable.
    """
    def __init__(self, input_size, hidden_size, gate_nonlinearity="sigmoid",
                 update_nonlinearity="tanh", wRank=None, uRank=None, 
                 wSparsity=1.0, uSparsity=1.0, zetaInit=1.0, nuInit=-4.0,
                 batch_first=False, name="FastGRNNCUDA"):
        super(FastGRNNCUDA, self).__init__()
        if utils.findCUDA() is None:
            raise Exception('FastGRNNCUDA is supported only on GPU devices.')
        NON_LINEARITY = {"sigmoid": 0, "relu": 1, "tanh": 2}
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._zetaInit = zetaInit
        self._nuInit = nuInit
        self._name = name
        self._num_W_matrices = 1
        self._num_U_matrices = 1
        self._num_biases = 2
        self._num_weight_matrices = [self._num_W_matrices, self._num_U_matrices, self._num_biases]
        self._wRank = wRank
        self._uRank = uRank
        self._wSparsity = wSparsity
        self._uSparsity = uSparsity
        self.oldmats = []
        self.device = torch.device("cuda")
        self.batch_first = batch_first
        if wRank is not None:
            self._num_W_matrices += 1
            self._num_weight_matrices[0] = self._num_W_matrices
        if uRank is not None:
            self._num_U_matrices += 1
            self._num_weight_matrices[1] = self._num_U_matrices
        self._name = name

        if wRank is None:
            self.W = nn.Parameter(0.1 * torch.randn([hidden_size, input_size], device=self.device))
            self.W1 = torch.empty(0)
            self.W2 = torch.empty(0)
        else:
            self.W = torch.empty(0)
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, input_size], device=self.device))
            self.W2 = nn.Parameter(0.1 * torch.randn([hidden_size, wRank], device=self.device))

        if uRank is None:
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, hidden_size], device=self.device))
            self.U1 = torch.empty(0)
            self.U2 = torch.empty(0)
        else:
            self.U = torch.empty(0)
            self.U1 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size], device=self.device))
            self.U2 = nn.Parameter(0.1 * torch.randn([hidden_size, uRank], device=self.device))

        self._gate_non_linearity = NON_LINEARITY[gate_nonlinearity]

        self.bias_gate = nn.Parameter(torch.ones([1, hidden_size], device=self.device))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size], device=self.device))
        self.zeta = nn.Parameter(self._zetaInit * torch.ones([1, 1], device=self.device))
        self.nu = nn.Parameter(self._nuInit * torch.ones([1, 1], device=self.device))

    def forward(self, input, hiddenState=None, cell_state=None):
        '''
            input: [timesteps, batch, features]; hiddenState: [batch, state_size]
            hiddenState is set to zeros if not provided. 
        '''
        if self.batch_first is True:
            input = input.transpose(0, 1).contiguous()
        if not input.is_cuda:
            input = input.to(self.device)
        if hiddenState is None:
            hiddenState = torch.zeros(
                [input.shape[1], self._hidden_size]).to(self.device)
        if not hiddenState.is_cuda:
            hiddenState = hiddenState.to(self.device)
        result = FastGRNNUnrollFunction.apply(input, self.bias_gate, self.bias_update, self.zeta, self.nu, hiddenState,
            self.W, self.U, self.W1, self.W2, self.U1, self.U2, self._gate_non_linearity)
        if self.batch_first is True:
            return result.transpose(0, 1)
        else:
            return result

    def getVars(self):
        Vars = []
        if self._num_W_matrices == 1:
            Vars.append(self.W)
        else:
            Vars.extend([self.W1, self.W2])

        if self._num_U_matrices == 1:
            Vars.append(self.U)
        else:
            Vars.extend([self.U1, self.U2])

        Vars.extend([self.bias_gate, self.bias_update, self.zeta, self.nu])
        return Vars

    def get_model_size(self):
        '''
        Function to get aimed model size
        '''
        mats = self.getVars()
        endW = self._num_W_matrices
        endU = endW + self._num_U_matrices

        totalnnz = 2  # For Zeta and Nu
        for i in range(0, endW):
            device = mats[i].device
            totalnnz += utils.countNNZ(mats[i].cpu(), self._wSparsity)
            mats[i].to(device)
        for i in range(endW, endU):
            device = mats[i].device
            totalnnz += utils.countNNZ(mats[i].cpu(), self._uSparsity)
            mats[i].to(device)
        for i in range(endU, len(mats)):
            device = mats[i].device
            totalnnz += utils.countNNZ(mats[i].cpu(), False)
            mats[i].to(device)
        return totalnnz * 4

    def copy_previous_UW(self):
        mats = self.getVars()
        num_mats = self._num_W_matrices + self._num_U_matrices
        if len(self.oldmats) != num_mats:
            for i in range(num_mats):
                self.oldmats.append(torch.FloatTensor())
        for i in range(num_mats):
            self.oldmats[i] = torch.FloatTensor(mats[i].detach().clone().to(mats[i].device))

    def sparsify(self):
        mats = self.getVars()
        endW = self._num_W_matrices
        endU = endW + self._num_U_matrices
        for i in range(0, endW):
            mats[i] = utils.hardThreshold(mats[i], self._wSparsity)
        for i in range(endW, endU):
            mats[i] = utils.hardThreshold(mats[i], self._uSparsity)
        self.copy_previous_UW()

    def sparsifyWithSupport(self):
        mats = self.getVars()
        endU = self._num_W_matrices + self._num_U_matrices
        for i in range(0, endU):
            mats[i] = utils.supportBasedThreshold(mats[i], self.oldmats[i])

class SRNN2(nn.Module):

    def __init__(self, inputDim, outputDim, hiddenDim0, hiddenDim1, cellType,
                 dropoutProbability0 = None, dropoutProbability1 = None,
                 **cellArgs):
        '''
        A 2 Layer Shallow RNN.

        inputDim: Input data's feature dimension.
        hiddenDim0: Hidden state dimension of the lower layer RNN cell.
        hiddenDim1: Hidden state dimension of the second layer RNN cell.
        cellType: The type of RNN cell to use. Options are ['LSTM', 'FastRNNCell',
        'FastGRNNCell', 'GRULRCell']
        '''
        super(SRNN2, self).__init__()

        # Create two RNN Cells
        self.inputDim = inputDim
        self.hiddenDim0 = hiddenDim0
        self.hiddenDim1 = hiddenDim1
        self.outputDim = outputDim
        self.dropoutProbability0 = dropoutProbability0
        self.dropoutProbability1 = dropoutProbability1
        if dropoutProbability0 != None:
            assert 0 < dropoutProbability0 <= 1.0
        if dropoutProbability1 != None:
            assert 0 < dropoutProbability1 <= 1.0
        # Setting batch_first = False to ensure compatibility of parameters across nn.LSTM and the
        # other low-rank implementations
        self.cellArgs = {
            'batch_first': False
        }
        self.cellArgs.update(cellArgs)
        supportedCells = ['LSTM', 'FastRNNCell', 'FastGRNNCell', 'GRULRCell']
        assert cellType in supportedCells, 'Currently supported cells: %r' % supportedCells
        self.cellType = cellType

        if self.cellType == 'LSTM':
            self.rnnClass = nn.LSTM
        elif self.cellType == 'FastRNNCell':
            self.rnnClass = FastRNN
        elif self.cellType == 'FastGRNNCell':
            self.rnnClass = FastGRNN
        else:
            self.rnnClass = GRU

        self.rnn0 = self.rnnClass(input_size=inputDim, hidden_size=hiddenDim0, **self.cellArgs)
        self.rnn1 = self.rnnClass(input_size=hiddenDim0, hidden_size=hiddenDim1, **self.cellArgs)
        self.W = torch.randn([self.hiddenDim1, self.outputDim])
        self.W = nn.Parameter(self.W)
        self.B = torch.randn([self.outputDim])
        self.B = nn.Parameter(self.B)

    def getBrickedData(self, x, brickSize):
        '''
        Takes x of shape [timeSteps, batchSize, featureDim] and returns bricked
        x of shape [numBricks, brickSize, batchSize, featureDim] by chunking
        along 0-th axes.
        '''
        timeSteps = list(x.size())[0]
        numSplits = int(timeSteps / brickSize)
        batchSize = list(x.size())[1]
        featureDim = list(x.size())[2]
        numBricks = int(timeSteps/brickSize)
        eqlen = numSplits * brickSize
        x = x[:eqlen]
        x_bricked = torch.split(x, numSplits, dim = 0)
        x_bricked_batched = torch.cat(x_bricked)
        x_bricked_batched = torch.reshape(x_bricked_batched, (numBricks,brickSize,batchSize,featureDim))
        return x_bricked_batched

    def forward(self, x, brickSize):
        '''
        x: Input data in numpy. Expected to be a 3D tensor  with shape
            [timeStep, batchSize, featureDim]. Note that this is different from
            the convention followed in the TF codebase.
        brickSize: The brick size for the lower dimension. The input data will
            be divided into bricks along the timeStep axis (axis=0) internally
            and fed into the lowest layer RNN. Note that if the last brick has
            fewer than 'brickSize' steps, it will be ignored (no internal
            padding is done).
        '''
        assert x.ndimension() == 3
        assert list(x.size())[2] == self.inputDim
        x_bricks = self.getBrickedData(x, brickSize)
        # x bricks: [numBricks, brickSize, batchSize, featureDim]
        x_bricks = x_bricks.permute(1,0,2,3)
        # x bricks: [brickSize, numBricks, batchSize, featureDim]
        oldShape = list(x_bricks.size())
        x_bricks = torch.reshape(x_bricks, [oldShape[0], oldShape[1] * oldShape[2], oldShape[3]])
        # x bricks: [brickSize, numBricks * batchSize, featureDim]
        # x_bricks = torch.Tensor(x_bricks)

        self.dropoutLayer0 = None
        self.dropoutLayer1 = None

        if self.cellType == 'LSTM':
            hidd0, out0 = self.rnn0(x_bricks)
        else:
            hidd0 = self.rnn0(x_bricks)

        if self.dropoutProbability0 != None:
            self.dropoutLayer0 = nn.Dropout(p=self.dropoutProbability0)
            hidd0 = self.dropoutLayer0(hidd0)
        hidd0 = torch.squeeze(hidd0[-1])
        # [numBricks * batchSize, hiddenDim0]
        inp1 = hidd0.view(oldShape[1], oldShape[2], self.hiddenDim0)
        # [numBricks, batchSize, hiddenDim0]
        if self.cellType == 'LSTM':
            hidd1, out1 = self.rnn1(inp1)
        else:
            hidd1 = self.rnn1(inp1)
        if self.dropoutProbability1 != None:
            self.dropoutLayer1 = nn.Dropout(p=self.dropoutProbability1)
            hidd1 = self.dropoutLayer1(hidd1)
        hidd1 = torch.squeeze(hidd1[-1])
        out = torch.matmul(hidd1, self.W) + self.B
        return out

class FastGRNNFunction(Function):
    @staticmethod
    def forward(ctx, input, bias_gate, bias_update, zeta, nu, old_h, w, u, w1, w2, u1, u2, gate_non_linearity):
        outputs = fastgrnn_cuda.forward(input, w, u, bias_gate, bias_update, zeta, nu, old_h, gate_non_linearity, w1, w2, u1, u2)
        new_h = outputs[0]
        variables = [input, old_h, zeta, nu, w, u] + outputs[1:] + [w1, w2, u1, u2]
        ctx.save_for_backward(*variables)
        ctx.non_linearity = gate_non_linearity
        return new_h

    @staticmethod
    def backward(ctx, grad_h):
        outputs = fastgrnn_cuda.backward(
            grad_h.contiguous(), *ctx.saved_variables, ctx.non_linearity)
        return tuple(outputs + [None])

class FastGRNNUnrollFunction(Function):
    @staticmethod
    def forward(ctx, input, bias_gate, bias_update, zeta, nu, old_h, w, u, w1, w2, u1, u2, gate_non_linearity):
        outputs = fastgrnn_cuda.forward_unroll(input, w, u, bias_gate, bias_update, zeta, nu, old_h, gate_non_linearity, w1, w2, u1, u2)
        hidden_states = outputs[0]
        variables = [input, hidden_states, zeta, nu, w, u] + outputs[1:] + [old_h, w1, w2, u1, u2]
        ctx.save_for_backward(*variables)
        ctx.gate_non_linearity = gate_non_linearity
        return hidden_states

    @staticmethod
    def backward(ctx, grad_h):
        outputs = fastgrnn_cuda.backward_unroll(
            grad_h.contiguous(), *ctx.saved_variables, ctx.gate_non_linearity)
        return tuple(outputs + [None])
