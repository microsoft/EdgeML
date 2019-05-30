# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.rnn_cell_impl import RNNCell


def gen_non_linearity(A, non_linearity):
    '''
    Returns required activation for a tensor based on the inputs

    non_linearity is either a callable or a value in
        ['tanh', 'sigmoid', 'relu', 'quantTanh', 'quantSigm', 'quantSigm4']
    '''
    if non_linearity == "tanh":
        return math_ops.tanh(A)
    elif non_linearity == "sigmoid":
        return math_ops.sigmoid(A)
    elif non_linearity == "relu":
        return gen_math_ops.maximum(A, 0.0)
    elif non_linearity == "quantTanh":
        return gen_math_ops.maximum(gen_math_ops.minimum(A, 1.0), -1.0)
    elif non_linearity == "quantSigm":
        A = (A + 1.0) / 2.0
        return gen_math_ops.maximum(gen_math_ops.minimum(A, 1.0), 0.0)
    elif non_linearity == "quantSigm4":
        A = (A + 2.0) / 4.0
        return gen_math_ops.maximum(gen_math_ops.minimum(A, 1.0), 0.0)
    else:
        # non_linearity is a user specified function
        if not callable(non_linearity):
            raise ValueError("non_linearity is either a callable or a value " +
                             + "['tanh', 'sigmoid', 'relu', 'quantTanh', " +
                             "'quantSigm'")
        return non_linearity(A)


class FastGRNNCell(RNNCell):
    '''
    FastGRNN Cell with Both Full Rank and Low Rank Formulations
    Has multiple activation functions for the gates
    hidden_size = # hidden units

    gate_non_linearity = nonlinearity for the gate can be chosen from
    [tanh, sigmoid, relu, quantTanh, quantSigm]
    update_non_linearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    wRank = rank of W matrix (creates two matrices if not None)
    uRank = rank of U matrix (creates two matrices if not None)
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

    def __init__(self, hidden_size, gate_non_linearity="sigmoid",
                 update_non_linearity="tanh", wRank=None, uRank=None,
                 zetaInit=1.0, nuInit=-4.0, name="FastGRNN", reuse=None):
        super(FastGRNNCell, self).__init__(_reuse=reuse)
        self._hidden_size = hidden_size
        self._gate_non_linearity = gate_non_linearity
        self._update_non_linearity = update_non_linearity
        self._num_weight_matrices = [1, 1]
        self._wRank = wRank
        self._uRank = uRank
        self._zetaInit = zetaInit
        self._nuInit = nuInit
        if wRank is not None:
            self._num_weight_matrices[0] += 1
        if uRank is not None:
            self._num_weight_matrices[1] += 1
        self._name = name
        self._reuse = reuse

    @property
    def state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def gate_non_linearity(self):
        return self._gate_non_linearity

    @property
    def update_non_linearity(self):
        return self._update_non_linearity

    @property
    def wRank(self):
        return self._wRank

    @property
    def uRank(self):
        return self._uRank

    @property
    def num_weight_matrices(self):
        return self._num_weight_matrices

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "FastGRNN"

    def call(self, inputs, state):
        with vs.variable_scope(self._name + "/FastGRNNcell"):

            if self._wRank is None:
                W_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W = vs.get_variable(
                    "W", [inputs.get_shape()[-1], self._hidden_size],
                    initializer=W_matrix_init)
                wComp = math_ops.matmul(inputs, self.W)
            else:
                W_matrix_1_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W1 = vs.get_variable(
                    "W1", [inputs.get_shape()[-1], self._wRank],
                    initializer=W_matrix_1_init)
                W_matrix_2_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W2 = vs.get_variable(
                    "W2", [self._wRank, self._hidden_size],
                    initializer=W_matrix_2_init)
                wComp = math_ops.matmul(
                    math_ops.matmul(inputs, self.W1), self.W2)

            if self._uRank is None:
                U_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U = vs.get_variable(
                    "U", [self._hidden_size, self._hidden_size],
                    initializer=U_matrix_init)
                uComp = math_ops.matmul(state, self.U)
            else:
                U_matrix_1_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U1 = vs.get_variable(
                    "U1", [self._hidden_size, self._uRank],
                    initializer=U_matrix_1_init)
                U_matrix_2_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U2 = vs.get_variable(
                    "U2", [self._uRank, self._hidden_size],
                    initializer=U_matrix_2_init)
                uComp = math_ops.matmul(
                    math_ops.matmul(state, self.U1), self.U2)
            # Init zeta to 6.0 and nu to -6.0 if this doesn't give good
            # results. The inits are hyper-params.
            zeta_init = init_ops.constant_initializer(
                self._zetaInit, dtype=tf.float32)
            self.zeta = vs.get_variable("zeta", [1, 1], initializer=zeta_init)

            nu_init = init_ops.constant_initializer(
                self._nuInit, dtype=tf.float32)
            self.nu = vs.get_variable("nu", [1, 1], initializer=nu_init)

            pre_comp = wComp + uComp

            bias_gate_init = init_ops.constant_initializer(
                1.0, dtype=tf.float32)
            self.bias_gate = vs.get_variable(
                "B_g", [1, self._hidden_size], initializer=bias_gate_init)
            z = gen_non_linearity(pre_comp + self.bias_gate,
                                  self._gate_non_linearity)

            bias_update_init = init_ops.constant_initializer(
                1.0, dtype=tf.float32)
            self.bias_update = vs.get_variable(
                "B_h", [1, self._hidden_size], initializer=bias_update_init)
            c = gen_non_linearity(
                pre_comp + self.bias_update, self._update_non_linearity)
            new_h = z * state + (math_ops.sigmoid(self.zeta) * (1.0 - z) +
                                 math_ops.sigmoid(self.nu)) * c
        return new_h, new_h

    def getVars(self):
        Vars = []
        if self._num_weight_matrices[0] == 1:
            Vars.append(self.W)
        else:
            Vars.extend([self.W1, self.W2])

        if self._num_weight_matrices[1] == 1:
            Vars.append(self.U)
        else:
            Vars.extend([self.U1, self.U2])

        Vars.extend([self.bias_gate, self.bias_update])
        Vars.extend([self.zeta, self.nu])

        return Vars


class FastRNNCell(RNNCell):
    '''
    FastRNN Cell with Both Full Rank and Low Rank Formulations
    Has multiple activation functions for the gates
    hidden_size = # hidden units

    update_non_linearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    wRank = rank of W matrix (creates two matrices if not None)
    uRank = rank of U matrix (creates two matrices if not None)
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

    def __init__(self, hidden_size, update_non_linearity="tanh",
                 wRank=None, uRank=None, alphaInit=-3.0, betaInit=3.0,
                 name="FastRNN", reuse=None):
        super(FastRNNCell, self).__init__(_reuse=reuse)
        self._hidden_size = hidden_size
        self._update_non_linearity = update_non_linearity
        self._num_weight_matrices = [1, 1]
        self._wRank = wRank
        self._uRank = uRank
        self._alphaInit = alphaInit
        self._betaInit = betaInit
        if wRank is not None:
            self._num_weight_matrices[0] += 1
        if uRank is not None:
            self._num_weight_matrices[1] += 1
        self._name = name
        self._reuse = reuse

    @property
    def state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def update_non_linearity(self):
        return self._update_non_linearity

    @property
    def wRank(self):
        return self._wRank

    @property
    def uRank(self):
        return self._uRank

    @property
    def num_weight_matrices(self):
        return self._num_weight_matrices

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "FastRNN"

    def call(self, inputs, state):
        with vs.variable_scope(self._name + "/FastRNNcell"):

            if self._wRank is None:
                W_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W = vs.get_variable(
                    "W", [inputs.get_shape()[-1], self._hidden_size],
                    initializer=W_matrix_init)
                wComp = math_ops.matmul(inputs, self.W)
            else:
                W_matrix_1_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W1 = vs.get_variable(
                    "W1", [inputs.get_shape()[-1], self._wRank],
                    initializer=W_matrix_1_init)
                W_matrix_2_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W2 = vs.get_variable(
                    "W2", [self._wRank, self._hidden_size],
                    initializer=W_matrix_2_init)
                wComp = math_ops.matmul(
                    math_ops.matmul(inputs, self.W1), self.W2)

            if self._uRank is None:
                U_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U = vs.get_variable(
                    "U", [self._hidden_size, self._hidden_size],
                    initializer=U_matrix_init)
                uComp = math_ops.matmul(state, self.U)
            else:
                U_matrix_1_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U1 = vs.get_variable(
                    "U1", [self._hidden_size, self._uRank],
                    initializer=U_matrix_1_init)
                U_matrix_2_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U2 = vs.get_variable(
                    "U2", [self._uRank, self._hidden_size],
                    initializer=U_matrix_2_init)
                uComp = math_ops.matmul(
                    math_ops.matmul(state, self.U1), self.U2)

            alpha_init = init_ops.constant_initializer(
                self._alphaInit, dtype=tf.float32)
            self.alpha = vs.get_variable(
                "alpha", [1, 1], initializer=alpha_init)

            beta_init = init_ops.constant_initializer(
                self._betaInit, dtype=tf.float32)
            self.beta = vs.get_variable("beta", [1, 1], initializer=beta_init)

            pre_comp = wComp + uComp

            bias_update_init = init_ops.constant_initializer(
                1.0, dtype=tf.float32)
            self.bias_update = vs.get_variable(
                "B_h", [1, self._hidden_size], initializer=bias_update_init)
            c = gen_non_linearity(
                pre_comp + self.bias_update, self._update_non_linearity)

            new_h = math_ops.sigmoid(self.beta) * \
                state + math_ops.sigmoid(self.alpha) * c
        return new_h, new_h

    def getVars(self):
        Vars = []
        if self._num_weight_matrices[0] == 1:
            Vars.append(self.W)
        else:
            Vars.extend([self.W1, self.W2])

        if self._num_weight_matrices[1] == 1:
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

    gate_non_linearity = nonlinearity for the gate can be chosen from
    [tanh, sigmoid, relu, quantTanh, quantSigm]
    update_non_linearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    wRank = rank of all W matrices
    (creates 5 matrices if not None else creates 4 matrices)
    uRank = rank of all U matrices
    (creates 5 matrices if not None else creates 4 matrices)

    LSTM architecture and compression techniques are found in
    LSTM paper

    Basic architecture is like:

    f_t = gate_nl(W1x_t + U1h_{t-1} + B_f)
    i_t = gate_nl(W2x_t + U2h_{t-1} + B_i)
    C_t^ = update_nl(W3x_t + U3h_{t-1} + B_c)
    o_t = gate_nl(W4x_t + U4h_{t-1} + B_o)
    C_t = f_t*C_{t-1} + i_t*C_t^
    h_t = o_t*update_nl(C_t)

    Wi and Ui can further parameterised into low rank version by
    Wi = matmul(W, W_i) and Ui = matmul(U, U_i)
    '''

    def __init__(self, hidden_size, gate_non_linearity="sigmoid",
                 update_non_linearity="tanh", wRank=None, uRank=None,
                 name="LSTMLR", reuse=None):
        super(LSTMLRCell, self).__init__(_reuse=reuse)
        self._hidden_size = hidden_size
        self._gate_non_linearity = gate_non_linearity
        self._update_non_linearity = update_non_linearity
        self._num_weight_matrices = [4, 4]
        self._wRank = wRank
        self._uRank = uRank
        if wRank is not None:
            self._num_weight_matrices[0] += 1
        if uRank is not None:
            self._num_weight_matrices[1] += 1
        self._name = name
        self._reuse = reuse

    @property
    def state_size(self):
        return 2 * self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def gate_non_linearity(self):
        return self._gate_non_linearity

    @property
    def update_non_linearity(self):
        return self._update_non_linearity

    @property
    def wRank(self):
        return self._wRank

    @property
    def uRank(self):
        return self._uRank

    @property
    def num_weight_matrices(self):
        return self._num_weight_matrices

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "LSTMLR"

    def call(self, inputs, state):
        c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)
        with vs.variable_scope(self._name + "/LSTMLRCell"):

            if self._wRank is None:
                W1_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W1 = vs.get_variable(
                    "W1", [inputs.get_shape()[-1], self._hidden_size],
                    initializer=W1_matrix_init)
                W2_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W2 = vs.get_variable(
                    "W2", [inputs.get_shape()[-1], self._hidden_size],
                    initializer=W2_matrix_init)
                W3_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W3 = vs.get_variable(
                    "W3", [inputs.get_shape()[-1], self._hidden_size],
                    initializer=W3_matrix_init)
                W4_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W4 = vs.get_variable(
                    "W4", [inputs.get_shape()[-1], self._hidden_size],
                    initializer=W4_matrix_init)
                wComp1 = math_ops.matmul(inputs, self.W1)
                wComp2 = math_ops.matmul(inputs, self.W2)
                wComp3 = math_ops.matmul(inputs, self.W3)
                wComp4 = math_ops.matmul(inputs, self.W4)
            else:
                W_matrix_r_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W = vs.get_variable(
                    "W", [inputs.get_shape()[-1], self._wRank],
                    initializer=W_matrix_r_init)
                W1_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W1 = vs.get_variable(
                    "W1", [self._wRank, self._hidden_size],
                    initializer=W1_matrix_init)
                W2_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W2 = vs.get_variable(
                    "W2", [self._wRank, self._hidden_size],
                    initializer=W2_matrix_init)
                W3_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W3 = vs.get_variable(
                    "W3", [self._wRank, self._hidden_size],
                    initializer=W3_matrix_init)
                W4_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W4 = vs.get_variable(
                    "W4", [self._wRank, self._hidden_size],
                    initializer=W4_matrix_init)
                wComp1 = math_ops.matmul(
                    math_ops.matmul(inputs, self.W), self.W1)
                wComp2 = math_ops.matmul(
                    math_ops.matmul(inputs, self.W), self.W2)
                wComp3 = math_ops.matmul(
                    math_ops.matmul(inputs, self.W), self.W3)
                wComp4 = math_ops.matmul(
                    math_ops.matmul(inputs, self.W), self.W4)
            if self._uRank is None:
                U1_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U1 = vs.get_variable(
                    "U1", [self._hidden_size, self._hidden_size],
                    initializer=U1_matrix_init)
                U2_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U2 = vs.get_variable(
                    "U2", [self._hidden_size, self._hidden_size],
                    initializer=U2_matrix_init)
                U3_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U3 = vs.get_variable(
                    "U3", [self._hidden_size, self._hidden_size],
                    initializer=U3_matrix_init)
                U4_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U4 = vs.get_variable(
                    "U4", [self._hidden_size, self._hidden_size],
                    initializer=U4_matrix_init)
                uComp1 = math_ops.matmul(h, self.U1)
                uComp2 = math_ops.matmul(h, self.U2)
                uComp3 = math_ops.matmul(h, self.U3)
                uComp4 = math_ops.matmul(h, self.U4)
            else:
                U_matrix_r_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U = vs.get_variable(
                    "U", [self._hidden_size, self._uRank],
                    initializer=U_matrix_r_init)
                U1_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U1 = vs.get_variable(
                    "U1", [self._uRank, self._hidden_size],
                    initializer=U1_matrix_init)
                U2_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U2 = vs.get_variable(
                    "U2", [self._uRank, self._hidden_size],
                    initializer=U2_matrix_init)
                U3_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U3 = vs.get_variable(
                    "U3", [self._uRank, self._hidden_size],
                    initializer=U3_matrix_init)
                U4_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U4 = vs.get_variable(
                    "U4", [self._uRank, self._hidden_size],
                    initializer=U4_matrix_init)

                uComp1 = math_ops.matmul(
                    math_ops.matmul(h, self.U), self.U1)
                uComp2 = math_ops.matmul(
                    math_ops.matmul(h, self.U), self.U2)
                uComp3 = math_ops.matmul(
                    math_ops.matmul(h, self.U), self.U3)
                uComp4 = math_ops.matmul(
                    math_ops.matmul(h, self.U), self.U4)

            pre_comp1 = wComp1 + uComp1
            pre_comp2 = wComp2 + uComp2
            pre_comp3 = wComp3 + uComp3
            pre_comp4 = wComp4 + uComp4

            bias_gate_init = init_ops.constant_initializer(
                1.0, dtype=tf.float32)
            self.bias_f = vs.get_variable(
                "B_f", [1, self._hidden_size], initializer=bias_gate_init)
            self.bias_i = vs.get_variable(
                "B_i", [1, self._hidden_size], initializer=bias_gate_init)
            self.bias_c = vs.get_variable(
                "B_c", [1, self._hidden_size], initializer=bias_gate_init)
            self.bias_o = vs.get_variable(
                "B_o", [1, self._hidden_size], initializer=bias_gate_init)

            f = gen_non_linearity(pre_comp1 + self.bias_f,
                                  self._gate_non_linearity)
            i = gen_non_linearity(pre_comp2 + self.bias_i,
                                  self._gate_non_linearity)
            o = gen_non_linearity(pre_comp4 + self.bias_o,
                                  self._gate_non_linearity)

            c_ = gen_non_linearity(
                pre_comp3 + self.bias_c, self._update_non_linearity)

            new_c = f * c + i * c_
            new_h = o * gen_non_linearity(new_c, self._update_non_linearity)
            new_state = array_ops.concat([new_c, new_h], 1)

        return new_h, new_state

    def getVars(self):
        Vars = []
        if self._num_weight_matrices[0] == 4:
            Vars.extend([self.W1, self.W2, self.W3, self.W4])
        else:
            Vars.extend([self.W, self.W1, self.W2, self.W3, self.W4])

        if self._num_weight_matrices[1] == 4:
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

    gate_non_linearity = nonlinearity for the gate can be chosen from
    [tanh, sigmoid, relu, quantTanh, quantSigm]
    update_non_linearity = nonlinearity for final rnn update
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

    def __init__(self, hidden_size, gate_non_linearity="sigmoid",
                 update_non_linearity="tanh", wRank=None, uRank=None,
                 name="GRULR", reuse=None):
        super(GRULRCell, self).__init__(_reuse=reuse)
        self._hidden_size = hidden_size
        self._gate_non_linearity = gate_non_linearity
        self._update_non_linearity = update_non_linearity
        self._num_weight_matrices = [3, 3]
        self._wRank = wRank
        self._uRank = uRank
        if wRank is not None:
            self._num_weight_matrices[0] += 1
        if uRank is not None:
            self._num_weight_matrices[1] += 1
        self._name = name
        self._reuse = reuse

    @property
    def state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def gate_non_linearity(self):
        return self._gate_non_linearity

    @property
    def update_non_linearity(self):
        return self._update_non_linearity

    @property
    def wRank(self):
        return self._wRank

    @property
    def uRank(self):
        return self._uRank

    @property
    def num_weight_matrices(self):
        return self._num_weight_matrices

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "GRULR"

    def call(self, inputs, state):
        with vs.variable_scope(self._name + "/GRULRCell"):

            if self._wRank is None:
                W1_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W1 = vs.get_variable(
                    "W1", [inputs.get_shape()[-1], self._hidden_size],
                    initializer=W1_matrix_init)
                W2_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W2 = vs.get_variable(
                    "W2", [inputs.get_shape()[-1], self._hidden_size],
                    initializer=W2_matrix_init)
                W3_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W3 = vs.get_variable(
                    "W3", [inputs.get_shape()[-1], self._hidden_size],
                    initializer=W3_matrix_init)
                wComp1 = math_ops.matmul(inputs, self.W1)
                wComp2 = math_ops.matmul(inputs, self.W2)
                wComp3 = math_ops.matmul(inputs, self.W3)
            else:
                W_matrix_r_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W = vs.get_variable(
                    "W", [inputs.get_shape()[-1], self._wRank],
                    initializer=W_matrix_r_init)
                W1_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W1 = vs.get_variable(
                    "W1", [self._wRank, self._hidden_size],
                    initializer=W1_matrix_init)
                W2_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W2 = vs.get_variable(
                    "W2", [self._wRank, self._hidden_size],
                    initializer=W2_matrix_init)
                W3_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W3 = vs.get_variable(
                    "W3", [self._wRank, self._hidden_size],
                    initializer=W3_matrix_init)
                wComp1 = math_ops.matmul(
                    math_ops.matmul(inputs, self.W), self.W1)
                wComp2 = math_ops.matmul(
                    math_ops.matmul(inputs, self.W), self.W2)
                wComp3 = math_ops.matmul(
                    math_ops.matmul(inputs, self.W), self.W3)

            if self._uRank is None:
                U1_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U1 = vs.get_variable(
                    "U1", [self._hidden_size, self._hidden_size],
                    initializer=U1_matrix_init)
                U2_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U2 = vs.get_variable(
                    "U2", [self._hidden_size, self._hidden_size],
                    initializer=U2_matrix_init)
                U3_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U3 = vs.get_variable(
                    "U3", [self._hidden_size, self._hidden_size],
                    initializer=U3_matrix_init)
                uComp1 = math_ops.matmul(state, self.U1)
                uComp2 = math_ops.matmul(state, self.U2)
            else:
                U_matrix_r_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U = vs.get_variable(
                    "U", [self._hidden_size, self._uRank],
                    initializer=U_matrix_r_init)
                U1_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U1 = vs.get_variable(
                    "U1", [self._uRank, self._hidden_size],
                    initializer=U1_matrix_init)
                U2_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U2 = vs.get_variable(
                    "U2", [self._uRank, self._hidden_size],
                    initializer=U2_matrix_init)
                U3_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U3 = vs.get_variable(
                    "U3", [self._uRank, self._hidden_size],
                    initializer=U3_matrix_init)
                uComp1 = math_ops.matmul(
                    math_ops.matmul(state, self.U), self.U1)
                uComp2 = math_ops.matmul(
                    math_ops.matmul(state, self.U), self.U2)

            pre_comp1 = wComp1 + uComp1
            pre_comp2 = wComp2 + uComp2

            bias_r_init = init_ops.constant_initializer(
                1.0, dtype=tf.float32)
            self.bias_r = vs.get_variable(
                "B_r", [1, self._hidden_size], initializer=bias_r_init)
            r = gen_non_linearity(pre_comp1 + self.bias_r,
                                  self._gate_non_linearity)

            bias_gate_init = init_ops.constant_initializer(
                1.0, dtype=tf.float32)
            self.bias_gate = vs.get_variable(
                "B_g", [1, self._hidden_size], initializer=bias_gate_init)
            z = gen_non_linearity(pre_comp2 + self.bias_gate,
                                  self._gate_non_linearity)

            if self._uRank is None:
                pre_comp3 = wComp3 + math_ops.matmul(r * state, self.U3)
            else:
                pre_comp3 = wComp3 + \
                    math_ops.matmul(math_ops.matmul(
                        r * state, self.U), self.U3)

            bias_update_init = init_ops.constant_initializer(
                1.0, dtype=tf.float32)
            self.bias_update = vs.get_variable(
                "B_h", [1, self._hidden_size], initializer=bias_update_init)
            c = gen_non_linearity(
                pre_comp3 + self.bias_update, self._update_non_linearity)

            new_h = z * state + (1.0 - z) * c

        return new_h, new_h

    def getVars(self):
        Vars = []
        if self._num_weight_matrices[0] == 3:
            Vars.extend([self.W1, self.W2, self.W3])
        else:
            Vars.extend([self.W, self.W1, self.W2, self.W3])

        if self._num_weight_matrices[1] == 3:
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

    gate_non_linearity = nonlinearity for the gate can be chosen from
    [tanh, sigmoid, relu, quantTanh, quantSigm]
    update_non_linearity = nonlinearity for final rnn update
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

    def __init__(self, hidden_size, gate_non_linearity="sigmoid",
                 update_non_linearity="tanh", wRank=None, uRank=None,
                 name="UGRNNLR", reuse=None):
        super(UGRNNLRCell, self).__init__(_reuse=reuse)
        self._hidden_size = hidden_size
        self._gate_non_linearity = gate_non_linearity
        self._update_non_linearity = update_non_linearity
        self._num_weight_matrices = [2, 2]
        self._wRank = wRank
        self._uRank = uRank
        if wRank is not None:
            self._num_weight_matrices[0] += 1
        if uRank is not None:
            self._num_weight_matrices[1] += 1
        self._name = name
        self._reuse = reuse

    @property
    def state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def gate_non_linearity(self):
        return self._gate_non_linearity

    @property
    def update_non_linearity(self):
        return self._update_non_linearity

    @property
    def wRank(self):
        return self._wRank

    @property
    def uRank(self):
        return self._uRank

    @property
    def num_weight_matrices(self):
        return self._num_weight_matrices

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "UGRNNLR"

    def call(self, inputs, state):
        with vs.variable_scope(self._name + "/UGRNNLRCell"):

            if self._wRank is None:
                W1_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W1 = vs.get_variable(
                    "W1", [inputs.get_shape()[-1], self._hidden_size],
                    initializer=W1_matrix_init)
                W2_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W2 = vs.get_variable(
                    "W2", [inputs.get_shape()[-1], self._hidden_size],
                    initializer=W2_matrix_init)
                wComp1 = math_ops.matmul(inputs, self.W1)
                wComp2 = math_ops.matmul(inputs, self.W2)
            else:
                W_matrix_r_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W = vs.get_variable(
                    "W", [inputs.get_shape()[-1], self._wRank],
                    initializer=W_matrix_r_init)
                W1_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W1 = vs.get_variable(
                    "W1", [self._wRank, self._hidden_size],
                    initializer=W1_matrix_init)
                W2_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W2 = vs.get_variable(
                    "W2", [self._wRank, self._hidden_size],
                    initializer=W2_matrix_init)
                wComp1 = math_ops.matmul(
                    math_ops.matmul(inputs, self.W), self.W1)
                wComp2 = math_ops.matmul(
                    math_ops.matmul(inputs, self.W), self.W2)

            if self._uRank is None:
                U1_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U1 = vs.get_variable(
                    "U1", [self._hidden_size, self._hidden_size],
                    initializer=U1_matrix_init)
                U2_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U2 = vs.get_variable(
                    "U2", [self._hidden_size, self._hidden_size],
                    initializer=U2_matrix_init)
                uComp1 = math_ops.matmul(state, self.U1)
                uComp2 = math_ops.matmul(state, self.U2)
            else:
                U_matrix_r_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U = vs.get_variable(
                    "U", [self._hidden_size, self._uRank],
                    initializer=U_matrix_r_init)
                U1_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U1 = vs.get_variable(
                    "U1", [self._uRank, self._hidden_size],
                    initializer=U1_matrix_init)
                U2_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U2 = vs.get_variable(
                    "U2", [self._uRank, self._hidden_size],
                    initializer=U2_matrix_init)
                uComp1 = math_ops.matmul(
                    math_ops.matmul(state, self.U), self.U1)
                uComp2 = math_ops.matmul(
                    math_ops.matmul(state, self.U), self.U2)

            pre_comp1 = wComp1 + uComp1
            pre_comp2 = wComp2 + uComp2

            bias_gate_init = init_ops.constant_initializer(
                1.0, dtype=tf.float32)
            self.bias_gate = vs.get_variable(
                "B_g", [1, self._hidden_size], initializer=bias_gate_init)
            z = gen_non_linearity(pre_comp1 + self.bias_gate,
                                  self._gate_non_linearity)

            bias_update_init = init_ops.constant_initializer(
                1.0, dtype=tf.float32)
            self.bias_update = vs.get_variable(
                "B_h", [1, self._hidden_size], initializer=bias_update_init)
            c = gen_non_linearity(
                pre_comp2 + self.bias_update, self._update_non_linearity)

            new_h = z * state + (1.0 - z) * c

        return new_h, new_h

    def getVars(self):
        Vars = []
        if self._num_weight_matrices[0] == 2:
            Vars.extend([self.W1, self.W2])
        else:
            Vars.extend([self.W, self.W1, self.W2])

        if self._num_weight_matrices[1] == 2:
            Vars.extend([self.U1, self.U2])
        else:
            Vars.extend([self.U, self.U1, self.U2])

        Vars.extend([self.bias_gate, self.bias_update])

        return Vars


class EMI_DataPipeline():
    '''
    The data input block for EMI-RNN training. Since EMI-RNN is an expensive
    algorithm due to the multiple rounds of updates that are to be performed,
    we avoid using feed dict to feed data into tensorflow and rather,
    exploit the dataset API. This class abstracts away most of the dataset API
    implementation details and provides a module that ingests data in numpy
    matrices and serves them to the remainder of the computation graph.

    This class uses reinitializable iterators. Please refer to the dataset API
    docs for more information.

    This class supports resuming from checkpoint files. Provide the restored
    meta graph as an argument to __init__ to enable this behaviour.

    Usage:
        Step 1: Create a data input pipeline object and obtain the x_batch and
        y_batch tensors. These should be fed to other parts of the graph that
        are supposed to act on the input data.
        ```
            inputPipeline = EMI_DataPipeline(NUM_SUBINSTANCE, NUM_TIMESTEPS,
                                             NUM_FEATS, NUM_OUTPUT)
            x_batch, y_batch = inputPipeline()
            # feed to emiLSTM or some other computation subgraph
            y_cap = emiLSTM(x_batch)
        ```

        Step 2:  Create other parts of the computation graph (loss operations,
        training ops etc). After the graph construction is complete and after
        initializing the Tensorflow graph with global_variables_initializer,
        initialize the iterator with the input data by calling:
            inputPipeline.runInitializer(x_train, y_trian, ...)

        Step 3: You can now iterate over batches by running some computation
        operation as you would normally do in seesion.run(..). Att the end of
        the data, tf.errors.OutOfRangeError will be
        thrown.
        ```
        while True:
            try:
                sess.run(y_cap)
            except tf.errors.OutOfRangeError:
                break
        ```
    '''

    def __init__(self, numSubinstance, numTimesteps, numFeats, numOutput,
                 graph=None, prefetchNum=5):
        '''
        numSubinstance, numTimeSteps, numFeats, numOutput:
            Dataset characteristics. Please refer to the data preparation
            documentation for more information provided in `examples/EMI-RNN`
        graph: This module supports resuming/restoring from a saved metagraph. To
            enable this behaviour, pass the restored graph as an argument. A
            saved metagraph can be restored using the edgeml.utils.GraphManager
            module.
        prefetchNum: The number of asynchronous prefetch to do when iterating over
            the data. Please refer to 'prefetching' in Tensorflow dataset API
        '''

        self.numSubinstance = numSubinstance
        self.numTimesteps = numTimesteps
        self.numFeats = numFeats
        self.graph = graph
        self.prefetchNum = prefetchNum
        self.numOutput = numOutput
        self.graphCreated = False
        # Either restore or create the following
        self.X = None
        self.Y = None
        self.batchSize = None
        self.numEpochs = None
        self.dataset_init = None
        self.x_batch = None
        self.y_batch = None
        # Internal
        self.scope = 'EMI/'

    def _createGraph(self):
        assert self.graphCreated is False
        dim = [None, self.numSubinstance, self.numTimesteps, self.numFeats]
        scope = self.scope + 'input-pipeline/'
        with tf.name_scope(scope):
            X = tf.placeholder(tf.float32, dim, name='inpX')
            Y = tf.placeholder(tf.float32, [None, self.numSubinstance,
                                            self.numOutput], name='inpY')
            batchSize = tf.placeholder(tf.int64, name='batch-size')
            numEpochs = tf.placeholder(tf.int64, name='num-epochs')

            dataset_x_target = tf.data.Dataset.from_tensor_slices(X)
            dataset_y_target = tf.data.Dataset.from_tensor_slices(Y)
            couple = (dataset_x_target, dataset_y_target)
            ds_target = tf.data.Dataset.zip(couple).repeat(numEpochs)
            ds_target = ds_target.batch(batchSize)
            ds_target = ds_target.prefetch(self.prefetchNum)
            ds_iterator_target = tf.data.Iterator.from_structure(ds_target.output_types,
                                                                 ds_target.output_shapes)
            ds_next_target = ds_iterator_target
            ds_init_target = ds_iterator_target.make_initializer(ds_target,
                                                                 name='dataset-init')
            x_batch, y_batch = ds_iterator_target.get_next()
            tf.add_to_collection('next-x-batch', x_batch)
            tf.add_to_collection('next-y-batch', y_batch)
        self.X = X
        self.Y = Y
        self.batchSize = batchSize
        self.numEpochs = numEpochs
        self.dataset_init = ds_init_target
        self.x_batch, self.y_batch = x_batch, y_batch
        self.graphCreated = True

    def _restoreGraph(self, graph):
        assert self.graphCreated is False
        scope = 'EMI/input-pipeline/'
        self.X = graph.get_tensor_by_name(scope + "inpX:0")
        self.Y = graph.get_tensor_by_name(scope + "inpY:0")
        self.batchSize = graph.get_tensor_by_name(scope + "batch-size:0")
        self.numEpochs = graph.get_tensor_by_name(scope + "num-epochs:0")
        self.dataset_init = graph.get_operation_by_name(scope + "dataset-init")
        self.x_batch = graph.get_collection('next-x-batch')
        self.y_batch = graph.get_collection('next-y-batch')
        msg = 'More than one tensor named next-x-batch/next-y-batch. '
        msg += 'Are you not resetting your graph?'
        assert len(self.x_batch) == 1, msg
        assert len(self.y_batch) == 1, msg
        self.x_batch = self.x_batch[0]
        self.y_batch = self.y_batch[0]
        self.graphCreated = True

    def __call__(self):
        '''
        The call method performs graph construction either by
        creating a new graph or, if a restored meta graph is provided, by
        restoring operators from this meta graph.

        returns iterators (x_batch, y_batch)
        '''
        if self.graphCreated is True:
            return self.x_batch, self.y_batch
        if self.graph is None:
            self._createGraph()
        else:
            self._restoreGraph(self.graph)
        assert self.graphCreated is True
        return self.x_batch, self.y_batch

    def restoreFromGraph(self, graph, *args, **kwargs):
        '''
        This method provides an alternate way of restoring
        from a saved meta graph - without having to provide the restored meta
        graph as a parameter to __init__. This is useful when, in between
        training, you want to reset the entire computation graph and reload a
        new meta graph from disk. This method allows you to attach to this
        newly loaded meta graph without having to create a new EMI_DataPipeline
        object. Use this method only when you want to clear/reset the existing
        computational graph.
        '''
        self.graphCreated = False
        self.graph = graph
        self._restoreGraph(graph)
        assert self.graphCreated is True

    def runInitializer(self, sess, x_data, y_data, batchSize, numEpochs):
        '''
        This method is used to ingest data by the dataset API. Call this method
        with the data matrices after the graph has been initialized.

        x_data, y_data, batchSize: Self explanatory.
        numEpochs: The Tensorflow dataset API implements iteration over epochs
            by appending the data to itself numEpochs times and then iterating
            over the resulting data as if it was a single data set.
        '''
        assert self.graphCreated is True
        msg = 'X shape should be [-1, numSubinstance, numTimesteps, numFeats]'
        assert x_data.ndim == 4, msg
        assert x_data.shape[1] == self.numSubinstance, msg
        assert x_data.shape[2] == self.numTimesteps, msg
        assert x_data.shape[3] == self.numFeats, msg
        msg = 'X and Y sould have same first dimension'
        assert y_data.shape[0] == x_data.shape[0], msg
        msg = 'Y shape should be [-1, numSubinstance, numOutput]'
        assert y_data.shape[1] == self.numSubinstance, msg
        assert y_data.shape[2] == self.numOutput, msg
        feed_dict = {
            self.X: x_data,
            self.Y: y_data,
            self.batchSize: batchSize,
            self.numEpochs: numEpochs
        }
        assert self.dataset_init is not None, 'Internal error!'
        sess.run(self.dataset_init, feed_dict=feed_dict)


class EMI_RNN():

    def __init__(self, *args, **kwargs):
        """
        Abstract base class for RNN architectures compatible with EMI-RNN.
        This class is extended by specific architectures like LSTM/GRU/FastGRNN
        etc.

        Note: We are not using the PEP recommended abc module since it is
        difficult to support in both python 2 and 3 """
        self.graphCreated = False
        # Model specific matrices, parameter should be saved
        self.graph = None
        self.varList = []
        self.output = None
        self.assignOps = []
        raise NotImplementedError("This is intended to act similar to an " +
                                  "abstract class. Instantiating is not " +
                                  "allowed.")

    def __call__(self, x_batch, **kwargs):
        '''
        The call method performs graph construction either by
        creating a new graph or, if a restored meta graph is provided, by
        restoring operators from this meta graph.

        x_batch: Dataset API iterators to the data.

        returns forward computation output tensor
        '''
        if self.graphCreated is True:
            assert self.output is not None
            return self.output
        if self.graph is None:
            output = self._createBaseGraph(x_batch, **kwargs)
            assert self.graphCreated is False
            self._createExtendedGraph(output, **kwargs)
        else:
            self._restoreBaseGraph(self.graph, **kwargs)
            assert self.graphCreated is False
            self._restoreExtendedGraph(self.graph, **kwargs)
        assert self.graphCreated is True
        return self.output

    def restoreFromGraph(self, graph, **kwargs):
        '''
        This method provides an alternate way of restoring
        from a saved meta graph - without having to provide the restored meta
        graph as a parameter to __init__. This is useful when, in between
        training, you want to reset the entire computation graph and reload a
        new meta graph from disk. This method allows you to attach to this
        newly loaded meta graph without having to create a new EMI_DataPipeline
        object. Use this method only when you want to clear/reset the existing
        computational graph.
        '''
        self.graphCreated = False
        self.varList = []
        self.output = None
        self.assignOps = []
        self.graph = graph
        self._restoreBaseGraph(self.graph, **kwargs)
        assert self.graphCreated is False
        self._restoreExtendedGraph(self.graph, **kwargs)
        assert self.graphCreated is True

    def getModelParams(self):
        raise NotImplementedError("Subclass does not implement this method")

    def _createBaseGraph(self, x_batch, **kwargs):
        raise NotImplementedError("Subclass does not implement this method")

    def _createExtendedGraph(self, baseOutput, **kwargs):
        raise NotImplementedError("Subclass does not implement this method")

    def _restoreBaseGraph(self, graph, **kwargs):
        raise NotImplementedError("Subclass does not implement this method")

    def _restoreExtendedGraph(self, graph, **kwargs):
        raise NotImplementedError("Subclass does not implement this method")

    def addBaseAssignOps(self, graph, initVarList, **kwargs):
        raise NotImplementedError("Subclass does not implement this method")

    def addExtendedAssignOps(self, graph, **kwargs):
        raise NotImplementedError("Subclass does not implement this method")


class EMI_BasicLSTM(EMI_RNN):

    def __init__(self, numSubinstance, numHidden, numTimeSteps,
                 numFeats, graph=None, forgetBias=1.0, useDropout=False):
        '''
        EMI-RNN using LSTM cell. The architecture consists of a single LSTM
        layer followed by a secondary classifier. The secondary classifier is
        not defined as part of this module and is left for the user to define,
        through the redefinition of the '_createExtendedGraph' and
        '_restoreExtendedGraph' methods.

        This class supports restoring from a meta-graph. Provide the restored
        graph as an argument to the graph keyword to enable this behaviour.

        numSubinstance: Number of sub-instance.
        numHidden: The dimension of the hidden state.
        numTimeSteps: The number of time steps of the RNN.
        numFeats: The feature vector dimension for each time step.
        graph: A restored metagraph. Provide a graph if restoring form a meta
            graph is required.
        forgetBias: Bias for the forget gate of the LSTM.
        useDropout: Set to True if a dropout layer is to be added between
            inputs and outputs to the LSTM.
        '''
        self.numHidden = numHidden
        self.numTimeSteps = numTimeSteps
        self.numFeats = numFeats
        self.useDropout = useDropout
        self.forgetBias = forgetBias
        self.numSubinstance = numSubinstance
        self.graph = graph
        self.graphCreated = False
        # Restore or initialize
        self.keep_prob = None
        self.varList = []
        self.output = None
        self.assignOps = []
        # Internal
        self._scope = 'EMI/BasicLSTM/'

    def _createBaseGraph(self, X, **kwargs):
        assert self.graphCreated is False
        msg = 'X should be of form [-1, numSubinstance, numTimeSteps, numFeatures]'
        assert X.get_shape().ndims == 4, msg
        assert X.shape[1] == self.numSubinstance
        assert X.shape[2] == self.numTimeSteps
        assert X.shape[3] == self.numFeats
        # Reshape into 3D such that the first dimension is -1 * numSubinstance
        # where each numSubinstance segment corresponds to one bag
        # then shape it back in into 4D
        scope = self._scope
        keep_prob = None
        with tf.name_scope(scope):
            x = tf.reshape(X, [-1, self.numTimeSteps, self.numFeats])
            x = tf.unstack(x, num=self.numTimeSteps, axis=1)
            # Get the LSTM output
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.numHidden,
                                                forget_bias=self.forgetBias,
                                                name='EMI-LSTM-Cell')
            wrapped_cell = cell
            if self.useDropout is True:
                keep_prob = tf.placeholder(dtype=tf.float32, name='keep-prob')
                wrapped_cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                             input_keep_prob=keep_prob,
                                                             output_keep_prob=keep_prob)
            outputs__, states = tf.nn.static_rnn(
                wrapped_cell, x, dtype=tf.float32)
            outputs = []
            for output in outputs__:
                outputs.append(tf.expand_dims(output, axis=1))
            # Convert back to bag form
            outputs = tf.concat(outputs, axis=1, name='concat-output')
            dims = [-1, self.numSubinstance, self.numTimeSteps, self.numHidden]
            output = tf.reshape(outputs, dims, name='bag-output')

        LSTMVars = cell.variables
        self.varList.extend(LSTMVars)
        if self.useDropout:
            self.keep_prob = keep_prob
        self.output = output
        return self.output

    def _restoreBaseGraph(self, graph, **kwargs):
        assert self.graphCreated is False
        assert self.graph is not None
        scope = self._scope
        if self.useDropout:
            self.keep_prob = graph.get_tensor_by_name(scope + 'keep-prob:0')
        self.output = graph.get_tensor_by_name(scope + 'bag-output:0')
        kernel = graph.get_tensor_by_name("rnn/EMI-LSTM-Cell/kernel:0")
        bias = graph.get_tensor_by_name("rnn/EMI-LSTM-Cell/bias:0")
        assert len(self.varList) is 0
        self.varList = [kernel, bias]

    def getModelParams(self):
        '''
        Returns the LSTM kernel and bias tensors.
        returns [kernel, bias]
        '''
        assert self.graphCreated is True, "Graph is not created"
        assert len(self.varList) == 2
        return self.varList

    def addBaseAssignOps(self, graph, initVarList, **kwargs):
        '''
        Adds Tensorflow assignment operations to all of the model tensors.
        These operations can then be used to initialize these tensors from
        numpy matrices by running these operators

        initVarList: A list of numpy matrices that will be used for
            initialization by the assignment operation. For EMI_BasicLSTM, this
            should be [kernel, bias] matrices.
        '''
        assert initVarList is not None
        assert len(initVarList) == 2
        k_ = graph.get_tensor_by_name('rnn/EMI-LSTM-Cell/kernel:0')
        b_ = graph.get_tensor_by_name('rnn/EMI-LSTM-Cell/bias:0')
        kernel, bias = initVarList[-2], initVarList[-1]
        k_op = tf.assign(k_, kernel)
        b_op = tf.assign(b_, bias)
        self.assignOps.extend([k_op, b_op])


class EMI_GRU(EMI_RNN):

    def __init__(self, numSubinstance, numHidden, numTimeSteps,
                 numFeats, graph=None, useDropout=False):
        '''
        EMI-RNN using GRU cell. The architecture consists of a single GRU
        layer followed by a secondary classifier. The secondary classifier is
        not defined as part of this module and is left for the user to define,
        through the redefinition of the '_createExtendedGraph' and
        '_restoreExtendedGraph' methods.

        This class supports restoring from a meta-graph. Provide the restored
        graph as value to the graph keyword to enable this behaviour.

        numSubinstance: Number of sub-instance.
        numHidden: The dimension of the hidden state.
        numTimeSteps: The number of time steps of the RNN.
        numFeats: The feature vector dimension for each time step.
        graph: A restored metagraph. Provide a graph if restoring form a meta
            graph is required.
        useDropout: Set to True if a dropout layer is to be added between
            inputs and outputs to the RNN.
        '''
        self.numHidden = numHidden
        self.numTimeSteps = numTimeSteps
        self.numFeats = numFeats
        self.useDropout = useDropout
        self.numSubinstance = numSubinstance
        self.graph = graph
        self.graphCreated = False
        # Restore or initialize
        self.keep_prob = None
        self.varList = []
        self.output = None
        self.assignOps = []
        # Internal
        self._scope = 'EMI/GRU/'

    def _createBaseGraph(self, X, **kwargs):
        assert self.graphCreated is False
        msg = 'X should be of form [-1, numSubinstance, numTimeSteps, numFeatures]'
        assert X.get_shape().ndims == 4, msg
        assert X.shape[1] == self.numSubinstance
        assert X.shape[2] == self.numTimeSteps
        assert X.shape[3] == self.numFeats
        # Reshape into 3D suself.h that the first dimension is -1 * numSubinstance
        # where each numSubinstance segment corresponds to one bag
        # then shape it back in into 4D
        scope = self._scope
        keep_prob = None
        with tf.name_scope(scope):
            x = tf.reshape(X, [-1, self.numTimeSteps, self.numFeats])
            x = tf.unstack(x, num=self.numTimeSteps, axis=1)
            # Get the GRU output
            cell = tf.nn.rnn_cell.GRUCell(self.numHidden, name='EMI-GRU-Cell')
            wrapped_cell = cell
            if self.useDropout is True:
                keep_prob = tf.placeholder(dtype=tf.float32, name='keep-prob')
                wrapped_cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                             input_keep_prob=keep_prob,
                                                             output_keep_prob=keep_prob)
            outputs__, states = tf.nn.static_rnn(
                wrapped_cell, x, dtype=tf.float32)
            outputs = []
            for output in outputs__:
                outputs.append(tf.expand_dims(output, axis=1))
            # Convert back to bag form
            outputs = tf.concat(outputs, axis=1, name='concat-output')
            dims = [-1, self.numSubinstance, self.numTimeSteps, self.numHidden]
            output = tf.reshape(outputs, dims, name='bag-output')

        GRUVars = cell.variables
        self.varList.extend(GRUVars)
        if self.useDropout:
            self.keep_prob = keep_prob
        self.output = output
        return self.output

    def _restoreBaseGraph(self, graph, **kwargs):
        assert self.graphCreated is False
        assert self.graph is not None
        scope = self._scope
        if self.useDropout:
            self.keep_prob = graph.get_tensor_by_name(scope + 'keep-prob:0')
        self.output = graph.get_tensor_by_name(scope + 'bag-output:0')
        kernel1 = graph.get_tensor_by_name("rnn/EMI-GRU-Cell/gates/kernel:0")
        bias1 = graph.get_tensor_by_name("rnn/EMI-GRU-Cell/gates/bias:0")
        kernel2 = graph.get_tensor_by_name(
            "rnn/EMI-GRU-Cell/candidate/kernel:0")
        bias2 = graph.get_tensor_by_name("rnn/EMI-GRU-Cell/candidate/bias:0")
        assert len(self.varList) is 0
        self.varList = [kernel1, bias1, kernel2, bias2]

    def getModelParams(self):
        '''
        Returns the GRU kernel and bias tensors.
        returns [kernel1, bias1, kernel2, bias2]
        '''
        assert self.graphCreated is True, "Graph is not created"
        assert len(self.varList) == 4
        return self.varList

    def addBaseAssignOps(self, graph, initVarList, **kwargs):
        '''
        Adds Tensorflow assignment operations to all of the model tensors.
        These operations can then be used to initialize these tensors from
        numpy matrices by running these operators

        initVarList: A list of numpy matrices that will be used for
            initialization by the assignment operation. For EMI_GRU, this
            should be list of numpy matrices corresponding to  [kernel1, bias1,
            kernel2, bias2]
        '''
        assert initVarList is not None
        assert len(initVarList) == 2
        kernel1_ = graph.get_tensor_by_name("rnn/EMI-GRU-Cell/gates/kernel:0")
        bias1_ = graph.get_tensor_by_name("rnn/EMI-GRU-Cell/gates/bias:0")
        kernel2_ = graph.get_tensor_by_name(
            "rnn/EMI-GRU-Cell/candidate/kernel:0")
        bias2_ = graph.get_tensor_by_name("rnn/EMI-GRU-Cell/candidate/bias:0")
        kernel1, bias1, kernel2, bias2 = initVarList[
            0], initVarList[1], initVarList[2], initVarList[3]
        kernel1_op = tf.assign(kernel1_, kernel1)
        bias1_op = tf.assign(bias1_, bias1)
        kernel2_op = tf.assign(kernel2_, kernel2)
        bias2_op = tf.assign(bias2_, bias2)
        self.assignOps.extend([kernel1_op, bias1_op, kernel2_op, bias2_op])


class EMI_FastRNN(EMI_RNN):

    def __init__(self, numSubinstance, numHidden, numTimeSteps,
                 numFeats, graph=None, useDropout=False,
                 update_non_linearity="tanh", wRank=None,
                 uRank=None, alphaInit=-3.0, betaInit=3.0):
        '''
        EMI-RNN using FastRNN cell. The architecture consists of a single
        FastRNN layer followed by a secondary classifier. The secondary
        classifier is not defined as part of this module and is left for the
        user to define, through the redefinition of the '_createExtendedGraph'
        and '_restoreExtendedGraph' methods.

        This class supports restoring from a meta-graph. Provide the restored
        graph as value to the graph keyword to enable this behaviour.

        numSubinstance: Number of sub-instance.
        numHidden: The dimension of the hidden state.
        numTimeSteps: The number of time steps of the RNN.
        numFeats: The feature vector dimension for each time step.
        graph: A restored metagraph. Provide a graph if restoring form a meta
            graph is required.
        useDropout: Set to True if a dropout layer is to be added
            between inputs and outputs to the RNN.
        update_non_linearity, wRank, uRank, _alphaInit, betaInit:
            These are FastRNN parameters. Please refer to FastRNN documentation
            for more information.
        '''
        self.numHidden = numHidden
        self.numTimeSteps = numTimeSteps
        self.numFeats = numFeats
        self.useDropout = useDropout
        self.numSubinstance = numSubinstance
        self.graph = graph
        self.update_non_linearity = update_non_linearity
        self.wRank = wRank
        self.uRank = uRank
        self.alphaInit = alphaInit
        self.betaInit = betaInit
        self.graphCreated = False
        # Restore or initialize
        self.keep_prob = None
        self.varList = []
        self.output = None
        self.assignOps = []
        # Internal
        self._scope = 'EMI/FastRNN/'

    def _createBaseGraph(self, X, **kwargs):
        assert self.graphCreated is False
        msg = 'X should be of form [-1, numSubinstance, numTimeSteps,'
        msg += ' numFeatures]'
        assert X.get_shape().ndims == 4, msg
        assert X.shape[1] == self.numSubinstance
        assert X.shape[2] == self.numTimeSteps
        assert X.shape[3] == self.numFeats
        # Reshape into 3D suself.h that the first dimension is -1 *
        # numSubinstance where each numSubinstance segment corresponds to one
        # bag then shape it back in into 4D
        scope = self._scope
        keep_prob = None
        with tf.name_scope(scope):
            x = tf.reshape(X, [-1, self.numTimeSteps, self.numFeats])
            x = tf.unstack(x, num=self.numTimeSteps, axis=1)
            # Get the FastRNN output
            cell = FastRNNCell(self.numHidden, self.update_non_linearity,
                               self.wRank, self.uRank, self.alphaInit,
                               self.betaInit, name='EMI-FastRNN-Cell')
            wrapped_cell = cell
            if self.useDropout is True:
                keep_prob = tf.placeholder(dtype=tf.float32, name='keep-prob')
                wrapped_cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                             input_keep_prob=keep_prob,
                                                             output_keep_prob=keep_prob)
            outputs__, states = tf.nn.static_rnn(wrapped_cell, x,
                                                 dtype=tf.float32)
            outputs = []
            for output in outputs__:
                outputs.append(tf.expand_dims(output, axis=1))
            # Convert back to bag form
            outputs = tf.concat(outputs, axis=1, name='concat-output')
            dims = [-1, self.numSubinstance, self.numTimeSteps, self.numHidden]
            output = tf.reshape(outputs, dims, name='bag-output')

        FastRNNVars = cell.variables
        self.varList.extend(FastRNNVars)
        if self.useDropout:
            self.keep_prob = keep_prob
        self.output = output
        return self.output

    def _restoreBaseGraph(self, graph, **kwargs):
        assert self.graphCreated is False
        assert self.graph is not None
        scope = self._scope
        if self.useDropout:
            self.keep_prob = graph.get_tensor_by_name(scope + 'keep-prob:0')
        self.output = graph.get_tensor_by_name(scope + 'bag-output:0')

        assert len(self.varList) is 0
        if self.wRank is None:
            W = graph.get_tensor_by_name(
                "rnn/fast_rnn_cell/EMI-FastRNN-Cell/FastRNNcell/W:0")
            self.varList = [W]
        else:
            W1 = graph.get_tensor_by_name(
                "rnn/fast_rnn_cell/EMI-FastRNN-Cell/FastRNNcell/W1:0")
            W2 = graph.get_tensor_by_name(
                "rnn/fast_rnn_cell/EMI-FastRNN-Cell/FastRNNcell/W2:0")
            self.varList = [W1, W2]

        if self.uRank is None:
            U = graph.get_tensor_by_name(
                "rnn/fast_rnn_cell/EMI-FastRNN-Cell/FastRNNcell/U:0")
            self.varList.extend([U])
        else:
            U1 = graph.get_tensor_by_name(
                "rnn/fast_rnn_cell/EMI-FastRNN-Cell/FastRNNcell/U1:0")
            U2 = graph.get_tensor_by_name(
                "rnn/fast_rnn_cell/EMI-FastRNN-Cell/FastRNNcell/U2:0")
            self.varList.extend([U1, U2])

        alpha = graph.get_tensor_by_name(
            "rnn/fast_rnn_cell/EMI-FastRNN-Cell/FastRNNcell/alpha:0")
        beta = graph.get_tensor_by_name(
            "rnn/fast_rnn_cell/EMI-FastRNN-Cell/FastRNNcell/beta:0")
        bias = graph.get_tensor_by_name(
            "rnn/fast_rnn_cell/EMI-FastRNN-Cell/FastRNNcell/B_h:0")
        self.varList.extend([alpha, beta, bias])

    def getModelParams(self):
        '''
        Returns the FastRNN model tensors.
        In the order of  [W(W1, W2), U(U1,U2), alpha, beta, B_h]
        () implies that the matrix can be replaced with the matrices inside.
        '''
        assert self.graphCreated is True, "Graph is not created"
        return self.varList

    def addBaseAssignOps(self, graph, initVarList, **kwargs):
        '''
        Adds Tensorflow assignment operations to all of the model tensors.
        These operations can then be used to initialize these tensors from
        numpy matrices by running these operators

        initVarList: A list of numpy matrices that will be used for
            initialization by the assignment operation. For EMI_FastRNN, this
            should be list of numpy matrices corresponding to  [W(W1, W2),
            U(U1,U2), alpha, beta, B_h]
        '''
        assert initVarList is not None
        index = 0
        if self.wRank is None:
            W_ = graph.get_tensor_by_name(
                "rnn/fast_rnn_cell/EMI-FastRNN-Cell/FastRNNcell/W:0")
            W = initVarList[0]
            w_op = tf.assign(W_, W)
            self.assignOps.extend([w_op])
            index += 1
        else:
            W1_ = graph.get_tensor_by_name(
                "rnn/fast_rnn_cell/EMI-FastRNN-Cell/FastRNNcell/W1:0")
            W2_ = graph.get_tensor_by_name(
                "rnn/fast_rnn_cell/EMI-FastRNN-Cell/FastRNNcell/W2:0")
            W1, W2 = initVarList[0], initVarList[1]
            w1_op = tf.assign(W1_, W1)
            w2_op = tf.assign(W2_, W2)
            self.assignOps.extend([w1_op, w2_op])
            index += 2

        if self.uRank is None:
            U_ = graph.get_tensor_by_name(
                "rnn/fast_rnn_cell/EMI-FastRNN-Cell/FastRNNcell/U:0")
            U = initVarList[index]
            u_op = tf.assign(U_, U)
            self.assignOps.extend([u_op])
            index += 1
        else:
            U1_ = graph.get_tensor_by_name(
                "rnn/fast_rnn_cell/EMI-FastRNN-Cell/FastRNNcell/U1:0")
            U2_ = graph.get_tensor_by_name(
                "rnn/fast_rnn_cell/EMI-FastRNN-Cell/FastRNNcell/U2:0")
            U1, U2 = initVarList[index], initVarList[index + 1]
            u1_op = tf.assign(U1_, U1)
            u2_op = tf.assign(U2_, U2)
            self.assignOps.extend([u1_op, u2_op])
            index += 2

        alpha_ = graph.get_tensor_by_name(
            "rnn/fast_rnn_cell/EMI-FastRNN-Cell/FastRNNcell/alpha:0")
        beta_ = graph.get_tensor_by_name(
            "rnn/fast_rnn_cell/EMI-FastRNN-Cell/FastRNNcell/beta:0")
        bias_ = graph.get_tensor_by_name(
            "rnn/fast_rnn_cell/EMI-FastRNN-Cell/FastRNNcell/B_h:0")

        alpha, beta, bias = initVarList[index], initVarList[
            index + 1], initVarList[index + 2]
        alpha_op = tf.assign(alpha_, alpha)
        beta_op = tf.assign(beta_, beta)
        bias_op = tf.assign(bias_, bias)

        self.assignOps.extend([alpha_op, beta_op, bias_op])


class EMI_UGRNN(EMI_RNN):

    def __init__(self, numSubinstance, numHidden, numTimeSteps,
                 numFeats, graph=None, forgetBias=1.0, useDropout=False):
        '''
        EMI-RNN using UGRNN cell. The architecture consists of a single UGRNN
        layer followed by a secondary classifier. The secondary classifier is
        not defined as part of this module and is left for the user to define,
        through the redefinition of the '_createExtendedGraph' and
        '_restoreExtendedGraph' methods.

        This class supports restoring from a meta-graph. Provide the restored
        graph as value to the graph keyword to enable this behaviour.

        numSubinstance: Number of sub-instance.
        numHidden: The dimension of the hidden state.
        numTimeSteps: The number of time steps of the RNN.
        numFeats: The feature vector dimension for each time step.
        graph: A restored metagraph. Provide a graph if restoring form a meta
            graph is required.
        forgetBias: Bias for the forget gate of the UGRNN.
        useDropout: Set to True if a dropout layer is to be added between
            inputs and outputs to the RNN.
        '''
        self.numHidden = numHidden
        self.numTimeSteps = numTimeSteps
        self.numFeats = numFeats
        self.useDropout = useDropout
        self.forgetBias = forgetBias
        self.numSubinstance = numSubinstance
        self.graph = graph
        self.graphCreated = False
        # Restore or initialize
        self.keep_prob = None
        self.varList = []
        self.output = None
        self.assignOps = []
        # Internal
        self._scope = 'EMI/UGRNN/'

    def _createBaseGraph(self, X, **kwargs):
        assert self.graphCreated is False
        msg = 'X should be of form [-1, numSubinstance, numTimeSteps, numFeatures]'
        assert X.get_shape().ndims == 4, msg
        assert X.shape[1] == self.numSubinstance
        assert X.shape[2] == self.numTimeSteps
        assert X.shape[3] == self.numFeats
        # Reshape into 3D such that the first dimension is -1 * numSubinstance
        # where each numSubinstance segment corresponds to one bag
        # then shape it back in into 4D
        scope = self._scope
        keep_prob = None
        with tf.name_scope(scope):
            x = tf.reshape(X, [-1, self.numTimeSteps, self.numFeats])
            x = tf.unstack(x, num=self.numTimeSteps, axis=1)
            # Get the UGRNN output
            cell = tf.contrib.rnn.UGRNNCell(self.numHidden,
                                            forget_bias=self.forgetBias)
            wrapped_cell = cell
            if self.useDropout is True:
                keep_prob = tf.placeholder(dtype=tf.float32, name='keep-prob')
                wrapped_cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                             input_keep_prob=keep_prob,
                                                             output_keep_prob=keep_prob)
            outputs__, states = tf.nn.static_rnn(
                wrapped_cell, x, dtype=tf.float32)
            outputs = []
            for output in outputs__:
                outputs.append(tf.expand_dims(output, axis=1))
            # Convert back to bag form
            outputs = tf.concat(outputs, axis=1, name='concat-output')
            dims = [-1, self.numSubinstance, self.numTimeSteps, self.numHidden]
            output = tf.reshape(outputs, dims, name='bag-output')

        UGRNNVars = cell.variables
        self.varList.extend(UGRNNVars)
        if self.useDropout:
            self.keep_prob = keep_prob
        self.output = output
        return self.output

    def _restoreBaseGraph(self, graph, **kwargs):
        assert self.graphCreated is False
        assert self.graph is not None
        scope = self._scope
        if self.useDropout:
            self.keep_prob = graph.get_tensor_by_name(scope + 'keep-prob:0')
        self.output = graph.get_tensor_by_name(scope + 'bag-output:0')
        kernel = graph.get_tensor_by_name("rnn/ugrnn_cell/kernel:0")
        bias = graph.get_tensor_by_name("rnn/ugrnn_cell/bias:0")
        assert len(self.varList) is 0
        self.varList = [kernel, bias]

    def getModelParams(self):
        '''
        Returns the FastRRNN model tensors.
        returns [kernel, bias]
        '''
        assert self.graphCreated is True, "Graph is not created"
        assert len(self.varList) == 2
        return self.varList

    def addBaseAssignOps(self, graph, initVarList, **kwargs):
        '''
        Adds Tensorflow assignment operations to all of the model tensors.
        These operations can then be used to initialize these tensors from
        numpy matrices by running these operators

        initVarList: A list of numpy matrices that will be used for
            initialization by the assignment operation. For EMI_UGRNN, this
            should be list of numpy matrices corresponding to  [kernel, bias]
        '''
        assert initVarList is not None
        assert len(initVarList) == 2
        k_ = graph.get_tensor_by_name('rnn/ugrnn_cell/kernel:0')
        b_ = graph.get_tensor_by_name('rnn/ugrnn_cell/bias:0')
        kernel, bias = initVarList[-2], initVarList[-1]
        k_op = tf.assign(k_, kernel)
        b_op = tf.assign(b_, bias)
        self.assignOps.extend([k_op, b_op])


class EMI_FastGRNN(EMI_RNN):

    def __init__(self, numSubinstance, numHidden, numTimeSteps, numFeats,
                 graph=None, useDropout=False, gate_non_linearity="sigmoid",
                 update_non_linearity="tanh", wRank=None, uRank=None,
                 zetaInit=1.0, nuInit=-4.0):
        '''
        EMI-RNN using FastGRNN cell. The architecture consists of a single
        FastGRNN layer followed by a secondary classifier. The secondary
        classifier is not defined as part of this module and is left for the
        user to define, through the redefinition of the '_createExtendedGraph'
        and '_restoreExtendedGraph' methods.

        This class supports restoring from a meta-graph. Provide the restored
        graph as value to the graph keyword to enable this behaviour.

        numSubinstance: Number of sub-instance.
        numHidden: The dimension of the hidden state.
        numTimeSteps: The number of time steps of the RNN.
        numFeats: The feature vector dimension for each time step.
        graph: A restored metagraph. Provide a graph if restoring form a meta
            graph is required.
        useDropout: Set to True if a dropout layer is to be added
            between inputs and outputs to the RNN.

        gate_non_linearity, update_non_linearity, wRank, uRank, zetaInit,
        nuInit:
            These are FastGRNN parameters. Please refer to FastGRNN documentation
            for more information.
        '''
        self.numHidden = numHidden
        self.numTimeSteps = numTimeSteps
        self.numFeats = numFeats
        self.useDropout = useDropout
        self.numSubinstance = numSubinstance
        self.graph = graph

        self.gate_non_linearity = gate_non_linearity
        self.update_non_linearity = update_non_linearity
        self.wRank = wRank
        self.uRank = uRank
        self.zetaInit = zetaInit
        self.nuInit = nuInit

        self.graphCreated = False
        # Restore or initialize
        self.keep_prob = None
        self.varList = []
        self.output = None
        self.assignOps = []
        # Internal
        self._scope = 'EMI/FastGRNN/'

    def _createBaseGraph(self, X, **kwargs):
        assert self.graphCreated is False
        msg = 'X should be of form [-1, numSubinstance, numTimeSteps, numFeatures]'
        assert X.get_shape().ndims == 4, msg
        assert X.shape[1] == self.numSubinstance
        assert X.shape[2] == self.numTimeSteps
        assert X.shape[3] == self.numFeats
        # Reshape into 3D suself.h that the first dimension is -1 * numSubinstance
        # where each numSubinstance segment corresponds to one bag
        # then shape it back in into 4D
        scope = self._scope
        keep_prob = None
        with tf.name_scope(scope):
            x = tf.reshape(X, [-1, self.numTimeSteps, self.numFeats])
            x = tf.unstack(x, num=self.numTimeSteps, axis=1)
            # Get the FastGRNN output
            cell = FastGRNNCell(self.numHidden, self.gate_non_linearity,
                                self.update_non_linearity, self.wRank,
                                self.uRank, self.zetaInit, self.nuInit,
                                name='EMI-FastGRNN-Cell')
            wrapped_cell = cell
            if self.useDropout is True:
                keep_prob = tf.placeholder(dtype=tf.float32, name='keep-prob')
                wrapped_cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                             input_keep_prob=keep_prob,
                                                             output_keep_prob=keep_prob)
            outputs__, states = tf.nn.static_rnn(
                wrapped_cell, x, dtype=tf.float32)
            outputs = []
            for output in outputs__:
                outputs.append(tf.expand_dims(output, axis=1))
            # Convert back to bag form
            outputs = tf.concat(outputs, axis=1, name='concat-output')
            dims = [-1, self.numSubinstance, self.numTimeSteps, self.numHidden]
            output = tf.reshape(outputs, dims, name='bag-output')

        FastGRNNVars = cell.variables
        self.varList.extend(FastGRNNVars)
        if self.useDropout:
            self.keep_prob = keep_prob
        self.output = output
        return self.output

    def _restoreBaseGraph(self, graph, **kwargs):
        assert self.graphCreated is False
        assert self.graph is not None
        scope = self._scope
        if self.useDropout:
            self.keep_prob = graph.get_tensor_by_name(scope + 'keep-prob:0')
        self.output = graph.get_tensor_by_name(scope + 'bag-output:0')

        assert len(self.varList) is 0
        if self.wRank is None:
            W = graph.get_tensor_by_name(
                "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/W:0")
            self.varList = [W]
        else:
            W1 = graph.get_tensor_by_name(
                "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/W1:0")
            W2 = graph.get_tensor_by_name(
                "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/W2:0")
            self.varList = [W1, W2]

        if self.uRank is None:
            U = graph.get_tensor_by_name(
                "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/U:0")
            self.varList.extend([U])
        else:
            U1 = graph.get_tensor_by_name(
                "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/U1:0")
            U2 = graph.get_tensor_by_name(
                "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/U2:0")
            self.varList.extend([U1, U2])

        zeta = graph.get_tensor_by_name(
            "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/zeta:0")
        nu = graph.get_tensor_by_name(
            "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/nu:0")
        gate_bias = graph.get_tensor_by_name(
            "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/B_g:0")
        update_bias = graph.get_tensor_by_name(
            "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/B_h:0")
        self.varList.extend([zeta, nu, gate_bias, update_bias])

    def getModelParams(self):
        '''
        Returns the FastGRNN model tensors.
        In the order of  [W(W1, W2), U(U1,U2), zeta, nu, B_g, B_h]
        () implies that the matrix can be replaced with the matrices inside.
        '''
        assert self.graphCreated is True, "Graph is not created"
        return self.varList

    def addBaseAssignOps(self, graph, initVarList, **kwargs):
        '''
        Adds Tensorflow assignment operations to all of the model tensors.
        These operations can then be used to initialize these tensors from
        numpy matrices by running these operators

        initVarList: A list of numpy matrices that will be used for
            initialization by the assignment operation. For EMI_FastGRNN, this
            should be list of numpy matrices corresponding to  [W(W1, W2),
            U(U1,U2), zeta, nu, B_g, B_h]
        '''
        assert initVarList is not None
        index = 0
        if self.wRank is None:
            W_ = graph.get_tensor_by_name(
                "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/W:0")
            W = initVarList[0]
            w_op = tf.assign(W_, W)
            self.assignOps.extend([w_op])
            index += 1
        else:
            W1_ = graph.get_tensor_by_name(
                "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/W1:0")
            W2_ = graph.get_tensor_by_name(
                "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/W2:0")
            W1, W2 = initVarList[0], initVarList[1]
            w1_op = tf.assign(W1_, W1)
            w2_op = tf.assign(W2_, W2)
            self.assignOps.extend([w1_op, w2_op])
            index += 2

        if self.uRank is None:
            U_ = graph.get_tensor_by_name(
                "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/U:0")
            U = initVarList[index]
            u_op = tf.assign(U_, U)
            self.assignOps.extend([u_op])
            index += 1
        else:
            U1_ = graph.get_tensor_by_name(
                "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/U1:0")
            U2_ = graph.get_tensor_by_name(
                "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/U2:0")
            U1, U2 = initVarList[index], initVarList[index + 1]
            u1_op = tf.assign(U1_, U1)
            u2_op = tf.assign(U2_, U2)
            self.assignOps.extend([u1_op, u2_op])
            index += 2

        zeta_ = graph.get_tensor_by_name(
            "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/zeta:0")
        nu_ = graph.get_tensor_by_name(
            "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/nu:0")
        gate_bias_ = graph.get_tensor_by_name(
            "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/B_g:0")
        update_bias_ = graph.get_tensor_by_name(
            "rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/B_h:0")

        zeta, nu, gate_bias, update_bias = initVarList[index], initVarList[
            index + 1], initVarList[index + 2], initVarList[index + 3]
        zeta_op = tf.assign(zeta_, zeta)
        nu_op = tf.assign(nu_, nu)
        gate_bias_op = tf.assign(gate_bias_, gate_bias)
        update_bias_op = tf.assign(update_bias_, update_bias)

        self.assignOps.extend([zeta_op, nu_op, gate_bias_op, update_bias_op])
