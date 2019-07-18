import numpy as np
from lstmtemplate import getFile

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x).astype(np.float32))

class LSTM:
    def __init__(self, kernel, bias, forgetBias = 1.0):
        assert bias.ndim == 1
        hidDim = int(len(bias) / 4)
        assert kernel.ndim == 2
        assert kernel.shape[0] == 4 * hidDim
        assert kernel.shape[1] - hidDim > 0
        self.kernel = kernel
        self.bias = bias
        self.forgetBias = forgetBias
        self.hidDim = hidDim
        self.featLen = self.kernel.shape[1] - self.hidDim

    def cell(self, x, h, c):
        '''
        Non batched version for simplicity
        '''
        assert x.ndim == 1
        assert x.shape[0] == self.featLen
        h_ = h.copy()
        x_ = np.concatenate([x, h_], axis=0)
        combOut = np.matmul(self.kernel, x_)
        combOut = combOut + self.bias
        i, j, f, o = np.split(combOut, 4, axis=0)
        new_c = c * sigmoid(f + self.forgetBias) + sigmoid(i) * np.tanh(j)
        new_h = np.tanh(new_c) * sigmoid(o)
        new_o = sigmoid(o)
        c = new_c
        h = new_h
        o = new_o
        return h, c


def main():
    inputDim = 32
    hiddenDim = 32
    # [hid + feat, 4 * hid]
    kernel =  np.random.normal(size=(4 * hiddenDim, hiddenDim + inputDim))
    bias = np.random.normal(size=hiddenDim * 4)
    x =  np.random.normal(size=inputDim)
    h0, c0 = np.zeros(hiddenDim), np.zeros(hiddenDim)

    lstm = LSTM(kernel, bias)
    h_final, c = lstm.cell(x, h0, c0)
    print(getFile(inputDim, kernel, bias, x, h_final))

main()