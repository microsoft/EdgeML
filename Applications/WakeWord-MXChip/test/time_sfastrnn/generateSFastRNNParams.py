import numpy as np
from template import getTemplate as getTemplateC
from template_MXChip import getTemplate as getTemplateMXChip

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x).astype(np.float32))

class FastRNN:
    def __init__(self, W, B, alpha, beta, timeSteps, featLen, statesLen):
        self.timeSteps = timeSteps
        self.featLen = featLen
        self.statesLen = statesLen
        self.alpha = alpha
        self.beta =  beta
        self.W = W
        self.B = B

    def cell(self, x, h):
        W = self.W
        B = self.B
        hx = np.concatenate([h, x])
        hcomb = np.matmul(W, hx)
        h_ = hcomb  + B
        h_ = sigmoid(h_)
        h_ = self.alpha * h_ + self.beta * h
        return h_

    def unroll(self, x_list):
        h = np.zeros(self.statesLen)
        for x in x_list:
            h = self.cell(x, h)
        return h


def main(device='C'):
    assert device in ['C', 'MXChip']
    inputDim = 32
    hiddenDim0 = 32
    timeSteps0 = 8
    hiddenDim1 = 16
    timeSteps1 = 8
    alpha0, beta0 = 0.2, 0.8
    alpha1, beta1 = 0.1, 0.9
    W0 = np.random.normal(size=[hiddenDim0, hiddenDim0 +  inputDim])
    B0 = np.random.normal(size=hiddenDim0)
    W1 = np.random.normal(size=[hiddenDim1, hiddenDim1 + hiddenDim0])
    B1 = np.random.normal(size=hiddenDim1)
    x0 = np.random.normal(size=[timeSteps0, inputDim])
    x1 = np.random.normal(size=[timeSteps0, inputDim])
    fastrnn0 = FastRNN(W0, B0, alpha0, beta0, timeSteps0, inputDim, hiddenDim0)
    fastrnn1 = FastRNN(W1, B1, alpha1, beta1, timeSteps1, hiddenDim0, hiddenDim1)

    hList = []
    for i in range(timeSteps1):
        h = fastrnn0.unroll(x0)
        hList.append(h)
    inp = np.array(hList)
    assert inp.shape[0] == timeSteps1
    assert inp.shape[1] == hiddenDim0
    hout0 = fastrnn1.unroll(inp)

    h = fastrnn0.unroll(x1)
    hList.append(h)
    inp = np.array(hList[-timeSteps1:])
    assert inp.shape[0] == timeSteps1
    assert inp.shape[1] == hiddenDim0
    hout1 = fastrnn1.unroll(inp)

    h = fastrnn0.unroll(x1)
    hList.append(h)
    inp = np.array(hList[-timeSteps1:])
    assert inp.shape[0] == timeSteps1
    assert inp.shape[1] == hiddenDim0
    hout2 = fastrnn1.unroll(inp)

    expected = ''
    for tx in hout0:
        expected += '%f, ' % tx
    expected += '\\n'
    for tx in hout1:
        expected += '%f, ' % tx
    expected += '\\n'
    for tx in hout2:
        expected += '%f, ' % tx
    expected += '\\n'
    if device == 'C':
        ret = getTemplateC(W0, B0, alpha0, beta0, W1, B1, alpha1, beta1,
                    timeSteps0, timeSteps1, x0, x1, expected)
    elif device == 'MXChip':
        ret = getTemplateMXChip(W0, B0, alpha0, beta0, W1, B1, alpha1, beta1,
                    timeSteps0, timeSteps1, x0, x1, expected)
    print(ret)




device = 'MXChip'
main(device)
