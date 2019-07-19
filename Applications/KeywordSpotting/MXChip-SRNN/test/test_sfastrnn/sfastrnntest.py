import numpy as np

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x).astype(np.float32))

class FastRNN0:
    def __init__(self):
        self.timesteps = 4
        self.featLen = 6
        self.statesLen = 4
        self.alpha = 0.2
        self.beta = 0.8
        W = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
            0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
            0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
        ]
        B = [1, 2, 3, 4,]
        W = np.array(W)
        self.W = np.reshape(W, [self.statesLen, self.statesLen + self.featLen])
        self.B = np.array(B)

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

class FastRNN1:
    def __init__(self):
        self.timesteps = 3
        self.featLen = 4
        self.statesLen = 4
        self.alpha = 0.3
        self.beta = 0.7
        W = [
            0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
            0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        ]
        B = [0.1, 0.2, 0.3, 0.4,]
        W = np.array(W)
        self.W = np.reshape(W, [self.statesLen, self.statesLen + self.featLen])
        self.B = np.array(B)

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



def main():
    fastrnn0 = FastRNN0()
    fastrnn1 = FastRNN1()
    h0 = np.zeros(fastrnn0.statesLen)
    inp1 = []
    xx0 = np.array([
        0.0,0.01,0.02,0.03,0.04,0.05,   0.06,0.07,0.08,0.09,0.1,0.11,
        0.12,0.13,0.14,0.15,0.16,0.17,  0.18,0.19,0.2,0.21,0.22,0.23,
    ])
    xx1 = np.array([
        0.12,0.13,0.14,0.15,0.16,0.17,  0.18,0.19,0.2,0.21,0.22,0.23,
        0.24,0.25,0.26,0.27,0.28,0.29,  0.3,0.31,0.32,0.33,0.34,0.35,
    ])
    xx2 = np.array([
        0.24,0.25,0.26,0.27,0.28,0.29,  0.3,0.31,0.32,0.33,0.34,0.35,
        0.36,0.37,0.38,0.39,0.4,0.41,   0.42,0.43,0.44,0.45,0.46,0.47,
    ])
    xx3 = np.array([
        0.36,0.37,0.38,0.39,0.4,0.41,   0.42,0.43,0.44,0.45,0.46,0.47,
        0.0,0.01,0.02,0.03,0.04,0.05,   0.06,0.07,0.08,0.09,0.1,0.11,
    ])
    xx4 = np.array([
        0.0,0.01,0.02,0.03,0.04,0.05,   0.06,0.07,0.08,0.09,0.1,0.11,
        0.24,0.25,0.26,0.27,0.28,0.29,  0.3,0.31,0.32,0.33,0.34,0.35,
    ])
    xx5 = np.array([
        0.12,0.13,0.14,0.15,0.16,0.17,  0.18,0.19,0.2,0.21,0.22,0.23,
        0.36,0.37,0.38,0.39,0.4,0.41,   0.42,0.43,0.44,0.45,0.46,0.47,
    ])
    h0List = []
    def predict(xx):
        xx = np.reshape(xx, [-1, fastrnn0.featLen])
        h0 = fastrnn0.unroll(xx)
        if len(h0List) > 2:
            del h0List[0]
        h0List.append(h0)
        if len(h0List) <= 2:
            return h0, None
        inp1 = np.array(h0List)
        h1 = fastrnn1.unroll(inp1)
        return h0, h1
    h0, h1 = predict(xx0)
    h0, h1 = predict(xx1)
    h0, h1 = predict(xx2)
    print(h1)
    h0, h1 = predict(xx3)
    print(h1)
    h0, h1 = predict(xx4)
    print(h1)
    h0, h1 = predict(xx5)
    print(h1)


main()


