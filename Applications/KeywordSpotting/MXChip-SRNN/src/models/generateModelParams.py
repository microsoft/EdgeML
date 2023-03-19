import numpy as np
from template_MXChip import getTemplate as getTemplateMXChip
import python_speech_features as sp


LABELMAP13 = {
    'go': 1, 'no': 2, 'on': 3, 'up': 4, 'bed': 5, 'cat': 6,
    'dog': 7, 'off': 8, 'one': 9, 'six': 10, 'two': 11,
    'yes': 12,
    'wow': 0, 'bird': 0, 'down': 0, 'five': 0, 'four': 0,
    'left': 0, 'nine': 0, 'stop': 0, 'tree': 0, 'zero': 0,
    'eight': 0, 'happy': 0, 'house': 0, 'right': 0, 'seven': 0,
    'three': 0, 'marvin': 0, 'sheila': 0, '_background_noise_': 0
}

np.random.seed(42)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x).astype(np.float32))

class FastRNN:
    def __init__(self, W, U, b, alpha, beta, statesLen):
        self.statesLen = statesLen
        self.alpha = alpha
        self.beta =  beta
        # Wx + Uh + b
        self.W, self.U = W, U
        self.Wfused = np.concatenate([U, W], axis=1)
        self.b = b

    def cell(self, x, h):
        Wfused = self.Wfused
        b = self.b
        hx = np.concatenate([h, x])
        h_= np.squeeze(np.matmul(Wfused, hx))
        h_ = np.squeeze(h_) + b
        h_ = sigmoid(h_)
        h_ = self.alpha * h_ + self.beta * h
        return h_

    def unroll(self, x_list):
        h = np.zeros(self.statesLen)
        for x in x_list:
            h = self.cell(x, h)
        return h

class Stacked2LayerRNN:
    def __init__(self, rnn0, rnn1, fcW, fcB):
        self.rnn0 = rnn0
        self.rnn1 = rnn1
        self.fcW = fcW
        self.fcB = fcB
    
    def infer(self, x):
        '''
        x: [NUM_BRICKS, NUM_BRICK_TIMESTEPS, NUM_INPUT]
        '''
        rnn0, rnn1 = self.rnn0, self.rnn1
        assert x.ndim == 3
        h_list = []
        for brick in x:
            h = rnn0.unroll(brick)
            h_list.append(h)
        h_final = rnn1.unroll(h_list)
        predictions = np.matmul(self.fcW, h_final) + self.fcB
        return predictions


def main():
    # Configuration
    # -------------
    paramsDir = './params/'
    timeSteps0 = 8
    timeSteps1 = 6
    # -------------
    # Load the model and initialize two fast RNN cells
    x_sample = np.load(paramsDir + 'x_sample.npy')
    target = np.load(paramsDir + 'target_sample.npy')
    predictions = np.load(paramsDir + 'predicted_sample.npy')
    mean = np.load(paramsDir + 'mean.npy')
    std = np.load(paramsDir + 'std.npy')

    fcW = np.load(paramsDir + 'fcW.npy').T
    fcB = np.load(paramsDir + 'fcB.npy') 
    W0 = np.load(paramsDir + 'W0.npy').T
    U0 = np.load(paramsDir + 'U0.npy').T
    b0 = np.squeeze(np.load(paramsDir + 'h0.npy'))
    alpha0 = sigmoid(np.squeeze(np.load(paramsDir + 'alpha0.npy')))
    beta0 = sigmoid(np.squeeze(np.load(paramsDir + 'beta0.npy')))
    fastRNN0 = FastRNN(W0, U0, b0, alpha0, beta0, W0.shape[0])
    W1 = np.load(paramsDir + 'W1.npy').T
    U1 = np.load(paramsDir + 'U1.npy').T
    b1 = np.squeeze(np.load(paramsDir + 'h1.npy'))
    alpha1 = sigmoid(np.squeeze(np.load(paramsDir + 'alpha1.npy')))
    beta1 = sigmoid(np.squeeze(np.load(paramsDir + 'beta1.npy')))
    fastRNN1 = FastRNN(W1, U1, b1, alpha1, beta1, W1.shape[0])

    srnn2 = Stacked2LayerRNN(fastRNN0, fastRNN1, fcW, fcB)
    errorCount = 0
    print("Verifying outputs")
    for i in range(len(x_sample)):
        ret = srnn2.infer(x_sample[i])
        errorCount += int(abs(np.sum(ret - predictions[i])) >= 0.0001)
    print("Done! Error cound %d for %d tests" % (errorCount, len(x_sample)))

    W0 = np.concatenate([U0, W0], axis=1) 
    W1 = np.concatenate([U1, W1], axis=1) 
    assert x_sample.shape[2] == timeSteps0
    assert x_sample.shape[1] == timeSteps1
    numOutput = fcW.shape[0]
    ret = getTemplateMXChip(W0=W0, B0=b0, alpha0=alpha0, beta0=beta0,
                            W1=W1, B1=b1, alpha1=alpha1, beta1=beta1,
                            timeSteps0=timeSteps0, timeSteps1=timeSteps1,
                            fcW=fcW, fcB=fcB, numOutput=numOutput, 
                            labelMap=LABELMAP13,
                            normalize=True, mean=mean, std=std)
    print("Generating model.h")
    f = open('model.h', 'w+')
    print(ret, file=f)
    f.close()
    print("Done")



main()
