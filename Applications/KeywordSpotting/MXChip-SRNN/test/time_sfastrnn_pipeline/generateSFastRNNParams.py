import numpy as np
from template_MXChip import getTemplate as getTemplateMXChip
import python_speech_features as sp

np.random.seed(42)

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
    audioLen = 16000
    inputDim = 32
    hiddenDim0 = 16
    timeSteps0 = 8
    hiddenDim1 = 16
    timeSteps1 = 8
    alpha0, beta0 = 0.2, 0.8
    alpha1, beta1 = 0.1, 0.9
    numOutput = 6

    # Define the FastRNN model
    # Replace with actual model and data later.
    # -----------------------------------------
    preemph = 0.97
    W0 = np.random.normal(size=[hiddenDim0, hiddenDim0 +  inputDim]) * 0.1
    B0 = np.random.normal(size=hiddenDim0) * 0.01
    W1 = np.random.normal(size=[hiddenDim1, hiddenDim1 + hiddenDim0])
    B1 = np.random.normal(size=hiddenDim1)
    fastrnn0 = FastRNN(W0, B0, alpha0, beta0, timeSteps0,
        inputDim, hiddenDim0)
    fastrnn1 = FastRNN(W1, B1, alpha1, beta1,
        timeSteps1, hiddenDim0, hiddenDim1)
    fcW = np.random.normal(size=[numOutput, hiddenDim1])
    fcB = np.random.normal(size=[numOutput])
    audioSamples = (np.random.normal(size=audioLen) * 2000).astype(int)
    # -----------------------------------------

    # Create the template
    ret = getTemplateMXChip(W0, B0, alpha0, beta0, W1, B1, alpha1, beta1,
                timeSteps0, timeSteps1, fcW, fcB, audioSamples, numOutput)
    fp = open('testdata.h', 'w+')
    print(ret, file=fp)
    # Check if framing is correct
    frames = sp.sigproc.framesig(audioSamples, frame_len=400, frame_step=160)
    tt = audioSamples[1:] - preemph * audioSamples[:-1]
    tt = np.append(audioSamples[0], tt)
    pre_frames = sp.sigproc.framesig(tt, frame_len=400, frame_step=160)

    for i in range(5):
        tt = frames[i]
        # print("input", tt[:5])
        # print("pr-ed input", pre_frames[i, :5])
        ttt = np.zeros(512)
        ttt[:len(tt)] = pre_frames[i, :]
        cfft = np.fft.fft(ttt, n=512)
        # print("fft", cfft[:5])
        # print()
    logfbank, energy = sp.fbank(audioSamples, nfilt=32, preemph=preemph)
    logfbank = np.log(logfbank)
    for feat in logfbank[:10]:
        # print(feat[:5])
        pass

    # run fastRNN0 on the input for each brick and 
    # collect all the hidden satates 0
    h0_list = []
    h1_list = []
    logits_list = []
    input1List = []
    for i in range(0, len(logfbank), timeSteps0):
        if i + timeSteps0 > len(logfbank):
            continue
        input0 = logfbank[i:i+timeSteps0]
        # print(input0[7, -10:])
        assert len(input0) == timeSteps0
        h = fastrnn0.unroll(input0)
        # print('--> %2d' % (i / timeSteps0), h)
        h0_list.append(h)
        input1List.append(h)
        assert len(input1List) <= timeSteps1
        if len(input1List) == timeSteps1:
            input1 = np.array(input1List)
            assert len(input1) == timeSteps1
            assert input1.shape[1] == hiddenDim0
            h = fastrnn1.unroll(input1)
            h1_list.append(h)
            logits = np.matmul(fcW, h) + fcB
            logits_list.append(logits)
            print("--> h1   ", h)
            print("--> fcout", logits)
            print()
            del input1List[0]


device = 'MXChip'
main(device)
