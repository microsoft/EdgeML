import numpy as np
import python_speech_features as sp


def hexToQ31(hexStr):
    '''
    hexStr in format 0x
    '''
    h = hexStr[2:]
    value = bin(int(h, 16))
    value = value[2:]
    value = value.zfill(32)
    valuef = ' '.join(value[i:i + 4] for i in range(0, 32, 4))
    return value, valuef


def q31ToF32(value):
    num = 0.0
    num += float(value[0])
    for i, x in enumerate(value[1:]):
        num += float(x) * (2 ** (-(i + 1)))
    return num

def createDataFile(inputData, expectedOutput, inputLen=512, outputLen=32):
    assert len(inputData) == inputLen, len(inputData)
    assert len(expectedOutput) == outputLen, len(expectedOutput)
    inputStr = ''
    for i, x in enumerate(inputData):
        inputStr += '%.0f, ' % x
        if (i+1) % 10 == 0:
            inputStr += '\n'
    outputStr = ''
    for i, x in enumerate(expectedOutput):
        outputStr += '%.3f, ' % x
        if (i + 1) % 10 == 0:
            outputStr += '\n'

    template = '''
static int32_t inputData[%d] = {%s};

static int32_t expectedOutput[%d] = {%s};
''' % (inputLen, inputStr, outputLen, outputStr)
    return template

def main():
    np.random.seed(42)
    # inputLen assumed to be same as frame len
    inputLen = 512
    frameLen = 400
    preemph = 0.97
    x = np.random.normal(size=frameLen)
    x = x * 100
    x = x.astype(int)
    pre = np.concatenate([x[:1], x[1:] - preemph * x[:-1]])
    pre_ = np.zeros(inputLen)
    pre_[:frameLen] = pre[:]
    pre = pre_
    frames = sp.sigproc.framesig(pre, inputLen, inputLen)
    assert len(frames) == 1
    print("FFT")
    print(np.fft.fft(frames[0], n=512)[:10])
    powSpec = sp.sigproc.powspec(frames, inputLen)
    powSpec = np.reshape(powSpec, -1)
    print("powSpec")
    # t = list(powSpec[-10:])
    # t.reverse()
    print(powSpec[:10])
    fb, energy = sp.fbank(x, winlen=512/16000.0, nfilt=32, preemph=preemph)
    print("FBank")
    log = np.log(fb)
    print(log)
    x_inp = np.zeros(inputLen)
    x_inp[:frameLen] = x[:]
    template = createDataFile(x_inp, powSpec, inputLen=inputLen,
                              outputLen=inputLen/2 + 1)
    fp = open('data.h', 'w+')
    print(template, file=fp)
    fp.close()

main()
