
# Google Speech data feature extraction

# Note that the 'testing_list.txt' and 'validation_list.txt'
# that provided is used to create test and validation
# sets. Everything that is not in these sets is considered
# for training.

# The testing_list and validation_list and by extension
# the training set has the following property.

#     If one audio sample of a user is in either one of these
#     sets, then all audio samples of that user will also be
#     in that set.

#     As long as the same methodology of creating testing
#     and validation set that google used - as outlined in
#     their README is used, the testing and validation set
#     will be consistent. That is, the will always contain
#     the same set of examples

# Sampling is not supported yet.

from python_speech_features import fbank
import os
import glob
import numpy as np
import scipy.io.wavfile as r
import random


# Various version can be created depending on which labels are chosen and which
# are moved to the negative (noise) set. We use LABELMAP13 for most of our
# experiments.
LABELMAP30 = {
    '_background_noise_': 1, 'bed': 2, 'bird': 3,
    'cat': 4, 'dog': 5, 'down': 6, 'eight': 7,
    'five': 8, 'four': 9, 'go': 10, 'happy': 11,
    'house': 12, 'left': 13, 'marvin': 14, 'nine': 15,
    'no': 16, 'off': 17, 'on': 18, 'one': 19,
    'right': 20, 'seven': 21, 'sheila': 22, 'six': 23,
    'stop': 24, 'three': 25, 'tree': 26, 'two': 27,
    'up': 28, 'wow': 29, 'yes': 30, 'zero': 31
}


LABELMAP13 = {
    'go': 1, 'no': 2, 'on': 3, 'up': 4, 'bed': 5, 'cat': 6,
    'dog': 7, 'off': 8, 'one': 9, 'six': 10, 'two': 11,
    'yes': 12,
    'wow': 0, 'bird': 0, 'down': 0, 'five': 0, 'four': 0,
    'left': 0, 'nine': 0, 'stop': 0, 'tree': 0, 'zero': 0,
    'eight': 0, 'happy': 0, 'house': 0, 'right': 0, 'seven': 0,
    'three': 0, 'marvin': 0, 'sheila': 0, '_background_noise_': 0
}

LABELMAP12 = {
    'yes': 1, 'no': 2, 'up': 3, 'down': 4, 'left': 5, 'right': 6,
    'on': 7, 'off': 8, 'stop': 9, 'go': 10,
    'bed':0, 'cat':0, 'dog':0, 'one':0, 'six':0, 'two':0,
    'wow':0, 'bird':0, 'five':0, 'four':0, 'nine':0, 'tree':0,
    'zero':0, 'eight':0, 'happy':0, 'house':0, 'seven':0, 'three':0,
    'marvin':0, 'sheila':0, '_background_noise_':0
}

def createFileList(audioFileDir, testingList,
                   validationList, outPrefix,
                   labelMap):
    '''
    audioFileDir: The directory containing the directories
        with audio files.
    testingList: the `testing_list.txt` file
    validationList: the `validation_list.txt` file

    Reads all the files in audioFileDir and creates
    a list of files that are not part of testingList
    or validationList.

    WARNING: _background_noise_ is ignored

    Then testingList, validationList and trainginList
    are converted into numpy arrays with their labels

    This is written as
        outPrefix + '_testList.npy'
        outPrefix + '_trainList.npy'
        outPrefix + '_validationList.npy'
    '''
    dirs = os.listdir(audioFileDir)
    dirs = [x for x in dirs if os.path.isdir(os.path.join(audioFileDir, x))]
    assert(len(dirs) == 31), (len(dirs))
    for x in dirs:
        msg = '%s found without label map' % x
        assert x in labelMap, msg

    allFileList = []
    for fol in dirs:
        if fol == '_background_noise_':
            print("Ignoring %s" % fol)
            continue
        path = audioFileDir + '/' + fol + '/'
        files = []
        for w in os.listdir(path):
            if not w.endswith('.wav'):
                print("Ignoring %s" % w)
                continue
            files.append(fol + '/' + w)
        allFileList.extend(files)
    assert(len(allFileList) == len(set(allFileList)))

    fil = open(testingList, 'r')
    testingList = fil.readlines()
    testingList = [x.strip() for x in testingList]
    fil.close()
    fil = open(validationList, 'r')
    validationList = fil.readlines()
    validationList = [x.strip() for x in validationList]
    originalLen = len(allFileList)
    allFileList = set(allFileList) - set(validationList)
    assert len(allFileList) < originalLen
    assert originalLen == len(allFileList) + len(validationList)
    originalLen = len(allFileList)
    allFileList = set(allFileList) - set(testingList)
    assert len(allFileList) < originalLen
    assert originalLen == len(allFileList) + len(testingList)

    trainingList = list(allFileList)
    testingList = list(testingList)
    validationList = list(validationList)
    np.save(outPrefix + 'file_train.npy', trainingList)
    np.save(outPrefix + 'file_test.npy', testingList)
    np.save(outPrefix + 'file_val.npy', validationList)


def extractFeatures(fileList, LABELMAP, maxlen, numFilt, samplerate, winlen,
                    winstep):
    '''
    Reads audio from files specified in fileList, extracts features and assigns
    labels to them.

    fileList: List of audio file names.
    LABELMAP: The label map to use.
    maxlen: maximum length of the audio file. Every other
        files is zero padded to maxlen
    numFilt: number of filters to use in MFCC
    samplerate: sample rate of the audio file. All files are
        assumed to be of same sample rate
    winLen: winLen to use for fbank in seconds
    winstep: winstep for fbank in seconds
    '''
    def __extractFeatures(stackedWav, numSteps, numFilt,
                          samplerate, winlen, winstep):
        '''
        [number of waves, Len(wave)]
        returns [number of waves, numSteps, numFilt]
        All waves are assumed to be of fixed length
        '''
        assert stackedWav.ndim == 2, 'Should be [number of waves, len(wav)]'
        extractedList = []
        eps = 1e-10
        for sample in stackedWav:
            temp, _ = fbank(sample, samplerate=samplerate, winlen=winlen,
                            winstep=winstep, nfilt=numFilt,
                            winfunc=np.hamming)
            temp = np.log(temp + eps)
            assert temp.ndim == 2, 'Should be [numSteps, numFilt]'
            assert temp.shape[0] == numSteps, 'Should be [numSteps, numFilt]'
            extractedList.append(temp)
        return np.array(extractedList)

    fileList = np.array(fileList)
    assert(fileList.ndim == 1)
    allSamples = np.zeros((len(fileList), maxlen))
    i = 0
    for i,file in enumerate(fileList):
        _, x = r.read(file)
        assert(len(x) <= maxlen)
        allSamples[i, maxlen - len(x):maxlen] += x
        i += 1
    assert allSamples.ndim == 2
    winstepSamples = winstep * samplerate
    winlenSamples = winlen * samplerate
    assert(winstepSamples.is_integer())
    assert(winlenSamples.is_integer())
    numSteps = int(np.ceil((maxlen - winlenSamples)/winstepSamples) + 1)
    x = __extractFeatures(allSamples, numSteps, numFilt, samplerate, winlen,
                          winstep)
    y_ = [t.split('/') for t in fileList]
    y_ = [t[-2] for t in y_]
    y = []
    for t in y_:
        assert t in LABELMAP
        y.append(LABELMAP[t])

    def to_onehot(indices, numClasses):
        assert indices.ndim == 1
        n = max(indices) + 1
        assert numClasses <= n
        b = np.zeros((len(indices), numClasses))
        b[np.arange(len(indices)), indices] = 1
        return b
    y = to_onehot(np.array(y), np.max(y) + 1)
    return x, y

if __name__=='__main__':
    # ----------------------------------------- #
    # Configuration
    # ----------------------------------------- #
    seed = 42
    maxlen = 16000
    numFilt = 32
    samplerate = 16000
    winlen = 0.025
    winstep = 0.010
    # 13 for google 13, 11 for google 12
    numLabels = 13 # 0 not assigned
    samplerate=16000
    # For creation of training file list, testing file list
    # and validation list. 
    audioFileDir = './GoogleSpeech/Raw/'
    testingList = './GoogleSpeech/Raw/testing_list.txt'
    validationList = './GoogleSpeech/Raw/validation_list.txt'
    outDir = './GoogleSpeech/Extracted/'
    # ----------------------------------------- #
    np.random.seed(seed)
    random.seed(seed)
    assert(numLabels in [13, 11])
    if numLabels == 13:
        values = [LABELMAP13[x] for x in LABELMAP13]
        values = set(values)
        assert(len(values) == 13)
        LABELMAP = LABELMAP13
    if numLabels == 11:
        values = [LABELMAP12[x] for x in LABELMAP12]
        values = set(values)
        assert(len(values) == 11)
        LABELMAP = LABELMAP12

    print("Peforming file creation")
    createFileList(audioFileDir, testingList, validationList,
                   outDir, LABELMAP)
    trainFileList = np.load(outDir + 'file_train.npy')
    testFileList = np.load(outDir + 'file_test.npy')
    valFileList = np.load(outDir + 'file_val.npy')
    print("Number of train files:", len(trainFileList))
    print("Number of test files", len(testFileList))
    print("Number of val files", len(valFileList))
    print("Performing feature extraction")
    trainFileList_ = [audioFileDir + x for x in trainFileList]
    valFileList_ = [audioFileDir + x for x in valFileList]
    testFileList_ = [audioFileDir + x for x in testFileList]
    x_test, y_test = extractFeatures(testFileList_, LABELMAP, maxlen, numFilt,
                                     samplerate, winlen, winstep)
    x_val, y_val = extractFeatures(valFileList_, LABELMAP, maxlen, numFilt,
                                   samplerate, winlen, winstep)
    x_train, y_train = extractFeatures(trainFileList_, LABELMAP, maxlen,
                                       numFilt, samplerate, winlen, winstep)
    np.save(outDir + 'x_train', x_train);np.save(outDir + 'y_train', y_train)
    np.save(outDir + 'x_test', x_test);np.save(outDir + 'y_test', y_test)
    np.save(outDir + 'x_val', x_val);np.save(outDir + 'y_val', y_val)
    print("Shape train", x_train.shape, y_train.shape)
    print("Shape test", x_test.shape, y_test.shape)
    print("Shape val", x_val.shape, y_val.shape)


