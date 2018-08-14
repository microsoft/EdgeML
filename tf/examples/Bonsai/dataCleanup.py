# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
from sklearn.datasets import load_svmlight_file
import sys


def loadLibSVMFile(file):
    data = load_svmlight_file(file)
    features = data[0]
    labels = data[1]

    retMat = np.zeros([features.shape[0], features.shape[1] + 1])

    retMat[:, 0] = labels
    retMat[:, 1:] = features.todense()

    return retMat


dataDir = sys.argv[1]

train = loadLibSVMFile(dataDir + '/train.txt')
test = loadLibSVMFile(dataDir + '/test.txt')

np.save(dataDir + '/train.npy', train)
np.save(dataDir + '/test.npy', test)
