# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
#
# Setting up the HAR data for EMI-RNN. It is assumed that the data has already
# been downloaded and is placed in downloadDir.

import subprocess
import os
import numpy as np
import sys
from helpermethods import *


if __name__ == '__main__':
    ## Config
    workingDir = './'
    downloadDir = 'HAR'
    subinstanceLen = 48
    subinstanceStride = 16
    ## End Config

    print("Processing data")
    path = workingDir + '/' + downloadDir
    path = os.path.abspath(path)
    extractedDir = generateData(path)

    rawDir = extractedDir + '/RAW/'
    print("Extracting features")
    sourceDir = rawDir
    outDir = extractedDir + '/%d_%d/' % (subinstanceLen, subinstanceStride)
    if subinstanceLen == 128:
        outDir = extractedDir + '/%d/' % (subinstanceLen)
    print('subinstanceLen', subinstanceLen)
    print('subinstanceStride', subinstanceStride)
    print('sourceDir', sourceDir)
    print('outDir', outDir)
    try:
        os.mkdir(outDir)
    except OSError:
        exit("Could not create %s" % outDir)
    assert len(os.listdir(outDir)) == 0
    makeEMIData(subinstanceLen, subinstanceStride, sourceDir, outDir)
    print("Done")

