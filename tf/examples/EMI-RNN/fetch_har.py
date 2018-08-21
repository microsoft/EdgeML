# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
# 
# Setting up the HAR Data for EMI-RNN. This scripts calls bash commands.
# If bash is not available, 
#   - download the train and test files from the web link provided in linkData
#   variable
#   - extract the downloaded zip files
#   - run this script with BASH = False

import subprocess
import os
import numpy as np
import sys
from helpermethods import *

## Config
BASH = True
workingDir = './'
downloadDir = 'HAR'
linkData = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
subinstanceLen = 48
subinstanceStride = 16

def downloadData(workingDir, downloadDir):
    def runcommand(command, splitChar=' '):
        p = subprocess.Popen(command.split(splitChar), stdout=subprocess.PIPE)
        output, error = p.communicate()
        assert(p.returncode == 0), 'Command failed: %s' % command

    path = workingDir + '/' + downloadDir
    path = os.path.abspath(path)
    try:
        os.mkdir(path)
    except OSError:
        print("Could not create %s. Make sure the path does" % path)
        print("not already exist and you have permisions to create it.")
        return False
    cwd = os.getcwd()
    os.chdir(path)
    print("Downloading data")
    command = 'wget %s' % linkData
    runcommand(command)
    print("Extracting data")
    print(os.getcwd())
    runcommand('ls')
    command = "unzip#UCI HAR Dataset.zip"
    runcommand(command, splitChar='#')
    os.chdir(cwd)
    return True

def processData(workingDir, downloadDir):
    path = workingDir + '/' + downloadDir
    path = os.path.abspath(path)
    return generateData(path)

if __name__ == '__main__':
    if BASH is True:
        print("Attempting to use bash. If bash is not available, please")
        print("refer to instructions for downloading files manually")
        print("provided in this script.")
        if not downloadData(workingDir, downloadDir):
            exit('Download failed')
    print("Procesing data")
    extractedDir = processData(workingDir, downloadDir)
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

