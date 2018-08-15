# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
# 
# Setting up the USPS Data for ProtoNN. This scripts calls bash commands.
# If bash is not available, 
#   - manually create a usps10 subdirectory
#   - download the train and test files from the `linkTrain` and `linkTest`
#     provided below.
#   - extract the downloaded .bz2 files
#   - rename usps to train.txt
#   - rename usps.t to test.txt
#   - run this script with BASH = False

import subprocess
import os
import numpy as np
from sklearn.datasets import load_svmlight_file
import sys

BASH = True

workingDir = './'
downloadDir = 'usps10'
linkTrain = 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2'
linkTest = 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2'

def downloadData(workingDir, downloadDir):
    def runcommand(command):
        p = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
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
    command = 'wget %s' % linkTrain
    runcommand(command)
    command = 'wget %s' % linkTest
    runcommand(command)
    print("Extracting data")
    command = 'bzip2 -d usps.bz2'
    runcommand(command)
    command = 'bzip2 -d usps.t.bz2'
    runcommand(command)
    command = 'mv usps train.txt'
    runcommand(command)
    command = 'mv usps.t test.txt'
    runcommand(command)
    os.chdir(cwd)
    return True

def processData(workingDir, downloadDir):
    def loadLibSVMFile(file):
        data = load_svmlight_file(file)
        features = data[0]
        labels = data[1]
        retMat = np.zeros([features.shape[0], features.shape[1] + 1])
        retMat[:, 0] = labels
        retMat[:, 1:] = features.todense()
        return retMat

    path = workingDir + '/' + downloadDir
    path = os.path.abspath(path)
    trf = path + '/train.txt'
    tsf = path + '/test.txt'
    assert os.path.isfile(trf), 'File not found: %s' % trf
    assert os.path.isfile(tsf), 'File not found: %s' % tsf
    train = loadLibSVMFile(trf)
    test = loadLibSVMFile(tsf)
    np.save(path + '/train.npy', train)
    np.save(path + '/test.npy', test)


if __name__ == '__main__':
    if BASH is True:
        print("Attempting to use bash. If bash is not available, please")
        print("refer to instructions for downloading files manually")
        print("provided in this script.")
        if not downloadData(workingDir, downloadDir):
            exit('Download failed')
    print("Procesing data")
    processData(workingDir, downloadDir)
    print("Done")
