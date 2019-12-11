# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
#
# Setting up the USPS Data.

import bz2
import os
import subprocess
import sys

import requests
import numpy as np
from sklearn.datasets import load_svmlight_file
from helpermethods import download_file, decompress



def downloadData(workingDir, downloadDir, linkTrain, linkTest):
    path = workingDir + '/' + downloadDir
    path = os.path.abspath(path)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        print("Could not create %s. Make sure the path does" % path)
        print("not already exist and you have permissions to create it.")
        return False

    training_data_bz2 = download_file(linkTrain, path)
    test_data_bz2 = download_file(linkTest, path)

    training_data = decompress(training_data_bz2)
    test_data = decompress(test_data_bz2)
    
    train = os.path.join(path, "train.txt")
    test = os.path.join(path, "test.txt")
    if os.path.isfile(train):
        os.remove(train)
    if os.path.isfile(test):
        os.remove(test)

    os.rename(training_data, train)
    os.rename(test_data, test)
    os.remove(training_data_bz2)
    os.remove(test_data_bz2)
    return True

if __name__ == '__main__':
    workingDir = './'
    downloadDir = 'usps10'
    linkTrain = 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2'
    linkTest = 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2'
    failureMsg = '''
Download Failed!
To manually perform the download
\t1. Create a new empty directory named `usps10`.
\t2. Download the data from the following links into the usps10 directory.
\t\tTest: %s
\t\tTrain: %s
\t3. Extract the downloaded files.
\t4. Rename `usps` to `train.txt` and,
\t5. Rename `usps.t` to `test.txt
''' % (linkTrain, linkTest)

    if not downloadData(workingDir, downloadDir, linkTrain, linkTest):
        exit(failureMsg)
    print("Done: see ", downloadDir)
