# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
#
# Setting up the USPS Data.

import subprocess
import os
import numpy as np
from sklearn.datasets import load_svmlight_file
import sys

def downloadData(workingDir, downloadDir, linkTrain, linkTest):
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
    print("Done")
