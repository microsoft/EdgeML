# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
# 
# Script to fetch the HAR data

import subprocess
import os
import numpy as np
import sys
from helpermethods import *


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


if __name__ == '__main__':
    workingDir = './'
    downloadDir = 'HAR'
    linkData = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
    exitMessage ='''
Download Failed!

To manually download the HAR data,
\t1. Download the compressed folder with the data from:
\t\t%s
\t2. Extract the downloaded files into a directory named `HAR`
''' % linkData
    if not downloadData(workingDir, downloadDir):
        exit('Download failed')
    print("Done")
