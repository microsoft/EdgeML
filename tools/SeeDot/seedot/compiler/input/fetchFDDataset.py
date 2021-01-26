# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import os
import requests
import subprocess
from tqdm import tqdm
import numpy as np

'''
The python file to obtain the dataset for face Detection.
'''


class FetchFaceDetectionDataset:
    def __init__(self):
        self.modelFiles = ['Bg1.npy', 'Bg2.npy', 'Bh1.npy', 'Bh2.npy', 'CBR1B.npy', 'CBR1F.npy', 'CBR1W.npy', 
                    'conf0b.npy', 'conf0w.npy', 'conf1b.npy', 'conf1w.npy', 'conf2b.npy', 'conf2w.npy', 
                    'conf3b.npy', 'conf3w.npy', 'L0B1.npy', 'L0B2.npy', 'L0B3.npy', 'L0F1.npy', 
                    'L0F2.npy', 'L0F3.npy', 'L0W1.npy', 'L0W2.npy', 'L0W3.npy', 'L1B1.npy', 'L1B2.npy', 
                    'L1B3.npy', 'L1F1.npy', 'L1F2.npy', 'L1F3.npy', 'L1W1.npy', 'L1W2.npy', 'L1W3.npy', 
                    'L2B1.npy', 'L2B2.npy', 'L2B3.npy', 'L2F1.npy', 'L2F2.npy', 'L2F3.npy', 'L2W1.npy', 
                    'L2W2.npy', 'L2W3.npy', 'L3B1.npy', 'L3B2.npy', 'L3B3.npy', 'L3F1.npy', 'L3F2.npy', 
                    'L3F3.npy', 'L3W1.npy', 'L3W2.npy', 'L3W3.npy', 'loc0b.npy', 'loc0w.npy', 'loc1b.npy', 
                    'loc1w.npy', 'loc2b.npy', 'loc2w.npy', 'loc3b.npy', 'loc3w.npy', 'normW1.npy', 
                    'normW2.npy', 'normW3.npy', 'nu1.npy', 'nu2.npy', 'U1.npy', 'U2.npy', 'W1.npy', 
                    'W2.npy', 'zeta1.npy', 'zeta2.npy']
        self.datasetFiles = ['train.npy', 'test.npy']

        self.modelDir = "../../../model/rnnpool/face-4/"
        self.datasetDir = "../../../datasets/rnnpool/face-4/"
        self.curdir = os.getcwd()

    def createDirectories(self):
        try:
            print("Creating model and dataset directories...")
            self.mkdirModel = "mkdir -p %s"%(self.modelDir)
            self.mkdirDatasets = "mkdir -p %s"%(self.datasetDir) 
            proc = subprocess.Popen(self.mkdirModel, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _ = proc.communicate()
            proc = subprocess.Popen(self.mkdirDatasets, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _ = proc.communicate()
        except: # Directory already exists
            print("Directories already exist, bypassing creating...")
            pass
    
    def fetchModelFiles(self):
        os.chdir(self.modelDir)

        cmd = "https://raw.githubusercontent.com/krantikiran68/EdgeML/newer-seedot/tools/SeeDot/seedot/compiler/input/face-4/model/modelShape.txt"
        r = requests.get(cmd)
        modelshapeparams = r.content.decode().split("\n")[:-1]

        for i in tqdm(range(len(self.modelFiles))):
            filename = self.modelFiles[i]
            shapestr = modelshapeparams[i][1:-1].split(",")
            shape = []
            for num in shapestr:
                if num == "":
                    shape.append(1)
                else:
                    shape.append(int(num))

            csvfilename = filename.split(".")[0] + ".csv"
            cmd = "https://raw.githubusercontent.com/krantikiran68/EdgeML/newer-seedot/tools/SeeDot/seedot/compiler/input/face-4/model/%s"%(csvfilename)
            
            r = requests.get(cmd)
            assert r.status_code == 200, "Fetching the model weights failed, fetch manually"

            f = open(filename.split(".")[0] + ".csv", "wb")
            f.write(r.content)
            f.close()
            f = np.loadtxt(filename.split(".")[0] + ".csv", delimiter=",")
            f = np.reshape(f, shape)
            np.save(filename, f)
        
        #Cleanup
        proc = subprocess.Popen("rm -f *.csv", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _ = proc.communicate()
        
        os.chdir(self.curdir)

    def fetchDatasetFiles(self):
        os.chdir(self.datasetDir)

        cmd = "https://raw.githubusercontent.com/krantikiran68/EdgeML/newer-seedot/tools/SeeDot/seedot/compiler/input/face-4/datasets/datasetsShape.txt"
        r = requests.get(cmd)
        datasetsshapeparams = r.content.decode().split("\n")[:-1]

        for i in tqdm(range(len(self.datasetFiles))):
            filename = self.datasetFiles[i]
            
            shapestr = datasetsshapeparams[i][1:-1].split(",")
            shape = []
            for num in shapestr:
                if num == "":
                    shape.append(1)
                else:
                    shape.append(int(num))

            r = requests.get("https://raw.githubusercontent.com/krantikiran68/EdgeML/newer-seedot/tools/SeeDot/seedot/compiler/input/face-4/datasets/%s"%(filename.split(".")[0] + ".csv"))
            assert r.status_code == 200, "Fetching the datasets failed, fetch manually"

            f = open(filename.split(".")[0] + ".csv", "wb")
            f.write(r.content)
            f.close()
            f = np.loadtxt(filename.split(".")[0] + ".csv", delimiter=",")
            
            f = np.reshape(f, shape)
            np.save(filename, f)

            proc = subprocess.Popen("rm -f *.csv", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _ = proc.communicate()

            os.chdir(self.curdir)

    def copySeeDotfile(self):
        os.popen("cp rnnpool.sd %s/input.sd"%(self.modelDir))
    
    def checkCorrectDir(self):
        errormessage = \
            "Run this python script from ${EdgeML Repo Folder}/tools/SeeDot/seedot/compiler/input/ directory..."
        
        if not ("rnnpool.sd" in os.listdir(os.getcwd())):
            assert False, errormessage

if __name__ == '__main__':
    obj = FetchFaceDetectionDataset()

    obj.checkCorrectDir()

    obj.createDirectories()
    obj.fetchModelFiles()
    obj.fetchDatasetFiles()
    obj.copySeeDotfile()