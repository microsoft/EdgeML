# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import os
import requests
import subprocess
from tqdm import tqdm
import numpy as np

'''
The python file to obtain the dataset for Face Detection.
'''


class FetchFaceDetectionDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        if dataset == "face-4":
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
        elif dataset == "face-2":
            self.modelFiles = ['Bg1.npy', 'Bg2.npy', 'Bh1.npy', 'Bh2.npy', 
                            'CBR1B.npy', 'CBR1F.npy', 'CBR1W.npy', 'conf0b.npy', 
                            'conf0w.npy', 'conf1b.npy', 'conf1w.npy', 'conf2b.npy',
                            'conf2w.npy', 'conf3b.npy', 'conf3w.npy', 
                            'L0B1.npy', 'L0B2.npy', 'L0B3.npy', 'L0F1.npy', 'L0F2.npy',
                            'L0F3.npy', 'L0W1.npy', 'L0W2.npy', 'L0W3.npy', 'L10B1.npy',
                            'L10B2.npy', 'L10B3.npy', 'L10F1.npy', 'L10F2.npy', 
                            'L10F3.npy', 'L10W1.npy', 'L10W2.npy', 'L10W3.npy', 
                            'L11B1.npy', 'L11B2.npy', 'L11B3.npy', 'L11F1.npy', 'L11F2.npy',
                            'L11F3.npy', 'L11W1.npy', 'L11W2.npy', 'L11W3.npy', 'L12B1.npy',
                            'L12B2.npy', 'L12B3.npy', 'L12F1.npy', 'L12F2.npy', 'L12F3.npy',
                            'L12W1.npy', 'L12W2.npy', 'L12W3.npy', 'L13B1.npy', 'L13B2.npy', 
                            'L13B3.npy', 'L13F1.npy', 'L13F2.npy', 'L13F3.npy', 'L13W1.npy', 
                            'L13W2.npy', 'L13W3.npy', 'L1B1.npy', 'L1B2.npy', 'L1B3.npy', 
                            'L1F1.npy', 'L1F2.npy', 'L1F3.npy', 'L1W1.npy', 'L1W2.npy', 
                            'L1W3.npy', 'L2B1.npy', 'L2B2.npy', 'L2B3.npy', 'L2F1.npy',
                            'L2F2.npy', 'L2F3.npy', 'L2W1.npy', 'L2W2.npy', 'L2W3.npy', 
                            'L3B1.npy', 'L3B2.npy', 'L3B3.npy', 'L3F1.npy', 'L3F2.npy', 
                            'L3F3.npy', 'L3W1.npy', 'L3W2.npy', 'L3W3.npy', 'L4B1.npy', 
                            'L4B2.npy', 'L4B3.npy', 'L4F1.npy', 'L4F2.npy', 'L4F3.npy', 
                            'L4W1.npy', 'L4W2.npy', 'L4W3.npy', 'L5B1.npy', 'L5B2.npy', 
                            'L5B3.npy', 'L5F1.npy', 'L5F2.npy', 'L5F3.npy', 'L5W1.npy', 
                            'L5W2.npy', 'L5W3.npy', 'L6B1.npy', 'L6B2.npy', 'L6B3.npy', 
                            'L6F1.npy', 'L6F2.npy', 'L6F3.npy', 'L6W1.npy', 'L6W2.npy', 
                            'L6W3.npy', 'L7B1.npy', 'L7B2.npy', 'L7B3.npy', 'L7F1.npy', 
                            'L7F2.npy', 'L7F3.npy', 'L7W1.npy', 'L7W2.npy', 'L7W3.npy', 
                            'L8B1.npy', 'L8B2.npy', 'L8B3.npy', 'L8F1.npy', 'L8F2.npy', 
                            'L8F3.npy', 'L8W1.npy', 'L8W2.npy', 'L8W3.npy', 'L9B1.npy', 
                            'L9B2.npy', 'L9B3.npy', 'L9F1.npy', 'L9F2.npy', 'L9F3.npy', 
                            'L9W1.npy', 'L9W2.npy', 'L9W3.npy', 'loc0b.npy', 'loc0w.npy', 
                            'loc1b.npy', 'loc1w.npy', 'loc2b.npy', 'loc2w.npy', 'loc3b.npy', 
                            'loc3w.npy', 'normW1.npy', 'normW2.npy', 'normW3.npy', 
                            'nu1.npy', 'nu2.npy', 'U1.npy', 'U2.npy', 'W1.npy', 'W2.npy', 
                            'zeta1.npy', 'zeta2.npy']
            self.datasetFiles = ['train.npy', 'test.npy']

            
            self.modelDir = "../../../model/rnnpool/face-2/"
            self.datasetDir = "../../../datasets/rnnpool/face-2/"
        self.curdir = os.getcwd()

    def createDirectories(self):
        try:
            print("Creating model and dataset directories for %s..."%(self.dataset))
            self.mkdirModel = "mkdir -p %s"%(self.modelDir)
            self.mkdirDatasets = "mkdir -p %s"%(self.datasetDir) 
            proc = subprocess.Popen(self.mkdirModel, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _ = proc.communicate()
            proc = subprocess.Popen(self.mkdirDatasets, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _ = proc.communicate()
            print("Done")
        except: # Directory already exists
            print("Directories already exist, bypassing creation...")
            pass
    
    def fetchModelFiles(self):
        print("Fetching Model weights for %s dataset"%(self.dataset))
        os.chdir(self.modelDir)

        cmd = "https://raw.githubusercontent.com/krantikiran68/EdgeML/newer-seedot/tools/SeeDot/seedot/compiler/input/%s/model/modelShape.txt"%(self.dataset)
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
            cmd = "https://raw.githubusercontent.com/krantikiran68/EdgeML/newer-seedot/tools/SeeDot/seedot/compiler/input/%s/model/%s"%(self.dataset,csvfilename)
            
            r = requests.get(cmd)
            assert r.status_code == 200, "Fetching the model weights failed, fetch manually by visiting https://github.com/krantikiran68/EdgeML/newer-seedot/tools/SeeDot/seedot/compiler/input/"

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
        print("Done")

    def fetchDatasetFiles(self):
        print("Fetching %s dataset files"%(self.dataset))
        os.chdir(self.datasetDir)

        cmd = "https://raw.githubusercontent.com/krantikiran68/EdgeML/newer-seedot/tools/SeeDot/seedot/compiler/input/%s/datasets/datasetsShape.txt"%(self.dataset)
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

            r = requests.get("https://raw.githubusercontent.com/krantikiran68/EdgeML/newer-seedot/tools/SeeDot/seedot/compiler/input/%s/datasets/%s"%(self.dataset, filename.split(".")[0] + ".csv"))
            assert r.status_code == 200, "Fetching the datasets failed, fetch manually by visiting https://github.com/krantikiran68/EdgeML/newer-seedot/tools/SeeDot/seedot/compiler/input/"

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
        print("Done")

    def copySeeDotfile(self):
        os.popen("cp rnnpool-%s.sd %s/input.sd"%(self.dataset, self.modelDir))
    
    def checkCorrectDir(self):
        errormessage = \
            "Run this python script from ${EdgeML Repo Folder}/tools/SeeDot/seedot/compiler/input/ directory..."
        
        if not ("rnnpool-%s.sd"%(self.dataset) in os.listdir(os.getcwd())):
            assert False, errormessage


def run(obj):
    obj.checkCorrectDir()

    obj.createDirectories()
    obj.fetchModelFiles()
    obj.fetchDatasetFiles()
    obj.copySeeDotfile()


if __name__ == '__main__':
    obj = FetchFaceDetectionDataset("face-4")
    run(obj)

    obj = FetchFaceDetectionDataset("face-2")
    run(obj)
