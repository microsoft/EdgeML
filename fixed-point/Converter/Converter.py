import os
import argparse

from Utils import *
from Protonn import *
from Bonsai import *


class Main:

    def __init__(self, algo, version, datasetType, target, modelDir, datasetOutputDir=None, outputDir=None):
        setAlgo(algo)
        setVersion(version)
        setDatasetType(datasetType)
        setTarget(target)

        setModelDir(modelDir)

        if outputDir == None:
            outputDir = os.path.join(
                getCurDir(), "output", algo + "-" + version + "-" + datasetType)
            os.makedirs(outputDir, exist_ok=True)
        setOutputDir(outputDir)

        if datasetOutputDir == None:
            datasetOutputDir = outputDir
        setDatasetOutputDir(datasetOutputDir)

    def setTSVinput(self, trainingFile, testingFile):
        setTSVinputFiles(trainingFile, testingFile)

    def setCSVinput(self, trainingDir, testingDir):
        setCSVinputDirs(trainingDir, testingDir)

    def run(self):
        algo, version = getAlgo(), getVersion()

        if algo == "bonsai" and version == "fixed":
            obj = BonsaiFixed()
        elif algo == "bonsai" and version == "float":
            obj = BonsaiFloat()
        elif algo == "protonn" and version == "fixed":
            obj = ProtonnFixed()
        elif algo == "protonn" and version == "float":
            obj = ProtonnFloat()

        obj.run()
