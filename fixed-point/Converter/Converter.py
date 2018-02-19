import os
import argparse

from Utils import *
from Protonn import *
from Bonsai import *

# Main file which sets the configurations and creates the corresponding object


class Main:

    def __init__(self, algo, version, datasetType, target, modelDir, datasetOutputDir=None, outputDir=None):
        setAlgo(algo)
        setVersion(version)
        setDatasetType(datasetType)
        setTarget(target)

        setModelDir(modelDir)

        # Setup the output directory
        if outputDir == None:
            outputDir = os.path.join(
                getCurDir(), "output", algo + "-" + version + "-" + datasetType)
            os.makedirs(outputDir, exist_ok=True)
        setOutputDir(outputDir)

        # If the dataset output directory is not set, it is defaulted to the output directory
        if datasetOutputDir == None:
            datasetOutputDir = outputDir
        setDatasetOutputDir(datasetOutputDir)

    # Set the TSV training and testing file paths
    def setTSVinput(self, trainingFile, testingFile):
        setTSVinputFiles(trainingFile, testingFile)

    # Set the CSV training and testing directory paths containing the corresponding X.csv and Y.csv
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
