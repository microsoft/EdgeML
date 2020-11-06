# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os

from seedot.compiler.converter.quantizer import *
from seedot.compiler.converter.util import *

import seedot.config as config

# Main file which sets the configurations and creates the corresponding object


class Converter:

    def __init__(self, algo, version, datasetType, target, source, datasetOutputDir, outputDir, varsForBitwidth={}, allScales={}, numOutputs=1, biasShifts={}, scaleForY=None):
        setAlgo(algo)
        setVersion(version)
        setDatasetType(datasetType)
        setTarget(target)

        # Set output directories
        setDatasetOutputDir(datasetOutputDir)
        setOutputDir(outputDir)

        self.sparseMatrixSizes = {}
        self.varsForBitwidth = varsForBitwidth
        self.allScales = allScales
        self.numOutputs = numOutputs
        self.source = source
        self.biasShifts = biasShifts
        self.scaleForY = scaleForY

    def setInput(self, inputFile, modelDir, trainingInput, testingInput):
        setInputFile(inputFile)
        setModelDir(modelDir)
        setDatasetInput(trainingInput, testingInput)
        self.inputSet = True

    def run(self):
        if self.inputSet != True:
            raise Exception("Set input paths before running Converter")

        if getVersion() == config.Version.fixed:
            obj = QuantizerFixed(self.varsForBitwidth, self.allScales, self.numOutputs, self.biasShifts, self.scaleForY)
        elif getVersion() == config.Version.floatt:
            obj = QuantizerFloat(self.numOutputs)

        obj.run(self.source)

        self.sparseMatrixSizes = obj.sparseMatSizes
