# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os

from Converter.Quantizer import *
from Converter.Util import *

import Common

# Main file which sets the configurations and creates the corresponding object
class Converter:

	def __init__(self, algo, version, datasetType, target, datasetOutputDir, outputDir, workers):
		setAlgo(algo)
		setVersion(version)
		setDatasetType(datasetType)
		setTarget(target)
		setNumWorkers(workers)

		# Set output directories
		setDatasetOutputDir(datasetOutputDir)
		setOutputDir(outputDir)

	def setInput(self, inputFile, modelDir, trainingInput, testingInput):
		setInputFile(inputFile)
		setModelDir(modelDir)
		setDatasetInput(trainingInput, testingInput)
		
		self.inputSet = True

	def run(self):
		if self.inputSet != True:
			raise Exception("Set input paths before running Converter")

		if getVersion() == Common.Version.Fixed:
			obj = QuantizerFixed()
		elif getVersion() == Common.Version.Float:
			obj = QuantizerFloat()

		obj.run()
