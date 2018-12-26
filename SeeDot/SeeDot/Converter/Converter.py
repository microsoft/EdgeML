# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os

import Common
from Converter.Util import *
from Converter.Protonn import *
from Converter.Bonsai import *
from Converter.Lenet import *

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

	def setInput(self, modelDir, inputType, trainingInput, testingInput):
		setModelDir(modelDir)
		
		# Type of normalization: 0 - No norm, 1 - MinMax norm, 2 - L2 norm, 3 - MeanVar norm
		if os.path.isfile(os.path.join(modelDir, "minMaxParams")):
			setNormType(1)
		else:
			setNormType(0)
		
		if inputType == "tsv":
			# Set the TSV training and testing file paths
			setDatasetTSVInput(trainingInput, testingInput)
		elif inputType == "csv":
			# Set the CSV training and testing directory paths containing the corresponding X.csv and Y.csv
			setDatasetCSVInput(trainingInput, testingInput)
		else:
			raise Exception("Unsupported value for input type")
		
		self.inputSet = True

	def run(self):
		if self.inputSet != True:
			raise Exception("Set input paths before running Converter")

		algo, version = getAlgo(), getVersion()

		if algo == Common.Algo.Bonsai and version == Common.Version.Fixed:
			obj = BonsaiFixed()
		elif algo == Common.Algo.Bonsai and version == Common.Version.Float:
			obj = BonsaiFloat()
		elif algo == Common.Algo.Lenet and version == Common.Version.Fixed:
			obj = LenetFixed()
		elif algo == Common.Algo.Lenet and version == Common.Version.Float:
			obj = LenetFloat()
		elif algo == Common.Algo.Protonn and version == Common.Version.Fixed:
			obj = ProtonnFixed()
		elif algo == Common.Algo.Protonn and version == Common.Version.Float:
			obj = ProtonnFloat()
		else:
			assert False

		obj.run()
