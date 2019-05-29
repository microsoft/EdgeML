# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse
import os
import shutil
import sys
import operator
import traceback

from Converter.Converter import Converter
from Converter.Bonsai import Bonsai
from Converter.Protonn import Protonn

import Common
from Compiler import Compiler
from Predictor import Predictor


class Main:

	def __init__(self, algo, version, target, trainingFile, testingFile, modelDir, sf, workers):
		self.algo, self.version, self.target, self.trainingFile, self.testingFile, self.modelDir, self.sf, self.numWorkers = algo, version, target, trainingFile, testingFile, modelDir, sf, workers
		self.accuracy = {}

	# Generate the fixed-point code using the input generated from the Converter
	# project
	def compile(self, version, target, sf):
		print("Generating code...", end='')

		# Set input and output files
		inputFile = os.path.join(self.modelDir, "input.sd")
		profileLogFile = os.path.join("..", "Predictor", "output", "float", "profile.txt")
		
		logDir = os.path.join("output")
		os.makedirs(logDir, exist_ok=True)
		if version == Common.Version.Float:
			outputLogFile = os.path.join(logDir, "log-float.txt")
		else:
			outputLogFile = os.path.join(logDir, "log-fixed-" + str(abs(sf)) + ".txt")

		if target == Common.Target.Arduino:
			outputDir = os.path.join("..", "arduino")
		elif target == Common.Target.Hls:
			outputDir = os.path.join("..", "hls")
		elif target == Common.Target.Verilog:
			outputDir = os.path.join("..", "verilog")
		elif target == Common.Target.X86:
			outputDir = os.path.join("..", "Predictor")
		
		try:
			obj = Compiler(self.algo, version, target, inputFile, outputDir, profileLogFile, sf, outputLogFile, self.numWorkers)
			obj.run()
		except:
			print("failed!\n")
			traceback.print_exc()
			return False

		self.scaleForX = obj.scaleForX

		print("completed")
		return True

	# Run the converter project to generate the input files using reading the
	# training model
	def convert(self, version, datasetType, target):
		print("Generating input files for %s %s dataset..." % (version, datasetType), end='')

		# Create output dirs
		if target == Common.Target.Arduino:
			outputDir = os.path.join("..", "Streamer", "input")
			datasetOutputDir = outputDir
		elif target == Common.Target.Hls:
			outputDir = os.path.join("..", "hls", "input")
			datasetOutputDir = os.path.join("..", "Streamer", "input")
		elif target == Common.Target.Verilog:
			outputDir = os.path.join("..", "verilog", "input")
			datasetOutputDir = os.path.join("..", "Streamer", "input")
		elif target == Common.Target.X86:
			outputDir = os.path.join("..", "Predictor")
			datasetOutputDir = os.path.join("..", "Predictor", "input")
		else:
			assert False
		
		os.makedirs(datasetOutputDir, exist_ok=True)
		os.makedirs(outputDir, exist_ok=True)

		inputFile = os.path.join(self.modelDir, "input.sd")

		try:
			obj = Converter(self.algo, version, datasetType, target, datasetOutputDir, outputDir, self.numWorkers)
			obj.setInput(inputFile, self.modelDir, self.trainingFile, self.testingFile)
			obj.run()
		except Exception as e:
			traceback.print_exc()
			return False

		print("done\n")
		return True

	# Build and run the Predictor project
	def predict(self, version, datasetType):
		outputDir = os.path.join("..", "Predictor", "output", version)

		curDir = os.getcwd()
		os.chdir(os.path.join("..", "Predictor"))

		obj = Predictor(self.algo, version, datasetType, outputDir, self.scaleForX)
		acc = obj.run()

		os.chdir(curDir)

		return acc

	# Compile and run the generated code once for a given scaling factor
	def runOnce(self, version, datasetType, target, sf):
		res = self.compile(version, target, sf)
		if res == False:
			return False, False

		acc = self.predict(version, datasetType)
		if acc == None:
			return False, True

		self.accuracy[sf] = acc
		print("Accuracy is %.3f%%\n" % (acc))

		return True, False

	# Iterate over multiple scaling factors and store their accuracies
	def performSearch(self):
		start, end = Common.maxScaleRange
		searching = False

		for i in range(start, end, -1):
			print("Testing with max scale factor of " + str(i))

			res, exit = self.runOnce(Common.Version.Fixed, Common.DatasetType.Training, Common.Target.X86, i)

			if exit == True:
				return False

			# The iterator logic is as follows:
			# Search begins when the first valid scaling factor is found (runOnce
			# returns True)
			# Search ends when the execution fails on a particular scaling factor
			# (runOnce returns False)
			# This is the window where valid scaling factors exist and we select the one
			# with the best accuracy
			if res == True:
				searching = True
			elif searching == True:
				#break
				pass

		# If search didn't begin at all, something went wrong
		if searching == False:
			return False

		print("\nSearch completed\n")
		print("----------------------------------------------")
		print("Best performing scaling factors with accuracy:")

		self.sf = self.getBestScale()

		return True

	# Reverse sort the accuracies, print the top 5 accuracies and return the best
	# scaling factor
	def getBestScale(self):
		sorted_accuracy = dict(sorted(self.accuracy.items(), key=operator.itemgetter(1), reverse=True)[:5])
		print(sorted_accuracy)
		return next(iter(sorted_accuracy))

	# Find the scaling factor which works best on the training dataset and predict
	# on the testing dataset
	def findBestScalingFactor(self):
		print("-------------------------------------------------")
		print("Performing search to find the best scaling factor")
		print("-------------------------------------------------\n")

		# Generate input files for training dataset
		res = self.convert(Common.Version.Fixed, Common.DatasetType.Training, Common.Target.X86)
		if res == False:
			return False

		# Search for the best scaling factor
		res = self.performSearch()
		if res == False:
			return False

		print("Best scaling factor = %d" % (self.sf))

		return True

	def runOnTestingDataset(self):
		print("\n-------------------------------")
		print("Prediction on testing dataset")
		print("-------------------------------\n")

		print("Setting max scaling factor to %d\n" % (self.sf))

		# Generate files for the testing dataset
		res = self.convert(Common.Version.Fixed, Common.DatasetType.Testing, Common.Target.X86)
		if res == False:
			return False

		# Compile and run code using the best scaling factor
		res = self.runOnce(Common.Version.Fixed, Common.DatasetType.Testing, Common.Target.X86, self.sf)
		if res == False:
			return False

		return True

	# Generate files for training dataset and perform a profiled execution
	def collectProfileData(self):
		print("-----------------------")
		print("Collecting profile data")
		print("-----------------------")

		res = self.convert(Common.Version.Float, Common.DatasetType.Training, Common.Target.X86)
		if res == False:
			return False

		res = self.compile(Common.Version.Float, Common.Target.X86, self.sf)
		if res == False:
			return False

		acc = self.predict(Common.Version.Float, Common.DatasetType.Training)
		if acc == None:
			return False

		print("Accuracy is %.3f%%\n" % (acc))

	# Generate code for Arduino
	def compileFixedForTarget(self):
		print("------------------------------")
		print("Generating code for %s..." % (self.target))
		print("------------------------------\n")

		res = self.convert(Common.Version.Fixed, Common.DatasetType.Testing, self.target)
		if res == False:
			return False

		# Copy model.h file
		srcFile = os.path.join("..", "Streamer", "input", "model_fixed.h")
		destFile = os.path.join("..", self.target, "model.h")
		shutil.copyfile(srcFile, destFile)

		# Copy library.h file
		srcFile = os.path.join("..", self.target, "library", "library_fixed.h")
		destFile = os.path.join("..", self.target, "library.h")
		shutil.copyfile(srcFile, destFile)

		res = self.compile(Common.Version.Fixed, self.target, self.sf)
		if res == False:
			return False

		return True

	def runForFixed(self):
		# Collect runtime profile
		res = self.collectProfileData()
		if res == False:
			return False

		# Obtain best scaling factor
		if self.sf == None:
			res = self.findBestScalingFactor()
			if res == False:
				return False

		res = self.runOnTestingDataset()
		if res == False:
			return False
		else:
			self.testingAccuracy = self.accuracy[self.sf]

		# Generate code for target
		if self.target == Common.Target.Arduino:
			return self.compileFixedForTarget()

		return True

	def compileFloatForTarget(self):
		print("------------------------------")
		print("Generating code for %s..." % (self.target))
		print("------------------------------\n")

		res = self.convert(Common.Version.Float, Common.DatasetType.Testing, self.target)
		if res == False:
			return False

		res = self.compile(Common.Version.Float, self.target, self.sf)
		if res == False:
			return False

		# Copy model.h
		srcFile = os.path.join("..", "Streamer", "input", "model_float.h")
		destFile = os.path.join("..", self.target, "model.h")
		shutil.copyfile(srcFile, destFile)

		# Copy library.h file
		srcFile = os.path.join("..", self.target, "library", "library_float.h")
		destFile = os.path.join("..", self.target, "library.h")
		shutil.copyfile(srcFile, destFile)

		return True

	def runForFloat(self):
		print("---------------------------")
		print("Executing for X86 target...")
		print("---------------------------\n")

		res = self.convert(Common.Version.Float, Common.DatasetType.Testing, Common.Target.X86)
		if res == False:
			return False

		res = self.compile(Common.Version.Float, Common.Target.X86, self.sf)
		if res == False:
			return False

		acc = self.predict(Common.Version.Float, Common.DatasetType.Testing)
		if acc == None:
			return False
		else:
			self.testingAccuracy = acc

		print("Accuracy is %.3f%%\n" % (acc))

		if self.target == Common.Target.Arduino:
			return self.compileFloatForTarget()

		return True

	def run(self):

		sys.setrecursionlimit(10000)

		if self.version == Common.Version.Fixed:
			return self.runForFixed()
		else:
			return self.runForFloat()


class MainDriver:

	def parseArgs(self):
		parser = argparse.ArgumentParser()

		parser.add_argument("-a", "--algo", choices = Common.Algo.All, metavar = '', help = "Algorithm to run")
		parser.add_argument("--train", required=True, metavar='', help="Training set file")
		parser.add_argument("--test", required=True, metavar='', help="Testing set file")
		parser.add_argument("--model", required=True, metavar='', help="Directory containing trained model")
		parser.add_argument("--convert", action = "store_true", help = "Standardize the Bonsai/ProtoNN trained models for SeeDot")
		
		self.args = parser.parse_args()

		# Verify the input files and directory exists
		if not os.path.isfile(self.args.train):
			raise Exception("Training set doesn't exist")
		if not os.path.isfile(self.args.test):
			raise Exception("Testing set doesn't exist")
		if not os.path.isdir(self.args.model):
			raise Exception("Model directory doesn't exist")

	def checkMSBuildPath(self):
		found = False
		for path in Common.msbuildPathOptions:
			if os.path.isfile(path):
				found = True
				Common.msbuildPath = path
		
		if not found:
			raise Exception("Msbuild.exe not found at the following locations:\n%s\nPlease change the path and run again" % (Common.msbuildPathOptions))

	def run(self):
		self.checkMSBuildPath()

		algo, trainingInput, testingInput, modelDir = self.args.algo, self.args.train, self.args.test, self.args.model

		print("\n================================")
		print("Executing on %s for Arduino" % (algo))
		print("--------------------------------")
		print("Train file: %s" % (trainingInput))
		print("Test file: %s" % (testingInput))
		print("Model directory: %s" % (modelDir))
		print("================================\n")

		if self.args.convert:
			datasetDir = os.path.join("..", "datasets", "datasets", dataset)
			modelDir = os.path.join("..", "model", dataset)

			if algo == Common.Algo.Bonsai:
				modelDir = os.path.join(modelDir, "BonsaiResults", "Params")
			elif algo == Common.Algo.Lenet:
				modelDir = os.path.join(modelDir, "LenetModel")
			else:
				modelDir = os.path.join(modelDir, "ProtoNNResults")
							
			trainingInput = os.path.join(datasetDir, "training-full.tsv")
			testingInput = os.path.join(datasetDir, "testing.tsv")
				
			datasetOutputDir = os.path.join("temp", "dataset-processed", algo, dataset)
			modelOutputDir = os.path.join("temp", "model-processed", algo, dataset)

			os.makedirs(datasetOutputDir, exist_ok=True)
			os.makedirs(modelOutputDir, exist_ok=True)
			
			if algo == Common.Algo.Bonsai:
				obj = Bonsai(trainingInput, testingInput, modelDir, datasetOutputDir, modelOutputDir)
				obj.run()
			elif algo == Common.Algo.Protonn:
				obj = Protonn(trainingInput, testingInput, modelDir, datasetOutputDir, modelOutputDir)
				obj.run()

			trainingInput = os.path.join(datasetOutputDir, "train.npy")
			testingInput = os.path.join(datasetOutputDir, "test.npy")
			modelDir = modelOutputDir

		obj = Main(algo, Common.Version.Fixed, Common.Target.Arduino, trainingInput, testingInput, modelDir, None, 1)
		obj.run()

if __name__ == "__main__":
	obj = MainDriver()
	obj.parseArgs()
	obj.run()
