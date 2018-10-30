import os
import subprocess
import sys
import operator
import traceback
import argparse
import shutil
from itertools import product
import json

import Common
from Compiler import Compiler
from Converter.Converter import Converter
from Predictor import Predictor


class Main:

	def __init__(self, algo, version, target, trainingFile, testingFile, modelDir, sf, workers):
		self.algo, self.version, self.target, self.trainingFile, self.testingFile, self.modelDir, self.sf, self.numWorkers = algo, version, target, trainingFile, testingFile, modelDir, sf, workers
		self.accuracy = {}

	# Generate the fixed-point code using the input generated from the Converter project
	def compile(self, outputPragmas, sf):
		print("Generating code...", end='')

		# Set input and output files
		inputFile = os.path.join("..", "Predictor", self.algo, "fixed-testing", "input.txt")
		profileLogFile = os.path.join("..", "Predictor", "output", self.algo + "-float", "profile.txt")

		if outputPragmas:
			if self.target == Common.Target.Arduino:
				outputFile = os.path.join("..", "arduino", "predict.cpp")			
			elif self.target == Common.Target.Hls:
				outputFile = os.path.join("..", "fpga", "predict.cpp")			
		else:
			outputFile = os.path.join("..", "Predictor", self.algo + "_fixed.cpp")
		
		try:
			obj = Compiler(self.algo, self.target, outputPragmas, inputFile, outputFile, profileLogFile, sf, self.numWorkers)
			obj.run()
		except:
			print("failed!\n")
			#traceback.print_exc()
			return False

		print("completed")
		return True

	# Run the converter project to generate the input files using reading the training model
	def convert(self, version, datasetType, target):
		print("Generating input files for %s %s dataset..." % (version, datasetType), end='')

		# Create output dirs
		if target == Common.Target.X86:
			datasetOutputDir = os.path.join("..", "Predictor", self.algo, version + "-" + datasetType)
			outputDir = os.path.join("..", "Predictor", self.algo, version + "-testing")
		elif target == Common.Target.Arduino:
			outputDir = os.path.join("..", "Streamer", "input")
			datasetOutputDir = outputDir
		elif target == Common.Target.Hls:
			outputDir = os.path.join("..", "fpga", "input")
			datasetOutputDir = os.path.join("..", "Streamer", "input")
		os.makedirs(datasetOutputDir, exist_ok=True)
		os.makedirs(outputDir, exist_ok=True)

		try:
			obj = Converter(self.algo, version, datasetType, target, datasetOutputDir, outputDir, self.numWorkers)
			obj.setInput(self.modelDir, "tsv", self.trainingFile, self.testingFile)
			obj.run()
		except Exception as e:
			traceback.print_exc()
			return False

		print("done\n")
		return True

	# Build and run the Predictor project
	def predict(self, version, datasetType):
		outputDir = os.path.join("..", "Predictor", "output", self.algo + "-" + version)

		curDir = os.getcwd()
		os.chdir(os.path.join("..", "Predictor"))

		obj = Predictor(self.algo, version, datasetType, outputDir)
		acc = obj.run()

		os.chdir(curDir)

		return acc

	# Compile and run the generated code once for a given scaling factor
	def runOnce(self, version, datasetType, outputPragmas, sf):
		res = self.compile(outputPragmas, sf)
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

			##
			res, exit = self.runOnce(Common.Version.Fixed, Common.DatasetType.Training, False, i)

			if exit == True:
				return False

			# The iterator logic is as follows:
			# Search begins when the first valid scaling factor is found (runOnce returns True)
			# Search ends when the execution fails on a particular scaling factor (runOnce returns False)
			# This is the window where valid scaling factors exist and we select the one with the best accuracy
			if res == True:
				searching = True
			elif searching == True:
				break

		# If search didn't begin at all, something went wrong
		if searching == False:
			return False

		print("\nSearch completed\n")
		print("----------------------------------------------")
		print("Best performing scaling factors with accuracy:")

		self.sf = self.getBestScale()

		return True

	# Reverse sort the accuracies, print the top 5 accuracies and return the best scaling factor
	def getBestScale(self):
		sorted_accuracy = dict(sorted(self.accuracy.items(), key=operator.itemgetter(1), reverse=True)[:5])
		print(sorted_accuracy)
		return next(iter(sorted_accuracy))

	# Find the scaling factor which works best on the training dataset and predict on the testing dataset
	def findBestScalingFactor(self):
		print("-------------------------------------------------")
		print("Performing search to find the best scaling factor")
		print("-------------------------------------------------\n")

		# Generate input files for training dataset
		##
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
		##
		res = self.convert(Common.Version.Fixed, Common.DatasetType.Testing, Common.Target.X86)
		if res == False:
			return False

		##
		# Compile and run code using the best scaling factor
		res = self.runOnce(Common.Version.Fixed, Common.DatasetType.Testing, False, self.sf)
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

		acc = self.predict(Common.Version.Float, Common.DatasetType.Training)
		if acc == None:
			return False

		print("Accuracy is %.3f%%\n" % (acc))

	# Generate code for Arduino
	def compileForTarget(self):
		print("------------------------------")
		print("Generating code for ",self.target,"...")
		print("------------------------------\n")

		res = self.convert(Common.Version.Fixed, Common.DatasetType.Testing, self.target)
		if res == False:
			return False

		# Copy file
		srcFile = os.path.join("..", "Streamer", "input", "model.h")
		destFile = os.path.join("..", self.target, "model.h")
		shutil.copyfile(srcFile, destFile)

		res = self.compile(True, self.sf)
		if res == False:
			return False

	def runForFixed(self):
		# Collect runtime profile for ProtoNN
		if self.algo == Common.Algo.Protonn:
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
		self.compileForTarget()

		return True

	def runForFloat(self):
		print("---------------------------")
		print("Executing for X86 target...")
		print("---------------------------\n")

		res = self.convert(Common.Version.Float, Common.DatasetType.Testing, Common.Target.X86)
		if res == False:
			return False

		acc = self.predict(Common.Version.Float, Common.DatasetType.Testing)
		if acc == None:
			return False
		else:
			self.testingAccuracy = acc

		print("Accuracy is %.3f%%\n" % (acc))

		print("------------------------------")
		print("Generating code for Arduino...")
		print("------------------------------\n")

		res = self.convert(Common.Version.Float, Common.DatasetType.Testing, Common.Target.Arduino)
		if res == False:
			return False

		# Copy model.h
		srcFile = os.path.join("..", "Streamer", "input", "model.h")
		destFile = os.path.join("..", "arduino", "model.h")
		shutil.copyfile(srcFile, destFile)

		# Copy predict.cpp
		srcFile = os.path.join("..", "arduino", "floating-point", self.algo + "_float.cpp")
		destFile = os.path.join("..", "arduino", "predict.cpp")
		shutil.copyfile(srcFile, destFile)

		return True

	def run(self):
		if self.version == Common.Version.Fixed:
			return self.runForFixed()
		else:
			return self.runForFloat()


class MainDriver:

	def __init__(self):
		self.driversAll = ["compiler", "converter", "predictor"]

	def parseArgs(self):
		parser = argparse.ArgumentParser()

		parser.add_argument("--driver", choices = self.driversAll, metavar = '', help = "Driver to use")
		parser.add_argument("-a", "--algo", choices = Common.Algo.All, default = Common.Algo.Default, metavar = '', help = "Algorithm to run")
		parser.add_argument("-v", "--version", choices = Common.Version.All, default = Common.Version.All, metavar = '', help = "Floating point code or fixed point code")
		parser.add_argument("-d", "--dataset", choices = Common.Dataset.All, default = Common.Dataset.Default, metavar = '', help = "Dataset to run")
		parser.add_argument("-dt", "--datasetType", choices = Common.DatasetType.All, default = [Common.DatasetType.Default], metavar = '', help = "Training dataset or testing dataset")
		parser.add_argument("-t", "--target", choices = Common.Target.All, default = [Common.Target.Default], metavar = '', help = "Desktop code or Arduino code or Fpga HLS code")
		parser.add_argument("-sf", "--max-scale-factor", type = int, metavar = '', help = "Max scaling factor for code generation")
		parser.add_argument("--output-pragmas", action = "store_true", help = "Add relevant pragmas to the generated code")
		parser.add_argument("--load-sf", action = "store_true", help = "Verify the accuracy of the generated code")
		parser.add_argument("--workers", type=int,default = 1, metavar = '', help = "number of worker threads to parallelize SparseMul on FPGAs only")
		
		self.args = parser.parse_args()

		if not isinstance(self.args.algo, list): self.args.algo = [self.args.algo]
		if not isinstance(self.args.version, list):	self.args.version = [self.args.version]
		if not isinstance(self.args.dataset, list): self.args.dataset = [self.args.dataset]
		if not isinstance(self.args.datasetType, list):	self.args.datasetType = [self.args.datasetType]
		if not isinstance(self.args.target, list): self.args.target = [self.args.target]

	def writeVivadoScripts(self):

		#script for HLS Synthesis
		FPGAHLSSynAutoFile = os.path.join("..","fpga","scriptSyn.tcl")
		fp = open(FPGAHLSSynAutoFile, 'w')
		fp.write("##################################################################\n")
		fp.write("## Generated Script to run Vivado HLS C Synthesis\n")
		fp.write("## PLEASE DO NOT EDIT\n")
		fp.write("##################################################################\n")
		fp.write("cd ../fpga/ \n")
		fp.write("open_project -reset SeeDotFpga_%s\n" % self.args.algo[0])
		fp.write("set_top %sFixed\n" % self.args.algo[0])
		fp.write("add_files ../fpga/predict.cpp\n")
		fp.write("add_files ../fpga/model.h\n")
		fp.write("open_solution -reset \"solution1\"\n")
		fp.write("set_part {xc7a35ticsg324-1l} -tool vivado\n")
		fp.write("create_clock -period 100 -name default\n")
		fp.write("config_array_partition -auto_partition_threshold 9 -auto_promotion_threshold 64\n")
		fp.write("csynth_design\n")
		fp.write("\n\n\n")
		fp.write("exit")
		fp.close

		##script for Vivado Synthesis, P&R and IP Gen
		#FPGAHLSIPGenAutoFile = os.path.join ("..","Predictor","fpgaOutput","scriptExportIP.tcl")
		#fp = open(FPGAHLSIPGenAutoFile, 'w')
		#fp.write("##################################################################\n")
		#fp.write("## Generated Script to run Vivado Synthesis, P&R and IP Gen\n")
		#fp.write("## PLEASE DO NOT EDIT\n")
		#fp.write("##################################################################\n")
		#fp.write("open_project FltFpga_%s\n" % self.args.algo[0])
		#fp.write("open_solution \"solution1\"\n")
		#fp.write("export_design -flow syn -rtl verilog -format ip_catalog -description \"IP of predictor generated from FLT\" -display_name \"predictor\" \n")
		#fp.write("\n\n\n")
		#fp.write("exit")
		#fp.close

		#Create batch files for execution
		SynGenFile = os.path.join("..","fpga","synGen.bat")
		fp = open(SynGenFile, 'w')
		fp.write("@echo off\n\n")
		fp.write("set PATH=%%~dp0;%%PATH%%;%%~dp0..\\msys\\bin;%s\n" % Common.vivadoInstallPath)
		fp.write("vivado_hls -f ../fpga/scriptSyn.tcl")
		fp.close

		IPGenFile = os.path.join("..","fpga","IPGen.bat")
		fp = open(IPGenFile, 'w')
		fp.write("@echo off\n\n")
		fp.write("set PATH=%%~dp0;%%PATH%%;%%~dp0..\\msys\\bin;%s\n" % Common.vivadoInstallPath)
		fp.write("vivado_hls -f ../fpga/scriptExportIP.tcl")
		fp.close

	def runVivadoScripts(self):
		FPGAHLSSynAutoFile = os.path.join("..","fpga","synGen.bat")
		FPGAHLSIPGenAutoFile = os.path.join("..","fpga","IPGen.bat")

		#automate tasks of Vivado HLS and Vivado IP generation
		print("Automatic Generation of Vivado HLS synthesized code started")
		process = subprocess.call(FPGAHLSSynAutoFile)
		if process == 1:
			print("FAILED Vivado HLS synthesis!!\n")
		else:
			print("success Vivado HLS synthesis")
		
		#print("Automatic Vivado IP generation started")
		#process = subprocess.call(FPGAHLSIPGenAutoFile)
		#if process == 1:
		#	print("FAILED  Vivado IP generation!!\n")
		#else:
		#	print("success  Vivado IP generation")

	def copyOutputs(self):
		#srcfilePath = os.path.join("FltFpga" + self.args.algo[0]+ "\\")
		srcfilePath = os.path.join("scriptSyn.tcl")
		destFilePath = os.path.join("..","Predictor","fpgaOutput\\",)
		print(srcfilePath, destFilePath)
		process = subprocess.call(["xcopy" , srcfilePath, destFilePath, "/s", "/o", "/i", "/x", "/e", "/h" , "/k"])
		if process == 1:
			print("FAILED  Copy!!\n")
		else:
			print("success  Copy")

	def run(self):
		if not os.path.isfile(Common.msbuildPath):
			raise Exception("Msbuild.exe not found at the following location:\n%s\nPlease change the path and run again" % (Common.msbuildPath))
		Common.LUTCount = 0
		if self.args.driver is None:
			self.runMainDriver()
		elif self.args.driver == "compiler":
			self.runCompilerDriver()
		elif self.args.driver == "converter":
			self.runConverterDriver()
		elif self.args.driver == "predictor":
			self.runPredictorDriver()

		#AUTOMATE Vivado HLS Syntheis and Vivado IP Generation for Fpga
		if self.args.target[0] == Common.Target.Hls:
			if not os.path.isdir(Common.vivadoInstallPath):
				raise Exception("Vivado not found at the following location:\n%s\nPlease change the path or check Installation and run again" % (Common.vivadoInstallPath))
			self.writeVivadoScripts()
			self.runVivadoScripts()
			#self.copyOutputs()

	def runMainDriver(self):
		results = self.loadResultsFile()

		for iter in product(self.args.algo, self.args.version, self.args.dataset, self.args.target):
			algo, version, dataset, target = iter

			print("\n========================================")
			print("Executing on %s %s %s %s" % (algo, version, dataset, target))
			print("========================================\n")

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

			try:
				if version == Common.Version.Float:
					key = 'float32'
				elif Common.wordLength == 16:
					key = 'int16'
				elif Common.wordLength == 32:
					key = 'int32'
				else:
					assert False
				
				curr = results[algo][key][dataset]
				
				expectedAcc = curr['accuracy']
				if version == Common.Version.Fixed:
					bestScale = curr['sf']

			except Exception as e:
				assert self.args.load_sf == False
				expectedAcc = 0

			if self.args.load_sf:
				sf = bestScale
			else:
				sf = self.args.max_scale_factor

			obj = Main(algo, version, target, trainingInput, testingInput, modelDir, sf, self.args.workers)
			obj.run()
			
			acc = obj.testingAccuracy
			if acc != expectedAcc:
				print("FAIL: Expected accuracy %f%%" % (expectedAcc))
				return
			elif version == Common.Version.Fixed and obj.sf != bestScale:
				print("FAIL: Expected best scale %d" % (bestScale))
				return
			else:
				print("PASS")

	def runCompilerDriver(self):
		for iter in product(self.args.algo, self.args.target):
			algo, target = iter

			print("\nGenerating code for " + algo + " " + target + "...")
					
			inputFile = os.path.join("input", algo + ".sd")
			profileLogFile = os.path.join("input", "profile.txt")
					
			outputDir = os.path.join("output")
			os.makedirs(outputDir, exist_ok=True)

			outputFile = os.path.join(outputDir, algo + "-fixed.cpp")
			obj = Compiler(algo, target, self.args.output_pragmas, inputFile, outputFile, profileLogFile, self.args.max_scale_factor, self.args.workers)
			obj.run()

	def runConverterDriver(self):
		for iter in product(self.args.algo, self.args.version, self.args.dataset, self.args.datasetType, self.args.target):
			algo, version, dataset, datasetType, target = iter

			print("\nGenerating input files for \"" + algo + " " + version + " " + dataset + " " + datasetType + " " + target + "\"...")

			outputDir = os.path.join("Converter", "output", algo + "-" + version + "-" + datasetType, dataset)
			os.makedirs(outputDir, exist_ok=True)

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
							
			obj = Converter(algo, version, datasetType, target, outputDir, outputDir,self.args.workers)
			obj.setInput(modelDir, "tsv", trainingInput, testingInput)
			obj.run()

	def runPredictorDriver(self):
		for iter in product(self.args.algo, self.args.version, self.args.dataset, self.args.datasetType):
			algo, version, dataset, datasetType = iter
			
			print("\nGenerating input files for \"" + algo + " " + version + " " + dataset + " " + datasetType + "\"...")

			datasetOutputDir = os.path.join("..", "Predictor", algo, version + "-" + datasetType)
			outputDir = os.path.join("..", "Predictor", algo, version + "-testing")

			os.makedirs(datasetOutputDir, exist_ok=True)
			os.makedirs(outputDir, exist_ok=True)

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
							
			obj = Converter(algo, version, datasetType, Common.Target.X86, datasetOutputDir, outputDir, self.args.workers)
			obj.setInput(modelDir, "tsv", trainingInput, testingInput)
			obj.run()

			print("Building and executing " + algo + " " + version + " " + dataset + " " + datasetType + "...")

			outputDir = os.path.join("..", "Predictor", "output", algo + "-" + version)

			curDir = os.getcwd()
			os.chdir(os.path.join("..", "Predictor"))

			obj = Predictor(algo, version, datasetType, outputDir)
			acc = obj.run()

			os.chdir(curDir)

			if acc != None:
				print("Accuracy is %.3f" % (acc))

	def loadResultsFile(self):
		with open(os.path.join("..", "Results", "Results.json")) as data:
			return json.load(data)

class MainDriverOld:

	algosAll = ["bonsai", "protonn"]

	def __init__(self):
		# Parser to accept command line arguments
		parser = argparse.ArgumentParser()

		parser.add_argument("-a", "--algo", choices=self.algosAll,
								  required=True, metavar='', help="Algorithm to run")
		parser.add_argument("--train", required=True,
							metavar='', help="Training dataset file")
		parser.add_argument("--test", required=True,
							metavar='', help="Testing dataset file")
		parser.add_argument("--model", required=True,
							metavar='', help="Directory containing model")

		self.args = parser.parse_args()

		# Verify the input files and directory exists
		if not os.path.isfile(self.args.train):
			raise Exception("Training dataset file doesn't exist")
		if not os.path.isfile(self.args.test):
			raise Exception("Testing dataset file doesn't exist")
		if not os.path.isdir(self.args.model):
			raise Exception("Model directory doesn't exist")

	def run(self):
		if not os.path.isfile(Common.msbuildPath):
			raise Exception("Msbuild.exe not found at the following locaiton:\n%s\nPlease change the path and run again" % (Common.msbuildPath))

		print("\n====================")
		print("Executing on %s" % (self.args.algo))
		print("====================\n")
		obj = Main(self.args.algo, self.args.train,
				   self.args.test, self.args.model)
		obj.run()


if __name__ == "__main__":
	obj = MainDriver()
	obj.parseArgs()
	obj.run()
