# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse
import datetime
from itertools import product
import json
import os
import sys
import tempfile

sys.path.insert(0, '../../')

import edgeml.tools.seedot.common as Common
from edgeml.tools.seedot.compiler import Compiler
from edgeml.tools.seedot.seedot import Main
import edgeml.tools.seedot.util as Util


class Dataset:
    Common = ["cifar-binary", "cr-binary", "cr-multiclass", "curet-multiclass",
              "letter-multiclass", "mnist-binary", "mnist-multiclass",
              "usps-binary", "usps-multiclass", "ward-binary"]
    Default = Common
    All = Common


class MainDriver:

    def __init__(self):
        self.driversAll = ["compiler", "converter", "predictor"]

    def parseArgs(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--driver", choices=self.driversAll,
                            metavar='', help="Driver to use")
        parser.add_argument("-a", "--algo", choices=Common.Algo.All,
                            default=Common.Algo.Default, metavar='', help="Algorithm to run")
        parser.add_argument("-v", "--version", choices=Common.Version.All,
                            default=Common.Version.All, metavar='', help="Floating point code or fixed point code")
        parser.add_argument("-d", "--dataset", choices=Dataset.All,
                            default=Dataset.Default, metavar='', help="Dataset to run")
        parser.add_argument("-dt", "--datasetType", choices=Common.DatasetType.All, default=[
                            Common.DatasetType.Default], metavar='', help="Training dataset or testing dataset")
        parser.add_argument("-t", "--target", choices=Common.Target.All, default=[
                            Common.Target.Default], metavar='', help="X86 code or Arduino code")
        parser.add_argument("-sf", "--max-scale-factor", type=int,
                            metavar='', help="Max scaling factor for code generation")
        parser.add_argument("--load-sf", action="store_true",
                            help="Verify the accuracy of the generated code")
        parser.add_argument("--tempdir", metavar='', help="Scratch directory")
        parser.add_argument("-o", "--outdir", metavar='',
                            help="Directory to output the generated Arduino sketch")

        self.args = parser.parse_args()

        if not isinstance(self.args.algo, list):
            self.args.algo = [self.args.algo]
        if not isinstance(self.args.version, list):
            self.args.version = [self.args.version]
        if not isinstance(self.args.dataset, list):
            self.args.dataset = [self.args.dataset]
        if not isinstance(self.args.datasetType, list):
            self.args.datasetType = [self.args.datasetType]
        if not isinstance(self.args.target, list):
            self.args.target = [self.args.target]

        if self.args.tempdir is not None:
            assert os.path.isdir(
                self.args.tempdir), "Scratch directory doesn't exist"
            Common.tempdir = self.args.tempdir
        else:
            Common.tempdir = os.path.join(tempfile.gettempdir(
            ), "SeeDot", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            os.makedirs(Common.tempdir, exist_ok=True)

        if self.args.outdir is not None:
            assert os.path.isdir(
                self.args.outdir), "Output directory doesn't exist"
            Common.outdir = self.args.outdir
        else:
            Common.outdir = os.path.join(Common.tempdir, "arduino")
            os.makedirs(Common.outdir, exist_ok=True)

    def checkMSBuildPath(self):
        found = False
        for path in Common.msbuildPathOptions:
            if os.path.isfile(path):
                found = True
                Common.msbuildPath = path

        if not found:
            raise Exception("Msbuild.exe not found at the following locations:\n%s\nPlease change the path and run again" % (
                Common.msbuildPathOptions))

    def run(self):
        if Util.windows():
            self.checkMSBuildPath()

        if self.args.driver is None:
            self.runMainDriver()
        elif self.args.driver == "compiler":
            self.runCompilerDriver()
        elif self.args.driver == "converter":
            self.runConverterDriver()
        elif self.args.driver == "predictor":
            self.runPredictorDriver()

    def runMainDriver(self):

        results = self.loadResultsFile()

        for iter in product(self.args.algo, self.args.version, self.args.dataset, self.args.target):
            algo, version, dataset, target = iter

            print("\n========================================")
            print("Executing on %s %s %s %s" %
                  (algo, version, dataset, target))
            print("========================================\n")

            datasetDir = os.path.join("..", "datasets", dataset)
            modelDir = os.path.join("..", "model", dataset)

            if algo == Common.Algo.Bonsai:
                modelDir = os.path.join(modelDir, "BonsaiResults", "Params")
            else:
                modelDir = os.path.join(modelDir, "ProtoNNResults")

            trainingInput = os.path.join(datasetDir, "train.npy")
            testingInput = os.path.join(datasetDir, "test.npy")

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

            obj = Main(algo, version, target, trainingInput,
                       testingInput, modelDir, sf)
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
            obj = Compiler(algo, target, inputFile, outputFile,
                           profileLogFile, self.args.max_scale_factor)
            obj.run()

    def runConverterDriver(self):
        for iter in product(self.args.algo, self.args.version, self.args.dataset, self.args.datasetType, self.args.target):
            algo, version, dataset, datasetType, target = iter

            print("\nGenerating input files for \"" + algo + " " + version +
                  " " + dataset + " " + datasetType + " " + target + "\"...")

            outputDir = os.path.join(
                "Converter", "output", algo + "-" + version + "-" + datasetType, dataset)
            os.makedirs(outputDir, exist_ok=True)

            datasetDir = os.path.join("..", "datasets", dataset)
            modelDir = os.path.join("..", "model", dataset)

            if algo == Common.Algo.Bonsai:
                modelDir = os.path.join(modelDir, "BonsaiResults", "Params")
            elif algo == Common.Algo.Lenet:
                modelDir = os.path.join(modelDir, "LenetModel")
            else:
                modelDir = os.path.join(modelDir, "ProtoNNResults")

            trainingInput = os.path.join(datasetDir, "train.npy")
            testingInput = os.path.join(datasetDir, "test.npy")

            obj = Converter(algo, version, datasetType, target,
                            outputDir, outputDir)
            obj.setInput(modelDir, "tsv", trainingInput, testingInput)
            obj.run()

    def runPredictorDriver(self):
        for iter in product(self.args.algo, self.args.version, self.args.dataset, self.args.datasetType):
            algo, version, dataset, datasetType = iter

            print("\nGenerating input files for \"" + algo + " " +
                  version + " " + dataset + " " + datasetType + "\"...")

            if version == Common.Version.Fixed:
                outputDir = os.path.join(
                    "Predictor", "seedot_fixed", "testing")
                datasetOutputDir = os.path.join(
                    "Predictor", "seedot_fixed", datasetType)
            elif version == Common.Version.Float:
                outputDir = os.path.join(
                    "Predictor", self.algo + "_float", "testing")
                datasetOutputDir = os.path.join(
                    "Predictor", self.algo + "_float", datasetType)

            os.makedirs(datasetOutputDir, exist_ok=True)
            os.makedirs(outputDir, exist_ok=True)

            datasetDir = os.path.join("..", "datasets", dataset)
            modelDir = os.path.join("..", "model", dataset)

            if algo == Common.Algo.Bonsai:
                modelDir = os.path.join(modelDir, "BonsaiResults", "Params")
            elif algo == Common.Algo.Lenet:
                modelDir = os.path.join(modelDir, "LenetModel")
            else:
                modelDir = os.path.join(modelDir, "ProtoNNResults")

            trainingInput = os.path.join(datasetDir, "train.npy")
            testingInput = os.path.join(datasetDir, "test.npy")

            obj = Converter(algo, version, datasetType, Common.Target.X86,
                            datasetOutputDir, outputDir)
            obj.setInput(modelDir, "tsv", trainingInput, testingInput)
            obj.run()

            print("Building and executing " + algo + " " +
                  version + " " + dataset + " " + datasetType + "...")

            outputDir = os.path.join(
                "Predictor", "output", algo + "-" + version)

            curDir = os.getcwd()
            os.chdir(os.path.join("Predictor"))

            obj = Predictor(algo, version, datasetType, outputDir)
            acc = obj.run()

            os.chdir(curDir)

            if acc != None:
                print("Accuracy is %.3f" % (acc))

    def loadResultsFile(self):
        with open(os.path.join("Results", "Results.json")) as data:
            return json.load(data)

if __name__ == "__main__":
    obj = MainDriver()
    obj.parseArgs()
    obj.run()
