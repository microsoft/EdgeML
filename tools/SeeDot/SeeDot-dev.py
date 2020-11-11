# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse
import csv
import datetime
from itertools import product
import json
import numpy as np
import os
import shutil
import tempfile

import seedot.config as config
import seedot.main as main
import seedot.predictor as predictor
import seedot.util as util

import seedot.compiler.converter.bonsai as bonsai
import seedot.compiler.converter.converter as converter
import seedot.compiler.converter.protonn as protonn


class Dataset:
    common = ["cifar-binary", "cr-binary", "cr-multiclass", "curet-multiclass",
              "letter-multiclass", "mnist-binary", "mnist-multiclass",
              "usps-binary", "usps-multiclass", "ward-binary"]
    extra = ["cifar-multiclass", "dsa", "eye-binary", "farm-beats",
             "interactive-cane", "spectakoms", "usps10", "whale-binary",
             "HAR-2", "HAR-6", "MNIST-10", "Google-12", "Google-30", "Wakeword-2",
             "wider-regression", "wider-mbconv", "face-1", "face-2", "face-2-rewrite", 
             "face-3", "face-4", "test"]
    default = common
    #default = ["spectakoms", "usps10", "HAR-2", "HAR-6", "dsa", "MNIST-10", "Google-12", "Google-30", "Wakeword-2"]
    all = common + extra

    datasetDir = os.path.join("..", "datasets", "datasets")
    modelDir = os.path.join("..", "model")

    datasetProcessedDir = os.path.join("datasets")
    modelProcessedDir = os.path.join("model")


class MainDriver:

    def parseArgs(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("-a", "--algo", choices=config.Algo.all,
                            default=config.Algo.default, metavar='', help="Algorithm to run")
        parser.add_argument("-v", "--version", choices=config.Version.all,
                            default=config.Version.default, metavar='', help="Floating-point or fixed-point")
        parser.add_argument("-d", "--dataset", choices=Dataset.all,
                            default=Dataset.default, metavar='', help="Dataset to use")
        parser.add_argument("-m", "--maximisingMetric", choices=config.MaximisingMetric.all, metavar='', 
                            help="What metric to maximise during exploration",default=config.MaximisingMetric.default)
        parser.add_argument("-n", "--numOutputs", type=int, metavar='', 
                            help="Number of simultaneous outputs of the inference procedure",default=1)
        parser.add_argument("-dt", "--datasetType", choices=config.DatasetType.all,
                            default=config.DatasetType.default, metavar='', help="Training dataset or testing dataset")
        parser.add_argument("-t", "--target", choices=config.Target.all,
                            default=config.Target.default, metavar='', help="X86 code or Arduino sketch")
        parser.add_argument("-s", "--source", metavar='', choices=config.Source.all, 
                            default=config.Source.default, help="model source type seedot/onnx/tf")                    
        parser.add_argument("-sf", "--max-scale-factor", type=int,
                            metavar='', help="Max scaling factor for code generation")
        parser.add_argument("--load-sf", action="store_true",
                            help="Use a pre-determined value for max scale factor")

        parser.add_argument("--convert", action="store_true",
                            help="Convert raw input and model files to standard numpy format")

        parser.add_argument("--tempdir", metavar='',
                            help="Scratch directory for intermediate files")
        parser.add_argument("-o", "--outdir", metavar='',
                            help="Directory to output the generated Arduino sketch")

        parser.add_argument("--driver", choices=["compiler", "converter", "predictor"],
                            metavar='', help="Invoke specific components of the tool instead of the entire workflow")

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
        if not isinstance(self.args.maximisingMetric, list):
            self.args.maximisingMetric = [self.args.maximisingMetric]

        if self.args.tempdir is not None:
            assert os.path.isdir(
                self.args.tempdir), "Scratch directory doesn't exist"
            config.tempdir = self.args.tempdir
        else:
            # config.tempdir = os.path.join(tempfile.gettempdir(
            # ), "SeeDot", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            config.tempdir = "temp"
            if os.path.exists(config.tempdir):
                shutil.rmtree(config.tempdir)
            os.makedirs(config.tempdir)

        if self.args.outdir is not None:
            assert os.path.isdir(
                self.args.outdir), "Output directory doesn't exist"
            config.outdir = self.args.outdir
        else:
            if self.args.target == [config.Target.arduino]:
                config.outdir = os.path.join("arduinodump", "arduino")
            elif self.args.target == [config.Target.m3]:
                config.outdir = os.path.join("m3")
            else:
                config.outdir = os.path.join(config.tempdir, "arduino")
            os.makedirs(config.outdir, exist_ok=True)

    def checkMSBuildPath(self):
        found = False
        for path in config.msbuildPathOptions:
            if os.path.isfile(path):
                found = True
                config.msbuildPath = path

        if not found:
            raise Exception("Msbuild.exe not found at the following locations:\n%s\nPlease change the path and run again" % (
                config.msbuildPathOptions))

    def setGlobalFlags(self):
        np.seterr(all='warn')

    def run(self):
        if util.windows():
            self.checkMSBuildPath()

        self.setGlobalFlags()

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

        for iter in product(self.args.algo, self.args.version, self.args.dataset, self.args.target, self.args.maximisingMetric, [16]):
            algo, version, dataset, target, maximisingMetric, wordLength = iter

            #config.wordLength = wordLength
            #config.maxScaleRange = 0, -wordLength

            print("\n========================================")
            print("Executing on %s %s %s %s" %
                  (algo, version, dataset, target))
            print("========================================\n")

            if self.args.convert:
                datasetDir = os.path.join(Dataset.datasetDir, dataset)
                modelDir = os.path.join(Dataset.modelDir, dataset)

                if algo == config.Algo.bonsai:
                    modelDir = os.path.join(
                        modelDir, "BonsaiResults", "Params")
                elif algo == config.Algo.lenet:
                    modelDir = os.path.join(modelDir, "LenetModel")
                else:
                    modelDir = os.path.join(modelDir, "ProtoNNResults")

                trainingInput = os.path.join(datasetDir, "training-full.tsv")
                testingInput = os.path.join(datasetDir, "testing.tsv")

                datasetOutputDir = os.path.join(
                    "temp", "dataset-processed", algo, dataset)
                modelOutputDir = os.path.join(
                    "temp", "model-processed", algo, dataset)

                os.makedirs(datasetOutputDir, exist_ok=True)
                os.makedirs(modelOutputDir, exist_ok=True)

                if algo == config.Algo.bonsai:
                    obj = bonsai.Bonsai(trainingInput, testingInput,
                                        modelDir, datasetOutputDir, modelOutputDir)
                    obj.run()
                elif algo == config.Algo.protonn:
                    obj = protonn.Protonn(trainingInput, testingInput,
                                          modelDir, datasetOutputDir, modelOutputDir)
                    obj.run()

                source_update = ""
                if self.args.source == config.Source.onnx:
                    source_update = "_onnx"

                trainingInput = os.path.join(datasetOutputDir, "train"+source_update+".npy")
                testingInput = os.path.join(datasetOutputDir, "test"+source_update+".npy")
                modelDir = modelOutputDir
            else:
                datasetDir = os.path.join(
                    Dataset.datasetProcessedDir, algo, dataset)
                modelDir = os.path.join(
                    Dataset.modelProcessedDir, algo, dataset)

                source_update = ""
                if self.args.source == config.Source.onnx:
                    source_update = "_onnx"

                trainingInput = os.path.join(datasetDir, "train"+source_update+".npy")
                testingInput = os.path.join(datasetDir, "test"+source_update+".npy")

            try:
                if version == config.Version.floatt:
                    bitwidth = 'float'
                elif config.wordLength == 8:
                    bitwidth = 'int8'
                elif config.wordLength == 16:
                    bitwidth = 'int16'
                elif config.wordLength == 32:
                    bitwidth = 'int32'
                else:
                    assert False

                curr = results[algo][bitwidth][dataset]

                expectedAcc = curr['accuracy']
                if version == config.Version.fixed:
                    bestScale = curr['scaleFactor']
                else:
                    bestScale = results[algo]['int16'][dataset]['scaleFactor']

            except Exception as _:
                assert self.args.load_sf == False
                expectedAcc = 0

            if self.args.load_sf:
                sf = bestScale
            else:
                sf = self.args.max_scale_factor

            numOutputs = self.args.numOutputs

            obj = main.Main(algo, version, target, trainingInput,
                            testingInput, modelDir, sf, maximisingMetric, dataset, numOutputs, self.args.source)
            obj.run()

            acc = obj.testingAccuracy

            if acc != expectedAcc:
                print("FAIL: Expected accuracy %f%%" % (expectedAcc))
                # return
            elif version == config.Version.fixed and obj.sf != bestScale:
                print("FAIL: Expected best scale %d" % (bestScale))
                # return
            else:
                print("PASS")

    def runCompilerDriver(self):
        for iter in product(self.args.algo, self.args.version, self.args.target):
            algo, version, target = iter

            print("\nGenerating code for " + algo + " " + target + "...")

            inputFile = os.path.join("input", algo + ".sd")
            #inputFile = os.path.join("input", algo + ".pkl")
            profileLogFile = os.path.join("input", "profile.txt")

            outputDir = os.path.join("output")
            os.makedirs(outputDir, exist_ok=True)

            outputFile = os.path.join(outputDir, algo + "-fixed.cpp")
            obj = main.Main(algo, version, target, inputFile, outputFile,
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

            datasetDir = os.path.join("..", "datasets", "datasets", dataset)
            modelDir = os.path.join("..", "model", dataset)

            if algo == config.Algo.bonsai:
                modelDir = os.path.join(modelDir, "BonsaiResults", "Params")
            elif algo == config.Algo.lenet:
                modelDir = os.path.join(modelDir, "LenetModel")
            else:
                modelDir = os.path.join(modelDir, "ProtoNNResults")

            inputFile = os.path.join(modelDir, "input.sd")

            trainingInput = os.path.join(datasetDir, "training-full.tsv")
            testingInput = os.path.join(datasetDir, "testing.tsv")

            obj = converter.Converter(algo, version, datasetType, target,
                                      outputDir, outputDir)
            obj.setInput(inputFile, modelDir, trainingInput, testingInput)
            obj.run()

    def runPredictorDriver(self):
        for iter in product(self.args.algo, self.args.version, self.args.dataset, self.args.datasetType):
            algo, version, dataset, datasetType = iter

            print("\nGenerating input files for \"" + algo + " " +
                  version + " " + dataset + " " + datasetType + "\"...")

            #outputDir = os.path.join("..", "Predictor", algo, version + "-testing")
            #datasetOutputDir = os.path.join("..", "Predictor", algo, version + "-" + datasetType)

            if version == config.Version.fixed:
                outputDir = os.path.join(
                    "..", "Predictor", "seedot_fixed", "testing")
                datasetOutputDir = os.path.join(
                    "..", "Predictor", "seedot_fixed", datasetType)
            elif version == config.Version.floatt:
                outputDir = os.path.join(
                    "..", "Predictor", algo + "_float", "testing")
                datasetOutputDir = os.path.join(
                    "..", "Predictor", algo + "_float", datasetType)

            os.makedirs(datasetOutputDir, exist_ok=True)
            os.makedirs(outputDir, exist_ok=True)

            datasetDir = os.path.join("..", "datasets", "datasets", dataset)
            modelDir = os.path.join("..", "model", dataset)

            if algo == config.Algo.bonsai:
                modelDir = os.path.join(modelDir, "BonsaiResults", "Params")
            elif algo == config.Algo.lenet:
                modelDir = os.path.join(modelDir, "LenetModel")
            else:
                modelDir = os.path.join(modelDir, "ProtoNNResults")

            inputFile = os.path.join(modelDir, "input.sd")

            trainingInput = os.path.join(datasetDir, "training-full.tsv")
            testingInput = os.path.join(datasetDir, "testing.tsv")

            obj = converter.Converter(algo, version, datasetType, config.Target.x86,
                                      datasetOutputDir, outputDir)
            obj.setInput(inputFile, modelDir, trainingInput, testingInput)
            obj.run()

            print("Building and executing " + algo + " " +
                  version + " " + dataset + " " + datasetType + "...")

            outputDir = os.path.join(
                "..", "Predictor", "output", algo + "-" + version)

            curDir = os.getcwd()
            os.chdir(os.path.join("..", "Predictor"))

            obj = predictor.Predictor(
                algo, version, datasetType, outputDir, self.args.max_scale_factor)
            acc = obj.run()

            os.chdir(curDir)

            if acc != None:
                print("Accuracy is %.3f" % (acc))

    def loadResultsFile(self):
        results = {}
        with open(os.path.join("Results.csv")) as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                algo, bitwidth, dataset = row[0], row[1], row[2]

                if algo not in results:
                    results[algo] = {}

                if bitwidth not in results[algo]:
                    results[algo][bitwidth] = {}

                if dataset not in results[algo][bitwidth]:
                    results[algo][bitwidth][dataset] = {}

                accuracy, scaleFactor = row[3], row[4]

                if not accuracy:
                    accuracy = 100
                if not scaleFactor:
                    scaleFactor = 9999

                results[algo][bitwidth][dataset] = {"accuracy": float(accuracy), "scaleFactor": int(scaleFactor)}
                
        return results


if __name__ == "__main__":
    obj = MainDriver()
    obj.parseArgs()
    obj.run()
