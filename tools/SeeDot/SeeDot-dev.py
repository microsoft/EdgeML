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
import logging
import seedot.compiler.converter.converter as converter

# This is the file which is invoked to run the compiler (Refer to README.md).
#
# Sanity checks are carried out and the main compiler arguments are taken from the user
# which is then used to invoke the main compiler code, 'main.py'.
#
# Note there are 3 different ways to change compiler arguments:
#   1) the arguments used by the user to invoke the compiler
#   2) seedot/config.py
#   3) seedot/util.py
# Different parameters are controlled in different files, refer to each one of them to
# find out how to change one parameter.

class Dataset:
    common = ["cifar-binary", "cr-binary", "cr-multiclass", "curet-multiclass",
              "letter-multiclass", "mnist-binary", "mnist-multiclass",
              "usps-binary", "usps-multiclass", "ward-binary"]
    extra = ["cifar-multiclass", "dsa", "eye-binary", "farm-beats",
             "interactive-cane", "spectakoms", "usps10", "whale-binary",
             "HAR-2", "HAR-6", "MNIST-10", "Google-12", "Google-30", "Wakeword-2",
             "wider-regression", "wider-mbconv", "face-1", "face-2", "face-2-rewrite", 
             "face-3", "face-4", "test"]
    # Datasets for ProtoNN and Bonsai.
    default = common
    # Datasets for FastGRNN.
    # default = ["spectakoms", "usps10", "HAR-2", "HAR-6", "dsa", "MNIST-10", "Google-12", "Google-30", "Wakeword-2"]
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

        parser.add_argument("--tempdir", metavar='',
                            help="Scratch directory for intermediate files")
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
        if not isinstance(self.args.maximisingMetric, list):
            self.args.maximisingMetric = [self.args.maximisingMetric]

        if self.args.tempdir is not None:
            assert os.path.isdir(
                self.args.tempdir), "Scratch directory doesn't exist"
            config.tempdir = self.args.tempdir
        else:
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
                config.outdir = os.path.join("m3dump")
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
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "ERROR"))

        if util.windows():
            self.checkMSBuildPath()

        self.setGlobalFlags()
        self.runMainDriver()

    def runMainDriver(self):
        results = self.loadResultsFile()

        for iter in product(self.args.algo, self.args.version, self.args.dataset, self.args.target, self.args.maximisingMetric, [16]):
            algo, version, dataset, target, maximisingMetric, wordLength = iter

            print("\n========================================")
            print("Executing on %s %s %s %s" %
                  (algo, version, dataset, target))
            print("========================================\n")

            datasetDir = os.path.join(
                Dataset.datasetProcessedDir, algo, dataset)
            modelDir = os.path.join(
                Dataset.modelProcessedDir, algo, dataset)

            source_update = ""
            if self.args.source == config.Source.onnx:
                source_update = "_onnx"

            trainingInput = os.path.join(datasetDir, "train" + source_update + ".npy")
            testingInput = os.path.join(datasetDir, "test" + source_update + ".npy")

            try:
                # The following is particularly for old SeeDot (PLDI '19).
                # In the new version of SeeDot (named Shiftry, OOPSLA '20), config.wordLength is ALWAYS expected to be 16, which is the base bit-width.
                # Some variables are demoted to 8 bits, and intermediate variables for multiplication may use 32 bits.
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
            elif version == config.Version.fixed and obj.sf != bestScale:
                print("FAIL: Expected best scale %d" % (bestScale))
            else:
                print("PASS")

    def loadResultsFile(self):
        results = {}
        # Results.csv contains some benchmark results for old SeeDot (PLDI '19).
        # The CSV file can be updated with better accuracy numbers if a better model is obtained.
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
