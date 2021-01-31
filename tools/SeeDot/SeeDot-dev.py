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

'''
This is the file which is invoked to run the compiler (Refer to README.md).

Sanity checks are carried out and the main compiler arguments are taken from the user
which is then used to invoke the main compiler code, 'main.py'.

Note there are 3 different ways to change compiler arguments:
  1) the arguments used by the user to invoke the compiler
  2) seedot/config.py
  3) seedot/util.py
Different parameters are controlled in different files, refer to each one of them to
find out how to change one parameter.
'''


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
    default = ["usps10"]
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
                            default=config.Algo.default, metavar='', help="Algorithm to run ['bonsai' or 'protonn' or 'fastgrnn' or 'rnnpool'] \
                           (Default: 'fastgrnn')")
        parser.add_argument("-e", "--encoding", choices=config.Encoding.all,
                            default=config.Encoding.default, metavar='', help="Floating-point ['float'] or Fixed-point ['fixed'] \
                           (Default: 'fixed')")
        parser.add_argument("-d", "--dataset", choices=Dataset.all,
                            default=Dataset.default, metavar='', help="Dataset to use\
                            (Default: 'usps10')")
        parser.add_argument("-m", "--metric", choices=config.Metric.all, metavar='',
                            help="Select the metric that will be used to measure the correctness of an inference, to obtain the \
                            best quantization of variables. \
                                ['acc', 'disagree', 'red_diagree'] (Default: 'red_disagree')",default=config.Metric.default)
        parser.add_argument("-n", "--numOutputs", type=int, metavar='',
                            help="Number of outputs (e.g., classification problems have only 1 output, i.e., the class label)\
                           (Default: 1)",default=1)
        parser.add_argument("-dt", "--datasetType", choices=config.DatasetType.all,
                            default=config.DatasetType.default, metavar='', help="Dataset type being used ['training', 'testing']\
                           (Default: 'testing')")
        parser.add_argument("-t", "--target", choices=config.Target.all,
                            default=config.Target.default, metavar='', help="Target device ['x86', 'arduino', 'm3'] \
                            (Default: 'x86')")
        parser.add_argument("-s", "--source", metavar='', choices=config.Source.all,
                            default=config.Source.default, help="Model source type ['seedot', 'onnx', 'tf']\
                           (Default: 'seedot')")
        parser.add_argument("-sf", "--max-scale-factor", type=int,
                            metavar='', help="Use the old max-scale mechanism of SeeDot's PLDI’19 paper to determine the scales (If not specified then it will be inferred from data)")
        parser.add_argument("-l", "--log", choices=config.Log.all,
                            default=config.Log.default, metavar='', help="Logging level (in increasing order)\
                             ['error', 'critical', 'warning', 'info', 'debug'] (Default: 'error')")
        parser.add_argument("-lsf", "--load-sf", action="store_true",
                            help=argparse.SUPPRESS)
        parser.add_argument("-tdr", "--tempdir", metavar='',
                            help="Scratch directory for intermediate files\
                           (Default: 'temp/')")
        parser.add_argument("-o", "--outdir", metavar='',
                            help="Directory to output the generated targetdevice sketch\
                           (Default: 'arduinodump/' for Arduino, 'temp/' for x86 and, 'm3dump/' for M3)")
        
        self.args = parser.parse_args()

        if not isinstance(self.args.algo, list):
            self.args.algo = [self.args.algo]
        if not isinstance(self.args.encoding, list):
            self.args.encoding = [self.args.encoding]
        if not isinstance(self.args.dataset, list):
            self.args.dataset = [self.args.dataset]
        if not isinstance(self.args.datasetType, list):
            self.args.datasetType = [self.args.datasetType]
        if not isinstance(self.args.target, list):
            self.args.target = [self.args.target]
        if not isinstance(self.args.metric, list):
            self.args.metric = [self.args.metric]

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

    def setLogLevel(self):
        logging.basicConfig(level=os.environ.get("LOGLEVEL", self.args.log.upper()))

    def run(self):
        self.setLogLevel()

        if util.windows():
            self.checkMSBuildPath()

        self.setGlobalFlags()
        self.runMainDriver()

    def runMainDriver(self):
        legacy_scales = self.loadScalesFile()

        for iter in product(self.args.algo, self.args.encoding, self.args.dataset, self.args.target, self.args.metric, [16]):
            algo, encoding, dataset, target, metric, wordLength = iter

            print("\n========================================")
            print("Executing on %s %s %s %s" %
                  (algo, encoding, dataset, target))
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
                if encoding == config.Encoding.floatt:
                    bitwidth = 'float'
                elif config.wordLength == 8:
                    bitwidth = 'int8'
                elif config.wordLength == 16:
                    bitwidth = 'int16'
                elif config.wordLength == 32:
                    bitwidth = 'int32'
                else:
                    assert False

                curr = legacy_scales[algo][bitwidth][dataset]

                expectedAcc = curr['accuracy']
                if encoding == config.Encoding.fixed:
                    bestScale = curr['scaleFactor']
                else:
                    bestScale = legacy_scales[algo]['int16'][dataset]['scaleFactor']

            except Exception as _:
                assert self.args.load_sf == False
                expectedAcc = 0

            if self.args.load_sf:
                sf = bestScale
            else:
                sf = self.args.max_scale_factor

            numOutputs = self.args.numOutputs

            obj = main.Main(algo, encoding, target, trainingInput,
                            testingInput, modelDir, sf, metric, dataset, numOutputs, self.args.source)
            obj.run()

            acc = obj.testingAccuracy

            if self.args.load_sf:
                if acc != expectedAcc:
                    print("FAIL: Expected accuracy %f%%" % (expectedAcc))
                elif encoding == config.Encoding.fixed and obj.sf != bestScale:
                    print("FAIL: Expected best scale %d" % (bestScale))
                else:
                    print("PASS")

    def loadScalesFile(self):
        scales = {}
        # legacy_scales.csv contains some benchmark results for old SeeDot (PLDI '19).
        # The CSV file can be updated with better accuracy numbers if a better model is obtained.
        with open(os.path.join("legacy_scales.csv")) as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                algo, bitwidth, dataset = row[0], row[1], row[2]

                if algo not in scales:
                    scales[algo] = {}

                if bitwidth not in scales[algo]:
                    scales[algo][bitwidth] = {}

                if dataset not in scales[algo][bitwidth]:
                    scales[algo][bitwidth][dataset] = {}

                accuracy, scaleFactor = row[3], row[4]

                if not accuracy:
                    accuracy = 100
                if not scaleFactor:
                    scaleFactor = 9999

                scales[algo][bitwidth][dataset] = {"accuracy": float(accuracy), "scaleFactor": int(scaleFactor)}

        return scales

if __name__ == "__main__":
    obj = MainDriver()
    obj.parseArgs()
    obj.run()
