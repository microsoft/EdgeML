# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse
import datetime
from distutils.dir_util import copy_tree
import os
import shutil
import sys
import operator
import tempfile
import traceback

sys.path.insert(0, '../../')

from edgeml.tools.seedot.converter.converter import Converter

import edgeml.tools.seedot.common as Common
from edgeml.tools.seedot.compiler import Compiler
from edgeml.tools.seedot.predictor import Predictor
from edgeml.tools.seedot.seedot import Main
import edgeml.tools.seedot.util as Util


class MainDriver:

    def parseArgs(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("-a", "--algo", choices=Common.Algo.All,
                            metavar='', help="Algorithm to run")
        parser.add_argument("--train", required=True,
                            metavar='', help="Training set file")
        parser.add_argument("--test", required=True,
                            metavar='', help="Testing set file")
        parser.add_argument("--model", required=True, metavar='',
                            help="Directory containing trained model")
        parser.add_argument("--tempdir", metavar='', help="Scratch directory")
        parser.add_argument("-o", "--outdir", metavar='',
                            help="Directory to output the generated Arduino sketch")

        self.args = parser.parse_args()

        # Verify the input files and directory exists
        assert os.path.isfile(self.args.train), "Training set doesn't exist"
        assert os.path.isfile(self.args.test), "Testing set doesn't exist"
        assert os.path.isdir(self.args.model), "Model directory doesn't exist"

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

        algo, trainingInput, testingInput, modelDir = self.args.algo, self.args.train, self.args.test, self.args.model

        print("\n================================")
        print("Executing on %s for Arduino" % (algo))
        print("--------------------------------")
        print("Train file: %s" % (trainingInput))
        print("Test file: %s" % (testingInput))
        print("Model directory: %s" % (modelDir))
        print("================================\n")

        obj = Main(algo, Common.Version.Fixed, Common.Target.Arduino,
                   trainingInput, testingInput, modelDir, None)
        obj.run()

if __name__ == "__main__":
    obj = MainDriver()
    obj.parseArgs()
    obj.run()
