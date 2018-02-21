import os
import sys
import operator
import traceback
import argparse
import shutil

sys.path.insert(0, 'Compiler')
sys.path.insert(0, 'Converter')
sys.path.insert(0, 'Predictor')

import Compiler
import Converter
import Predictor


class Algo:
    Bonsai = "bonsai"
    Protonn = "protonn"


class Version:
    Fixed = "fixed"
    Float = "float"


class DatasetType:
    Training = "training"
    Testing = "testing"


class Target:
    Desktop = "desktop"
    Arduino = "arduino"


class Main:

    def __init__(self, algo, trainingFile, testingFile, modelDir, msbuildPath):
        self.algo, self.trainingFile, self.testingFile, self.modelDir, self.msbuildPath = algo, trainingFile, testingFile, modelDir, msbuildPath

    # Run the converter project to generate the input files using reading the training model
    def convert(self, version, datasetType, target):
        print("Generating input files for %s %s dataset...\n" %
              (version, datasetType))

        # Create output dirs
        if target == Target.Desktop:
            datasetOutputDir = os.path.join(
                "Predictor", self.algo, version + "-" + datasetType)
            outputDir = os.path.join(
                "Predictor", self.algo, version + "-testing")
        else:
            outputDir = os.path.join("Streamer", "input")
            datasetOutputDir = outputDir

        os.makedirs(datasetOutputDir, exist_ok=True)
        os.makedirs(outputDir, exist_ok=True)

        try:
            obj = Converter.Main(self.algo, version, datasetType, target, self.modelDir,
                                 datasetOutputDir=datasetOutputDir, outputDir=outputDir)
            obj.setTSVinput(self.trainingFile, self.testingFile)
            obj.run()
        except Exception as e:
            traceback.print_exc()
            return False
        return True

    # Generate the fixed-point code using the input generated from the Converter project
    def compile(self, target, sf):
        print("Generating code...", end='')

        # Set input and output files
        inputFile = os.path.join(
            "Predictor", self.algo, "fixed-testing", "input.txt")
        profileLogFile = os.path.join(
            "Predictor", "output", self.algo + "-float", "profile.txt")

        if target == Target.Desktop:
            outputFile = os.path.join("Predictor", self.algo + "_fixed.cpp")
        else:
            outputFile = "predict.cpp"

        try:
            obj = Compiler.Main(self.algo, target, inputFile,
                                outputFile, profileLogFile, sf)
            obj.run()
        except:
            print("failed!\n")
            # traceback.print_exc()
            return False

        print("completed")
        return True

    # Build and run the Predictor project
    def predict(self, version, datasetType):
        curDir = os.getcwd()
        os.chdir("Predictor")

        obj = Predictor.Main(self.algo, version, datasetType,
                             verbose=False, msbuildPath=self.msbuildPath)
        res = obj.run()

        os.chdir(curDir)

        if res == False:
            return False, None

        # If success, read the stats file and return accuracy
        acc = self.readStatsFile(version, datasetType)

        return True, acc

    # Read statistics of execution (currently only accuracy)
    def readStatsFile(self, version, datasetType):
        statsFile = os.path.join(
            "Predictor", "output", self.algo + "-" + version, "stats-" + datasetType + ".txt")

        with open(statsFile, 'r') as file:
            content = file.readlines()

        stats = [x.strip() for x in content]

        return float(stats[0])

    # Compile and run the generated code once for a given scaling factor
    def runOnce(self, version, datasetType, target, sf):
        res = self.compile(target, sf)
        if res == False:
            return False, False

        res, acc = self.predict(version, datasetType)
        if res == False:
            return False, True

        self.accuracy[sf] = acc
        print("Accuracy is %.3f%%\n" % (acc))

        return True, False

    # Iterate over multiple scaling factors and store their accuracies
    def performSearch(self):
        start, end = 0, -16
        searching = False

        for i in range(start, end, -1):
            print("Testing with max scale factor of " + str(i))

            res, exit = self.runOnce(
                Version.Fixed, DatasetType.Training, Target.Desktop, i)

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
        sorted_accuracy = dict(
            sorted(self.accuracy.items(), key=operator.itemgetter(1), reverse=True)[:5])
        print(sorted_accuracy)
        return next(iter(sorted_accuracy))

    # Find the scaling factor which works best on the training dataset and predict on the testing dataset
    def findBestScalingFactor(self):
        print("-------------------------------------------------")
        print("Performing search to find the best scaling factor")
        print("-------------------------------------------------\n")

        # Generate input files for training dataset
        res = self.convert(Version.Fixed, DatasetType.Training, Target.Desktop)
        if res == False:
            return False

        self.accuracy = {}

        # Search for the best scaling factor
        res = self.performSearch()
        if res == False:
            return False

        print("Best scaling factor = %d" % (self.sf))

        print("\n-------------------------------")
        print("Prediction on testing dataset")
        print("-------------------------------\n")

        print("Setting max scaling factor to %d\n" % (self.sf))

        # Generate files for the testing dataset
        res = self.convert(Version.Fixed, DatasetType.Testing, Target.Desktop)
        if res == False:
            return False

        # Compiler and run code using the best scaling factor
        res = self.runOnce(Version.Fixed, DatasetType.Testing,
                           Target.Desktop, self.sf)
        if res == False:
            return False

        return True

    # Generate files for training dataset and perform a profiled execution
    def collectProfileData(self):
        print("-----------------------")
        print("Collecting profile data")
        print("-----------------------")

        res = self.convert(Version.Float, DatasetType.Training, Target.Desktop)
        if res == False:
            return False

        res, acc = self.predict(Version.Float, DatasetType.Training)
        if res == False:
            return False

        print("Accuracy is %.3f%%\n" % (acc))

    # Generate code for Arduino
    def compileForArduino(self):
        print("------------------------------")
        print("Generating code for Arduino...")
        print("------------------------------\n")

        res = self.convert(Version.Fixed, DatasetType.Testing, Target.Arduino)
        if res == False:
            return False

        # Copy file
        srcFile = os.path.join("Streamer", "input", "model.h")
        destFile = "model.h"
        shutil.copyfile(srcFile, destFile)

        res = self.compile(Target.Arduino, self.sf)
        if res == False:
            return False

    def run(self):

        # Collect runtime profile
        res = self.collectProfileData()
        if res == False:
            return False

        # Obtain best scaling factor
        res = self.findBestScalingFactor()
        if res == False:
            return False

        # Generate code for Arduino
        self.compileForArduino()

        return True


class MainDriver:

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

    def verifyMsbuildPath(self):
        # Location of msbuild
        # Edit the path if not present at the following location
        msbuildPath = r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\MSBuild\15.0\Bin\MSBuild.exe"

        if os.path.isfile(msbuildPath):
            self.msbuildPath = msbuildPath
        else:
            raise Exception(
                "Msbuild.exe not found at the following locaiton:\n%s\nPlease change the path and run again" % (msbuildPath))

    def run(self):

        # Verify that msbuild exists
        self.verifyMsbuildPath()

        print("\n====================")
        print("Executing on %s" % (self.args.algo))
        print("====================\n")
        obj = Main(self.args.algo, self.args.train,
                   self.args.test, self.args.model, self.msbuildPath)
        obj.run()


if __name__ == "__main__":
    obj = MainDriver()
    obj.run()
