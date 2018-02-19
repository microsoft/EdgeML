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


class Main:

    def __init__(self, algo, trainingFile, testingFile, modelDir):
        self.algo, self.trainingFile, self.testingFile, self.modelDir = algo, trainingFile, testingFile, modelDir
        self.datasetType = "testing"

    def convert(self, target, version, datasetType):
        print("Generating input files for %s %s dataset..." %
              (version, datasetType))

        if target == "desktop":
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

    def compile(self, target, sf):
        print("Generating code...", end='')

        inputFile = os.path.join(
            "Predictor", self.algo, "fixed-testing", "input.txt")
        profileLogFile = os.path.join(
            "Predictor", "output", self.algo + "-float", "profile.txt")

        if target == "desktop":
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

    def predict(self, version, datasetType):
        curDir = os.getcwd()
        os.chdir("Predictor")

        obj = Predictor.Main(self.algo, version, datasetType, verbose=False)
        res = obj.run()

        if res == False:
            os.chdir(curDir)
            return False

        os.chdir(curDir)
        return True

    def readStatsFile(self, version, datasetType, sf):
        statsFile = os.path.join(
            "Predictor", "output", self.algo + "-" + version, "stats-" + datasetType + ".txt")

        with open(statsFile, 'r') as file:
            content = file.readlines()

        stats = [x.strip() for x in content]

        self.accuracy[sf] = float(stats[0])

    def runOnce(self, version, datasetType, target, sf):
        res = self.compile(target, sf)
        if res == False:
            return False, False

        res = self.predict(version, datasetType)
        if res == False:
            return False, True

        self.readStatsFile(version, datasetType, sf)

        print("Accuracy is %.3f%%\n" % (self.accuracy[sf]))

        return True, False

    def runOnTraining(self):
        start, end = 0, -16
        searching = False
        for i in range(start, end, -1):
            print("Testing with max scale factor of " + str(i))

            res, exit = self.runOnce("fixed", "training", "desktop", i)

            if exit == True:
                return False

            if res == True:
                searching = True
            elif searching == True:
                break

        if searching == False:
            return False
        else:
            return True

    def getBestScale(self):
        sorted_accuracy = dict(
            sorted(self.accuracy.items(), key=operator.itemgetter(1), reverse=True)[:5])
        print(sorted_accuracy)
        return next(iter(sorted_accuracy))

    def compileAndPredict(self):
        print("\n-------------------------------")
        print("Predicting on training dataset")
        print("-------------------------------\n")
        print("Starting search to find the best scaling factor...\n\n")

        self.accuracy = {}

        res = self.runOnTraining()
        if res == False:
            return False

        print("\nSearch completed\n")
        print("----------------------------------------------")
        print("Best performing scaling factors with accuracy:")

        self.sf = self.getBestScale()

        print("Best scaling factor = " + str(self.sf))

        print("\n-------------------------------")
        print("Predicting on testing dataset")
        print("-------------------------------\n")

        print("Setting max scaling factor to " + str(self.sf))

        res = self.runOnce("fixed", "testing", "desktop", self.sf)
        if res == False:
            return False

        return True

    def collectProfile(self):

        res = self.convert("desktop", "float", "training")
        if res == False:
            return False

        print("-----------------------")
        print("Collecting profile data")
        print("-----------------------")

        res = self.predict("float", "training")
        if res == False:
            return False

        self.accuracy = {}

        self.readStatsFile("float", "training", 0)

        print("Accuracy is %.3f%%\n\n" % (self.accuracy[0]))

    def dumpForArduino(self):
        print("--------------------------------------")
        print("Generating Arduino prediction files...")
        print("--------------------------------------\n")

        res = self.convert("arduino", "fixed", "testing")
        if res == False:
            return False

        # Copy file
        srcFile = os.path.join("Streamer", "input", "model.h")
        destFile = "model.h"
        shutil.copyfile(srcFile, destFile)

        res = self.compile("arduino", self.sf)
        if res == False:
            return False

    def run(self):

        res = self.collectProfile()
        if res == False:
            return False

        res = self.convert("desktop", "fixed", "training")
        if res == False:
            return False

        res = self.convert("desktop", "fixed", "testing")
        if res == False:
            return False

        res = self.compileAndPredict()
        if res == False:
            return False

        self.dumpForArduino()

        return True


class MainDriver:

    algosAll = ["bonsai", "protonn"]

    def __init__(self):
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

        if not os.path.isfile(self.args.train):
            raise Exception("Training dataset file doesn't exist")
        if not os.path.isfile(self.args.test):
            raise Exception("Testing dataset file doesn't exist")
        if not os.path.isdir(self.args.model):
            raise Exception("Model directory doesn't exist")

    def run(self):
        print("\n====================")
        print("Executing on %s" % (self.args.algo))
        print("====================\n")
        obj = Main(self.args.algo, self.args.train,
                   self.args.test, self.args.model)
        obj.run()


if __name__ == "__main__":
    obj = MainDriver()
    obj.run()
