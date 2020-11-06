# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import subprocess

import seedot.config as config
import seedot.util as Util

# Program to build and run the predictor project using msbuild
# The accuracy and other statistics are written to the output file specified


class Predictor:

    def __init__(self, algo, version, datasetType, outputDir, scaleForX, scalesForX, scaleForY, scalesForY, problemType, numOutputs):
        self.algo, self.version, self.datasetType = algo, version, datasetType

        self.outputDir = outputDir
        os.makedirs(self.outputDir, exist_ok=True)

        self.scaleForX = scaleForX
        self.scalesForX = scalesForX
        self.scaleForY = scaleForY
        self.scalesForY = scalesForY

        self.problemType = problemType
        self.numOutputs = numOutputs

        self.genHeaderFile()

    def genHeaderFile(self):
        with open("datatypes.h", 'w') as file:
            file.write("#pragma once\n\n")

            if config.wordLength == 8:
                file.write("#define INT8\n")
                file.write("typedef int8_t MYINT;\n\n")
            elif config.wordLength == 16:
                file.write("#define INT16\n")
                file.write("typedef int16_t MYINT;\n\n")
            elif config.wordLength == 32:
                file.write("#define INT32\n")
                file.write("typedef int32_t MYINT;\n\n")

            file.write("typedef int16_t MYITE;\n")
            file.write("typedef uint16_t MYUINT;\n\n")

            file.write("const int scaleForX = %d;\n\n" % (self.scaleForX))
            if len(self.scalesForX) > 0:
                assert len(self.scalesForX) == max(list(self.scalesForX.keys())), "Malformed array scalesForX"
                file.write("const int scalesForX[%d] = {%s};\n" % (len(self.scalesForX), ', '.join([str(self.scalesForX[i+1]) for i in range(len(self.scalesForX))])))
            else:
                file.write("const int scalesForX[1] = {100}; //junk, needed for compilation\n")

            file.write("const int scaleForY = %d;\n\n" % (self.scaleForY))
            if len(self.scalesForY) > 0:
                assert len(self.scalesForY) == max(list(self.scalesForY.keys())), "Malformed array scalesForY"
                file.write("const int scalesForY[%d] = {%s};\n" % (len(self.scalesForY), ', '.join([str(self.scalesForY[i+1]) for i in range(len(self.scalesForY))])))
            else:
                file.write("const int scalesForY[1] = {100}; //junk, needed for compilation\n")

            if Util.debugMode():
                file.write("const bool debugMode = true;\n")
            else:
                file.write("const bool debugMode = false;\n")

            file.write("const bool logProgramOutput = true;\n")

            if Util.isSaturate():
                file.write("#define SATURATE\n")
            else:
                file.write("//#define SATURATE\n")
            
            if Util.isfastApprox():
                file.write("#define FASTAPPROX\n")
            else:
                file.write("//#define FASTAPPROX\n")
            
            if Util.useMathExp() or (Util.useNewTableExp()):
                file.write("#define FLOATEXP\n")
            else:
                file.write("//#define FLOATEXP\n")

    def buildForWindows(self):
        '''
        Builds using the Predictor.vcxproj project file and creates the executable
        The target platform is currently set to x64
        '''
        print("Build...", end='')

        projFile = "Predictor.vcxproj"
        args = [config.msbuildPath, projFile, r"/t:Build",
                r"/p:Configuration=Release", r"/p:Platform=x64"]

        logFile = os.path.join(self.outputDir, "msbuild.txt")
        with open(logFile, 'w') as file:
            process = subprocess.call(args, stdout=file, stderr=subprocess.STDOUT)

        if process == 1:
            print("FAILED!!\n")
            return False
        else:
            print("success")
            return True

    def buildForLinux(self):
        print("Build...", end='')

        args = ["make"]

        logFile = os.path.join(self.outputDir, "build.txt")
        with open(logFile, 'w') as file:
            process = subprocess.call(args, stdout=file, stderr=subprocess.STDOUT)

        if process == 1:
            print("FAILED!!\n")
            return False
        else:
            print("success")
            return True

    def build(self):
        if Util.windows():
            return self.buildForWindows()
        else:
            return self.buildForLinux()

    def executeForWindows(self):
        '''
        Invokes the executable with arguments
        '''
        print("Execution...", end='')

        exeFile = os.path.join("x64", "Release", "Predictor.exe")
        args = [exeFile, self.version, self.datasetType, self.problemType, str(self.numOutputs)]

        logFile = os.path.join(self.outputDir, "exec.txt")
        with open(logFile, 'w') as file:
            process = subprocess.call(args, stdout=file, stderr=subprocess.STDOUT)

        if process == 1:
            print("FAILED!!\n")
            return None
        else:
            print("success")
            execMap = self.readStatsFile()
            return execMap

    def executeForLinux(self):
        print("Execution...", end='')

        exeFile = os.path.join("./Predictor")
        args = [exeFile, self.version, self.datasetType, self.problemType, str(self.numOutputs)]

        logFile = os.path.join(self.outputDir, "exec.txt") 
        with open(logFile, 'w') as file:
            process = subprocess.call(args, stdout=file, stderr=subprocess.STDOUT)

        if process == 1:
            print("FAILED!!\n")
            return None
        else:
            print("success")
            execMap = self.readStatsFile()
            return execMap

    def execute(self):
        if Util.windows():
            return self.executeForWindows()
        else:
            return self.executeForLinux()

    # Read statistics of execution (currently only accuracy)
    def readStatsFile(self):
        statsFile = os.path.join(
            "output", self.version, "stats-" + self.datasetType + ".txt")

        with open(statsFile, 'r') as file:
            content = file.readlines()

        stats = [x.strip() for x in content]

        executionOutput = {}
        
        for i in range(len(stats) // 4):
            key = stats[4*i]
            value = (float(stats[4*i + 1]), float(stats[4*i + 2]), float(stats[4*i + 3]))
            executionOutput[key] = value

        return executionOutput

    def run(self):
        res = self.build()
        if res == False:
            return None

        execMap = self.execute()

        return execMap
