import sys
import os
import subprocess
import argparse

# Program to build and run the predictor project using msbuild
# The accuracy and other statistics are written to the output file specified


class Main:

    def __init__(self, algo, version, datasetType, outputDir=None, verbose=True, msbuildPath=None):
        self.algo, self.version, self.datasetType = algo, version, datasetType

        if outputDir == None:
            outputDir = os.path.join("output", self.algo + "-" + self.version)

        self.outputDir = outputDir
        os.makedirs(self.outputDir, exist_ok=True)

        self.verbose = verbose
        self.msbuildPath = msbuildPath

    def build(self):
        '''
        Builds using the Predictor.vcxproj project file and creates the executable
        The target platform is currently set to x64
        '''
        print("Build...", end='')

        projFile = "Predictor.vcxproj"
        args = [self.msbuildPath, projFile, r"/t:Build",
                r"/p:Configuration=Release", r"/p:Platform=x64"]

        logFile = os.path.join(self.outputDir, "msbuild.txt")
        with open(logFile, 'w') as file:
            process = subprocess.call(args, stdout=file)

        if process == 1:
            print("FAILED!!\n")
            return False
        else:
            print("success")
            return True

    def execute(self):
        '''
        Invokes the executable with arguments
        '''
        print("Execution...", end='')

        exeFile = os.path.join("x64", "Release", "Predictor.exe")
        args = [exeFile, self.algo, self.version, self.datasetType]

        logFile = os.path.join(self.outputDir, "exec.txt")
        with open(logFile, 'w') as file:
            process = subprocess.call(args, stdout=file)

        if process == 1:
            print("FAILED!!\n")
            return False
        else:
            print("success")
            if self.verbose:
                with open(logFile, 'r') as file:
                    print(file.read())
            return True

    def run(self):
        res = self.build()
        if res == False:
            return False

        res = self.execute()
        if res == False:
            return False
