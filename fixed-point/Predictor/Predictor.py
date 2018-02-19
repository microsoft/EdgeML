import sys
import os
import subprocess
import argparse


class Main:

    def __init__(self, algo, version, datasetType, outputDir=None, verbose=True):
        self.algo, self.version, self.datasetType = algo, version, datasetType

        if outputDir == None:
            outputDir = os.path.join("output", self.algo + "-" + self.version)

        self.outputDir = outputDir
        os.makedirs(self.outputDir, exist_ok=True)

        self.verbose = verbose

    def build(self):
        print("Build...", end='')

        msbuild = r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\MSBuild\15.0\Bin\MSBuild.exe"
        projFile = "Predictor.vcxproj"
        args = [msbuild, projFile, r"/t:Build",
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
