# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import os

from seedot.compiler.converter.util import *

# Class to read ProtoNN model dumps and generate input files (C header file and the compiler input)
# The two classes ProtonnFixed and ProtonnFloat are for generating fixed model and float model respectively
# The baseclass Protonn collects some of the common functions between them.


class Protonn:

    def __init__(self, trainFile, testFile, modelDir, datasetOutputDir, modelOutputDir):
        self.trainFile = trainFile
        self.testFile = testFile
        self.modelDir = modelDir
        self.datasetOutputDir = datasetOutputDir
        self.modelOutputDir = modelOutputDir

        self.seeDotProgram = os.path.join(self.modelOutputDir, "input.sd")

    def readDataset(self):
        train_ext = os.path.splitext(self.trainFile)[1]
        test_ext = os.path.splitext(self.testFile)[1]

        if train_ext == test_ext == ".npy":
            assert False
        elif train_ext == test_ext == ".tsv":
            self.train = np.loadtxt(self.trainFile, delimiter="\t", ndmin=2)
            self.test = np.loadtxt(self.testFile, delimiter="\t", ndmin=2)
        elif train_ext == test_ext == ".csv":
            # Check the length of X and Y
            #assert len(self.X) == len(self.Y)
            assert False
        elif train_ext == test_ext == ".txt":
            assert False
        else:
            assert False

    def computeTrainSetRange(self):
        self.X = self.train[:, 1:]
        X_list = self.X.tolist()

        X_trimmed, _ = trimMatrix(X_list)
        self.trainDatasetRange = matRange(X_trimmed)

    def writeDataset(self):
        np.save(os.path.join(self.datasetOutputDir, "train.npy"), self.train)
        np.save(os.path.join(self.datasetOutputDir, "test.npy"), self.test)

    def processDataset(self):
        self.readDataset()
        self.computeTrainSetRange()
        self.writeDataset()

    def readNormFile(self):
        if os.path.isfile(os.path.join(self.modelDir, "minMaxParams")):
            self.MinMax = np.loadtxt(os.path.join(
                self.modelDir, "minMaxParams"), delimiter="\t", ndmin=2)
            self.normType = "MinMax"
        else:
            self.normType = None

    def readModel(self):
        if os.path.isfile(os.path.join(getModelDir(), "W")):
            return self.readModelAsTxt()
        elif os.path.isfile(os.path.join(getModelDir(), "W.npy")):
            return self.readModelAsNpy()
        else:
            assert False

    def readModelAsNpy(self):
        self.W = np.load(os.path.join(getModelDir(), "W.npy"))
        self.W = self.W.transpose().tolist()
        self.B = np.load(os.path.join(getModelDir(), "B.npy")).tolist()
        self.Z = np.load(os.path.join(getModelDir(), "Z.npy")).tolist()
        self.gamma = np.load(os.path.join(getModelDir(), "gamma.npy")).tolist()
        self.gamma = [[self.gamma]]
        self.readNormFile()

    def readModelAsTxt(self):

        self.W = np.loadtxt(os.path.join(
            self.modelDir, "W"), delimiter="\t", ndmin=2)
        self.B = np.loadtxt(os.path.join(
            self.modelDir, "B"), delimiter="\t", ndmin=2)
        self.Z = np.loadtxt(os.path.join(
            self.modelDir, "Z"), delimiter="\t", ndmin=2)
        self.gamma = np.loadtxt(os.path.join(
            self.modelDir, "gamma"), delimiter="\t", ndmin=2)
        self.readNormFile()

    def validateNormFile(self):
        if self.normType == "MinMax":
            MinMax_m, MinMax_n = self.MinMax.shape
            W_m, W_n = self.W.shape

            assert MinMax_m == 2
            assert MinMax_n == W_n
        else:
            assert False

    def validateModel(self):
        self.validateNormFile()

        W_m, W_n = self.W.shape
        B_m, B_n = self.B.shape
        Z_m, Z_n = self.Z.shape

        assert W_m == B_m
        assert B_n == Z_n
        assert self.gamma.size == 1

    def computeVars(self):
        self.d, self.D = self.W.shape
        _, self.p = self.B.shape
        self.c, _ = self.Z.shape

    # Precompute some values to speedup prediction
    def formatModel(self):
        # Precompute g2
        self.g2 = self.gamma * self.gamma

        if self.normType == None:
            # Creating dummpy values
            self.Norm = np.ones([1, self.W.shape[0]])
        elif self.normType == "MinMax":
            # Extract Min and Max
            Min = self.MinMax[0].reshape(-1, 1)
            Max = self.MinMax[1].reshape(-1, 1)

            # Precompute W * (X-m)/(M-m) by absorbing m, M into W
            for i in range(self.W.shape[0]):
                for j in range(self.W.shape[1]):
                    self.W[i][j] = self.W[i][j] / (Max[j][0] - Min[j][0])

            self.Norm = self.W.dot(Min)

            assert self.Norm.shape[1] == 1
            #self.Norm = [x[0] for x in self.Norm]
            self.Norm = self.Norm.reshape(1, -1)
        else:
            assert False

        self.computeVars()

    # Write the ProtoNN algorithm in terms of the compiler DSL.
    # Compiler takes this as input and generates fixed-point code.
    def genSeeDotProgram(self):
        with open(self.seeDotProgram, 'w') as file:
            # Matrix declarations
            file.write("let X   = (%d, 1)   in [%.6f, %.6f] in\n" % (
                (self.X.shape[1],) + self.trainDatasetRange))
            file.write("let W  = (%d, %d)    in [%.6f, %.6f] in\n" % (
                self.d, self.D, np.amin(self.W), np.amax(self.W)))
            file.write("let B  = (%d, %d, 1) in [%.6f, %.6f] in\n" % (
                self.p, self.d, np.amin(self.B), np.amax(self.B)))
            file.write("let Z  = (%d, %d, 1) in [%.6f, %.6f] in\n" % (
                self.p, self.c, np.amin(self.Z), np.amax(self.Z)))
            if self.normType != None:
                file.write("let norm = (%d, 1)   in [%.6f, %.6f] in\n" % (
                    self.d, np.amin(self.Norm), np.amax(self.Norm)))
            file.write("let g2 = %.6f in\n\n" % (self.g2))

            # Algorithm
            if useSparseMat():
                s = "W |*| X"
            else:
                s = "W * X"

            if self.normType != None:
                s = s + " - norm"

            file.write("let WX = %s in\n" % (s))
            file.write("let res = $(i = [0:%d])\n" % (self.p))
            file.write("(\n")
            file.write("\tlet del = WX - B[i] in\n")
            file.write("\tZ[i] * exp(-g2 * (del^T * del))\n")
            file.write(") in\n")
            file.write("argmax(res)\n")

    # Writing the model as a bunch of variables, arrays and matrices to a file
    def writeModel(self):

        if self.normType != None:
            np.save(os.path.join(self.modelOutputDir, "norm.npy"), self.Norm)

        np.save(os.path.join(self.modelOutputDir, "W.npy"), self.W)

        # Transpose B and Z to satisfy the declarations in the generated DSL input
        B_transp = np.transpose(self.B)
        Z_transp = np.transpose(self.Z)

        np.save(os.path.join(self.modelOutputDir, "B.npy"), B_transp)
        np.save(os.path.join(self.modelOutputDir, "Z.npy"), Z_transp)

    def processModel(self):
        self.readModel()
        self.validateModel()
        self.formatModel()
        self.genSeeDotProgram()
        self.writeModel()

    def run(self):
        self.processDataset()

        self.processModel()

        assert self.X.shape[1] == self.W.shape[1]
