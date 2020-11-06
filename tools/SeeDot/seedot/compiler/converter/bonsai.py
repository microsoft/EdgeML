# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import math
import os

from seedot.compiler.converter.util import *

# Class to read Bonsai model dumps and generate input files (C header file and the compiler input)
# The two classes BonsaiFixed and BonsaiFloat are for generating fixed model and float model respectively
# The baseclass Bonsai collects some of the common functions between them.


class Bonsai:

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

    def formatDataset(self):
        ones = np.ones([self.train.shape[0], 1])
        self.train = np.append(self.train, ones, axis=1)

        ones = np.ones([self.test.shape[0], 1])
        self.test = np.append(self.test, ones, axis=1)

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
        self.formatDataset()
        self.computeTrainSetRange()
        self.writeDataset()

    def readModel(self):
        self.Z = np.loadtxt(os.path.join(
            self.modelDir, "Z"), delimiter="\t", ndmin=2)
        self.W = np.loadtxt(os.path.join(
            self.modelDir, "W"), delimiter="\t", ndmin=2)
        self.V = np.loadtxt(os.path.join(
            self.modelDir, "V"), delimiter="\t", ndmin=2)
        self.T = np.loadtxt(os.path.join(
            self.modelDir, "Theta"), delimiter="\t", ndmin=2)
        self.Sigma = np.loadtxt(os.path.join(
            self.modelDir, "Sigma"), delimiter="\t", ndmin=2)
        self.Mean = np.loadtxt(os.path.join(
            self.modelDir, "Mean"), delimiter="\t", ndmin=2)
        self.Variance = np.loadtxt(os.path.join(
            self.modelDir, "Variance"), delimiter="\t", ndmin=2)

    def validateModel(self):
        Z_m, Z_n = self.Z.shape
        W_m, W_n = self.W.shape
        V_m, V_n = self.V.shape
        # If T is empty
        if self.T.size == 0:
            T_m, T_n = 0, Z_m
        else:
            T_m, T_n = self.T.shape
        Mean_m, Mean_n = self.Mean.shape
        Variance_m, Variance_n = self.Variance.shape

        assert Z_n == Mean_m == Variance_m
        assert Z_m == W_n == V_n == T_n
        assert W_m == V_m
        assert self.Sigma.size == 1
        assert Mean_n == Variance_n == 1

    # Restructure the W and V matrix to be node ID major instead of class ID major
    def rearrangeWandV(self, mat):
        matNew = np.empty(mat.shape)
        for i in range(self.numClasses):
            for j in range(self.totalNodes):
                matNew[i + j * self.numClasses] = mat[i * self.totalNodes + j]
        return matNew

    # Note: Following computations assume that the tree is always complete (which is a property of Bonsai algorithm)
    def computeVars(self):
        self.D = self.Z.shape[1]
        self.d = self.Z.shape[0]
        self.internalNodes = self.T.shape[0]
        self.depth = int(math.log2(self.internalNodes + 1))
        self.totalNodes = 2 ** (self.depth + 1) - 1
        self.numClasses = len(self.W) // self.totalNodes

    def formatModel(self):
        # Manually set mean and variance of bias to 0 and 1 respectively
        self.Mean[-1, 0] = 0
        self.Variance[-1, 0] = 1

        sigma = self.Sigma

        self.computeVars()

        # Precompute some values to speedup prediction
        # Precompute V*sigma and Z/d
        self.V = self.V * sigma
        self.Z = self.Z / self.d

        # Precompute Z*(X-m)/v by absorbing m, v into Z
        #self.Z = [[x / v[0] for x, v in zip(y, self.Variance)] for y in self.Z]
        self.Z = self.Z / self.Variance.reshape(1, -1)

        self.mean = self.Z.dot(self.Mean)
        assert self.mean.shape[1] == 1
        self.mean = self.mean.reshape(1, -1)

        # Restructure W and V
        self.W = self.rearrangeWandV(self.W)
        self.V = self.rearrangeWandV(self.V)

    def verifyModel(self):
        # If T is empty, it is replaced with a matrix containing 2 elements (which is never used during prediction)
        if self.T.size == 0:
            print("Warning: Empty matrix T\n")
            self.T = np.array([[-0.000001, 0.000001]])

    # Write the Bonsai algorithm in terms of the compiler DSL.
    # Compiler takes this as input and generates fixed-point code.
    def genSeeDotProgram(self):
        # Threshold for mean
        m_mean, M_mean = np.amin(self.mean), np.amax(self.mean)
        if abs(m_mean) < 0.0000005:
            m_mean = -0.000001
        if abs(M_mean) < 0.0000005:
            M_mean = 0.000001

        with open(self.seeDotProgram, 'w') as file:
            # Matrix declarations
            file.write("let X   = (%d, 1)   in [%.6f, %.6f] in\n" % (
                (self.X.shape[1],) + self.trainDatasetRange))
            file.write("let Z   = (%d, %d)  in [%.6f, %.6f] in\n" % (
                self.d, self.D, np.amin(self.Z), np.amax(self.Z)))
            file.write("let W   = (%d, %d, %d) in [%.6f, %.6f] in\n" % (
                self.totalNodes, self.numClasses, self.d, np.amin(self.W), np.amax(self.W)))
            file.write("let V   = (%d, %d, %d) in [%.6f, %.6f] in\n" % (
                self.totalNodes, self.numClasses, self.d, np.amin(self.V), np.amax(self.V)))
            file.write("let T   = (%d, 1, %d) in [%.6f, %.6f] in\n" % (
                self.internalNodes, self.d, np.amin(self.T), np.amax(self.T)))
            file.write("let mean = (%d, 1) in [%.6f, %.6f] in\n\n" % (
                self.d, m_mean, M_mean))

            if useSparseMat():
                file.write("let ZX = Z |*| X - mean in\n\n")
            else:
                file.write("let ZX = Z * X - mean in\n\n")

            # Computing score for depth == 0
            file.write("// depth 0\n")
            file.write("let node0   = 0    in\n")
            file.write("let W0      = W[node0] * ZX in\n")
            file.write("let V0      = V[node0] * ZX in\n")
            file.write("let V0_tanh = tanh(V0) in\n")
            file.write("let score0  = W0 <*> V0_tanh in\n\n")

            # Computing score for depth > 0
            for i in range(1, self.depth + 1):
                file.write("// depth %d\n" % (i))
                file.write(
                    "let node%d   = (T[node%d] * ZX) >= 0? 2 * node%d + 1 : 2 * node%d + 2 in\n" % (i, i - 1, i - 1, i - 1))
                file.write("let W%d      = W[node%d] * ZX in\n" % (i, i))
                file.write("let V%d      = V[node%d] * ZX in\n" % (i, i))
                file.write("let V%d_tanh = tanh(V%d) in\n" % (i, i))
                file.write(
                    "let score%d  = score%d + W%d <*> V%d_tanh in\n\n" % (i, i - 1, i, i))

            # Predicting the class
            if self.numClasses <= 2:
                file.write("sgn(score%d)\n" % (self.depth))
            else:
                file.write("argmax(score%d)\n" % (self.depth))

    # Writing the model as a bunch of variables, arrays and matrices to a file
    def writeModel(self):
        np.save(os.path.join(self.modelOutputDir, "mean.npy"), self.mean)
        np.save(os.path.join(self.modelOutputDir, "Z.npy"), self.Z)
        np.save(os.path.join(self.modelOutputDir, "W.npy"), self.W)
        np.save(os.path.join(self.modelOutputDir, "V.npy"), self.V)
        np.save(os.path.join(self.modelOutputDir, "T.npy"), self.T)

    def processModel(self):
        self.readModel()
        self.validateModel()
        self.formatModel()
        self.verifyModel()
        self.genSeeDotProgram()
        self.writeModel()

    def run(self):
        self.processDataset()

        self.processModel()

        assert len(self.X[0]) == len(self.Z[0])
