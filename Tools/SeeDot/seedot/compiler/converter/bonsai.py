# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import math
import os

from seedot.compiler.converter.util import *

import seedot.common as Common

# Class to read Bonsai model dumps and generate input files (C header file and the compiler input)
# The two classes BonsaiFixed and BonsaiFloat are for generating fixed model and float model respectively
# The baseclass Bonsai collects some of the common functions between them.


class Bonsai:

    def readDataset(self):
        self.X, self.Y = readXandY()

    def formatDataset(self):
        assert len(self.X) == len(self.Y)
        for i in range(len(self.X)):
            self.X[i].append(1)

    def writeDataset(self):
        writeMatAsCSV(self.X, os.path.join(getDatasetOutputDir(), "X.csv"))
        writeMatAsCSV(self.Y, os.path.join(getDatasetOutputDir(), "Y.csv"))

    def processDataset(self):
        self.readDataset()
        self.formatDataset()
        self.transformDataset()
        self.writeDataset()

    def readTheta(self):
        if os.path.isfile(os.path.join(getModelDir(), "T")):
            return readFileAsMat(os.path.join(getModelDir(), "T"), "\t", float)
        elif os.path.isfile(os.path.join(getModelDir(), "Theta")):
            return readFileAsMat(os.path.join(getModelDir(), "Theta"), "\t", float)
        else:
            assert False

    def readModel(self):
        self.Z = readFileAsMat(os.path.join(getModelDir(), "Z"), "\t", float)
        self.W = readFileAsMat(os.path.join(getModelDir(), "W"), "\t", float)
        self.V = readFileAsMat(os.path.join(getModelDir(), "V"), "\t", float)
        self.T = self.readTheta()
        self.Sigma = readFileAsMat(os.path.join(
            getModelDir(), "Sigma"), "\t", float)
        self.Mean = readFileAsMat(os.path.join(
            getModelDir(), "Mean"), "\t", float)
        self.Variance = readFileAsMat(os.path.join(
            getModelDir(), "Std"), "\t", float)

    def validateModel(self):
        Z_m, Z_n = matShape(self.Z)
        W_m, W_n = matShape(self.W)
        V_m, V_n = matShape(self.V)
        # If T is empty
        if len(self.T) == 0:
            T_m, T_n = 0, Z_m
        else:
            T_m, T_n = matShape(self.T)
        Sigma_m, Sigma_n = matShape(self.Sigma)
        Mean_m, Mean_n = matShape(self.Mean)
        Variance_m, Variance_n = matShape(self.Variance)

        assert Z_n == Mean_m == Variance_m
        assert Z_m == W_n == V_n == T_n
        assert W_m == V_m
        assert Sigma_m == Sigma_n == 1
        assert Mean_n == Variance_n == 1

    # Restructure the W and V matrix to be node ID major instead of class ID
    # major
    def rearrangeWandV(self, mat):
        matNew = [[] for _ in range(len(mat))]
        for i in range(self.numClasses):
            for j in range(self.totalNodes):
                matNew[i + j * self.numClasses] = mat[i * self.totalNodes + j]
        return matNew

    # Note: Following computations assume that the tree is always complete
    # (which is a property of Bonsai algorithm)
    def computeVars(self):
        self.D = len(self.Z[0])
        self.d = len(self.Z)
        self.internalNodes = len(self.T)
        self.depth = int(math.log2(self.internalNodes + 1))
        self.totalNodes = 2 ** (self.depth + 1) - 1
        self.numClasses = len(self.W) // self.totalNodes

    def formatModel(self):
        # Manually set mean and variance of bias to 0 and 1 respectively
        self.Mean[len(self.Mean) - 1] = [0]
        self.Variance[len(self.Variance) - 1] = [1]

        sigma = self.Sigma[0][0]

        self.computeVars()

        # Precompute some values to speedup prediction
        # Precompute V*sigma and Z/d
        self.V = [[x * sigma for x in y] for y in self.V]
        self.Z = [[x / self.d for x in y] for y in self.Z]

        # Precompute Z*(X-m)/v by absorbing m, v into Z
        self.Z = [[x / v[0] for x, v in zip(y, self.Variance)] for y in self.Z]

        self.mean = matMul(self.Z, self.Mean)
        assert len(self.mean[0]) == 1
        self.mean = [x[0] for x in self.mean]

        # Restructure W and V
        self.W = self.rearrangeWandV(self.W)
        self.V = self.rearrangeWandV(self.V)

    def verifyModel(self):
        # If T is empty, it is replaced with a matrix containing 2 elements
        # (which is never used during prediction)
        if len(self.T) == 0:
            print("Warning: Empty matrix T\n")
            self.T = [[-0.000001, 0.000001]]

    def computeModelSize(self):
        mean_num = len(self.mean)
        if useSparseMat():
            Z_transp = matTranspose(self.Z)
            Zval, Zidx = convertToSparse(Z_transp)
            Z_num = len(Zval)
            Zidx_num = len(Zidx)
        else:
            Z_num = len(self.Z) * len(self.Z[0])
            Zidx_num = 0
        W_num = len(self.W) * len(self.W[0])
        V_num = len(self.V) * len(self.V[0])
        T_num = len(self.T) * len(self.T[0])

        total_num = mean_num + Z_num + W_num + V_num + T_num

        with open(self.infoFile, 'a') as file:
            file.write("nnz values: %d\n" % (total_num))
            file.write("# indexes: %d\n\n" % (Zidx_num))
            file.write("---------------------\n")
            file.write("Model size comparison\n")
            file.write("---------------------\n")
            file.write("32-bit floating-points (in KB): %.3f\n" %
                       (((total_num * 4) + (Zidx_num * 2)) / 1024))
            file.write(
                "(assuming 4 bytes for values and 2 bytes for indices)\n\n")
            file.write("16-bit fixed-points (in KB): %.3f\n" %
                       (((total_num * 2) + (Zidx_num * 2)) / 1024))
            file.write(
                "(assuming 2 bytes for values and 2 bytes for indices)\n\n")
            file.write("32-bit fixed-points (in KB): %.3f\n" %
                       (((total_num * 4) + (Zidx_num * 2)) / 1024))
            file.write(
                "(assuming 4 bytes for values and 2 bytes for indices)\n")
            file.write("--------------------------------------------\n\n")

    # Write macros and namespace declarations
    def writeHeader(self):
        with open(self.headerFile, 'a') as file:
            file.write("#pragma once\n\n")
            if useSparseMat():
                file.write("#define B_SPARSE_Z 1\n\n")
            else:
                file.write("#define B_SPARSE_Z 0\n\n")

            if forArduino():
                s = 'model'
            else:
                s = 'bonsai_' + getVersion()
            file.write("namespace %s {\n\n" % (s))

    def writeFooter(self):
        with open(self.headerFile, 'a') as file:
            file.write("}\n")

    def processModel(self):
        self.readModel()
        self.validateModel()
        self.formatModel()
        self.verifyModel()
        self.transformModel()
        self.computeModelSize()
        self.writeModel()

    def run(self):
        if getVersion() == Common.Version.Float:
            self.headerFile = os.path.join(
                getOutputDir(), "bonsai_float_model.h")
        else:
            self.headerFile = os.path.join(
                getOutputDir(), "seedot_fixed_model.h")
        self.inputFile = os.path.join(getOutputDir(), "input.sd")
        self.infoFile = os.path.join(getOutputDir(), "info.txt")

        open(self.headerFile, 'w').close()
        open(self.infoFile, 'w').close()

        if dumpDataset():
            self.processDataset()

        self.processModel()

        if dumpDataset():
            assert len(self.X[0]) == len(self.Z[0])


class BonsaiFixed(Bonsai):

    # The X matrix is quantized using a scale factor computed from the training dataset.
    # The range of X_train is used to compute the scale factor.
    # Since the range of X_train depends on its distribution, the scale computed may be imprecise.
    # To avoid this, any outliers in X_train is trimmed off using a threshold
    # to get a more precise range and a more precise scale.
    def transformDataset(self):
        # If X itself is X_train, reuse it. Otherwise, read it from file
        if usingTrainingDataset():
            self.X_train = list(self.X)
        else:
            self.X_train, _ = readXandY(useTrainingSet=True)

        # Trim some data points from X_train
        self.X_train, _ = trimMatrix(self.X_train)

        # Compute range and scale and quantize X
        testDatasetRange = matRange(self.X)
        self.trainDatasetRange = matRange(self.X_train)

        scale = computeScale(*self.trainDatasetRange)
        self.X, _ = scaleMat(self.X, scale)

        with open(self.infoFile, 'a') as file:
            file.write("Range of test dataset: [%.6f, %.6f]\n" % (
                testDatasetRange))
            file.write("Range of training dataset: [%.6f, %.6f]\n" % (
                self.trainDatasetRange))
            file.write("Test dataset scaled by: %d\n\n" % (scale))

    # Write the Bonsai algorithm in terms of the compiler DSL.
    # Compiler takes this as input and generates fixed-point code.
    def genInputForCompiler(self):
        # Threshold for mean
        m_mean, M_mean = listRange(self.mean)
        if abs(m_mean) < 0.0000005:
            m_mean = -0.000001
        if abs(M_mean) < 0.0000005:
            M_mean = 0.000001

        with open(self.inputFile, 'w') as file:
            # Matrix declarations
            file.write("let X   = (%d, 1)   in [%.6f, %.6f] in\n" % (
                (len(self.X[0]),) + self.trainDatasetRange))
            file.write("let Z   = (%d, %d)  in [%.6f, %.6f] in\n" % (
                (self.d, self.D) + matRange(self.Z)))
            file.write("let W   = (%d, %d, %d) in [%.6f, %.6f] in\n" % (
                (self.totalNodes, self.numClasses, self.d) + matRange(self.W)))
            file.write("let V   = (%d, %d, %d) in [%.6f, %.6f] in\n" % (
                (self.totalNodes, self.numClasses, self.d) + matRange(self.V)))
            file.write("let T   = (%d, 1, %d) in [%.6f, %.6f] in\n" % (
                (self.internalNodes, self.d) + matRange(self.T)))
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

    # Quantize the matrices
    def transformModel(self):
        if dumpDataset():
            self.genInputForCompiler()

        self.Z, _ = scaleMat(self.Z)
        self.W, _ = scaleMat(self.W)
        self.V, _ = scaleMat(self.V)
        self.T, _ = scaleMat(self.T)
        self.mean, _ = scaleList(self.mean)

    # Writing the model as a bunch of variables, arrays and matrices to a file
    def writeModel(self):
        self.writeHeader()

        if forArduino():
            writeListAsArray(self.X[0], 'X', self.headerFile)
            writeVars({'Y': self.Y[0][0]}, self.headerFile)

        writeListAsArray(self.mean, 'mean', self.headerFile,
                         shapeStr="[%d]" * 2 % (self.d, 1))

        # Sparse matrices are converted in to two arrays containing values and
        # indices to reduce space
        if useSparseMat():
            Z_transp = matTranspose(self.Z)
            Zval, Zidx = convertToSparse(Z_transp)
            writeListsAsArray({'Zval': Zval, 'Zidx': Zidx}, self.headerFile)
        else:
            writeMatAsArray(self.Z, 'Z', self.headerFile)

        # If T_m is 0, the generated code will throw an error
        # Hence, setting it to 1
        T_m = self.internalNodes
        if T_m == 0:
            T_m = 1

        writeMatAsArray(self.W, 'W', self.headerFile,
                        shapeStr="[%d]" * 3 % (self.totalNodes, self.numClasses, self.d))
        writeMatAsArray(self.V, 'V', self.headerFile,
                        shapeStr="[%d]" * 3 % (self.totalNodes, self.numClasses, self.d))
        writeMatAsArray(self.T, 'T', self.headerFile,
                        shapeStr="[%d]" * 3 % (T_m, 1, self.d))

        self.writeFooter()


class BonsaiFloat(Bonsai):

    # Float model is generated for for training dataset to profile the prediction
    # Hence, X is trimmed down to remove outliers. Prediction profiling is
    # performed on the trimmed X to generate more precise profile data
    def transformDataset(self):
        if usingTrainingDataset():
            beforeLen = len(self.X)
            beforeRange = matRange(self.X)

            self.X, self.Y = trimMatrix(self.X, self.Y)

            afterLen = len(self.X)
            afterRange = matRange(self.X)

            with open(self.infoFile, 'a') as file:
                file.write("Old range of X: [%.6f, %.6f]\n" % (beforeRange))
                file.write("Trimmed the dataset from %d to %d data points; %.3f%%\n" % (
                    beforeLen, afterLen, float(beforeLen - afterLen) / beforeLen * 100))
                file.write("New range of X: [%.6f, %.6f]\n" % (afterRange))

    def transformModel(self):
        pass

    # Writing the model as a bunch of variables, arrays and matrices to a file
    def writeModel(self):
        lists = {'mean': self.mean}
        mats = {}

        # Sparse matrices are converted in to two arrays containing values and
        # indices to reduce space
        if useSparseMat():
            Z_transp = matTranspose(self.Z)
            Zval, Zidx = convertToSparse(Z_transp)
            lists.update({'Zval': Zval, 'Zidx': Zidx})
        else:
            mats['Z'] = self.Z

        mats.update({'W': self.W, 'V': self.V, 'T': self.T})

        self.writeHeader()
        writeVars({'D': self.D, 'd': self.d, 'c': self.numClasses, 'depth': self.depth,
                   'totalNodes': self.totalNodes, 'internalNodes': self.internalNodes,
                   'tanh_limit': Common.tanh_limit}, self.headerFile)

        if forArduino():
            writeListAsArray(self.X[0], 'X', self.headerFile)
            writeVars({'Y': self.Y[0][0]}, self.headerFile)

        writeListsAsArray(lists, self.headerFile)
        writeMatsAsArray(mats, self.headerFile)
        self.writeFooter()
