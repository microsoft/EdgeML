# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os

from Converter.Util import *


class Lenet:

    def readDataset(self):
        self.X, self.Y = readXandY()

    def writeDataset(self):
        writeMatAsCSV(self.X, os.path.join(getDatasetOutputDir(), "X.csv"))
        writeMatAsCSV(self.Y, os.path.join(getDatasetOutputDir(), "Y.csv"))

    def processDataset(self):
        self.readDataset()
        assert len(self.X) == len(self.Y)
        self.transformDataset()
        self.writeDataset()

    def readModel(self):
        self.Wc1 = readFileAsMat(os.path.join(
            getModelDir(), "Wc1"), "\t", float)
        self.Wc2 = readFileAsMat(os.path.join(
            getModelDir(), "Wc2"), "\t", float)
        self.Wf1 = readFileAsMat(os.path.join(
            getModelDir(), "Wf1"), "\t", float)
        self.Wf2 = readFileAsMat(os.path.join(
            getModelDir(), "Wf2"), "\t", float)
        #self.Wf3 = readFileAsMat(os.path.join(getModelDir(), "Wf3"), "\t", float)
        self.Bc1 = readFileAsMat(os.path.join(
            getModelDir(), "Bc1"), "\t", float)
        self.Bc2 = readFileAsMat(os.path.join(
            getModelDir(), "Bc2"), "\t", float)
        self.Bf1 = readFileAsMat(os.path.join(
            getModelDir(), "Bf1"), "\t", float)
        self.Bf2 = readFileAsMat(os.path.join(
            getModelDir(), "Bf2"), "\t", float)
        #self.Bf3 = readFileAsMat(os.path.join(getModelDir(), "Bf3"), "\t", float)

    def validateModel(self):
        Wc1_m, Wc1_n = matShape(self.Wc1)
        Wc2_m, Wc2_n = matShape(self.Wc2)
        Wf1_m, Wf1_n = matShape(self.Wf1)
        Wf2_m, Wf2_n = matShape(self.Wf2)
        #Wf3_m, Wf3_n = matShape(self.Wf3)
        Bc1_m, Bc1_n = matShape(self.Bc1)
        Bc2_m, Bc2_n = matShape(self.Bc2)
        Bf1_m, Bf1_n = matShape(self.Bf1)
        Bf2_m, Bf2_n = matShape(self.Bf2)
        #Bf3_m, Bf3_n = matShape(self.Bf3)

        #assert Wc1_n == Bc1_n
        #assert Wc2_n == Bc2_n
        assert Wf1_n == Wf2_m
        #assert Wf2_n == Wf3_m
        assert Wf1_n == Bf1_n
        assert Wf2_n == Bf2_n
        #assert Wf3_n == Bf3_n

    def computeVars(self):
        pass

    # Precompute some values to speedup prediction
    def formatModel(self):
        self.computeVars()

    # Write macros and namespace declarations
    def writeHeader(self):
        with open(self.headerFile, 'a') as file:
            file.write("#pragma once\n\n")

            if forArduino():
                s = 'model'
            else:
                s = 'lenet_' + getVersion()
            file.write("namespace %s {\n\n" % (s))

    def writeFooter(self):
        with open(self.headerFile, 'a') as file:
            file.write("}\n")

    def processModel(self):
        self.readModel()
        self.validateModel()
        self.formatModel()
        self.transformModel()
        self.writeModel()

    def run(self):
        if getVersion() == Common.Version.Float:
            self.headerFile = os.path.join(
                getOutputDir(), "lenet_float_model.h")
        else:
            self.headerFile = os.path.join(
                getOutputDir(), "seedot_fixed_model.h")
        self.inputFile = os.path.join(getOutputDir(), "input.sd")
        self.infoFile = os.path.join(getOutputDir(), "info.txt")

        open(self.headerFile, 'w').close()

        if dumpDataset():
            self.processDataset()

        self.processModel()


class LenetFixed(Lenet):

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

        with open(self.infoFile, 'w') as file:
            file.write("Range of test dataset: [%.6f, %.6f]\n" % (
                testDatasetRange))
            file.write("Range of training dataset: [%.6f, %.6f]\n" % (
                self.trainDatasetRange))
            file.write("Test dataset scaled by: %d\n\n" % (scale))

    # Write the ProtoNN algorithm in terms of the compiler DSL.
    # Compiler takes this as input and generates fixed-point code.
    def genInputForCompiler(self):
        with open(self.inputFile, 'w') as file:
            # Matrix declarations
            file.write("let X   = (%d, 1)   in [%.6f, %.6f] in\n" % (
                (len(self.X[0]),) + self.trainDatasetRange))

            file.write("let %s  = (%d, %d, %d, %d)    in [%.6f, %.6f] in\n" % (
                ("Wc1", 5, 5, 3, 4) + matRange(self.Wc1)))
            file.write("let %s  = (%d, %d, %d, %d)    in [%.6f, %.6f] in\n" % (
                ("Wc2", 5, 5, 4, 12) + matRange(self.Wc2)))
            file.write("let %s  = (%d, %d)    in [%.6f, %.6f] in\n" % (
                ("Wf1", 768, 64) + matRange(self.Wf1)))
            file.write("let %s  = (%d, %d)    in [%.6f, %.6f] in\n" % (
                ("Wf2", 64, 10) + matRange(self.Wf2)))
            #file.write("let %s  = (%d, %d)    in [%.6f, %.6f] in\n" % (("Wf3", 84, 10) + matRange(self.Wf3)))
            file.write("let %s  = (%d)    in [%.6f, %.6f] in\n" % (
                ("Bc1", 4) + matRange(self.Bc1)))
            file.write("let %s  = (%d)    in [%.6f, %.6f] in\n" % (
                ("Bc2", 12) + matRange(self.Bc2)))
            file.write("let %s  = (%d)    in [%.6f, %.6f] in\n" % (
                ("Bf1", 64) + matRange(self.Bf1)))
            file.write("let %s  = (%d)    in [%.6f, %.6f] in\n" % (
                ("Bf2", 10) + matRange(self.Bf2)))
            #file.write("let %s  = (%d)    in [%.6f, %.6f] in\n" % (("Bf3", 10) + matRange(self.Bf3)))

            file.write("\n")

            # Algorithm

            file.write(
                "let X2    = reshape(X, (1, 32, 32, 3), (1, 2))     in\n")
            file.write("let Hc1   = relu   ((X2    # Wc1) <+> Bc1) in\n")
            file.write("let Hc1P  = maxpool(Hc1, 2)              in\n")
            file.write("let Hc2   = relu   ((Hc1P  # Wc2) <+> Bc2) in\n")
            file.write("let Hc2P  = maxpool(Hc2, 2)              in\n")
            file.write(
                "let Hc2PP = reshape(Hc2P, (1, 768), (1, 4, 2, 3))       in\n")
            file.write("let Hf1   = relu   ((Hc2PP * Wf1) <+> Bf1) in\n")
            file.write("let Hf2   =        ((Hf1   * Wf2) <+> Bf2) in\n")
            #file.write("let Hf3   =        ((Hf2   * Wf3) <+> Bf3) in\n")
            file.write("argmax(Hf2)\n")

    # Quantize the matrices
    def transformModel(self):
        if dumpDataset():
            self.genInputForCompiler()

        self.Wc1, _ = scaleMat(self.Wc1)
        self.Wc2, _ = scaleMat(self.Wc2)
        self.Wf1, _ = scaleMat(self.Wf1)
        self.Wf2, _ = scaleMat(self.Wf2)
        #self.Wf3, _ = scaleMat(self.Wf3)
        self.Bc1, _ = scaleMat(self.Bc1)
        self.Bc2, _ = scaleMat(self.Bc2)
        self.Bf1, _ = scaleMat(self.Bf1)
        self.Bf2, _ = scaleMat(self.Bf2)
        #self.Bf3, _ = scaleMat(self.Bf3)

    # Writing the model as a bunch of variables, arrays and matrices to a file
    def writeModel(self):
        self.writeHeader()

        if forArduino():
            writeListAsArray(self.X[0], 'X', self.headerFile)
            writeVars({'Y': self.Y[0][0]}, self.headerFile)

        writeMatAsArray(self.Wc1, 'Wc1', self.headerFile,
                        shapeStr="[%d]" * 4 % (5, 5, 3, 4))
        writeMatAsArray(self.Wc2, 'Wc2', self.headerFile,
                        shapeStr="[%d]" * 4 % (5, 5, 4, 12))
        writeMatAsArray(self.Wf1, 'Wf1', self.headerFile)
        writeMatAsArray(self.Wf2, 'Wf2', self.headerFile)
        #writeMatAsArray(self.Wf3, 'Wf3', self.headerFile)
        writeMatAsArray(self.Bc1, 'Bc1', self.headerFile,
                        shapeStr="[%d]" * 1 % (4))
        writeMatAsArray(self.Bc2, 'Bc2', self.headerFile,
                        shapeStr="[%d]" * 1 % (12))
        writeMatAsArray(self.Bf1, 'Bf1', self.headerFile,
                        shapeStr="[%d]" * 1 % (64))
        writeMatAsArray(self.Bf2, 'Bf2', self.headerFile,
                        shapeStr="[%d]" * 1 % (10))
        #writeMatAsArray(self.Bf3, 'Bf3', self.headerFile, shapeStr="[%d]" * 1 % (10))

        self.writeFooter()


class LenetFloat(Lenet):

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

            with open(self.infoFile, 'w') as file:
                file.write("Old range of X: [%.6f, %.6f]\n" % (beforeRange))
                file.write("Trimmed the dataset from %d to %d data points; %.3f%%\n" % (
                    beforeLen, afterLen, float(beforeLen - afterLen) / beforeLen * 100))
                file.write("New range of X: [%.6f, %.6f]\n" % (afterRange))

    def transformModel(self):
        pass

    # Writing the model as a bunch of variables, arrays and matrices to a file
    def writeModel(self):
        self.writeHeader()

        if forArduino():
            writeListAsArray(self.X[0], 'X', self.headerFile)
            writeVars({'Y': self.Y[0][0]}, self.headerFile)

        writeMatAsArray(self.Wc1, 'Wc1', self.headerFile,
                        shapeStr="[%d]" * 4 % (5, 5, 3, 4))
        writeMatAsArray(self.Wc2, 'Wc2', self.headerFile,
                        shapeStr="[%d]" * 4 % (5, 5, 4, 12))
        writeMatAsArray(self.Wf1, 'Wf1', self.headerFile)
        writeMatAsArray(self.Wf2, 'Wf2', self.headerFile)
        #writeMatAsArray(self.Wf3, 'Wf3', self.headerFile)
        writeMatAsArray(self.Bc1, 'Bc1', self.headerFile,
                        shapeStr="[%d]" * 1 % (4))
        writeMatAsArray(self.Bc2, 'Bc2', self.headerFile,
                        shapeStr="[%d]" * 1 % (12))
        writeMatAsArray(self.Bf1, 'Bf1', self.headerFile,
                        shapeStr="[%d]" * 1 % (64))
        writeMatAsArray(self.Bf2, 'Bf2', self.headerFile,
                        shapeStr="[%d]" * 1 % (10))
        #writeMatAsArray(self.Bf3, 'Bf3', self.headerFile, shapeStr="[%d]" * 1 % (10))

        self.writeFooter()
