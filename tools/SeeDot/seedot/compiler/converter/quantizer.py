# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from antlr4 import *
import os

import seedot.config as config

from seedot.compiler.antlr.seedotLexer import seedotLexer as SeeDotLexer
from seedot.compiler.antlr.seedotParser import seedotParser as SeeDotParser

import seedot.compiler.ast.astBuilder as ASTBuilder

import seedot.compiler.converter.paramsBuilder as ParamsBuilder
from seedot.compiler.converter.util import *

import seedot.compiler.ONNX.process_onnx as process_onnx
import seedot.compiler.ONNX.paramsBuilderOnnx as paramsBuilderOnnx

class Quantizer:

    def __init__(self):
        self.sparseMatSizes = {}

    def genASTFromFile(self, inputFile):
        # Parse and generate CST for the input
        lexer = SeeDotLexer(FileStream(inputFile))
        tokens = CommonTokenStream(lexer)
        parser = SeeDotParser(tokens)
        tree = parser.expr()

        # Generate AST
        ast = ASTBuilder.ASTBuilder().visit(tree)
        return ast

    def genAST(self, inputFile, source):
        ext = os.path.splitext(inputFile)[1]

        if source == config.Source.seedot:
            return self.genASTFromFile(inputFile)
        elif source == config.Source.onnx:
            ast = process_onnx.get_seedot_ast(inputFile)
            return ast
        else:    
            ast = TFMain.main()
            # with open(inputFile, 'rb') as file:
            #	ast = pic    

    def buildParams(self, source):
        ast = self.genAST(getInputFile(), source)

        # Generate params
        paramsBuilder = ParamsBuilder.ParamsBuilder()
        paramsBuilder.visit(ast)

        self.params = paramsBuilder.params.values()

    def readDataset(self):
        self.X, self.Y = readXandY(numOutputs=self.numOutputs)

    def writeDataset(self):
        writeMatAsCSV(self.X, os.path.join(getDatasetOutputDir(), "X.csv"))
        writeMatAsCSV(self.Y, os.path.join(getDatasetOutputDir(), "Y.csv"))

    def processDataset(self):
        self.readDataset()
        assert len(self.X) == len(self.Y)
        self.transformDataset()
        self.writeDataset()

    def readModel(self):
        for param in self.params:
            #param.data = readFileAsMat(os.path.join(getModelDir(), param.name), "\t", float)
            param.data = np.load(os.path.join(
                getModelDir(), param.name + ".npy"))

            if param.data.ndim == 1:
                param.data = param.data.reshape(-1, 1)

            param.data = param.data.tolist()

    def computeModelSize(self):
        totalVal = 0
        totalIndex = 0

        for param in self.params:
            if param.sparse:
                transp = matTranspose(param.data)
                val, idx = convertToSparse(transp)
                totalVal += len(val)
                totalIndex += len(idx)
            else:
                totalVal += len(param.data) * len(param.data[0])

        with open(self.infoFile, 'a') as file:
            file.write("nnz values: %d\n" % (totalVal))
            file.write("# indexes: %d\n\n" % (totalIndex))
            file.write("---------------------\n")
            file.write("Model size comparison\n")
            file.write("---------------------\n")
            file.write("32-bit floating-points (in KB): %.3f\n" %
                       (((totalVal * 4) + (totalIndex * 2)) / 1024))
            file.write(
                "(assuming 4 bytes for values and 2 bytes for indices)\n\n")
            file.write("16-bit fixed-points (in KB): %.3f\n" %
                       (((totalVal * 2) + (totalIndex * 2)) / 1024))
            file.write(
                "(assuming 2 bytes for values and 2 bytes for indices)\n\n")
            file.write("32-bit fixed-points (in KB): %.3f\n" %
                       (((totalVal * 4) + (totalIndex * 2)) / 1024))
            file.write(
                "(assuming 4 bytes for values and 2 bytes for indices)\n")
            file.write("--------------------------------------------\n\n")

    # Writing the model as a bunch of variables, arrays and matrices to a file
    def writeModel(self):
        self.writeHeader()

        if forArduino() and dumpDataset():
            scaleOfX = self.allScales['X'] #computeScale(*self.trainDatasetRange)
            Xint, _ = scaleList(self.X[0], scaleOfX)

            writeListAsArray(self.X[0], 'X', self.headerFile, None, self.varsForBitwidth['X'])
            writeListAsArray(Xint, 'Xint', self.headerFile, None, self.varsForBitwidth['X'])
            writeVars({'scaleOfX': scaleOfX}, self.headerFile)
            writeVars({'Y': self.Y[0][0]}, self.headerFile) 

        if forM3() and dumpDataset():
            writeVars({'scaleForX': self.allScales['X']}, self.scalesFile)
            assert self.scaleForY is not None, "Illegal state for M3 codegen"
            writeVars({'scaleForY': self.scaleForY}, self.scalesFile)

        for param in self.params:
            if param.sparse:
                transp = matTranspose(param.data)
                val, idx = convertToSparse(transp)
                self.sparseMatSizes[param.name + 'val'] = len(val)
                self.sparseMatSizes[param.name + 'idx'] = len(idx)
                if forArduino() or forM3():
                    bw = self.varsForBitwidth[param.name + 'val'] if param.name not in self.promoteParam else 2 * self.varsForBitwidth[param.name]
                    writeListAsArray(val, param.name + 'val', self.headerFile, None, bw)
                    writeListAsArray(idx, param.name + 'idx', self.headerFile, None, self.varsForBitwidth[param.name + 'idx'])
                else:
                    if hasattr(self, 'varsForBitwidth') and param.name + 'val' in self.varsForBitwidth:
                        writeListAsArray(val, param.name + 'val', self.headerFile, None, self.varsForBitwidth[param.name + 'val'])
                    else:
                        writeListAsArray(val, param.name + 'val', self.headerFile)
                    if hasattr(self, 'varsForBitwidth') and param.name + 'idx' in self.varsForBitwidth:
                        writeListAsArray(idx, param.name + 'idx', self.headerFile, None, self.varsForBitwidth[param.name + 'idx'])
                    else:
                        writeListAsArray(idx, param.name + 'idx', self.headerFile)
            else:
                if forArduino() or forM3():
                    bw = self.varsForBitwidth[param.name] if param.name not in self.promoteParam else 2 * self.varsForBitwidth[param.name]
                    writeMatAsArray(param.data, param.name, self.headerFile, shapeStr=("[%d]" * len(param.shape)) % tuple(param.shape), bw=bw)
                else:
                    if hasattr(self, 'varsForBitwidth') and param.name in self.varsForBitwidth:
                        writeMatAsArray(param.data, param.name, self.headerFile, shapeStr=("[%d]" * len(param.shape)) % tuple(param.shape), bw=self.varsForBitwidth[param.name])
                    else:
                        writeMatAsArray(param.data, param.name, self.headerFile, shapeStr=("[%d]" * len(param.shape)) % tuple(param.shape))

        self.writeFooter()

    # Write macros and namespace declarations
    def writeHeader(self):
        with open(self.headerFile, 'a') as file:
            file.write("#pragma once\n\n")

            if forArduino():
                file.write("namespace model {\n\n")
            elif forM3():
                file.write("")
            else:
                file.write("namespace seedot_%s {\n\n" % (getVersion()))

    def writeFooter(self):
        with open(self.headerFile, 'a') as file:
            if not forM3():
                file.write("}\n")

    def processModel(self):
        self.readModel()
        self.transformModel()
        self.computeModelSize()
        self.writeModel()

    def printDataRange(self):
        for param in self.params:
            print("%s = %.6f, %.6f" %
                  (param.name, np.amin(param.data), np.amax(param.data)))
        print("X = %.6f, %.6f" % self.trainDatasetRange)

    # Float model is generated for for training dataset to profile the prediction
    # Hence, X is trimmed down to remove outliers. Prediction profiling is performed on the trimmed X to generate more precise profile data
    def transformDataset(self):
        if getVersion() == config.Version.fixed:
            # If X itself is X_train, reuse it. Otherwise, read it from file
            if usingTrainingDataset():
                self.X_train = list(self.X)
            else:
                self.X_train, _ = readXandY(useTrainingSet=True)

            # Trim some data points from X_train
            self.X_train, _ = trimMatrix(self.X_train)

            self.trainDatasetRange = matRange(self.X_train)
        elif getVersion() == config.Version.floatt:
            if usingTrainingDataset():
                self.X, self.Y = trimMatrix(self.X, self.Y)

                self.trainDatasetRange = matRange(self.X)
            else:
                self.X_train, _ = readXandY(useTrainingSet=True)

                # Trim some data points from X_train
                self.X_train, _ = trimMatrix(self.X_train)

                self.trainDatasetRange = matRange(self.X_train)

    def generateParamFilesForOnnx(self):
        self.params = paramsBuilderOnnx.getParams(getInputFile())
        self.writeHeader()
        self.transformModel()
        for param in self.params:
            writeMatAsArray(param.data, param.name, self.headerFile, shapeStr=("[%d]" * len(param.shape)) % tuple(param.shape))
        self.writeFooter()              

    def run(self, source):
        self.headerFile = os.path.join(
            getOutputDir(), "model_%s.h" % (getVersion()))
        if forM3():
            self.scalesFile = os.path.join(getOutputDir(), "scales.h")
            open(self.scalesFile, 'w').close()
        self.infoFile = os.path.join(getOutputDir(), "info.txt")

        open(self.headerFile, 'w').close()
        open(self.infoFile, 'w').close()

        if dumpDataset():
            self.processDataset()

        if source == config.Source.seedot:    
            self.buildParams(source)
            self.processModel()
        elif source == config.Source.onnx:
            self.generateParamFilesForOnnx()    

        # self.printDataRange()
        
class QuantizerFixed(Quantizer):

    def __init__(self, varsForBitwidth, allScales, numOutputs, biasShifts, scaleForY):
        super().__init__()
        self.varsForBitwidth = varsForBitwidth
        self.allScales = allScales
        self.numOutputs = numOutputs
        self.biasShifts = biasShifts
        self.scaleForY = scaleForY
        self.promoteParam = []

    # The X matrix is quantized using a scale factor computed from the training dataset.
    # The range of X_train is used to compute the scale factor.
    # Since the range of X_train depends on its distribution, the scale computed may be imprecise.
    # To avoid this, any outliers in X_train is trimmed off using a threshold to get a more precise range and a more precise scale.
    def transformDatasetOld(self):
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

    # Quantize the matrices
    def transformModel(self):
        for param in self.params:
            #data = list(param.data)

            # print(param.name)

            # if data[0] is list:
            #	beforeRange = matRange(data)
            # else:
            #	beforeRange = listRange(data)

            #scale_old = computeScale(*beforeRange)
            #data, _ = trimMatrix(data)
            # if data[0] is list:
            #	afterRange = matRange(data)
            # else:
            #	afterRange = listRange(data)
            #scale_new = computeScale(*afterRange)

            #print("Old range = ", beforeRange, "Old scale = ", scale_old)
            #print("New range = ", afterRange, "New scale = ", scale_new)
            # print()

            if forArduino() or forM3():
                scale = self.allScales[param.name]
                if forM3() and param.name in self.biasShifts.keys():
                    if self.biasShifts[param.name] < 0:
                        self.promoteParam.append(param.name)
                    scale += self.biasShifts[param.name]
                param.data, _ = scaleMat(param.data, scale)
            else:
                if param.name in self.varsForBitwidth:
                    param.data, _ = scaleMat(param.data, self.allScales[param.name])
                else:
                    param.data, _ = scaleMat(param.data)


class QuantizerFloat(Quantizer):

    def __init__(self, numOutputs):
        super().__init__()
        self.numOutputs = numOutputs

    # Float model is generated for for training dataset to profile the prediction
    # Hence, X is trimmed down to remove outliers. Prediction profiling is performed on the trimmed X to generate more precise profile data
    def transformDatasetOld(self):
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

            self.trainDatasetRange = afterRange
        else:
            self.X_train, _ = readXandY(useTrainingSet=True)

            # Trim some data points from X_train
            self.X_train, _ = trimMatrix(self.X_train)

            self.trainDatasetRange = matRange(self.X_train)

    def transformModel(self):
        pass
