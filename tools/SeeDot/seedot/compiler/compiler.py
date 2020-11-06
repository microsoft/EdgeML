# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import antlr4 as antlr
import argparse
import os
import pickle


import seedot

import seedot.compiler.antlr.seedotLexer as seedotLexer
import seedot.compiler.antlr.seedotParser as seedotParser

import seedot.compiler.ast.ast as AST
import seedot.compiler.ast.astBuilder as astBuilder
import seedot.compiler.ast.printAST as printAST

import seedot.compiler.codegen.arduino as arduino
import seedot.compiler.codegen.x86 as x86
import seedot.compiler.codegen.m3 as m3

import seedot.compiler.ir.irBuilder as irBuilder
import seedot.compiler.ir.irUtil as irUtil

import seedot.compiler.TF.ProcessTFGraph as TFMain
import seedot.compiler.ONNX.process_onnx as process_onnx

import seedot.compiler.type as type
import seedot.util as util
import seedot.writer as writer

import seedot.config as config

class Compiler:

    def __init__(self, algo, version, target, inputFile, outputDir, profileLogFile, maxScale, source, outputLogFile, generateAllFiles=True, id=None, printSwitch=-1, substitutions={}, scaleForX=None, variableToBitwidthMap={}, sparseMatrixSizes={}, demotedVarsList=[], demotedVarsOffsets={}, paramInNativeBitwidth=True):
        if os.path.isfile(inputFile) == False:
            print(inputFile)
            raise Exception("Input file doesn't exist")

        util.setAlgo(algo)
        util.setVersion(version)
        util.setTarget(target)
        self.input = inputFile
        self.outputDir = outputDir
        util.setProfileLogFile(profileLogFile)
        self.outputLogFile = outputLogFile
        util.setMaxScale(maxScale)
        self.source = source
        self.generateAllFiles = generateAllFiles
        self.id = str(id) if id is not None else ""
        self.printSwitch = printSwitch

        self.intermediateScales = {}
        self.substitutions = substitutions
        self.scaleForX = scaleForX
        self.scaleForY = 0
        self.problemType = config.ProblemType.default

        self.variableToBitwidthMap = variableToBitwidthMap
        self.sparseMatrixSizes = sparseMatrixSizes

        self.demotedVarsList = demotedVarsList
        self.demotedVarsOffsets = demotedVarsOffsets

        self.paramInNativeBitwidth = paramInNativeBitwidth

        self.biasShifts = {}

    def genASTFromFile(self, inputFile):
        # Parse and generate CST for the input
        lexer = seedotLexer.seedotLexer(antlr.FileStream(inputFile))
        tokens = antlr.CommonTokenStream(lexer)
        parser = seedotParser.seedotParser(tokens)
        tree = parser.expr()

        # Generate AST
        ast = astBuilder.ASTBuilder().visit(tree)
        return ast

    def genAST(self, inputFile):
        ext = os.path.splitext(inputFile)[1]

        if self.source == config.Source.seedot:
            return self.genASTFromFile(inputFile)
        elif self.source == config.Source.onnx:
            ast = process_onnx.get_seedot_ast(inputFile)
            return ast
        else:    
            ast = TFMain.main()
            # with open(inputFile, 'rb') as file:
            #	ast = pickle.load(file)
            return ast

    def run(self):
        ast = self.genAST(self.input)

        # Pretty printing AST
        # printAST.PrintAST().visit(ast)

        # Perform type inference
        type.InferType().visit(ast)

        irUtil.init()

        res, state = self.compile(ast)

        if util.forArduino():
            codegen = arduino.Arduino(self.outputDir, *state)
        elif util.forM3():
            assert self.problemType == config.ProblemType.regression, "M3 codegen only for Regression problems"
            codegen = m3.M3(self.outputDir, *state)
        elif util.forX86():
            codegen = x86.X86(self.outputDir, self.generateAllFiles, self.printSwitch, self.id, self.paramInNativeBitwidth, *state)
        else:
            assert False

        codegen.printAll(*res)

    def compile(self, ast):
        return self.genCodeWithFuncCalls(ast)

    def genCodeWithFuncCalls(self, ast):

        outputLog = writer.Writer(self.outputLogFile)

        if util.getVersion() == config.Version.fixed and config.ddsEnabled:
            self.intermediateScales = self.readDataDrivenScales()

        compiler = irBuilder.IRBuilder(outputLog, self.intermediateScales, self.substitutions, self.scaleForX, self.variableToBitwidthMap, self.sparseMatrixSizes, self.demotedVarsList, self.demotedVarsOffsets)
        res = compiler.visit(ast)

        print(compiler.varScales)
        self.biasShifts = compiler.biasShifts
        self.varScales = dict(compiler.varScales)

        outputLog.close()

        state = [compiler.varDeclarations, compiler.varDeclarationsLocal, compiler.varScales, compiler.varIntervals, compiler.intConstants, compiler.expTables, compiler.globalVars, compiler.internalVars, compiler.floatConstants, compiler.substitutions, compiler.demotedVarsOffsets, compiler.varsForBitwidth, compiler.varLiveIntervals, compiler.notScratch, compiler.coLocatedVariables]

        state[12] = self.adjustLiveRanges(state[12], compiler.allDepths)

        for i in compiler.globalVars:
            state[13].append(i)

        if util.getVersion() == config.Version.floatt:
            self.independentVars = list(compiler.independentVars)
            self.independentVars += compiler.globalVars

        self.substitutions = compiler.substitutions # for profiling code, substitutions get updated and this variable is then read by main.py

        self.scaleForX = compiler.varScales['X']
        self.scaleForY = compiler.varScales[res[1].idf] if res[1].idf in compiler.varScales else 0
        self.problemType = config.ProblemType.classification if res[1].idf not in compiler.varScales else config.ProblemType.regression

        return res, state

    def readDataDrivenScales(self):
        tempScales = {}
        error = 0.01
        with open('temp/Predictor/dump.profile', 'r') as f:
            for line in f:
                entries = line.strip().split(",")
                var, m, M = entries
                m, M = float(m), float(M)
                tempScales[var] = util.computeScalingFactor(max(abs(m) + error, abs(M) + error)) 
        return tempScales

    def adjustLiveRanges(self, oldRanges, depthData):
        newRanges = {}
        for var in oldRanges:
            begIns = oldRanges[var][0]
            endIns = oldRanges[var][1]
            beginningDepth = depthData[begIns]
            endingDepth = depthData[endIns]
            if endingDepth > beginningDepth:
                while depthData[endIns] > beginningDepth:
                    endIns += 1
                endIns -= 1
            elif endingDepth < beginningDepth:
                while depthData[begIns] > endingDepth:
                    begIns -= 1
                begIns += 1
            newRanges[var] = [begIns, endIns]
        return newRanges
