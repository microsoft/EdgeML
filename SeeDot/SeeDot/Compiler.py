# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from antlr4 import *
import argparse
import os
import pickle

from Antlr.SeeDotLexer import SeeDotLexer
from Antlr.SeeDotParser import SeeDotParser

import AST.AST as AST
import AST.ASTBuilder as ASTBuilder
from AST.PrintAST import PrintAST

from Codegen.Arduino import Arduino as ArduinoCodegen
from Codegen.Hls import Hls as HlsCodegen
from Codegen.Verilog import Verilog as VerilogCodegen
from Codegen.X86 import X86 as X86Codegen

from IR.IRBuilder import IRBuilder
import IR.IRUtil as IRUtil

from IR.IRGen.Arduino import Arduino as ArduinoIRGen
from IR.IRGen.Hls import Hls as HlsIRGen

from TF.ProcessTFGraph import main as TFMain

from Type import InferType
from Util import *
from Writer import Writer

class Compiler:

	def __init__(self, algo, version, target, inputFile, outputDir, profileLogFile, maxScale, outputLogFile, numWorkers):
		if os.path.isfile(inputFile) == False:
			raise Exception("Input file doesn't exist")

		setAlgo(algo)
		setVersion(version)
		setTarget(target)
		setNumWorkers(numWorkers)
		self.input = inputFile
		self.outputDir = outputDir
		setProfileLogFile(profileLogFile)
		self.outputLogFile = outputLogFile
		setMaxScale(maxScale)
	
	def genASTFromFile(self, inputFile):
		# Parse and generate CST for the input
		lexer = SeeDotLexer(FileStream(inputFile))
		tokens = CommonTokenStream(lexer)
		parser = SeeDotParser(tokens)
		tree = parser.expr()

		# Generate AST
		ast = ASTBuilder.ASTBuilder().visit(tree)
		return ast

	def genAST(self, inputFile):
		ext = os.path.splitext(inputFile)[1]

		if ext == ".sd":
			return self.genASTFromFile(inputFile)
		elif ext == ".pkl":
			ast = TFMain()
			#with open(inputFile, 'rb') as file:
			#	ast = pickle.load(file)
			return ast

	def run(self):
		ast = self.genAST(self.input)

		# Pretty printing AST
		# PrintAST().visit(ast)

		# Perform type inference
		InferType().visit(ast)

		IRUtil.init()

		res, state = self.compile(ast)

		if forArduino():
			codegen = ArduinoCodegen(self.outputDir, *state)
		elif forHls():
			codegen = HlsCodegen(self.outputDir, *state)
		elif forVerilog():
			codegen = VerilogCodegen(self.outputDir, *state)
		elif forX86():
			codegen = X86Codegen(self.outputDir, *state)
		else:
			assert False

		codegen.printAll(*res)

	def compile(self, ast):
		if genFuncCalls():
			return self.genCodeWithFuncCalls(ast)
		else:
			return self.genCodeWithoutFuncCalls(ast)

	def genCodeWithFuncCalls(self, ast):

		outputLog = Writer(self.outputLogFile)

		compiler = IRBuilder(outputLog)
		res = compiler.visit(ast)

		outputLog.close()

		state = compiler.varDeclarations, compiler.varScales, compiler.varIntervals, compiler.intConstants, compiler.expTables, compiler.globalVars, compiler.internalVars, compiler.floatConstants

		self.scaleForX = compiler.varScales['X']

		return res, state

	def genCodeWithoutFuncCalls(self, ast):
		
		if forArduino() or forX86():
			compiler = ArduinoIRGen()
		elif forHls():
			compiler = HlsIRGen()
		else:
			assert False

		prog, expr,	decls, scales, intvs, cnsts = compiler.visit(ast)

		res = prog, expr
		state = decls, scales, intvs, cnsts, compiler.expTables, compiler.VAR_IDF_INIT

		self.scaleForX = scales['X']

		return res, state
