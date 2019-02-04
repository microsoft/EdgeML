# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from antlr4 import *
import argparse
import os

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

from Type import InferType
from Util import *
from Writer import Writer

class Compiler:

	def __init__(self, algo, version, target, inputFile, outputFile, profileLogFile, maxExpnt, numWorkers):
		if os.path.isfile(inputFile) == False:
			raise Exception("Input file doesn't exist")

		setAlgo(algo)
		setVersion(version)
		setTarget(target)
		setNumWorkers(numWorkers)
		self.input = FileStream(inputFile)
		self.outputFile = outputFile
		setProfileLogFile(profileLogFile)
		setMaxExpnt(maxExpnt)
	
	def run(self):
		# Parse and generate CST for the input
		lexer = SeeDotLexer(self.input)
		tokens = CommonTokenStream(lexer)
		parser = SeeDotParser(tokens)
		tree = parser.expr()

		# Generate AST
		ast = ASTBuilder.ASTBuilder().visit(tree)

		# Pretty printing AST
		# PrintAST().visit(ast)

		# Perform type inference
		InferType().visit(ast)

		IRUtil.init()

		res, state = self.compile(ast)

		writer = Writer(self.outputFile)

		if forArduino():
			codegen = ArduinoCodegen(writer, *state)
		elif forHls():
			codegen = HlsCodegen(writer, *state)
		elif forVerilog():
			codegen = VerilogCodegen(writer, *state)
		elif forX86():
			codegen = X86Codegen(writer, *state)
		else:
			assert False

		codegen.printAll(*res)

		writer.close()

	def compile(self, ast):
		if genFuncCalls():
			return self.genCodeWithFuncCalls(ast)
		else:
			return self.genCodeWithoutFuncCalls(ast)

	def genCodeWithFuncCalls(self, ast):

		compiler = IRBuilder()
		
		res = compiler.visit(ast)

		state = compiler.decls, compiler.scales, compiler.intvs, compiler.cnsts, compiler.expTables, compiler.globalVars, compiler.internalVars

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

		return res, state
