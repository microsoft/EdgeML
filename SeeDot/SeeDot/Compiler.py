import os
import argparse
from antlr4 import *

from Antlr.SeeDotLexer  import SeeDotLexer
from Antlr.SeeDotParser import SeeDotParser

import AST.AST        as AST
import AST.ASTBuilder as ASTBuilder
from AST.PrintAST  import PrintAST

from Type import InferType

from IR.IRBuilder import IRBuilder
import IR.IRUtil as IRUtil

from IR.IRGen.Arduino import Arduino
from IR.IRGen.Hls import Hls

from Codegen.Arduino import Arduino as ArduinoCodegen
from Codegen.Hls import Hls as HlsCodegen
from Codegen.Verilog import Verilog as VerilogCodegen

from Util import *
from Writer import Writer

class Compiler:

	def __init__(self, algo, target, outputPragmas, inputFile, outputFile, profileLogFile, maxExpnt, numWorkers):
		if os.path.isfile(inputFile) == False:
			raise Exception("Input file doesn't exist")

		setAlgo(algo)
		setTarget(target)
		setNumWorkers(numWorkers)
		setOutputPragmasFlag(outputPragmas)
		self.input = FileStream(inputFile)
		self.outputFile = outputFile
		setProfileLogFile(profileLogFile)
		setMaxExpnt(maxExpnt)
	
	def run(self):
		if genFuncCalls():
			self.runWithFuncCalls()
		else:
			self.runWithNewAST()

	def runWithFuncCalls(self):
		# Parse and generate CST for the input
		lexer = SeeDotLexer(self.input)
		tokens = CommonTokenStream(lexer)
		parser = SeeDotParser(tokens)
		tree = parser.expr()

		# Generate AST
		ast = ASTBuilder.ASTBuilder().visit(tree)

		PrintAST().visit(ast)

		# Perform type inference
		InferType().visit(ast)

		IRUtil.init()
		
		compiler = IRBuilder()
		res = compiler.visit(ast)

		writer = Writer(self.outputFile)

		if forArduino():
			codegen = ArduinoCodegen(writer, compiler.decls, compiler.expts, compiler.intvs, compiler.cnsts, compiler.expTables, compiler.VAR_IDF_INIT)
		elif forVerilog():
			codegen = VerilogCodegen(writer, compiler.decls, compiler.expts, compiler.intvs, compiler.cnsts, compiler.expTables, compiler.VAR_IDF_INIT)
		else:
			assert False

		codegen.printAll(*res)

		writer.close()

	def runWithNewAST(self):
		# Parse and generate CST for the input
		lexer = SeeDotLexer(self.input)
		tokens = CommonTokenStream(lexer)
		parser = SeeDotParser(tokens)
		tree = parser.expr()

		# Generate AST
		ast = ASTBuilder.ASTBuilder().visit(tree)

		# Perform type inference
		InferType().visit(ast)

		#PrintAST().visit(ast)

		IRUtil.init()
		
		if forArduino():
			compiler = Arduino()
		elif forHls():
			compiler = Hls()
		else:
			assert False

		res = compiler.visit(ast)
		writer = Writer(self.outputFile)

		prog, expr,	decls, expts, intvs, cnsts = res

		if forArduino():
			codegen = ArduinoCodegen(writer, decls, expts, intvs, cnsts, compiler.expTables, compiler.VAR_IDF_INIT)
		elif forHls():
			codegen = HlsCodegen(writer, decls, expts, intvs, cnsts, compiler.expTables, compiler.VAR_IDF_INIT)
		else:
			assert False

		codegen.printAll(prog, expr)

		writer.close()
