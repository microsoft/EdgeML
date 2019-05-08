# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import os

from Codegen.CodegenBase import CodegenBase

import IR.IR as IR
import IR.IRUtil as IRUtil

import Type
from Util import *
from Writer import Writer

class Verilog(CodegenBase):

	def __init__(self, outputDir, decls, scales, intvs, cnsts, expTables, globalVars, internalVars, floatConstants):
		outputFile = os.path.join(outputDir, "predict.cpp")
		self.out = Writer(outputFile)

		self.decls = decls
		self.scales = scales
		self.intvs = intvs
		self.cnsts = cnsts
		self.expTables = expTables
		self.globalVars = globalVars
		self.internalVars = internalVars
		self.floatConstants = floatConstants

	def printPrefix(self):

		self.out.printf("`timescale 1ns/1ps\n\n")

		self.out.printf("module main(X, clk, rst);\n\n")

		self.out.increaseIndent()

		self.printVarDecls()

		type = self.decls[expr.idf]
		if isinstance(type, Type.Tensor):
			shape_str = ''.join(['[' + str(n - 1) + ':0]' for n in type.shape])
		else:
			shape_str = ''

		self.out.printf("output logic [%d:0] %s%s;" % (Common.wordLength - 1, expr.idf, shape_str), indent=True)
		self.out.printf('\n')

	def printVarDecls(self):

		self.out.printf("input clk, rst;\n", indent=True)

		for decl in self.decls:
			if decl in self.globalVars:
				continue
			typ_str = IR.DataType.getIntStr()
			idf_str = decl
			type = self.decls[decl]
			if Type.isInt(type): shape_str = ''
			elif Type.isTensor(type): shape_str = ''.join(['[' + str(n - 1) + ':0]' for n in type.shape])
			self.out.printf('input [%d:0] %s%s;\n', Common.wordLength - 1, idf_str, shape_str, indent=True)
		self.out.printf('\n')

	def printSuffix(self, expr:IR.Expr):
		self.out.printf('\n', indent=True)
		self.out.decreaseIndent()
		self.out.printf('endmodule\n', indent=True)

		self.out.close()

	def printFuncCall(self, ir):
		self.out.printf("%s %s(\n" % (ir.name, ir.name.lower()), indent = True)
		self.out.increaseIndent()
		self.out.printf(".wordLength(%d),\n" % Common.wordLength, indent = True)
		self.out.printf(".clk(clk),\n", indent = True)
		self.out.printf(".rst(rst),\n", indent = True)
		for arg, id in ir.argList.items():
			if isinstance(arg, IR.Var) and arg.idf in self.decls.keys():
				x = self.decls[arg.idf].dim
			else:
				x = 0
			self.out.printf('.', indent=True)
			self.out.printf('%s', id)
			self.out.printf('(')
			self.print(arg)
			self.out.printf('),\n')
		self.out.printf(");\n", indent=True)
		self.out.decreaseIndent()
