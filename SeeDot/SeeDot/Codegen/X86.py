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

class X86(CodegenBase):

	def __init__(self, outputDir, decls, scales, intvs, cnsts, expTables, globalVars, internalVars, floatConstants):
		self.outputDir = outputDir
		cppFile = os.path.join(self.outputDir, "seedot_" + getVersion() + ".cpp")
		self.out = Writer(cppFile)

		self.decls = decls
		self.scales = scales
		self.intvs = intvs
		self.cnsts = cnsts
		self.expTables = expTables
		self.globalVars = globalVars
		self.internalVars = internalVars
		self.floatConstants = floatConstants

	def printPrefix(self):
		self.printCincludes()

		self.printExpTables()
		
		self.printVarDecls()

		self.printCHeader()

		self.printConstDecls()
		
		self.out.printf('\n')

	def printCincludes(self):
		self.out.printf('#include <iostream>\n\n', indent=True)
		self.out.printf('#include "datatypes.h"\n', indent=True)
		self.out.printf('#include "predictors.h"\n', indent=True)
		self.out.printf('#include "profile.h"\n', indent=True)
		self.out.printf('#include "library_%s.h"\n' % (getVersion()), indent=True)
		self.out.printf('#include "model_%s.h"\n' % (getVersion()), indent=True)
		self.out.printf('#include "vars_%s.h"\n\n' % (getVersion()), indent=True)
		self.out.printf('using namespace std;\n', indent=True)
		self.out.printf('using namespace seedot_%s;\n' % (getVersion()), indent=True)
		self.out.printf('using namespace vars_%s;\n\n' % (getVersion()), indent=True)

	def printExpTables(self):
		for exp, [table, [tableVarA, tableVarB]] in self.expTables.items():
			self.printExpTable(table[0], tableVarA)
			self.printExpTable(table[1], tableVarB)
			self.out.printf('\n')

	def printExpTable(self, table_row, var):
		self.out.printf('const MYINT %s[%d] = {\n' % (var.idf, len(table_row)), indent = True)
		self.out.increaseIndent()
		self.out.printf('', indent = True)
		for i in range(len(table_row)):
			self.out.printf('%d, ' % table_row[i])
		self.out.decreaseIndent()
		self.out.printf('\n};\n')

	def printCHeader(self):
		if forFloat():
			func = "Float"
			type = "float"
		else:
			func = "Fixed"
			type = "MYINT"
		self.out.printf('int seedot%s(%s **X) {\n' % (func, type), indent=True)
		self.out.increaseIndent()

	def printVarDecls(self):

		varsFilePath = os.path.join(self.outputDir, "vars_" + getVersion() + ".h")
		varsFile = Writer(varsFilePath)

		varsFile.printf("#pragma once\n\n")
		varsFile.printf("#include \"datatypes.h\"\n\n")
		varsFile.printf("namespace vars_%s {\n" % (getVersion()))
		varsFile.increaseIndent()

		for decl in self.decls:
			if decl in self.globalVars:
				continue
			
			if forFloat() and decl not in self.internalVars:
				typ_str = IR.DataType.getFloatStr()
			else:
				typ_str = IR.DataType.getIntStr()
			
			idf_str = decl
			type = self.decls[decl]
			if Type.isInt(type):
				shape_str = ''
			elif Type.isTensor(type):
				shape_str = ''.join(['[' + str(n) + ']' for n in type.shape])

			self.out.printf('%s vars_%s::%s%s;\n', typ_str, getVersion(), idf_str, shape_str, indent=True)
			varsFile.printf('extern %s %s%s;\n', typ_str, idf_str, shape_str, indent=True)

		self.out.printf('\n')
		
		varsFile.decreaseIndent()
		varsFile.printf("}\n")
		varsFile.close()

		self.generateDebugProgram()

	def generateDebugProgram(self):
		debugFilePath = os.path.join(self.outputDir, "debug.cpp")
		debugFile = Writer(debugFilePath)

		debugFile.printf("#include <iostream>\n\n")
		debugFile.printf("#include \"datatypes.h\"\n")
		debugFile.printf("#include \"profile.h\"\n")
		debugFile.printf("#include \"vars_fixed.h\"\n")
		debugFile.printf("#include \"vars_float.h\"\n\n")
		debugFile.printf("using namespace std;\n\n")
		debugFile.printf("void debug() {\n\n")

		if debugMode() and forFixed():
			debugFile.increaseIndent()

			for decl in self.decls:
				if decl in self.globalVars:
					continue

				type = self.decls[decl]
				if decl not in self.scales or not isinstance(type, Type.Tensor) or type.isShapeOne():
					continue

				scale = self.scales[decl]

				s = decl + "[0]" * type.dim
				shape_str = ''.join([str(n) + ', ' for n in type.shape])
				shape_str = shape_str.rstrip(', ')

				debugFile.printf("diff(&vars_float::%s, &vars_fixed::%s, %d, %s);\n\n" % (s, s, scale, shape_str), indent = True)

			debugFile.decreaseIndent()
		
		debugFile.printf("}\n")

		debugFile.close()

	def printSuffix(self, expr:IR.Expr):
		self.out.printf('\n')

		type = self.decls[expr.idf]

		if Type.isInt(type):
			self.out.printf('return ', indent = True)
			self.print(expr)
			self.out.printf(';\n')
		elif Type.isTensor(type):
			idfr = expr.idf
			exponent = self.scales[expr.idf]
			num = 2 ** exponent

			if type.dim == 0:
				self.out.printf('cout << ', indent = True)
				self.out.printf('float(' + idfr + ')*' + str(num))
				self.out.printf(' << endl;\n')
			else:
				iters = []
				for i in range(type.dim):
					s = chr(ord('i') + i)
					tempVar = IR.Var(s)
					iters.append(tempVar)
				expr_1 = IRUtil.addIndex(expr, iters)
				cmds = IRUtil.loop(type.shape, iters, [IR.PrintAsFloat(expr_1, exponent)])
				self.print(IR.Prog(cmds))
		else:
			assert False

		self.out.decreaseIndent()
		self.out.printf('}\n', indent=True)

		self.out.close()
