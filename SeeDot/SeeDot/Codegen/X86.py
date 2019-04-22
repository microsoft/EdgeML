# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np

from Codegen.CodegenBase import CodegenBase

import IR.IR as IR
import IR.IRUtil as IRUtil

import Type
from Util import *

class X86(CodegenBase):

	def __init__(self, writer, decls, scales, intvs, cnsts, expTables, globalVars, internalVars, floatConstants):
		self.out = writer
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
		
		self.printCHeader()

		self.printVarDecls()

		self.printConstDecls()
		
		self.out.printf('\n')

	def printCincludes(self):
		self.out.printf('#include <iostream>\n\n', indent=True)
		self.out.printf('#include "datatypes.h"\n', indent=True)
		self.out.printf('#include "predictors.h"\n', indent=True)
		self.out.printf('#include "library_%s.h"\n' % (getVersion()), indent=True)
		self.out.printf('#include "model_%s.h"\n\n' % (getVersion()), indent=True)
		self.out.printf('using namespace std;\n', indent=True)
		self.out.printf('using namespace %s_%s;\n\n' % (getAlgo(), getVersion()), indent=True)

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
