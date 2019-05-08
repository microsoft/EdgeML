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

class Hls(CodegenBase):

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
		self.printHlsIncludes()

		self.printExpTables()

		self.printHlsHeader()

		self.printVarDecls()

		self.printConstDecls()
		
		self.out.printf('\n')

	def printHlsIncludes(self):
		self.out.printf('#include <iostream>\n', indent=True)
		self.out.printf('#include <stdint.h>\n', indent=True)
		#self.out.printf('#include <ap_int.h> \n ', indent=True)
		#self.out.printf('#include "predict.h"\n', indent=True)
		self.out.printf('#ifdef __SYNTHESIS__\n #include <ap_int.h>\n#endif\n', indent=True)
		#self.out.printf('typedef ap_int<BITWIDTH> MYINT;\n', indent=True)
		self.out.printf('typedef int16_t MYINT;\n', indent=True)
		self.out.printf('#ifndef __SYNTHESIS__\n #include "%s.h"\n #endif \n ',getAlgo(),indent=True)
		#self.out.printf('#include "predict.h"\n', indent=True)
		self.out.printf('#include "model.h"\n\n', indent=True)
		self.out.printf('using namespace %s_fixed;\n\n' % getAlgo(), indent=True)

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

	def printHlsHeader(self):
		self.out.printf('int %sFixed(',getAlgo())
		#self.out.printf('X[D]){\n')
		self.out.printf('\n #ifndef __SYNTHESIS__ \n')
		self.out.printf('MYINT **X')
		#for i in range(0,getNumWorkers()):
		#	if(i==getNumWorkers()-1):
		#		self.out.printf('MYINT X' + str(i) +'[D]') 
		#	else:
		#		self.out.printf('MYINT X' + str(i) +'[D],') 
		self.out.printf('\n #else \n')

		#for i in range(0,getNumWorkers()):
		#	self.out.printf('MYINT ZX_t_ex' + '[10]' + '[d],')
				#for i in range(0,getNumWorkers()):
		self.out.printf('MYINT ZX_t_ex[' + str(getNumWorkers()) + '][d],')
		self.out.printf('ap_uint<1> doneSpMV')
		self.out.printf('\n  #endif \n')

		self.out.printf(') {\n', indent=True)
		self.out.printf('#pragma HLS INTERFACE ap_memory port=ZX_t_ex \n')
		self.out.printf('#pragma HLS ARRAY_PARTITION variable=ZX_t_ex block factor=' + str(getNumWorkers()) + ' dim=1 \n')
		self.out.increaseIndent()
		#self.out.printf('#pragma HLS INTERFACE ap_ctrl_none port=return \n ', indent=True)

		self.out.printf('#pragma HLS ALLOCATION instances=mul limit=75 operation \n ', indent=True)# to account for the constraint of 90 DSPs on device
		#printf('char buff[8];\n\n', indent=True)

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
				#iters = self.getTempIterators(type.dim)
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

		#self.out.decreaseIndent()
		self.out.printf('}\n', indent=True)

		self.out.close()

	def printMemset(self, ir):
		self.out.printf('//memset cant be used in HLS code; using for-loop instead\n', indent = True)
		if(self.dim == 1):
			self.out.printf('for(int i=0;i<%d;i++){ \n #pragma HLS UNROLL \n', ir.len, indent = True)
			self.out.increaseIndent()
			self.out.printf('',indent = True)
			self.print(ir.e)
			self.out.printf('[i][0] = 0;\n')
			self.out.decreaseIndent()
			self.out.printf('}\n \n', indent = True)
		else:#hardcoded for dim==2; change later
			self.out.printf('for(int i=0;i<%d;i++){ \n #pragma HLS UNROLL \n', ir.lens[0], indent = True)
			self.out.increaseIndent()
			self.out.printf('for(int j=0;j<%d;j++){ \n #pragma HLS UNROLL \n', ir.lens[1], indent = True)
			self.out.increaseIndent()
			self.out.printf('',indent = True)
			self.print(ir.e)
			self.out.printf('[i][j] = 0;\n')
			self.out.decreaseIndent()
			self.out.printf('}\n \n', indent = True)
			self.out.decreaseIndent()
			self.out.printf('}\n \n', indent = True)

	def printForHeader(self, ir):
		self.out.printf('for (%s ', IR.DataType.getIntStr(), indent=True)
		self.print(ir.var)
		self.out.printf(' = %d; ', ir.st)
		self.print(ir.cond)
		self.out.printf('; ')
		self.print(ir.var)
		# factor = -1 -- Complete UNROLL
		# factor = 0  -- NO UNROLL
		# factor = fac  -- UNROLL by a factor 'fac'
		if(ir.factor == -1):
			self.out.printf('++) {\n ')
			self.out.printf('#pragma HLS UNROLL\n')
		elif(ir.factor == 0):
			self.out.printf('++) {\n')
		else:
			self.out.printf('++) {\n ')
			self.out.printf('#pragma HLS UNROLL factor=%d\n', ir.factor)

	def printCExpr(self, ir):
		self.out.printf('(')
		self.out.printf('(MYINT)')
		self.print(ir.cond)
		self.out.printf(' ? ')
		self.out.printf('(MYINT)')
		self.print(ir.et)
		self.out.printf(' : ')
		self.out.printf('(MYINT)')
		self.print(ir.ef)
		self.out.printf(')')
