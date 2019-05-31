# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import operator

from Antlr.SeeDotParser import SeeDotParser
import AST.AST as AST
from AST.ASTVisitor import ASTVisitor
import IR.IR as IR
import IR.IRUtil as IRUtil
import Common
import Type
from Util import *

class IRBuilder(ASTVisitor):

	def __init__(self, outputLog):

		self.log = outputLog

		# MAX_SCALE is used at each operation to compute scale parameters
		# It is not used for floating-poing code generation
		if getMaxScale() == None:
			if forFloat():
				print("Setting MAX_SCALE = 0. This value will not affect the generated code.")
				self.MAX_SCALE = 0
			else:
				assert False, "MAX_SCALE not set for fixed-point code generation."
		else:
			self.MAX_SCALE = getMaxScale()

		# Variables used for exp() computation
		self.expTables = {}
		self.expProfileLoaded = False

		# Counter used while creating temp variables
		self.counter_var = 0
		self.counter_iter = 0

		# List of variables declared in the SeeDot program whose definitions will be present in the model.h file
		self.globalVars = []

		# Global variables
		#
		# varDeclarations: Map of local variables to their type used for declaring the variables in the generated C++ code
		# varScales: Map of variables to their scaling factors
		# varIntervals: Map of variables to the range of values stored in the variable, which is obtained from range analysis
		# internalVars: List of intermediate variables in the program whose type is always int irrespective of floating-point or fixed-point code generation
		# intConstants: Map of integer constant variables to their value
		# floatConstants: Map of float constant variables to their value
		self.varDeclarations = {}
		self.varScales = {}
		self.varIntervals = {}
		self.internalVars = []
		self.intConstants = {}
		self.floatConstants = {}

		# Mutable variables declared in the 'loop' operator is stored in mutableVars
		# The range of run-time values see by mutable variables is stored in mutableVarsProfile. This information is obtained from collecting run-time profile on the floating-point code
		self.mutableVars = []
		self.mutableVarsProfile = []

	def visitInt(self, node:AST.Int):
		val = node.value

		prog = IR.Prog([])
		expr = IR.Int(val)

		return (prog, expr)

	def visitFloat(self, node:AST.Float):
		val = node.value
		scale = self.getScale(abs(val))
		intv = self.getInterval(scale, val, val)
		val_int = IR.DataType.getInt(int(np.ldexp(val, -scale)))

		prog = IR.Prog([])
		expr = self.getTempVar()

		self.varDeclarations[expr.idf] = node.type
		self.varScales[expr.idf] = scale
		self.varIntervals[expr.idf] = intv
		self.intConstants[expr.idf] = val_int
		self.floatConstants[expr.idf] = val

		return (prog, expr)

	def visitId(self, node:AST.ID):
		idf = node.name

		prog = IR.Prog([])
		expr = IR.Var(idf, inputVar = True if idf in self.globalVars else False)

		return (prog, expr)

	def visitDecl(self, node:AST.Decl):
		minVal, maxVal = node.range

		assert minVal <= maxVal, "Range of a variable with values (%.6f, %.6f) is not valid" % (minVal, maxVal)
		
		scale = self.getScale(max(abs(minVal), abs(maxVal)))
		intv = self.getInterval(scale, minVal, maxVal)

		prog = IR.Prog([])
		expr = self.getTempVar()
		expr.inputVar = True
		
		self.varScales[expr.idf] = scale
		self.varIntervals[expr.idf] = intv

		return (prog, expr)

	def visitInit(self, node:AST.Init):
		if node.value == 0:
			# getScale() fails for 0. Hence, replacing it with a very low value.
			minVal, maxVal = -0.000001, 0.000001
		else:
			minVal, maxVal = node.value, node.value

		# Have to use loops to initialize non-zero values instead of memset
		assert node.value == 0, "'init' operator currently only supports initialization to 0"

		scale = self.getScale(max(abs(minVal), abs(maxVal)))
		intv = self.getInterval(scale, minVal, maxVal)

		expr = self.getTempVar()

		comment = IR.Comment('init([%s], %.6f)' % (', '.join(map(str, node.shape)), node.value))

		memset = IR.Memset(expr, node.type.size())

		prog_init = IR.Prog([comment, memset])

		prog_out = prog_init
		expr_out = expr

		self.varDeclarations[expr_out.idf] = node.type
		self.varScales[expr_out.idf] = scale
		self.varIntervals[expr_out.idf] = intv

		self.log.print(comment.msg)
		self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

		return (prog_out, expr)

	# out = in ^ T
	def visitTransp(self, node:AST.Transp):

		(prog_in, expr_in) = self.visit(node.expr)
		
		expr_out = self.getTempVar()

		type_out = node.type
		[I, J] = type_out.shape

		scale_out = self.varScales[expr_in.idf]
		intv_out = self.varIntervals[expr_in.idf]

		expr_in.inputVar = False
		expr_out.inputVar = False

		comment = IR.Comment(expr_in.idf + "^T")

		funcCall = IR.FuncCall("Transpose", {
								expr_in: "A",
								expr_out: "B",
								IR.Int(I): "I",
								IR.Int(J): "J"
								})

		prog_transp = IR.Prog([comment, funcCall])

		prog_out = IRUtil.concatPrograms(prog_in, prog_transp)
		
		self.varDeclarations[expr_out.idf] = type_out
		self.varScales[expr_out.idf] = scale_out
		self.varIntervals[expr_out.idf] = intv_out

		return (prog_out, expr_out)

	# out = reshape(in, shape, order)
	def visitReshape(self, node:AST.Reshape):

		(prog_in, expr_in) = self.visit(node.expr)

		'''
		reshape(A, (T1, T2), (N, H, W))

		cmd1:  t1 = t2 = 0;
		loop: for n in 0:N:
		         for h in 0:H:
		           for w in 0:W:
		cmd3:        B[t1][t2] = A[n][h][w]
		cmd5:        t2++;
			         if (t2 == T2)
		               t2 = 0;
		cmd5_:         t1++;
		'''

		type_in = node.expr.type
		type_out = node.type

		# Compute scaling factors
		scale_out = self.varScales[expr_in.idf]
		intv_out = self.varIntervals[expr_in.idf]

		# Declare variables
		expr_out = self.getTempVar()
		iters_in = self.getTempIterators(type_in.dim)
		iters_out = self.getTempVars(type_out.dim)

		# Initialize to 0
		cmd1 = [IR.Assn(var, IRUtil.zero) for var in iters_out]

		# Incrementing the first index
		first_iter = iters_out[0]
		cmd5_ = IRUtil.incCmd(first_iter)

		# Incrementing other indices using a loop
		cmd5 = [cmd5_]
		for i in range(1, type_out.dim):
			curr_iter = iters_out[i]
			curr_size = IR.Int(type_out.shape[i])
			cmd5 = [IRUtil.incCmd(curr_iter), IR.If(IRUtil.eq(curr_iter, curr_size), [IRUtil.initVarToZero(curr_iter)] + cmd5)]
		
		# Outer loop
		# The iterators are selected based on the selection order specified by the user
		loopShape = []
		loopIters = []
		for order in node.order:
			order = order - 1
			loopShape.append(type_in.shape[order])
			loopIters.append(iters_in[order])

		loop = IRUtil.loop(loopShape, loopIters, [IR.Assn(IRUtil.addIndex(expr_out, iters_out), IRUtil.addIndex(expr_in, iters_in))] + cmd5)

		# Finalize
		comment = IR.Comment("reshape(" + expr_in.idf + ", (" + ', '.join(str(e) for e in type_out.shape) + "), (" + ', '.join(str(e) for e in node.order))
		prog_reshape = IR.Prog([comment] + cmd1 + loop)
		
		prog_out = IRUtil.concatPrograms(prog_in, prog_reshape)

		# Update context
		self.varDeclarations[expr_out.idf] = type_out
		self.varScales[expr_out.idf] = scale_out
		self.varIntervals[expr_out.idf] = intv_out
		
		# Update declarations
		for var in iters_out:
			self.varDeclarations[var.idf] = Type.Int()
			self.internalVars.append(var.idf)

		return (prog_out, expr_out)
	
	# out = maxpool(in, stride)
	def visitMaxpool(self, node:AST.Maxpool):

		(prog_in, expr_in) = self.visit(node.expr)

		type_out = node.type
		stride = node.dim

		# Compute scaling factor
		scale_out = self.varScales[expr_in.idf]
		intv_out = self.varIntervals[expr_in.idf]

		# Declare variables
		expr_out = self.getTempVar()

		[N, H, W, C] = node.expr.type.shape

		expr_in.inputVar = False
		expr_out.inputVar = False

		comment = IR.Comment("maxpool(" + expr_in.idf + ", " + str(stride) + ")")

		funcCall = IR.FuncCall("Maxpool", {
								expr_in: "A",
								expr_out: "B",
								IR.Int(N): "N",
								IR.Int(H): "H",
								IR.Int(W): "W",
								IR.Int(C): "C",
								IR.Int(stride): "stride"
								})

		prog_maxpool = IR.Prog([comment, funcCall])
		
		prog_out = IRUtil.concatPrograms(prog_in, prog_maxpool)

		# Update declarations
		self.varDeclarations[expr_out.idf] = type_out
		self.varScales[expr_out.idf] = scale_out
		self.varIntervals[expr_out.idf] = intv_out

		return (prog_out, expr_out)
	
	# out = in[index]
	def visitIndex(self, node:AST.Index):

		(prog_in, expr_in) = self.visit(node.expr)
		(prog_idx, expr_idx) = self.visit(node.index)

		prog_out = IRUtil.concatPrograms(prog_in, prog_idx)
		expr_out = IRUtil.addIndex(expr_in, [expr_idx])

		return (prog_out, expr_out)
	
	# func(in_A, in_B, in_C, ... , in_n, out)
	# out.type = in_A.type = ... = in_n.type
	# out.scale = in_A.scale
	def visitFuncCall(self, node:AST.FuncCall):
		# The type of each argument is same and is equal to the type of the output
		# The compiler assumes that the output of the uninterpreted function call is the last argument to the function
		# Also assumes that the scale of the output is equal to the scale of the first argument

		progs = []
		exprs = []
		for expr in node.exprList:
			(prog_in, expr_in) = self.visit(expr)
			progs.append(prog_in)
			exprs.append(expr_in)

		prog_out = IR.Prog([])
		for prog_funcCall in progs:
			prog_out = IRUtil.concatPrograms(prog_out, prog_funcCall)

		expr_out = self.getTempVar()

		# Scale of the output is the scale of the first argument
		scale_out = self.varScales[exprs[0].idf]
		intv_out = self.varIntervals[exprs[0].idf]

		args = dict()
		ch = 'A'
		for expr in exprs:
			args[expr] = ch
			ch = chr(ord(ch) + 1)
		args[expr_out] = expr_out.idf

		ch = 'I'
		for i in node.type.shape:
			args[IR.Int(i)] = ch
			ch = chr(ord(ch) + 1)

		comment = IR.Comment(node.name + '(' + ', '.join(expr.idf for expr in exprs) + ')')

		funcCall = IR.FuncCall(node.name, args)

		prog_funcCall = IR.Prog([comment, funcCall])

		prog_out = IRUtil.concatPrograms(prog_out, prog_funcCall)

		self.varDeclarations[expr_out.idf] = node.type
		self.varScales[expr_out.idf] = scale_out
		self.varIntervals[expr_out.idf] = intv_out

		return (prog_out, expr_out)

	# out = +- in
	def visitUop(self, node:AST.Uop):

		(prog_in, expr_in) = self.visit(node.expr)
		
		if node.op == SeeDotParser.ADD:
			return (prog_in, expr_in)

		assert node.op == SeeDotParser.SUB
		
		type_out = node.type
		
		# e : Int
		if Type.isInt(type_out):
			prog_out = prog_in
			expr_out = IRUtil.negate(expr_in)

			# Just to be safe, check that the scaling factor of the integer variable is never tracked
			assert expr_in.idf not in self.varScales and expr_in.idf not in self.varIntervals

		# e: Tensor(), or Tensor(..)
		else:
			expr_out = self.getTempVar()
			iters = self.getTempIterators(type_out.dim)

			scale_out = self.varScales[expr_in.idf]
			(m, M) = self.varIntervals[expr_in.idf]
			intv_out = (-M, -m)

			lhs = IRUtil.addIndex(expr_out, iters)
			rhs = IRUtil.negate(IRUtil.addIndex(expr_in, iters))
			loop = IRUtil.loop(type_out.shape, iters, [IR.Assn(lhs, rhs)])
			prog_uop = IR.Prog(loop)

			prog_out = IRUtil.concatPrograms(prog_in, prog_uop)
			
			self.varDeclarations[expr_out.idf] = type_out
			self.varScales[expr_out.idf] = scale_out
			self.varIntervals[expr_out.idf] = intv_out

		return (prog_out, expr_out)

	# out = in_A op in_B
	def visitBop1(self, node:AST.Bop1):
		if    node.op == SeeDotParser.MUL:        return self.visitBopMul(node)
		elif  node.op == SeeDotParser.SPARSEMUL:  return self.visitBopSparseMul(node)
		elif  node.op == SeeDotParser.MULCIR:     return self.visitBopMulCir(node)
		elif  node.op == SeeDotParser.CONV:       return self.visitBopConv(node)
		elif  node.op == SeeDotParser.ADDCIR:     return self.visitBopAddOrSubCir(node)
		elif  node.op == SeeDotParser.SUBCIR:     return self.visitBopAddOrSubCir(node)
		else:                                     assert False

	# out = in_A * in_B
	def visitBopMul(self, node:AST.Bop1):
		type_in_A = node.expr1.type
		type_in_B = node.expr2.type
		type_out = node.type

		if    Type.isInt(type_out):  return self.visitBopMulInt(node)
		elif  type_in_A.dim == 0:    return self.visitBopMul1DTensor(node)
		elif  type_in_B.dim == 0:    return self.visitBopMul1DTensor(node)
		else:                        return self.visitBopMul2DTensor(node)

	# out = in_A * in_B
	def visitBopMulInt(self, node:AST.Bop1):
		
		(prog_in_A, expr_in_A) = self.visit(node.expr1)

		(prog_in_B, expr_in_B) = self.visit(node.expr2)

		prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B)
		expr_out = IRUtil.mul(expr_in_A, expr_in_B)

		# Just to be safe, check that the scaling factor of the integer variables is never tracked
		if isinstance(expr_in_A, IR.Var):
			assert expr_in_A.idf not in self.varScales and expr_in_A.idf not in self.varIntervals
		if isinstance(expr_in_B, IR.Var):
			assert expr_in_B.idf not in self.varScales and expr_in_B.idf not in self.varIntervals

		return (prog_out, expr_out)

	# out = in_A * in_B
	def visitBopMul1DTensor(self, node:AST.Bop1):

		(prog_in_A, expr_in_A) = self.visit(node.expr1)

		(prog_in_B, expr_in_B) = self.visit(node.expr2)

		type_in_A, type_in_B = node.expr1.type, node.expr2.type
		type_out = node.type

		expr_out = self.getTempVar()

		scale_in_A, scale_in_B = self.varScales[expr_in_A.idf], self.varScales[expr_in_B.idf]
		intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]

		[shr_A, shr_B] = self.getShrForMul(scale_in_A, scale_in_B)
		
		scale_out = self.getScaleForMul(scale_in_A, shr_A, scale_in_B, shr_B)
		intv_out = self.getIntvervalForMul(intv_in_A, shr_A, intv_in_B, shr_B)

		if type_in_A.dim == 0:
			a, b = expr_in_A, expr_in_B
			[I, J] = type_in_B.shape
			shr_a, shr_b = shr_A, shr_B
		else:
			a, b = expr_in_B, expr_in_A
			[I, J] = type_in_A.shape
			shr_a, shr_b = shr_B, shr_A

		shr_a = self.formatShr(shr_a)
		shr_b = self.formatShr(shr_b)

		a.inputVar = False
		b.inputVar = False
		expr_out.inputVar = False

		comment = IR.Comment(expr_in_A.idf + ' * ' + expr_in_B.idf)

		funcCall = IR.FuncCall("ScalarMul", {
								a: "A",
								b: "B",
								expr_out: "C",
								IR.Int(I): "I",
								IR.Int(J): "J",
								shr_a: "shr1",
								shr_b: "shr2"
								})

		prog_mul = IR.Prog([comment, funcCall])

		prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_mul)

		self.varDeclarations[expr_out.idf] = type_out
		self.varScales[expr_out.idf] = scale_out
		self.varIntervals[expr_out.idf] = intv_out

		self.log.print(comment.msg)
		self.log.print("\tInput1: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
		self.log.print("\tInput2: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
		self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

		return (prog_out, expr_out)

	# out = in_A * in_B
	def visitBopMul2DTensor(self, node:AST.Bop1):
		
		(prog_in_A, expr_in_A) = self.visit(node.expr1)

		(prog_in_B, expr_in_B) = self.visit(node.expr2)

		expr_treeSum = self.getTempVar()
		expr_out = self.getTempVar()

		# Compute scales
		scale_in_A, scale_in_B = self.varScales[expr_in_A.idf], self.varScales[expr_in_B.idf]
		intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]

		[shr_A, shr_B] = self.getShrForMul(scale_in_A, scale_in_B)

		type_in_A, type_in_B = node.expr1.type, node.expr2.type
		type_out = node.type

		[I, J] = type_in_A.shape
		[J, K] = type_in_B.shape
		type_treeSum = Type.Tensor([J])

		scale_treeSum = self.getScaleForMul(scale_in_A, shr_A, scale_in_B, shr_B)
		intv_treeSum = self.getIntvervalForMul(intv_in_A, shr_A, intv_in_B, shr_B)

		(scale_out, height_shr, height_noshr) = self.getScaleForTreeSum(scale_treeSum, J)
		intv_out = self.getIntervalForTreeSum(intv_treeSum, J, height_shr, height_noshr)

		shr_A = self.formatShr(shr_A)
		shr_B = self.formatShr(shr_B)

		c = ''
		if expr_in_A.idf in self.globalVars:
			c += 'C'
		else:
			c += 'N'
		if expr_in_B.idf in self.globalVars:
			c += 'C'
		else:
			c += 'N'

		expr_in_A.inputVar = False
		expr_in_B.inputVar = False
		expr_out.inputVar = False
		expr_treeSum.inputVar = False

		comment = IR.Comment(expr_in_A.idf + ' * ' + expr_in_B.idf)

		funcCall = IR.FuncCall("MatMul" + c, {
								expr_in_A: "A",
								expr_in_B: "B",
								expr_out: "C",
								expr_treeSum: "T",
								IR.Int(I): "I",
								IR.Int(J): "J",
								IR.Int(K): "K",
								shr_A: "shr1",
								shr_B: "shr2",
								IR.Int(height_shr): "H1",
								IR.Int(height_noshr): "H2"
								})

		prog_mul = IR.Prog([comment, funcCall])
		
		prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_mul)
		
		self.varDeclarations[expr_out.idf] = type_out
		self.varScales[expr_out.idf] = scale_out
		self.varIntervals[expr_out.idf] = intv_out
		
		self.varDeclarations[expr_treeSum.idf] = type_treeSum

		self.log.print(comment.msg)
		self.log.print("\tInput1: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
		self.log.print("\tInput2: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
		self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

		return (prog_out, expr_out)

	# out = in_A |*| in_B
	def visitBopSparseMul(self, node:AST.Bop1):

		(prog_in_A, expr_in_A) = self.visit(node.expr1)

		(prog_in_B, expr_in_B) = self.visit(node.expr2)

		[P, Q] = node.expr1.type.shape
		[Q, R] = node.expr2.type.shape
		
		assert R == 1, "Sparse matrix multiplication currently only support multiplication with a vector"

		expr_out = self.getTempVar()
		type_out = node.type

		scale_in_A, scale_in_B = self.varScales[expr_in_A.idf], self.varScales[expr_in_B.idf]
		intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]

		[shr_A, shr_B] = self.getShrForMul(scale_in_A, scale_in_B)

		scale_treeSum = self.getScaleForMul(scale_in_A, shr_A, scale_in_B, shr_B)
		intv_treeSum = self.getIntvervalForMul(intv_in_A, shr_A, intv_in_B, shr_B)

		(scale_out, height_shr, height_noshr) = self.getScaleForTreeSum(scale_treeSum, Q)
		intv_out = self.getIntervalForTreeSum(intv_treeSum, Q, height_shr, height_noshr)

		in_A_idx = IR.Var(expr_in_A.idf[0] + 'idx', expr_in_A.idx, inputVar = True)
		in_A_val = IR.Var(expr_in_A.idf[0] + 'val', expr_in_A.idx, inputVar = True)

		shr_A = self.formatShr(shr_A)
		shr_B = self.formatShr(shr_B)
		height_shr = self.formatShr(height_shr)

		in_A_idx.inputVar = False
		in_A_val.inputVar = False
		expr_in_B.inputVar = False
		expr_out.inputVar = False

		comment = IR.Comment(expr_in_A.idf + ' |*| ' + expr_in_B.idf)
		
		cmd1 = IR.Memset(expr_out, type_out.size())

		funcCall = IR.FuncCall("SparseMatMul", {
								in_A_idx: "Aidx",
								in_A_val: "Aval",
								expr_in_B: "B",
								expr_out: "C",
								IR.Int(Q): "K",
								shr_A: "shrA",
								shr_B: "shrB",
								height_shr: "shrC"
								})

		prog_mul = IR.Prog([comment, cmd1, funcCall])

		prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_mul)

		self.varDeclarations[expr_out.idf] = type_out
		self.varScales[expr_out.idf] = scale_out
		self.varIntervals[expr_out.idf] = intv_out

		# TODO: Length of Aidx and Aval hard coded to 100
		# This is safe as it will be ignored in the generated code
		self.varDeclarations.update({in_A_idx.idf : Type.Tensor([100]),
									in_A_val.idf : Type.Tensor([100]),
									})
		self.globalVars.append(in_A_idx.idf)
		self.globalVars.append(in_A_val.idf)

		self.log.print(comment.msg)
		self.log.print("\tInput1: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
		self.log.print("\tInput2: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
		self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

		return (prog_out, expr_out)

	# out = in_A <*> in_B
	def visitBopMulCir(self, node:AST.Bop1):

		(prog_in_A, expr_in_A) = self.visit(node.expr1)

		(prog_in_B, expr_in_B) = self.visit(node.expr2)

		type_in_A, type_in_B = node.expr1.type, node.expr2.type
		type_out = node.type

		expr_out = self.getTempVar()

		assert type_out.dim == 2

		[I, J] = type_out.shape

		scale_in_A, scale_in_B = self.varScales[expr_in_A.idf], self.varScales[expr_in_B.idf]
		intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]

		[shr_A, shr_B] = self.getShrForMul(scale_in_A, scale_in_B)
		
		scale_out = self.getScaleForMul(scale_in_A, shr_A, scale_in_B, shr_B)
		intv_out = self.getIntvervalForMul(intv_in_A, shr_A, intv_in_B, shr_B)

		shr_A = self.formatShr(shr_A)
		shr_B = self.formatShr(shr_B)

		expr_in_A.inputVar = False
		expr_in_B.inputVar = False
		expr_out.inputVar = False

		comment = IR.Comment(expr_in_A.idf + ' <*> ' + expr_in_B.idf)

		funcCall = IR.FuncCall("MulCir", {
								expr_in_A: "A",
								expr_in_B: "B",
								expr_out: "C",
								IR.Int(I): "I",
								IR.Int(J): "J",
								shr_A: "shrA",
								shr_B: "shrB"
								})

		prog_mul = IR.Prog([comment, funcCall])

		prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_mul)
		
		self.varDeclarations[expr_out.idf] = type_out
		self.varScales[expr_out.idf] = scale_out
		self.varIntervals[expr_out.idf] = intv_out

		self.log.print(comment.msg)
		self.log.print("\tInput1: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
		self.log.print("\tInput2: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
		self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

		return (prog_out, expr_out)

	# out = in_A # in_B
	def visitBopConv(self, node:AST.Bop1):

		(prog_in_A, expr_in_A) = self.visit(node.expr1)

		(prog_in_B, expr_in_B) = self.visit(node.expr2)
		
		[N , H , W , CI] = node.expr1.type.shape
		[HF, WF, CI, CO] = node.expr2.type.shape

		type_treeSum = Type.Tensor([HF * WF * CI])
		type_out = node.type

		# Declare variables
		[expr_treeSum, expr_out] = self.getTempVars(2)

		# Compute scale reductions and new scaling factors
		scale_in_A, scale_in_B = self.varScales[expr_in_A.idf], self.varScales[expr_in_B.idf]
		intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]

		[shr_A, shr_B] = self.getShrForMul(scale_in_A, scale_in_B)

		scale_treeSum = self.getScaleForMul(scale_in_A, shr_A, scale_in_B, shr_B)
		intv_treeSum = self.getIntvervalForMul(intv_in_A, shr_A, intv_in_B, shr_B)

		(scale_out, height_shr, height_noshr) = self.getScaleForTreeSum(scale_treeSum, HF * WF * CI)
		intv_out = self.getIntervalForTreeSum(intv_treeSum, HF * WF * CI, height_shr, height_noshr)

		shr_A = self.formatShr(shr_A)
		shr_B = self.formatShr(shr_B)

		expr_in_A.inputVar = False
		expr_in_B.inputVar = False
		expr_out.inputVar = False

		comment = IR.Comment(expr_in_A.idf + ' # ' + expr_in_B.idf)

		funcCall = IR.FuncCall("Conv", {
								expr_in_A: "A",
								expr_in_B: "B",
								expr_out: "C",
								expr_treeSum: "tmp",
								IR.Int(N): "N",
								IR.Int(H): "H",
								IR.Int(W): "W",
								IR.Int(CI): "CI",
								IR.Int(HF): "HF",
								IR.Int(WF): "WF",
								IR.Int(CO): "CO",
								shr_A: "shrA",
								shr_B: "shrB",
								IR.Int(height_shr): "H1",
								IR.Int(height_noshr): "H2"
								})

		prog_conv = IR.Prog([comment, funcCall])
		
		prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_conv)
		
		# Update context for output variable
		self.varDeclarations[expr_out.idf] = type_out
		self.varScales[expr_out.idf] = scale_out
		self.varIntervals[expr_out.idf] = intv_out
		
		# Update declarations
		self.varDeclarations[expr_treeSum.idf] = type_treeSum

		self.log.print(comment.msg)
		self.log.print("\tInput1: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
		self.log.print("\tInput2: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
		self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

		return (prog_out, expr_out)

	# out = in_A <+-> in_B
	def visitBopAddOrSubCir(self, node:AST.Bop1):

		(prog_in_A, expr_in_A) = self.visit(node.expr1)

		(prog_in_B, expr_in_B) = self.visit(node.expr2)

		type_in_A, type_in_B = node.expr1.type, node.expr2.type
		type_out = node.type

		if node.op == SeeDotParser.ADDCIR:
			(op_ir, op_fn) = (IR.Op.Op['+'], operator.add)
			add = True
		elif node.op == SeeDotParser.SUBCIR:
			(op_ir, op_fn) = (IR.Op.Op['-'], operator.sub)
			add = False
		
		assert op_fn == operator.add, "Compiler currently does not support convolution-like subtraction."

		scale_in_A, scale_in_B = self.varScales[expr_in_A.idf], self.varScales[expr_in_B.idf]
		intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]

		(scale_out, intv_out, [shr_A, shr_B, shr_out]) = self.getScaleAndIntervalForAddAndSub(scale_in_A, scale_in_B, intv_in_A, intv_in_B, op_fn)

		shr_A = self.formatShr(shr_A)
		shr_B = self.formatShr(shr_B)
		shr_out = self.formatShr(shr_out)

		expr_in_A.inputVar = False
		expr_in_B.inputVar = False

		comment = IR.Comment(expr_in_A.idf + " <" + op_ir.name + "> " + expr_in_B.idf)

		if node.type.dim == 4:
			[N, H, W, C] = node.type.shape
			funcCall = IR.FuncCall("AddOrSubCir4D", {
							expr_in_A: "A",
							expr_in_B: "B",
							IR.Int(N): "N",
							IR.Int(H): "H",
							IR.Int(W): "W",
							IR.Int(C): "C",
							shr_A: "shrA",
							shr_B: "shrB",
							shr_out: "shrC",
							IR.Bool(add): "add"
							})
		elif node.type.dim == 2:
			[H, W] = node.type.shape
			funcCall = IR.FuncCall("AddOrSubCir2D", {
							expr_in_A: "A",
							expr_in_B: "B",
							IR.Int(H): "H",
							IR.Int(W): "W",
							shr_A: "shrA",
							shr_B: "shrB",
							shr_out: "shrC",
							IR.Bool(add): "add"
							})
		else:
			assert False, "AddCir only supports 2D and 4D tensors."

		prog_cir = IR.Prog([comment, funcCall])
			
		prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_cir)
		
		self.varScales[expr_in_A.idf] = scale_out
		self.varIntervals[expr_in_A.idf] = intv_out

		self.log.print(comment.msg)
		self.log.print("\tInput1: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
		self.log.print("\tInput2: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
		self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

		return (prog_out, expr_in_A)

	# out = in_A 'op' in_B
	def visitBop2(self, node:AST.Bop2):

		(prog_in_A, expr_in_A) = self.visit(node.expr1)

		(prog_in_B, expr_in_B) = self.visit(node.expr2)

		type_out = node.type

		if node.op == SeeDotParser.ADD:
			(op_ir, op_fn) = (IR.Op.Op['+'], operator.add)
			funcName = "MatAdd"
		elif node.op == SeeDotParser.SUB:
			(op_ir, op_fn) = (IR.Op.Op['-'], operator.sub)
			funcName = "MatSub"

		# e : Int
		if Type.isInt(type_out):
			prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B)
			expr_out = IR.IntBop(expr_in_A, op_ir, expr_in_B)

			# Just to be safe that the scaling factor of the integer variable is never tracked
			if isinstance(expr_in_A, IR.Var):
				assert expr_in_A.idf not in self.varScales and expr_in_A.idf not in self.varIntervals
			if isinstance(expr_in_B, IR.Var):
				assert expr_in_B.idf not in self.varScales and expr_in_B.idf not in self.varIntervals

		# e : Tensor(), or Tensor(..)
		else:

			assert type_out.dim == 2, "Addition/subtraction of tensors is currently only supported for 2D tensors"

			c = ''
			if op_fn == operator.add:
				if expr_in_A.idf in self.globalVars:
					c += 'C'
				else:
					c += 'N'
				if expr_in_B.idf in self.globalVars:
					c += 'C'
				else:
					c += 'N'

			type_A = node.expr1.type
			type_B = node.expr2.type

			if type_A.dim == 0:
				funcName += 'BroadCastA'
				c = ''
			elif type_B.dim == 0:
				funcName += 'BroadCastB'
				c = ''

			expr_out = self.getTempVar()
			
			scale_in_A, scale_in_B = self.varScales[expr_in_A.idf], self.varScales[expr_in_B.idf]
			intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]

			(scale_out, intv_out, [shr_A, shr_B, shr_out]) = self.getScaleAndIntervalForAddAndSub(scale_in_A, scale_in_B, intv_in_A, intv_in_B, op_fn)

			[I, J] = type_out.shape

			shr_A = self.formatShr(shr_A)
			shr_B = self.formatShr(shr_B)
			shr_out = self.formatShr(shr_out)

			expr_in_A.inputVar = False
			expr_in_B.inputVar = False
			expr_out.inputVar = False

			comment = IR.Comment(expr_in_A.idf + ' ' + op_ir.name + ' ' + expr_in_B.idf)

			funcCall = IR.FuncCall(funcName + c, {
									expr_in_A: "A",
									expr_in_B: "B",
									expr_out: "C",
									IR.Int(I): "I",
									IR.Int(J): "J",
									shr_A: "shrA",
									shr_B: "shrB",
									shr_out: "shrC"
									})

			prog_bop = IR.Prog([comment, funcCall])

			prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_bop)
			
			self.varDeclarations[expr_out.idf] = type_out
			self.varScales[expr_out.idf] = scale_out
			self.varIntervals[expr_out.idf] = intv_out

			self.log.print(comment.msg)
			self.log.print("\tInput1: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
			self.log.print("\tInput2: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
			self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

		return (prog_out, expr_out)

	# out = func(in)
	def visitFunc(self, node:AST.Func):
		if    node.op == SeeDotParser.RELU:     return self.visitRelu(node)
		elif  node.op == SeeDotParser.EXP:      return self.visitExp(node)
		elif  node.op == SeeDotParser.ARGMAX:   return self.visitArgMax(node)
		elif  node.op == SeeDotParser.SGN:      return self.visitSgn(node)
		elif  node.op == SeeDotParser.TANH:     return self.visitTanh(node)
		elif  node.op == SeeDotParser.SIGMOID:  return self.visitSigmoid(node)
		else:                                   assert False

	# out = relu(in)
	def visitRelu(self, node:AST.Func):

		(prog_in, expr_in) = self.visit(node.expr)

		type_out = node.expr.type
		
		(m, M) = self.varIntervals[expr_in.idf]
		if m < 0: m = 0
		if M < 0: M = 0
		intv_out = (m, M)

		# Temp computation for POC. Remove later.
		scale_in = self.varScales[expr_in.idf]
		max_val = max(abs(m), abs(M))
		max_val_f = np.ldexp(max_val, scale_in)
		scale_new = self.getScale(max_val_f)
		print("Scale changes in relu operations: old = %d, new = %d, diff = %d" % (scale_in, scale_new, abs(scale_in - scale_new)))

		expr_in.inputVar = False

		comment = IR.Comment("relu(" + expr_in.idf + ")")

		if node.type.dim == 4:
			[N, H, W, C] = node.type.shape
			funcCall = IR.FuncCall("Relu4D", {
									expr_in: "A",
									IR.Int(N): "N",
									IR.Int(H): "H",
									IR.Int(W): "W",
									IR.Int(C): "C"
									})
		elif node.type.dim == 2:
			[H, W] = node.type.shape
			funcCall = IR.FuncCall("Relu2D", {
									expr_in: "A",
									IR.Int(H): "H",
									IR.Int(W): "W"
									})
		else:
			assert False, "Relu operator currently only supports 2D and 4D tensors."
		
		prog_relu = IR.Prog([comment, funcCall])
		
		prog_out = IRUtil.concatPrograms(prog_in, prog_relu)
		
		self.varIntervals[expr_in.idf] = intv_out

		return (prog_out, expr_in)

	# out = exp(in)
	def visitExp(self, node:AST.Func):
		
		if forFloat() or useMathExp():
			return self.visitMathExp(node)
		elif useTableExp():
			self.readExpProfileFile()
			return self.visitTableExp(node)
		else:
			assert False

	# Note: We assume e<=0 for exp(e)
	def visitMathExp(self, node:AST.Func):

		# Tunable parameter
		MIN = 0.1

		(prog_in, expr_in) = self.visit(node.expr)

		type_in = node.expr.type
		
		scale_in = self.varScales[expr_in.idf]
		intv_in = self.varIntervals[expr_in.idf]

		'''
		1.  y = ((int) (exp(((float)e) / shr1) * shr2))
		'''
		
		maxExp = np.exp(-MIN)

		expr_out = self.getTempVar()
		
		scale_out = self.getScale(maxExp)
		intv_out = self.getInterval(scale_out, maxExp, maxExp)

		[I, J] = type_in.shape

		shr1 = 2 ** -scale_in
		shr2 = 2 ** -scale_out

		shr1 = self.formatShr(shr1)
		shr2 = self.formatShr(shr2)

		cmd0 = IR.Comment('exp(' + expr_in.idf + ')')

		funcCall = IR.FuncCall("Exp", {
								expr_in: "A",
								IR.Int(I): "I",
								IR.Int(J): "J",
								shr1: "shrA",
								shr2: "shrB",
								expr_out: "B"
								})

		prog_exp = IR.Prog([cmd0, funcCall])

		prog_out = IRUtil.concatPrograms(prog_in, prog_exp)

		self.varDeclarations[expr_out.idf] = type_in
		self.varScales[expr_out.idf] = scale_out
		self.varIntervals[expr_out.idf] = intv_out

		return (prog_out, expr_out)

	# Note: We assume e<=0 for exp(e)
	def visitTableExp(self, node:AST.Func):

		(prog_in, expr_in) = self.visit(node.expr)

		# TODO: use MAX_VAL_EXP
		type_in = node.expr.type
		
		scale_in = self.varScales[expr_in.idf]
		intv_in = self.varIntervals[expr_in.idf]

		[m, M] = self.expRange
		[m_scale, M_scale] = [int(np.ldexp(m, -scale_in)), int(np.ldexp(M, -scale_in))]

		max = int(np.ldexp(M - m, -scale_in))
		shl = self.getShl(max)
		
		input = self.getTempVar()
		[i, j] = self.getTempVars(2)
		expr_out = self.getTempVar()

		'''
		1.  if ((-x) < min) {
		2.  	i = 0;
		3.  	j = 0;
		4.  }
		5.  else {
		6.  	y = ((-x) - min) << shl
		7.  	i = (y >> shrI) & (2^b-1)
		8.  	j = (y >> shrJ) & (2^b-1)
		9.  }
		10. ans = T[i] * U[j]
		'''
		
		mask = IR.Int(2 ** self.expB - 1)
		shrI = Common.wordLength - self.expB
		shrJ = Common.wordLength - self.expB * 2
		table = self.getExpTable(scale_in)

		scale1 = self.getScale(1)
		scale2 = self.getScale(abs(np.exp(-m)))

		[shr1, shr2] = self.getShrForMul(scale1, scale2)

		expr_1_elt = IRUtil.addIndex(expr_in, [IRUtil.zero] * type_in.dim)
		expr_2_elt = IRUtil.addIndex(expr_out, [IRUtil.zero] * type_in.dim)

		cond = IRUtil.lt(IRUtil.negate(expr_1_elt), IR.Int(m_scale))
		
		cmd2 = IR.Assn(i, IR.Int(0))
		cmd3 = IR.Assn(j, IR.Int(0))

		cmd6 = IR.Assn(input, IRUtil.shl(IRUtil.sub(IRUtil.negate(expr_1_elt), IR.Int(m_scale)), shl))
		cmd7 = IR.Assn(i, IRUtil.bitAnd(IRUtil.shrUint(input, shrI), mask))
		cmd8 = IR.Assn(j, IRUtil.bitAnd(IRUtil.shrUint(input, shrJ), mask))
		
		cmd1 = IR.If(cond, [cmd2, cmd3], [cmd6, cmd7, cmd8])
		cmd10 = IR.Assn(expr_2_elt, IRUtil.mul(IRUtil.shrUint(IRUtil.addIndex(table[0], [i]), shr1), IRUtil.shrUint(IRUtil.addIndex(table[1], [j]), shr2)))

		scale_out = self.getScaleForExp(scale1, shr1, scale2, shr2)
		intv_out = self.getIntervalForExp(scale_out, [-m_scale, -M_scale])
		
		cmd0 = IR.Comment('exp(' + expr_in.idf + ')')

		prog_exp = IR.Prog([cmd0, cmd1, cmd10])

		prog_out = IRUtil.concatPrograms(prog_in, prog_exp)
		
		self.varDeclarations[expr_out.idf] = type_in
		self.varScales[expr_out.idf] = scale_out
		self.varIntervals[expr_out.idf] = intv_out

		self.varDeclarations.update(dict((var.idf, Type.Int()) for var in [input, i, j]))

		return (prog_out, expr_out)

	def getShl(self, n:int):
		assert n != 0

		shl = 0
		while(n != 0):
			n = n >> 1
			shl += 1
		return min(Common.wordLength - shl, Common.wordLength - self.expB * 2)

	def getExpTable(self, p):
		table = self.expTables.get(p)
		if table == None:
			table = self.populateExpTable(p)
			self.expTables[p] = table

		return table[1]

	def populateExpTable(self, p):
		[table_m, table_n] = self.expTableShape
		b = np.log2(table_n)
		
		# Currently looking at only 2D arrays
		assert table_m == 2

		[m, M] = self.expRange
		max = int(np.ldexp(M - m, -p))
		shl = self.getShl(max)
		
		#alpha_count = self.getAlphaCount(max, shl)
		alpha_count = table_n
		beta_count = table_n

		table = [[0 for _ in range(alpha_count)], [0 for _ in range(beta_count)]]

		alpha = Common.wordLength - shl - b
		pRes = self.getScale(1)
		for i in range(alpha_count):
			num = i * 2 ** (alpha + p)
			exp = np.exp(-num)
			table[0][i] = int(np.ldexp(exp, -pRes))

		beta = alpha - b
		pRes = self.getScale(abs(np.exp(-m)))
		for i in range(beta_count):
			num = m + i * 2 ** (beta + p)
			exp = np.exp(-num)
			table[1][i] = int(np.ldexp(exp, -pRes))

		tableVar = [IR.Var('EXP' + str(abs(p)) + 'A', inputVar = True), IR.Var('EXP' + str(abs(p)) + 'B', inputVar = True)]

		return [table, tableVar]

	def getAlphaCount(self, max, shl):
		mask = 2 ** self.expB - 1
		shr = Common.wordLength - shl - self.expB
		return ((max >> shr) & mask) + 1

	def readExpProfileFile(self):
		'''
		This function reads the profile generated by the floating-point program.
		The profile will consist of the range of run-time values seen for the exp() function.
		This range is stored and will be used while generating look-up tables for exp() function.
		'''
		if self.expProfileLoaded == True:
			return
		self.expProfileLoaded = True

		inputFile = getProfileLogFile()

		data = []
		with open(inputFile, 'r') as f:
			for line in f:
				entries = line.strip().split(", ")
				row = list(map(float, entries))
				data.append(row)

		[min_exp, max_exp] = data[1]
		#[min_exp, max_exp] = [0.022, 15.012]
		
		assert max_exp >= min_exp >= 0, "The range of values for exp() is not as expected."

		# Data for computing exp
		self.expRange = [min_exp, max_exp]
		self.expB = getExpBitLength()
		self.expTableShape = [2, 2 ** self.expB]

		self.MAX_VAL_EXP = max_exp

	# out = argmax(in)
	def visitArgMax(self, node:AST.Func):

		(prog_in, expr_in) = self.visit(node.expr)

		type_out = node.expr.type

		assert type_out.dim == 2, "'argmax' operator currently only supports 2D tensors."

		[I, J] = type_out.shape

		expr_out = self.getTempVar()

		expr_in.inputVar = False

		comment = IR.Comment('argmax(' + expr_in.idf + ')')

		funcCall = IR.FuncCall("ArgMax", {
								expr_in: "A",
								IR.Int(I): "I",
								IR.Int(J): "J",
								expr_out: "index"
								})

		prog_argmax = IR.Prog([comment, funcCall])
			
		prog_out = IRUtil.concatPrograms(prog_in, prog_argmax)
		
		self.varDeclarations[expr_out.idf] = Type.Int()
		self.internalVars.append(expr_out.idf)

		return (prog_out, expr_out)

	# out = sgn(in)
	# if in > 0:
	#    out = 1
	# else
	#    out = 0
	def visitSgn(self, node:AST.Func):

		(prog_in, expr_in) = self.visit(node.expr)

		expr_out = self.getTempVar()
		type_in = node.expr.type

		expr_in_idx = IRUtil.addIndex(expr_in, [IRUtil.zero] * type_in.dim)

		comment = IR.Comment('sgn(' + expr_in.idf + ')')
		
		cmd1 = IR.Assn(expr_out, IRUtil.cond_zero(expr_in_idx, IRUtil.one, IRUtil.zero))

		prog_sgn = IR.Prog([comment, cmd1])

		prog_out = IRUtil.concatPrograms(prog_in, prog_sgn)
		
		self.varDeclarations[expr_out.idf] = Type.Int()
		self.internalVars.append(expr_out.idf)
		
		return (prog_out, expr_out)

	# out = tanh(in)
	def visitTanh(self, node:AST.Func):

		(prog_in, expr_in) = self.visit(node.expr)

		type_in = node.expr.type
		[I, J] = type_in.shape

		scale_in = self.varScales[expr_in.idf]
		intv_in = self.varIntervals[expr_in.idf]

		if forFloat():
			tanh_limit = IR.Float(Common.tanh_limit)
		else:
			# Scale tanh limit
			tanh_limit = self.getNumInFixedPoint(Common.tanh_limit, scale_in)

		tanh_intv = self.getInterval(scale_in, Common.tanh_limit, Common.tanh_limit)
		intv_out = self.updateTanhIntv(intv_in, tanh_intv)

		# Temp computation for POC. Remove later.
		scale_new = self.getScale(Common.tanh_limit)
		print("Scale changes in TanH operation: old = %d, new = %d, diff = %d" % (scale_in, scale_new, abs(scale_in - scale_new)))

		expr_in.inputVar = False

		comment = IR.Comment("tanh(" + expr_in.idf + ")")

		funcCall = IR.FuncCall("TanH", {
								expr_in: "A",
								IR.Int(I): "I",
								IR.Int(J): "J",
								tanh_limit: "threshold"
								})

		prog_tanh = IR.Prog([comment, funcCall])
		
		prog_out = IRUtil.concatPrograms(prog_in, prog_tanh)

		self.varIntervals[expr_in.idf] = intv_out
		expr_out = expr_in
		
		return (prog_out, expr_out)

	# out = sigmoid(in)
	def visitSigmoid(self, node:AST.Func):

		# y = max(min( x/4 + 2/4 , 1), 0), 1)

		denominator = 4
		addition = 0.5
		sigmoid_limit = 1

		(prog_in, expr_in) = self.visit(node.expr)

		type_in = node.expr.type
		[I, J] = type_in.shape

		scale_in = self.varScales[expr_in.idf]
		intv_in = self.varIntervals[expr_in.idf]
		
		# Scale sigmoid limit and other constants
		addition_int = self.getNumInFixedPoint(addition, scale_in)
		sigmoid_limit_int = self.getNumInFixedPoint(sigmoid_limit, scale_in)

		# Compute new interval
		[m, M] = intv_in
		m_new = max(min((m / denominator) + addition_int.n, sigmoid_limit_int.n), 0)
		M_new = max(min((M / denominator) + addition_int.n, sigmoid_limit_int.n), 0)
		assert m_new <= M_new, "The range of sigmoid has changed. Re-check the assertion."
		#if m_new > M_new:
			#m_new, M_new = M_new, m_new
		
		intv_out = (m_new, M_new)

		scale_out = self.getScale(1.5)

		# Compute new scale
		# Temp computation for POC. Remove later.
		max_val = max(abs(m_new), abs(M_new))
		max_val_f = np.ldexp(max_val, scale_in)
		scale_new = self.getScale(max_val_f)
		print("Scale changes in Sigmoid operation: old = %d, new = %d, diff = %d" % (scale_in, scale_new, abs(scale_in - scale_new)))

		if forFloat():
			addition_ir = IR.Float(addition)
			sigmoid_limit_ir = IR.Float(sigmoid_limit)
		else:
			addition_ir = addition_int
			sigmoid_limit_ir = sigmoid_limit_int

		scale_in_num = 2 ** -scale_in
		scale_out_num = 2 ** -scale_out

		expr_in.inputVar = False

		comment = IR.Comment("Sigmoid(" + expr_in.idf + ")")

		funcCall = IR.FuncCall("Sigmoid", {
								expr_in: "A",
								IR.Int(I): "I",
								IR.Int(J): "J",
								IR.Int(denominator): "div",
								addition_ir: "add",
								sigmoid_limit_ir: "sigmoid_limit",
								IR.Int(scale_in_num): "scale_in",
								IR.Int(scale_out_num): "scale_out"
								})

		prog_sigmoid = IR.Prog([comment, funcCall])
		
		prog_out = IRUtil.concatPrograms(prog_in, prog_sigmoid)

		expr_out = expr_in
		
		self.varScales[expr_in.idf] = scale_out
		self.varIntervals[expr_in.idf] = intv_out

		self.log.print(comment.msg)
		self.log.print("\tInput:  scale = %d, interval = [%d, %d]" % ((self.varScales[expr_in.idf],) + self.varIntervals[expr_in.idf]))
		self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

		return (prog_out, expr_out)

	# out = $x[start:end] in
	def visitSum(self, node:AST.Sum):
		'''
		expr_out
		i = 0
		for (j = 0; j < n; j++)
		  expr_in = prog_in
		  expr_out = expr_out + expr_in
		  i++

		1.  for i in [0, C]:
		2.    expr_out[i] = expr_out[i] + shr(expr_in[i])
		'''

		var_idf = node.name
		self.varDeclarations[var_idf] = Type.Int()
		self.internalVars.append(var_idf)

		(prog_in, expr_in) = self.visit(node.expr)

		start, end = node.start, node.end
		
		expr_out = self.getTempVar()
		type_out = node.type

		var = IR.Var(var_idf)
		var_iter = self.getTempIterator()
		iters = self.getTempIterators(type_out.dim)

		scale_in = self.varScales[expr_in.idf]
		intv_in = self.varIntervals[expr_in.idf]

		(scale_out, height_shr, height_noshr) = self.getScaleForTreeSum(scale_in, end - start)
		intv_out = self.getIntervalForTreeSum(intv_in, end - start, height_shr, height_noshr)

		# Tree sum to sum output of each iteration
		expr_in_idx = IRUtil.addIndex(expr_in, iters)
		expr_out_idx = IRUtil.addIndex(expr_out, iters)

		comment = IR.Comment("sum(i = [%d, %d])" % (start, end))
		
		cmd1 = IR.Memset(expr_out, type_out.size())
		cmd2 = IR.Assn(expr_out_idx, IRUtil.add(expr_out_idx, IRUtil.shr(expr_in_idx, height_shr)))
		treeSum = IRUtil.loop(type_out.shape, iters, [cmd2])

		# Final program to sum output of each iteration
		prog_sum = [cmd1,
					IR.Assn(var, IR.Int(start)),
					IR.For(var_iter, 0, IRUtil.lt(var_iter, IR.Int(end - start)),
					prog_in.cmd_l + treeSum + [IR.Assn(var, IRUtil.inc(var))])
					]

		prog_out = IR.Prog([comment] + prog_sum)
		
		self.varDeclarations[expr_out.idf] = type_out
		self.varScales[expr_out.idf] = scale_out
		self.varIntervals[expr_out.idf] = intv_out

		return (prog_out, expr_out)

	# out = loop(x[start:end]) (expr) in
	def visitLoop(self, node:AST.Loop):
		'''
		for (i = 0; i < n; i++)
		  prog_in
		'''

		idf = node.mutableVar.name
		self.mutableVars.append(idf)

		# Update the scale and interval of the mutable variable only during fixed-point code generation
		if forFixed():
			scale, intv = self.readProfileForMutableVars(idf)
			self.varScales[idf] = scale
			self.varIntervals[idf] = intv

		(prog_in, expr_in) = self.visit(node.expr)

		start, end = node.start, node.end
		assert start == 0, "'loop' operator currently supports only iterations starting from 0."

		var = IR.Var(node.name)

		comment = IR.Comment("loop(%s = [%d, %d], %s)" % (node.name, start, end, idf))

		loop = IR.For(var, 0, IRUtil.lt(var, IR.Int(end - start)), prog_in.cmd_l)

		# Generate code for profiling
		if forFloat() and getTarget() == Common.Target.X86:
			mVar = IR.Var(node.mutableVar.name)
			mVar_type = node.mutableVar.type
			profile_iters = self.getTempIterators(mVar_type.dim)
			mVar_idx = IRUtil.addIndex(mVar, profile_iters)
			funcCall = IR.FuncCall("updateRange", {
									mVar_idx: "A"
									})
			profile = IRUtil.loop(mVar_type.shape, profile_iters, [funcCall])
		else:
			profile = []

		prog_out = IR.Prog([comment, loop] + profile)

		return (prog_out, expr_in)

	# out = in_cond > 0? in_A: in_B
	def visitCond(self, node:AST.Cond):

		(prog_in_cond, expr_in_cond) = self.visit(node.expr)

		(prog_in_A, expr_in_A) = self.visit(node.trueBlock)

		(prog_in_B, expr_in_B) = self.visit(node.falseBlock)

		type_in_cond = node.expr.type
		type_in_A = node.trueBlock.type
		
		if Type.isInt(type_in_cond):
			expr_in_cond_idx = expr_in_cond
		else:
			expr_in_cond_idx = IRUtil.addIndex(expr_in_cond, [IRUtil.zero] * type_in_cond.dim)
		
		# e2, e3 : Int
		if Type.isInt(type_in_A):
			prog_out = IRUtil.concatPrograms(prog_in_cond, prog_in_A, prog_in_B)
			expr_out = IRUtil.cond_zero(expr_in_cond_idx, expr_in_A, expr_in_B)

			if isinstance(expr_in_A, IR.Var):
				assert expr_in_A.idf not in self.varScales and expr_in_A.idf not in self.varIntervals
			if isinstance(expr_in_B, IR.Var):
				assert expr_in_B.idf not in self.varScales and expr_in_B.idf not in self.varIntervals

		# e2, e3 : Tensor(), or Tensor(..)
		else:
			expr_out = self.getTempVar()
			iters = self.getTempIterators(type_in_A.dim)

			scale_in_A, scale_in_B = self.varScales[expr_in_A.idf], self.varScales[expr_in_B.idf]
			intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]
			
			m_A, M_A = intv_in_A
			m_B, M_B = intv_in_B

			if scale_in_A >= scale_in_B:
				shr_A, shr_B = 0, scale_in_A - scale_in_B
			else:
				shr_A, shr_B = scale_in_B - scale_in_A, 0

			scale_out = max(scale_in_A, scale_in_B)
			intv_out = (min(m_A >> shr_A, m_B >> shr_B),
						max(M_A >> shr_A, M_B >> shr_B))
				
			# prog_assn
			expr_in_A_idx = IRUtil.addIndex(expr_in_A, iters)
			expr_in_B_idx = IRUtil.addIndex(expr_in_B, iters)
			expr_out_idx = IRUtil.addIndex(expr_out, iters)
			
			rhs = IRUtil.cond_zero(expr_in_cond_idx,
							   IRUtil.shr(expr_in_A_idx, shr_A),
							   IRUtil.shr(expr_in_B_idx, shr_B))
			cmdl_assn = IRUtil.loop(type_in_A.shape, iters, [IR.Assn(expr_out_idx, rhs)])
			prog_cond = IR.Prog(cmdl_assn)
			
			prog_out = IRUtil.concatPrograms(prog_in_cond, prog_in_A, prog_in_B, prog_cond)
			
			self.varDeclarations[expr_out.idf] = type_in_A
			self.varScales[expr_out.idf] = scale_out
			self.varIntervals[expr_out.idf] = intv_out

		return (prog_out, expr_out)

	# let idf = decl 'in' in
	def visitLet(self, node:AST.Let):

		(prog_decl, expr_decl) = self.visit(node.decl)
		
		type_decl = node.decl.type
		idf = node.name

		# e1 : Int
		if Type.isInt(type_decl):
			self.varDeclarations[idf] = Type.Int()
			self.internalVars.append(idf)

			(prog_in, expr_in) = self.visit(node.expr)

			cmd = IR.Assn(IR.Var(idf), expr_decl)
			prog_let = IR.Prog([cmd])
			
			prog_out = IRUtil.concatPrograms(prog_decl, prog_let, prog_in)

			return (prog_out, expr_in)
		
		# e1 : Tensor{(),(..)}
		else:
			self.varScales[idf] = self.varScales[expr_decl.idf]
			self.varIntervals[idf] = self.varIntervals[expr_decl.idf]

			if isinstance(node.decl, AST.Decl):
				self.globalVars.append(idf)
				# TODO: do I need to update varDeclarations?
				self.varDeclarations[idf] = node.decl.type
				expr_decl.idf = idf
				expr_decl.inputVar = True

			if idf in self.mutableVars:
				expr_decl.idf = idf

			if forFixed() and idf in self.mutableVars:
				# add a loop to adjust the scale back to the original one
				curr_scale = self.varScales[idf]
				[minVal, maxVal] = self.mutableVarsProfile[0]
				new_scale = self.getScale(max(abs(minVal), abs(maxVal)))
				new_intv = self.getInterval(new_scale, minVal, maxVal)

				diff_scale = curr_scale - new_scale
				
				[I, J] = type_decl.shape

				if diff_scale > 0:
					diff_scale = self.formatShr(abs(diff_scale))

					funcCall_for_mutable = IR.FuncCall("AdjustScaleShl", {
											expr_decl: "A",
											IR.Int(I): "I",
											IR.Int(J): "J",
											diff_scale: "scale"
											})
					prog_for_mutable = IR.Prog([funcCall_for_mutable])
				elif diff_scale < 0:
					diff_scale = self.formatShr(abs(diff_scale))

					funcCall_for_mutable = IR.FuncCall("AdjustScaleShr", {
											expr_decl: "A",
											IR.Int(I): "I",
											IR.Int(J): "J",
											diff_scale: "scale"
											})

					prog_for_mutable = IR.Prog([funcCall_for_mutable])
				else:
					prog_for_mutable = IR.Prog([])

				# reset the self.scale value to the profile generated one
				self.varScales[idf] = new_scale
				self.varIntervals[idf] = new_intv
			else:
				prog_for_mutable = IR.Prog([])

			(prog_in, expr_in) = self.visit(node.expr)

			# TODO: When is this triggered and why is this required?
			if forFixed() and idf in self.mutableVars:
				print("TODO: Fix this if condition")
				#expr_decl.idf = idf
				[minVal, maxVal] = self.mutableVarsProfile[0]
				new_scale = self.getScale(max(abs(minVal), abs(maxVal)))
				new_intv = self.getInterval(new_scale, minVal, maxVal)
				self.varScales[expr_decl.idf] = new_scale
				self.varIntervals[expr_decl.idf] = new_intv

			prog_decl = IRUtil.concatPrograms(prog_decl, IR.Prog([prog_for_mutable]))

			prog_in = prog_in.subst(idf, expr_decl)
			expr_in = expr_in.subst(idf, expr_decl)

			prog_out = IRUtil.concatPrograms(prog_decl, prog_in)

			return (prog_out, expr_in)




	# TODO: The profile for the mutable variable is only in the first line of the profile dump
	# Currently this doesn't support recording the profile of multiple variables. Fix this.
	def readProfileForMutableVars(self, idf):

		# data-driven parameters
		inputFile = getProfileLogFile()

		data = []
		with open(inputFile, 'r') as f:
			for line in f:
				entries = line.strip().split(", ")
				row = list(map(float, entries))
				self.mutableVarsProfile.append(row)

		# REMOVE THIS. USED FOR DEBUGGING
		# -14 = 
		#self.mutableVarsProfile[0] = [-0.000001, 0.6]
		# -15 = 48.933
		#self.mutableVarsProfile[0] = [-0.000001, 0.3]
		# -16 = 50.277
		#self.mutableVarsProfile[0] = [-0.000001, 0.2]
		# -17 = 50.198
		#self.mutableVarsProfile[0] = [-0.000001, 0.12]
		# -18 = 15.6
		#self.mutableVarsProfile[0] = [-0.000001, 0.06]
		# -19 = 9.2
		#self.mutableVarsProfile[0] = [-0.000001, 0.02]

		[minVal, maxVal] = self.mutableVarsProfile[0]
		
		scale = self.getScale(max(abs(minVal), abs(maxVal)))
		intv = self.getInterval(scale, minVal, maxVal)
		
		return scale, intv

	# Computing exponent and intervals
	def getScale(self, val_max:float): # -> int
		return computeScalingFactor(val_max)

	# Takes range [val_min, val_max] and returns the interval in fixed-point
	def getInterval(self, scale:int, val_min:float, val_max:float):
		return (int(np.ldexp(val_min, -scale)), int(np.ldexp(val_max, -scale)))

	# A * B
	def getScaleForMul(self, scale_A:int, shr_A:int, scale_B:int, shr_B:int) -> int:
		return (scale_A + shr_A) + (scale_B + shr_B)

	def getIntvervalForMul(self, intv_A, shr_A:int, intv_B, shr_B:int): # int^2 * int^2 -> int^2
		(minVal_A, maxVal_A) = intv_A
		(minVal_A, maxVal_A) = (minVal_A >> shr_A, maxVal_A >> shr_A)

		(minVal_B, maxVal_B) = intv_B
		(minVal_B, maxVal_B) = (minVal_B >> shr_B, maxVal_B >> shr_B)

		values = [minVal_A * minVal_B, minVal_A * maxVal_B, maxVal_A * minVal_B, maxVal_A * maxVal_B]

		minVal_out, maxVal_out = min(values), max(values)

		return (minVal_out, maxVal_out)

	def getScaleForTreeSum(self, scale:int, length:int):
		height = int(np.ceil(np.log2(length)))
		
		if scale >= self.MAX_SCALE:
			scale_out = scale
		else:
			scale_out = min(scale + height, self.MAX_SCALE)
		
		height_shr = scale_out - scale
		assert height_shr >= 0
		
		height_noshr = height - height_shr
		assert height_noshr >= 0
		
		return (scale_out, height_shr, height_noshr)

	def getIntervalForTreeSum(self, intv, count, height_shr, height_noshr):
		(minVal, maxVal) = intv

		arr_min = [minVal for i in range(count)]
		arr_max = [maxVal for i in range(count)]

		minVal_out = self.treeSum(arr_min, count, height_shr, height_noshr)
		maxVal_out = self.treeSum(arr_max, count, height_shr, height_noshr)

		return (minVal_out, maxVal_out)

	def treeSum(self, arr, count, height_shr, height_noshr):
		# Return if only one element
		if count == 1:
			return arr[0]

		# Start in the scale down mode by performing shift rights
		shr = True

		for depth in range(height_shr + height_noshr):
			# Switch off scale down mode after reaching the desired height
			if depth >= height_shr:
				shr = False

			# Pair-wise sum the array elements
			# If odd number of elements, the last element is handled after the loop
			for p in range(count // 2):
				sum = arr[2 * p] + arr[(2 * p) + 1]

				# Perform scale down based on the mode
				if shr:
					arr[p] = sum // 2
				else:
					arr[p] = sum

			# Handling the last element if odd number of elements
			if count % 2 == 1:
				# Copy the last element adjacent to the new array
				index = count // 2 + 1
				if shr:
					arr[index - 1] = arr[count - 1] // 2
				else:
					arr[index - 1] = arr[count - 1]

			# Debugging statement
			# Adding a 0 after the end of the new array to seperate the old array
			if count % 2 == 1:
				index = count // 2 + 1
				arr[index - 1 + 1] = 0
			else:
				arr[count // 2] = 0
		
			count = (count + 1) >> 1

		return arr[0]

	def getScaleAndIntervalForAddAndSub(self, scale_A:int, scale_B:int, intv_A, intv_B, op_fn):
		if op_fn == operator.add:
			return self.getScaleAndIntervalForAdd(scale_A, scale_B, intv_A, intv_B)
		elif op_fn == operator.sub:
			return self.getScaleAndIntervalForSub(scale_A, scale_B, intv_A, intv_B)
		else:
			assert False, "Operator other than add and sub not supported"

	def getScaleAndIntervalForAdd(self, scale_A:int, scale_B:int, intv_A, intv_B):
		(minVal_A, maxVal_A) = intv_A
		(minVal_B, maxVal_B) = intv_B

		if scale_A >= scale_B:
			shr_all = [0, scale_A - scale_B, 0]
			scale_common = scale_A
		else:
			shr_all = [scale_B - scale_A, 0, 0]
			scale_common = scale_B
		
		minVal_out = (minVal_A >> shr_all[0]) + (minVal_B >> shr_all[1])
		maxVal_out = (maxVal_A >> shr_all[0]) + (maxVal_B >> shr_all[1])
		
		#if max(abs(minVal_out), abs(maxVal_out)) >= (1 << (Common.wordLength - 2)) and scale_common < self.MAX_SCALE:
		if scale_common < self.MAX_SCALE:
			shr_all[2] = 1
			scale_common += 1
		max_abs = (1 << Common.wordLength - 2) - 1
		
		minVal_out = max(minVal_out >> shr_all[2], -max_abs)
		maxVal_out = min(maxVal_out >> shr_all[2],  max_abs)
			
		return (scale_common, (minVal_out, maxVal_out), shr_all)

	def getScaleAndIntervalForSub(self, scale_A:int, scale_B:int, intv_A, intv_B):
		(minVal_A, maxVal_A) = intv_A
		(minVal_B, maxVal_B) = intv_B

		if scale_A >= scale_B:
			shr_all = [0, scale_A - scale_B, 0]
			scale_common = scale_A
		else:
			shr_all = [scale_B - scale_A, 0, 0]
			scale_common = scale_B
		
		minVal_out = (minVal_A >> shr_all[0]) - (minVal_B >> shr_all[1])
		maxVal_out = (maxVal_A >> shr_all[0]) - (maxVal_B >> shr_all[1])
		
		#if max(abs(minVal_out), abs(maxVal_out)) >= (1 << (Common.wordLength - 2)) and scale_common < self.MAX_SCALE:
		if scale_common < self.MAX_SCALE:
			shr_all[2] = 1
			scale_common += 1
		max_abs = (1 << Common.wordLength - 2) - 1
		
		minVal_out = max(minVal_out >> shr_all[2], -max_abs)
		maxVal_out = min(maxVal_out >> shr_all[2],  max_abs)
			
		return (scale_common, (minVal_out, maxVal_out), shr_all)

	def getScaleForExp(self, scale_A:int, shr_A:int, scale_B:int, shr_B:int):
		return (scale_A + shr_A) + (scale_B + shr_B)

	def getIntervalForExp(self, scale:int, intv): # int^2 -> int^2
		(m, M) = intv
		assert m < np.ldexp(self.MAX_VAL_EXP, -scale)
		M = min(M, np.ldexp(self.MAX_VAL_EXP, -scale))
		return self.getInterval(scale, np.exp(np.ldexp(m, scale)), np.exp(np.ldexp(M, scale)))

	def getShrForMulOld(self, scale_A, scale_B):
		shr = (Common.wordLength - 1) // 2
		pRes = (scale_A + shr) + (scale_B + shr)
		if pRes < self.MAX_SCALE:
			return [shr, shr]
		else:
			save = abs(abs(pRes) - abs(self.MAX_SCALE))
			save1 = save // 2
			save2 = save - save1
			shr1 = max(shr - save1, 0)
			shr2 = max(shr - save2, 0)
			return [shr1, shr2]

	def getShrForMul(self, scale_A, scale_B):
		shr1, shr2 = Common.wordLength // 2, (Common.wordLength // 2) - 1
		pRes = (scale_A + shr1) + (scale_B + shr2)

		if pRes <= self.MAX_SCALE:
			if scale_A <= scale_B:
				shrA, shrB = shr1, shr2
			else:
				shrA, shrB = shr2, shr1
			return [shrA, shrB]
		else:
			save = abs(abs(pRes) - abs(self.MAX_SCALE))
			if save % 2 == 1:
				shr1 -= 1
				save -= 1
			save = save // 2
			if scale_A <= scale_B:
				shrA = max(shr1 - save, 0)
				shrB = max(shr2 - save, 0)
			else:
				shrA = max(shr2 - save, 0)
				shrB = max(shr1 - save, 0)
		
			return [shrA, shrB]

	def getNumInFixedPoint(self, num_float, scale):
		# num_float as python int
		num_py = int(np.ldexp(num_float, -scale))
		assert np.iinfo(IR.DataType.getIntClass()).min <= num_py <= np.iinfo(IR.DataType.getIntClass()).max, "%.6f in fixed-point representation using numpy will overflow." % (num_float)
		# num as numpy int
		num_np = IR.DataType.getInt(num_py)
		# num as SeeDot int
		num_ir = IR.Int(num_np)
		return num_ir

	def updateTanhIntv(self, intv_A, intv_tanh):
		minVal_A, maxVal_A = intv_A
		minVal_tanh, maxVal_tanh = intv_tanh
		return min(minVal_A, minVal_tanh), min(maxVal_A, maxVal_tanh)

	# Variable and iterators creation
	def getTempVars(self, num:int):
		return [self.getTempVar() for i in range(num)]

	def getTempVar(self):
		var = IR.Var('tmp' + str(self.counter_var))
		self.counter_var += 1
		return var

	def getTempIterators(self, num:int):
		return [self.getTempIterator() for i in range(num)]

	def getTempIterator(self):
		var = IR.Var('i' + str(self.counter_iter))
		self.counter_iter += 1
		return var

	def formatShr(self, num):
		assert num >= 0
	
		shrType = getShrType()

		if shrType == "shr" or shrType == "shr+":
			return IR.Int(num)
		elif shrType == "div":
			if num >= Common.wordLength:
				return IR.Int(IR.Int.max())
			else:
				intVar = IR.Int(2 ** num)
				if intVar.n == 0:
					assert False
				return intVar
		else:
			assert False
