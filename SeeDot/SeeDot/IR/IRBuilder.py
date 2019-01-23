# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import operator
import os

from Antlr.SeeDotParser import SeeDotParser

import AST.AST as AST
from AST.ASTVisitor import ASTVisitor

import IR.IR as IR
import IR.IRUtil as IRUtil

import Common
import Type as Type
from Util import *

class IRBuilder(ASTVisitor):

	def __init__(self):
		
		self.profileLoaded = False

		if getMaxExpnt() == None:
			# data-driven parameters
			inputFile = getProfileLogFile()

			assert os.path.isfile(inputFile)

			data = []
			with open(inputFile, 'r') as f:
				for line in f:
					entries = line.strip().split(", ")
					row = list(map(float, entries))
					data.append(row)

			[m_all, M_all] = data[0]
			self.MAX_EXPNT_ALL = self.get_expnt(max(abs(m_all), abs(M_all)))
		else:
			self.MAX_EXPNT_ALL = getMaxExpnt()

		self.expTables = {}

		# fresh vars
		self._var_cnt = 0
		self._iter_cnt = 0

		# idf of vars that need to be init'ed
		self.VAR_IDF_INIT = []

		# Global variables
		self.decls = {}
		self.expts = {}
		self.intvs = {}
		self.cnsts = {}

	def readProfileFile(self):
		if self.profileLoaded == True:
			return
		
		self.profileLoaded = True

		# data-driven parameters
		inputFile = getProfileLogFile()

		data = []
		with open(inputFile, 'r') as f:
			for line in f:
				entries = line.strip().split(", ")
				row = list(map(float, entries))
				data.append(row)

		[m_all, M_all] = data[0]
		[m_exp, M_exp] = data[1]
		#[m_exp, M_exp] = [0.022, 15.012]
		
		expB = getExpBitLength()

		# Data for computing exp
		self.expRange = [m_exp, M_exp]
		self.expB = expB
		self.expTableShape = [2, 2 ** self.expB]

		self.MAX_VAL_EXP = M_exp

	# Variable and iterators creation
	def getTempVars(self, n:int):
		return [self.getTempVar() for i in range(n)]

	def getTempVar(self):
		var = IR.Var('tmp' + str(self._var_cnt))
		self._var_cnt += 1
		return var

	def getTempIterators(self, n:int):
		return [self.getTempIterator() for i in range(n)]

	def getTempIterator(self):
		var = IR.Var('i' + str(self._iter_cnt))
		self._iter_cnt += 1
		return var

	def get_intv_exp(self, p:int, intv): # int^2 -> int^2
		(m, M) = intv
		assert m < np.ldexp(self.MAX_VAL_EXP, -p)
		M = min(M, np.ldexp(self.MAX_VAL_EXP, -p))
		return self.get_intv(p, np.exp(np.ldexp(m, p)), np.exp(np.ldexp(M, p)))

	# Computing exponent and intervals
	def get_expnt(self, maxabs:float): # -> int
		return int(np.ceil(np.log2(maxabs) - np.log2((1 << (Common.wordLength - 2)) - 1)))

	# Takes range [r1, r2] and returns the interval scaled by p
	def get_intv(self, p:int, r1:float, r2:float):
		return (int(np.ldexp(r1, -p)), int(np.ldexp(r2, -p)))

	def get_shr_mul(self, p1, p2):
		shr = (Common.wordLength - 2) // 2
		pRes = (p1 + shr) + (p2 + shr)
		if pRes < self.MAX_EXPNT_ALL:
			return [shr, shr]
		else:
			save = abs(abs(pRes) - abs(self.MAX_EXPNT_ALL))
			save1 = save // 2
			save2 = save - save1
			shr1 = max(shr - save1, 0)
			shr2 = max(shr - save2, 0)
			return [shr1, shr2]
	
	def get_expnt_mul(self, p1:int, shr1:int, p2:int, shr2:int) -> int:
		return (p1 + shr1) + (p2 + shr2)

	def get_intv_mul(self, intv_1, shr1:int, intv_2, shr2:int): # int^2 * int^2 -> int^2
		(m_1, M_1) = intv_1
		(m_1, M_1) = (m_1 >> shr1, M_1 >> shr1)

		(m_2, M_2) = intv_2
		(m_2, M_2) = (m_2 >> shr2, M_2 >> shr2)

		m = min([m_1 * m_2, m_1 * M_2, M_1 * m_2, M_1 * M_2])
		M = max([m_1 * m_2, m_1 * M_2, M_1 * m_2, M_1 * M_2])

		return (m, M)

	def get_expnt_intv_add(self, p_1:int, p_2:int, intv_1, intv_2, op_fn):
		# int * int * int^2 * int^2 -> int * int^2 * int list (len 3)
		(m_1, M_1) = intv_1
		(m_2, M_2) = intv_2

		if p_1 >= p_2:
			shr_n = [0, p_1 - p_2, 0]
			p = p_1
		else         :
			shr_n = [p_2 - p_1, 0, 0]
			p = p_2
		m = op_fn(m_1 >> shr_n[0], m_2 >> shr_n[1])
		M = op_fn(M_1 >> shr_n[0], M_2 >> shr_n[1])
		
		if max(abs(m),abs(M)) >= (1 << (Common.wordLength - 2)) and p < self.MAX_EXPNT_ALL:
			shr_n[2] = 1
			p += 1
		max_abs = (1 << Common.wordLength - 2) - 1
		m = max(m >> shr_n[2], -max_abs)
		M = min(M >> shr_n[2],  max_abs)
			
		return (p, (m, M), shr_n)

	#=== sum: expnt, intv, IR ===#
	def get_expnt_sum(self, p:int, n:int): # -> int^3
		H_tot = int(np.ceil(np.log2(n)))
		if p >= self.MAX_EXPNT_ALL:
			p_res = p
		else:
			p_res = min(p + H_tot, self.MAX_EXPNT_ALL)
		H_1 = p_res - p
		assert H_1 >= 0
		H_2 = H_tot - H_1
		assert H_2 >= 0
		return (p_res, H_1, H_2)

	def get_intv_sum(self, intv, n:int): # int^2 -> int^2
		max_abs = (1 << Common.wordLength - 2) - 1
		(m, M) = intv
		m = max(n * m, -max_abs)
		M = min(n * M,  max_abs)
		return (m, M)

	def compile_sum(self, N:int, H_1:int, H_2:int, src:IR.Var, dst:IR.Var, idx_to_prefix:bool):
		# -> cmd list * decl dict
		n_cur = self.getTempVar()
		itmp = self.getTempVar()
		[h, i] = self.getTempIterators(2)

		def get_cmdl_body(shr_n:int) -> IR.CmdList:
			assert(0 <= shr_n <= 1)
			# rhs_1 =
			#     itmp < (n_cur>>1)
			#   ?  ((src[2*i]+src[2*i+1])>>@shr_n@)
			#   : itmp == (n_cur>>1)
			#   ?  ((src[2*i] )>>@shr_n@)
			#   : 0 //src[i]
			src_i = IRUtil.addIndex(src, [i], idx_to_prefix)
			src_2i = IRUtil.addIndex(src, [IRUtil.mul(IR.Int(2),i)], idx_to_prefix)
			src_2i1 = IRUtil.addIndex(src, [IRUtil.inc(IRUtil.mul(IR.Int(2),i))], idx_to_prefix)
			rhs_1 = IR.CExpr(IRUtil.lt(itmp, IRUtil.shrUint(n_cur, 1)),
					IRUtil.shr(IRUtil.add(src_2i, src_2i1), shr_n),
					IR.CExpr(IRUtil.andd(IRUtil.eq(itmp, IRUtil.shrUint(n_cur, 1)),
							  IRUtil.eq(IR.IntBop(n_cur, IR.Op.Op['&'], IRUtil.one), IRUtil.one)),
							 IRUtil.shr(src_2i, shr_n),
							 IRUtil.zero))
			'''# rhs_2 = (n_cur&1) == 1 ? (src[n_cur-1]>>@shr_n@) : 0
			src_ncnt1 = IR.addIndex(src, [IR.sub(n_cur,IR.one)], idx_to_prefix)
			rhs_2 = IR.CExpr(IR.eq(IR.IntBop(n_cur, IR.Op.Op['&'], IR.one), IR.one),
							 IR.shr(src_ncnt1, shr_n), IR.zero)'''
			# res =
			#   itmp = 0
			#   for i in [0,@N/2+1@):
			#     src[i] = rhs_1
			#     itmp = itmp+1
			#   // src[n_cur>>1] = rhs_2
			#   n_cur = (n_cur+1)>>1
			'''lhs_2 = IR.addIndex(src, [IR.shrUint(n_cur,1)], idx_to_prefix)'''
			res = \
				[IR.Assn(itmp, IRUtil.zero)] + \
				IRUtil.loop([N // 2 + 1], [i],
					[IR.Assn(src_i, rhs_1),
					 IR.Assn(itmp, IRUtil.inc(itmp))]) + \
				[# IR.Assn(lhs_2, rhs_2),
				 IR.Assn(n_cur, IRUtil.shrUint(IRUtil.inc(n_cur), 1))]
			return res
			
		# cmdl_res =
		#   n_cur = @N@
		#   for h in [0,@H_1@): get_cmdl_body(1)
		#   for h in [0,@H_2@): get_cmdl_body(0)
		#   dst = src[0]
		cmdl_res = \
			[IR.Assn(n_cur, IR.Int(N))] + \
			IRUtil.loop([H_1], [h], get_cmdl_body(1)) + \
			IRUtil.loop([H_2], [h], get_cmdl_body(0)) + \
			[IR.Assn(dst, IRUtil.addIndex(src, [IRUtil.zero], idx_to_prefix))]
		decls_res = {n_cur.idf : Type.Int(),
					 itmp .idf : Type.Int()}
		return (cmdl_res, decls_res)

	def updateTanhIntv(self, intv_1, intv_tanh):
		m_e, M_e = intv_1
		m_t, M_t = intv_tanh
		return min(m_e, m_t), min(M_e, M_t)

	def formatShr(self, n):
		assert n >= 0
	
		shrType = getShrType()

		if shrType == "shr" or shrType == "shr+":
			return IR.Int(n)
		elif shrType == "div":
			intVar = IR.Int(2 ** n)
			if intVar.n == 0:
				return IR.Int(IR.Int.max())
			return intVar
		else:
			assert False

	def visitInt(self, node:AST.Int):
		n = node.value

		prog = IR.Prog([])
		expr = IR.Int(n)

		return (prog, expr)

	def visitFloat(self, node:AST.Float):
		r = node.value
		p = self.get_expnt(abs(r))
		intv = self.get_intv(p, r, r)
		k = IR.DataType.getInt(np.ldexp(r, -p))

		prog = IR.Prog([])
		expr = self.getTempVar()

		self.decls[expr.idf] = node.type
		self.expts[expr.idf] = p
		self.intvs[expr.idf] = intv
		self.cnsts[expr.idf] = k

		return (prog, expr)

	def visitId(self, node:AST.ID):
		idf = node.name

		prog = IR.Prog([])
		
		expr = IR.Var(idf, inputVar = True if idf in self.VAR_IDF_INIT else False)

		return (prog, expr)

	def visitDecl(self, node:AST.Decl):
		r1, r2 = node.range
		p = self.get_expnt(max(abs(r1), abs(r2)))
		intv = self.get_intv(p, r1, r2)

		prog = IR.Prog([])
		expr = self.getTempVar()
		expr.inputVar = True
		#self.VAR_IDF_INIT.append(expr.idf)
		#expr = IR.Var(idf, inputVar = True)
		
		#self.decls[expr.idf] = node.type
		self.expts[expr.idf] = p
		self.intvs[expr.idf] = intv

		return (prog, expr)

	def visitTransp(self, node:AST.Transp):
		(prog_1, expr_1) = self.visit(node.expr)
		
		# decl fresh vars
		expr_2 = self.getTempVar()

		# cmdl_for
		typ_2 = node.type
		[I, J] = typ_2.shape

		cmd0 = IR.Comment(expr_1.idf + "^T")

		# prog_for, p_2, intv_2
		p_2 = self.expts[expr_1.idf]
		intv_2 = self.intvs[expr_1.idf]

		expr_1.inputVar = False
		expr_2.inputVar = False

		funcCall = IR.FuncCall("Transpose", {
								expr_1: "A",
								expr_2: "B",
								IR.Int(I): "I",
								IR.Int(J): "J"
								})

		prog_for = IR.Prog([cmd0, funcCall])

		prog_2 = IRUtil.prog_merge(prog_1, prog_for)
		
		self.decls[expr_2.idf] = typ_2
		self.expts[expr_2.idf] = p_2
		self.intvs[expr_2.idf] = intv_2

		return (prog_2, expr_2)

	def visitReshape(self, node:AST.Reshape):
		(prog_1, expr_1) = self.visit(node.expr)

		'''
		reshape(A, n, h, w)

		cmd1:  t1 = t2 = 0;
		loop2: for n in 0:N:
		         for h in 0:H:
		           for w in 0:W:
		cmd3:        B[n][h][w] = A[t1][t2][t3]
		cmd4:        t3++;
		cmd5:        if (t3 == WW)
		               t3 = 0;
		               t2++;
		               if (t2 == HH)
		                 t2 = 0;
		                 t1++;
		'''

		typ_1 = node.expr.type
		typ_2 = node.type

		# Compute scaling factors
		p_2 = self.expts[expr_1.idf]
		intv_2 = self.intvs[expr_1.idf]

		# Declare variables
		expr_2 = self.getTempVar()
		iters_1 = self.getTempIterators(typ_1.dim)
		iters_2 = self.getTempVars(typ_2.dim)

		# Initialize to 0
		cmd1 = [IR.Assn(var, IRUtil.zero) for var in iters_2]

		# Incrementing the first index
		first_iter = iters_2[0]
		cmd4 = IRUtil.incCmd(first_iter)

		# Incrementing other indices using a loop
		cmd5 = [cmd4]
		for i in range(1, typ_2.dim):
			curr_iter = iters_2[i]
			curr_size = IR.Int(typ_2.shape[i])
			cmd5 = [IRUtil.incCmd(curr_iter), IR.If(IRUtil.eq(curr_iter, curr_size), [IRUtil.initVarToZero(curr_iter)] + cmd5)]
		
		# Outer loop
		loopShape = []
		loopIters = []
		for order in node.order:
			order = order - 1
			loopShape.append(typ_1.shape[order])
			loopIters.append(iters_1[order])

		loop2 = IRUtil.loop(loopShape, loopIters, [IR.Assn(IRUtil.addIndex(expr_2, iters_2), IRUtil.addIndex(expr_1, iters_1))] + cmd5)

		# Finalize
		comment = IR.Comment("reshape(" + expr_1.idf + ", " + ', '.join(str(e) for e in typ_2.shape) + ")")
		reshape_prog = IR.Prog([comment] + cmd1 + loop2)
		prog_2 = IRUtil.prog_merge(prog_1, reshape_prog)

		# Update context
		self.decls[expr_2.idf] = typ_2
		self.expts[expr_2.idf] = p_2
		self.intvs[expr_2.idf] = intv_2
		
		# Update declarations
		self.decls.update(dict((var.idf, Type.Int()) for var in iters_2))

		return (prog_2, expr_2)
	
	def visitMaxpool(self, node:AST.Maxpool):

		(prog_1, expr_1) = self.visit(node.expr)

		typ_2 = node.type
		F = node.dim

		# Compute scaling factor
		p_2 = self.expts[expr_1.idf]
		intv_2 = self.intvs[expr_1.idf]

		# Declare variables
		expr_2 = self.getTempVar()

		[N_A, H_A, W_A, C_A] = node.expr.type.shape

		# Finalize
		comment = IR.Comment("maxpool(" + expr_1.idf + ", " + str(F) + ")")

		expr_1.inputVar = False
		expr_2.inputVar = False

		funcCall = IR.FuncCall("Maxpool", {
								expr_1: "A",
								expr_2: "B",
								IR.Int(N_A): "N",
								IR.Int(H_A): "H",
								IR.Int(W_A): "W",
								IR.Int(C_A): "C",
								IR.Int(F): "stride"
								})

		prog_for = IR.Prog([comment, funcCall])
		
		prog_2 = IRUtil.prog_merge(prog_1, prog_for)

		# Update declarations
		self.decls[expr_2.idf] = typ_2
		self.expts[expr_2.idf] = p_2
		self.intvs[expr_2.idf] = intv_2

		return (prog_2, expr_2)
	
	def visitIndex(self, node:AST.Index):

		(prog_1, expr_1) = self.visit(node.expr)
		(prog_2, expr_2) = self.visit(node.index)

		prog_3 = IRUtil.prog_merge(prog_1, prog_2)
		expr_3 = IRUtil.addIndex(expr_1, [expr_2])

		return (prog_3, expr_3)
	
	def visitFuncCall(self, node:AST.FuncCall):

		progs = []
		exprs = []
		for expr in node.exprList:
			(prog_1, expr_1) = self.visit(expr)
			progs.append(prog_1)
			exprs.append(expr_1)

		prog_ret = IR.Prog([])
		for prog in progs:
			prog_ret = IRUtil.prog_merge(prog_ret, prog)

		expr_ret = self.getTempVar()

		args = dict()
		ch = 'A'
		for expr in exprs:
			args[expr] = ch
			ch = chr(ord(ch) + 1)
		args[expr_ret] = expr_ret.idf

		ch = 'I'
		for i in node.type.shape:
			args[IR.Int(i)] = ch
			ch = chr(ord(ch) + 1)

		s = [expr.idf for expr in exprs]
		comment = IR.Comment(node.name + '(' + ', '.join(s) + ')')

		funcCall = IR.FuncCall(node.name, args)

		prog = IR.Prog([comment, funcCall])

		prog_ret = IRUtil.prog_merge(prog_ret, prog)

		self.decls[expr_ret.idf] = node.type
		self.expts[expr_ret.idf] = self.expts[exprs[0].idf]
		self.intvs[expr_ret.idf] = self.intvs[exprs[0].idf]

		return (prog_ret, expr_ret)

	def visitUop(self, node:AST.Uop):

		(prog_1, expr_1) = self.visit(node.expr)

		op = node.op
		
		if op == SeeDotParser.ADD:
			return (prog_1, expr_1)
		assert op == SeeDotParser.SUB
		
		typ_2 = node.type
		
		# e : Int
		if Type.isInt(typ_2):
			prog_2 = prog_1
			expr_2 = IRUtil.negate(expr_1)
			decls_2 = decls_1
			expts_2 = expts_1
			intvs_2 = intvs_1

		# e: Tensor(), or Tensor(..)
		else:
			# decl fresh vars
			expr_2 = self.getTempVar()
			iters = self.getTempIterators(typ_2.dim)

			# cmdl_assn
			expr_1_elt = IRUtil.addIndex(expr_1, iters)
			expr_2_elt = IRUtil.addIndex(expr_2, iters)
			rhs = IRUtil.negate(expr_1_elt)
			cmdl_assn = IRUtil.loop(typ_2.shape, iters, [IR.Assn(expr_2_elt, rhs)])

			# prog_assn, p_2, intv_2
			prog_assn = IR.Prog(cmdl_assn)
			p_2 = self.expts[expr_1.idf]
			(m, M) = self.intvs[expr_1.idf]
			intv_2 = (-M, -m)

			prog_2 = IRUtil.prog_merge(prog_1, prog_assn)
			
			self.decls[expr_2.idf] = typ_2
			self.expts[expr_2.idf] = p_2
			self.intvs[expr_2.idf] = intv_2

		return (prog_2, expr_2)

	def visitBop1(self, node:AST.Bop1):
		op = node.op

		if    op == SeeDotParser.MUL:        return self.visitBopMul(node)
		elif  op == SeeDotParser.SPARSEMUL:  return self.visitBopSparseMul(node)
		elif  op == SeeDotParser.MULCIR:     return self.visitBopMulCir(node)
		elif  op == SeeDotParser.CONV:       return self.visitBopConv(node)
		elif  op == SeeDotParser.ADDCIR:     return self.visitBopAddOrSubCir(node)
		elif  op == SeeDotParser.SUBCIR:     return self.visitBopAddOrSubCir(node)
		else:                                assert False

	def visitBopMul(self, node:AST.Bop1):
		typ_1 = node.expr1.type
		typ_2 = node.expr2.type
		typ_3 = node.type

		if    Type.isInt(typ_3):  return self.visitBopMulInt(node)
		elif  typ_1.dim == 0:     return self.visitBopMul1DTensor(node)
		elif  typ_2.dim == 0:     return self.visitBopMul1DTensor(node)
		else:                     return self.visitBopMul2DTensor(node)

	def visitBopMulInt(self, node:AST.Bop1):
		
		(prog_1, expr_1) = self.visit(node.expr1)

		(prog_2, expr_2) = self.visit(node.expr2)

		prog_3 = IRUtil.prog_merge(prog_1, prog_2)
		expr_3 = IRUtil.mul(expr_1, expr_2)

		return (prog_3, expr_3)

	def visitBopMul1DTensor(self, node:AST.Bop1):

		(prog_1, expr_1) = self.visit(node.expr1)

		(prog_2, expr_2) = self.visit(node.expr2)

		typ_1 = node.expr1.type
		typ_2 = node.expr2.type
		typ_3 = node.type

		# decl fresh vars
		expr_3 = self.getTempVar()

		p1, p2 = self.expts[expr_1.idf], self.expts[expr_2.idf]
		intv1, intv2 = self.intvs[expr_1.idf], self.intvs[expr_2.idf]

		[shr1, shr2] = self.get_shr_mul(p1, p2)
		
		p_3 = self.get_expnt_mul(p1, shr1, p2, shr2)
		intv_3 = self.get_intv_mul(intv1, shr1, intv2, shr2)

		if typ_1.dim == 0:
			a, b = expr_1, expr_2
			[I, J] = typ_2.shape
		else:
			a, b = expr_2, expr_1
			[I, J] = typ_1.shape

		shr1 = self.formatShr(shr1)
		shr2 = self.formatShr(shr2)

		a.inputVar = False
		b.inputVar = False
		expr_3.inputVar = False

		cmd0 = IR.Comment(expr_1.idf + ' * ' + expr_2.idf)

		funcCall = IR.FuncCall("ScalarMul", {
								a: "A",
								b: "B",
								expr_3: "C",
								IR.Int(I): "I",
								IR.Int(J): "J",
								shr1: "shr1",
								shr2: "shr2"
								})

		prog_assn = IR.Prog([cmd0, funcCall])

		prog_3 = IRUtil.prog_merge(prog_1, prog_2, prog_assn)

		self.decls[expr_3.idf] = typ_3
		self.expts[expr_3.idf] = p_3
		self.intvs[expr_3.idf] = intv_3

		return (prog_3, expr_3)

	def visitBopMul2DTensor(self, node:AST.Bop1):
		
		(prog_1, expr_1) = self.visit(node.expr1)

		(prog_2, expr_2) = self.visit(node.expr2)

		# decl fresh vars
		expr_3 = self.getTempVar()
		expr_mul = self.getTempVar()

		# Compute scales
		p1, p2 = self.expts[expr_1.idf], self.expts[expr_2.idf]
		intv1, intv2 = self.intvs[expr_1.idf], self.intvs[expr_2.idf]

		[shr1, shr2] = self.get_shr_mul(p1, p2)

		typ_1 = node.expr1.type
		typ_2 = node.expr2.type
		typ_3 = node.type

		[I, J] = typ_1.shape
		[J, K] = typ_2.shape
		typ_mul = Type.Tensor([J])

		p_mul = self.get_expnt_mul(p1, shr1, p2, shr2)
		intv_mul = self.get_intv_mul(intv1, shr1, intv2, shr2)

		# p_3, intv_3, cmdl_sum, decls_sum
		(p_3, H_1, H_2) = self.get_expnt_sum(p_mul, J)
		intv_3 = self.get_intv_sum(intv_mul, J)

		shr1 = self.formatShr(shr1)
		shr2 = self.formatShr(shr2)

		c = ''
		if expr_1.idf in self.VAR_IDF_INIT:
			c += 'C'
		else:
			c += 'N'
		if expr_2.idf in self.VAR_IDF_INIT:
			c += 'C'
		else:
			c += 'N'

		expr_1.inputVar = False
		expr_2.inputVar = False
		expr_3.inputVar = False
		expr_mul.inputVar = False

		cmd0 = IR.Comment(expr_1.idf + ' * ' + expr_2.idf)

		funcCall = IR.FuncCall("MatMul" + c, {
									expr_1: "A",
									expr_2: "B",
									expr_3: "C",
									expr_mul: "T",
									IR.Int(I): "I",
									IR.Int(J): "J",
									IR.Int(K): "K",
									shr1: "shr1",
									shr2: "shr2",
									IR.Int(H_1): "H1",
									IR.Int(H_2): "H2"
									})

		prog_assn = IR.Prog([cmd0, funcCall])
		
		prog_3 = IRUtil.prog_merge(prog_1, prog_2, prog_assn)
		
		self.decls[expr_3.idf] = typ_3
		self.expts[expr_3.idf] = p_3
		self.intvs[expr_3.idf] = intv_3
		
		self.decls[expr_mul.idf] = typ_mul

		return (prog_3, expr_3)

	def visitBopMul2DTensorOld(self, node:AST.Bop1):
		
		(prog_1, expr_1) = self.visit(node.expr1)

		(prog_2, expr_2) = self.visit(node.expr2)

		typ_1 = node.expr1.type
		typ_2 = node.expr2.type
		typ_3 = node.type

		[I, J] = typ_1.shape
		[J, K] = typ_2.shape
		typ_mul = Type.Tensor([J])

		# decl fresh vars
		expr_3 = self.getTempVar()
		expr_1_shr = self.getTempVar()
		expr_2_shr = self.getTempVar()
		expr_mul = self.getTempVar()
		[i, j, k] = self.getTempIterators(3)

		p1 = self.expts[expr_1.idf]
		p2 = self.expts[expr_2.idf]
		intv1 = self.intvs[expr_1.idf]
		intv2 = self.intvs[expr_2.idf]

		[shr1, shr2] = self.get_shr_mul(p1, p2)

		# cmdl_{shr1, shr2, mul}
		expr_1_elt = IRUtil.addIndex(expr_1, [i,j])
		expr_2_elt = IRUtil.addIndex(expr_2, [j,k])

		[tmp1, tmp2] = self.getTempVars(2)
		assn1 = IR.Assn(tmp1, expr_1_elt)
		assn2 = IR.Assn(tmp2, expr_2_elt)

		cmd_1 = IR.Assn(expr_1_shr, IRUtil.shr(tmp1, shr1))
		cmd_2 = IR.Assn(expr_2_shr, IRUtil.shr(tmp2, shr2))
		cmd_3 = IR.Assn(IRUtil.addIndex(expr_mul, [j]), IRUtil.mul(expr_1_shr, expr_2_shr))
		loop_1 = IRUtil.loop([J], [j], [assn1, assn2, cmd_1, cmd_2, cmd_3])

		p_mul = self.get_expnt_mul(p1, shr1, p2, shr2)
		intv_mul = self.get_intv_mul(intv1, shr1, intv2, shr2)

		# p_3, intv_3, cmdl_sum, decls_sum
		(p_3, H_1, H_2) = self.get_expnt_sum(p_mul, J)
		intv_3 = self.get_intv_sum(intv_mul, J)
		(cmdl_sum_body, decls_sum) = \
			self.compile_sum(J, H_1, H_2,
				expr_mul,
				IRUtil.addIndex(expr_3, [i,k]), False)
		cmdl_sum = IRUtil.loop([I,K], [i,k], loop_1 + cmdl_sum_body)
		
		# prog_assn
		cmd0 = IR.Comment(expr_1.idf + ' * ' + expr_2.idf)
		prog_assn = IR.Prog([cmd0] + cmdl_sum)
		
		prog_3 = IRUtil.prog_merge(prog_1, prog_2, prog_assn)
		
		self.decls[expr_3.idf] = typ_3
		self.expts[expr_3.idf] = p_3
		self.intvs[expr_3.idf] = intv_3
		
		self.decls.update({expr_1_shr.idf : Type.Int(),
						expr_2_shr.idf : Type.Int(),
						expr_mul  .idf : typ_mul,
						tmp1.idf : Type.Int(),
						tmp2.idf : Type.Int()
						})
		self.decls.update(decls_sum)

		return (prog_3, expr_3)

	def visitBopSparseMul(self, node:AST.Bop1):

		(prog_1, expr_1) = self.visit(node.expr1)

		(prog_2, expr_2) = self.visit(node.expr2)

		[P, Q] = node.expr1.type.shape
		[Q, R] = node.expr2.type.shape
		assert R == 1

		# Initialize C
		expr_3 = self.getTempVar()
		typ_3 = node.type

		p1, p2 = self.expts[expr_1.idf], self.expts[expr_2.idf]
		intv1, intv2 = self.intvs[expr_1.idf], self.intvs[expr_2.idf]

		[shr1, shr2] = self.get_shr_mul(p1, p2)

		p_mul = self.get_expnt_mul(p1, shr1, p2, shr2)
		intv_mul = self.get_intv_mul(intv1, shr1, intv2, shr2)

		(p_3, H_1, H_2) = self.get_expnt_sum(p_mul, Q)
		intv_3 = self.get_intv_sum(intv_mul, Q)

		Aidx = IR.Var(expr_1.idf[0] + 'idx', expr_1.idx, inputVar = True)
		Aval = IR.Var(expr_1.idf[0] + 'val', expr_1.idx, inputVar = True)

		# extra
		#iters = self.getTempIterators(typ_3.dim)
		#p = IRUtil.print_loop(typ_3.shape, iters, [IR.PrintAsFloat(IRUtil.addIndex(expr_3, iters), p_3)])
		
		cmd0 = IR.Comment(expr_1.idf + ' |*| ' + expr_2.idf)
		cmd1 = IR.Memset(expr_3, typ_3.size())

		shr1 = self.formatShr(shr1)
		shr2 = self.formatShr(shr2)
		H_1 = self.formatShr(H_1)

		Aidx.inputVar = False
		Aval.inputVar = False
		expr_2.inputVar = False
		expr_3.inputVar = False

		funcCall = IR.FuncCall("SparseMatMul",
								{Aidx: "Aidx",
								Aval: "Aval",
								expr_2: "B",
								expr_3: "C",
								IR.Int(Q): "K",
								shr1: "shrA",
								shr2: "shrB",
								H_1: "shrC"
								})

		prog_3 = IR.Prog([cmd0, cmd1, funcCall])

		self.decls[expr_3.idf] = typ_3
		self.expts[expr_3.idf] = p_3
		self.intvs[expr_3.idf] = intv_3

		# Hard coded the length of Aidx and Aval to 100. Need to change that
		self.decls.update({Aidx.idf : Type.Tensor([100]),
							Aval.idf : Type.Tensor([100]),
							})
		self.VAR_IDF_INIT.append(Aidx.idf)
		self.VAR_IDF_INIT.append(Aval.idf)

		return (prog_3, expr_3)

	def visitBopMulCir(self, node:AST.Bop1):

		(prog_1, expr_1) = self.visit(node.expr1)

		(prog_2, expr_2) = self.visit(node.expr2)

		typ_1 = node.expr1.type
		typ_2 = node.expr2.type
		typ_3 = node.type

		expr_3 = self.getTempVar()

		assert typ_3.dim == 2

		[I, J] = typ_3.shape

		p1, p2 = self.expts[expr_1.idf], self.expts[expr_2.idf]
		intv1, intv2 = self.intvs[expr_1.idf], self.intvs[expr_2.idf]

		[shr1, shr2] = self.get_shr_mul(p1, p2)
		
		p_3 = self.get_expnt_mul(p1, shr1, p2, shr2)
		intv_3 = self.get_intv_mul(intv1, shr1, intv2, shr2)

		cmd0 = IR.Comment(expr_1.idf + ' <*> ' + expr_2.idf)

		shr1 = self.formatShr(shr1)
		shr2 = self.formatShr(shr2)

		expr_1.inputVar = False
		expr_2.inputVar = False
		expr_3.inputVar = False

		funcCall = IR.FuncCall("MulCir", {
								expr_1: "A",
								expr_2: "B",
								expr_3: "C",
								IR.Int(I): "I",
								IR.Int(J): "J",
								shr1: "shrA",
								shr2: "shrB"
								})

		prog_assn = IR.Prog([cmd0, funcCall])

		prog_3 = IRUtil.prog_merge(prog_1, prog_2, prog_assn)
		
		self.decls[expr_3.idf] = typ_3
		self.expts[expr_3.idf] = p_3
		self.intvs[expr_3.idf] = intv_3

		return (prog_3, expr_3)

	def visitBopConv(self, node:AST.Bop1):

		(prog_1, expr_1) = self.visit(node.expr1)

		(prog_2, expr_2) = self.visit(node.expr2)
		
		[N , H , W , CI] = node.expr1.type.shape
		[HF, WF, CI, CO] = node.expr2.type.shape

		typ_mul = Type.Tensor([HF * WF * CI])
		typ_3 = node.type

		# Compute padding
		padH = (HF - 1) // 2
		padW = (WF - 1) // 2

		# Declare variables
		[expr_3, sum] = self.getTempVars(2)

		# Compute scale reductions and new scaling factors
		p1, p2 = self.expts[expr_1.idf], self.expts[expr_2.idf]
		intv1, intv2 = self.intvs[expr_1.idf], self.intvs[expr_2.idf]

		[shr1, shr2] = self.get_shr_mul(p1, p2)

		p_mul = self.get_expnt_mul(p1, shr1, p2, shr2)
		intv_mul = self.get_intv_mul(intv1, shr1, intv2, shr2)

		(p_3, H_1, H_2) = self.get_expnt_sum(p_mul, HF * WF * CI)
		intv_3 = self.get_intv_sum(intv_mul, HF * WF * CI)

		shr1 = self.formatShr(shr1)
		shr2 = self.formatShr(shr2)

		expr_1.inputVar = False
		expr_2.inputVar = False
		expr_3.inputVar = False

		comment = IR.Comment(expr_1.idf + ' # ' + expr_2.idf)

		funcCall = IR.FuncCall("Conv", {
								expr_1: "A",
								expr_2: "B",
								expr_3: "C",
								sum: "tmp",
								IR.Int(N): "N",
								IR.Int(H): "H",
								IR.Int(W): "W",
								IR.Int(CI): "CI",
								IR.Int(HF): "HF",
								IR.Int(WF): "WF",
								IR.Int(CO): "CO",
								shr1: "shrA",
								shr2: "shrB",
								IR.Int(H_1): "H1",
								IR.Int(H_2): "H2"
								})

		prog_conv = IR.Prog([comment, funcCall])
		prog_3 = IRUtil.prog_merge(prog_1, prog_2, prog_conv)
		
		# Update context for output variable
		self.decls[expr_3.idf] = typ_3
		self.expts[expr_3.idf] = p_3
		self.intvs[expr_3.idf] = intv_3
		
		# Update declarations
		self.decls.update({sum.idf : typ_mul})

		return (prog_3, expr_3)

	def visitBopAddOrSubCir(self, node:AST.Bop1):

		(prog_1, expr_1) = self.visit(node.expr1)

		(prog_2, expr_2) = self.visit(node.expr2)

		op = node.op
		typ_1 = node.expr1.type
		typ_2 = node.expr2.type
		typ_3 = node.type

		# decl fresh vars
		expr_1_shr = self.getTempVar()
		expr_2_shr = self.getTempVar()
		sum = self.getTempVar()
		iters = self.getTempIterators(typ_3.dim)

		# p_3, intv_3, shr_n*
		if   op == SeeDotParser.ADDCIR:
			(op_ir, op_fn) = (IR.Op.Op['+'], operator.add)
			add = True
		elif op == SeeDotParser.SUBCIR:
			(op_ir, op_fn) = (IR.Op.Op['-'], operator.sub)
			add = False
		
		assert add == True

		(p_3, intv_3, [shr_n1, shr_n2, shr_n3]) = self.get_expnt_intv_add(self.expts[expr_1.idf], self.expts[expr_2.idf],
										self.intvs[expr_1.idf], self.intvs[expr_2.idf], op_fn)

		shr_n1 = self.formatShr(shr_n1)
		shr_n2 = self.formatShr(shr_n2)
		shr_n3 = self.formatShr(shr_n3)

		comment = IR.Comment(expr_1.idf + " <" + op_ir.name + "> " + expr_2.idf)

		expr_1.inputVar = False
		expr_2.inputVar = False

		if node.type.dim == 4:
			[N, H, W, C] = node.type.shape
			funcCall = IR.FuncCall("AddOrSubCir4D", {
							expr_1: "A",
							expr_2: "B",
							IR.Int(N): "N",
							IR.Int(H): "H",
							IR.Int(W): "W",
							IR.Int(C): "C",
							shr_n1: "shrA",
							shr_n2: "shrB",
							shr_n3: "shrC",
							IR.Bool(True): "add"
							})
		elif node.type.dim == 2:
			[H, W] = node.type.shape
			funcCall = IR.FuncCall("AddOrSubCir2D", {
							expr_1: "A",
							expr_2: "B",
							IR.Int(H): "H",
							IR.Int(W): "W",
							shr_n1: "shrA",
							shr_n2: "shrB",
							shr_n3: "shrC",
							IR.Bool(True): "add"
							})
		else:
			assert False

		prog_assn = IR.Prog([comment, funcCall])
			
		prog_3 = IRUtil.prog_merge(prog_1, prog_2, prog_assn)
		
		self.expts[expr_1.idf] = p_3
		self.intvs[expr_1.idf] = intv_3

		return (prog_3, expr_1)

	def visitBop2(self, node:AST.Bop2):

		(prog_1, expr_1) = self.visit(node.expr1)

		(prog_2, expr_2) = self.visit(node.expr2)

		# op_ir, typ_3
		op = node.op
		if   op == SeeDotParser.ADD:
			(op_ir, op_fn) = (IR.Op.Op['+'], operator.add)
			funcName = "MatAdd"
		elif op == SeeDotParser.SUB:
			(op_ir, op_fn) = (IR.Op.Op['-'], operator.sub)
			funcName = "MatSub"

		typ_3 = node.type

		# e : Int
		if Type.isInt(typ_3):
			prog_3 = IRUtil.prog_merge(prog_1, prog_2)
			expr_3 = IR.IntBop(expr_1, op_ir, expr_2)

		# e : Tensor(), or Tensor(..)
		else:
			# decl fresh vars
			expr_3 = self.getTempVar()
			
			# p_3, intv_3, shr_n*
			(p_3, intv_3, [shr_n1, shr_n2, shr_n3]) = self.get_expnt_intv_add(self.expts[expr_1.idf], self.expts[expr_2.idf],
										  self.intvs[expr_1.idf], self.intvs[expr_2.idf], op_fn)

			cmd0 = IR.Comment(expr_1.idf + ' ' + op_ir.name + ' ' + expr_2.idf)

			assert typ_3.dim == 2

			[I, J] = typ_3.shape

			shr_n1 = self.formatShr(shr_n1)
			shr_n2 = self.formatShr(shr_n2)
			shr_n3 = self.formatShr(shr_n3)

			expr_1.inputVar = False
			expr_2.inputVar = False
			expr_3.inputVar = False

			funcCall = IR.FuncCall(funcName, {
									expr_1: "A",
									expr_2: "B",
									expr_3: "C",
									IR.Int(I): "I",
									IR.Int(J): "J",
									shr_n1: "shrA",
									shr_n2: "shrB",
									shr_n3: "shrC"
									})

			prog_assn = IR.Prog([cmd0, funcCall])

			prog_3 = IRUtil.prog_merge(prog_1, prog_2, prog_assn)
			
			self.decls[expr_3.idf] = typ_3
			self.expts[expr_3.idf] = p_3
			self.intvs[expr_3.idf] = intv_3

		return (prog_3, expr_3)

	def visitFunc(self, node:AST.Func):
		op = node.op
		
		if    op == SeeDotParser.RELU:    return self.visitRelu(node)
		elif  op == SeeDotParser.EXP:     return self.visitExp(node)
		elif  op == SeeDotParser.ARGMAX:  return self.visitArgMax(node)
		elif  op == SeeDotParser.SGN:     return self.visitSgn(node)
		elif  op == SeeDotParser.TANH:    return self.visitTanh(node)
		else:                             assert False

	def visitRelu(self, node:AST.Func):

		(prog_1, expr_1) = self.visit(node.expr)

		typ_1 = node.expr.type
		
		comment = IR.Comment("relu(" + expr_1.idf + ")")
		
		expr_1.inputVar = False

		if node.type.dim == 4:
			[N, H, W, C] = node.type.shape
			funcCall = IR.FuncCall("Relu4D", {
									expr_1: "A",
									IR.Int(N): "N",
									IR.Int(H): "H",
									IR.Int(W): "W",
									IR.Int(C): "C"
									})
		elif node.type.dim == 2:
			[H, W] = node.type.shape
			funcCall = IR.FuncCall("Relu2D", {
									expr_1: "A",
									IR.Int(H): "H",
									IR.Int(W): "W"
									})
		else:
			assert False
		
		prog_for = IR.Prog([comment, funcCall])

		# p_2, intv_2
		(m, M) = self.intvs[expr_1.idf]
		intv_2 = (0, M)
		
		prog_2 = IRUtil.prog_merge(prog_1, prog_for)
		
		self.intvs[expr_1.idf] = intv_2

		return (prog_2, expr_1)

	def visitExp(self, node:AST.Func):
		
		self.readProfileFile()

		if useMathExp():
			return self.visitMathExp(node)
		elif useTableExp():
			return self.visitTableExp(node)
		else:
			assert False

	# Note: We assume e<=0 for exp(e)
	def visitMathExp(self, node:AST.Func):

		# Tunable parameter
		MIN = 0.1

		self.set_arg(node.expr, node)
		(prog_1, expr_1) = self.visit(node.expr)

		typ_1 = node.expr.type
		p_1 = self.expts[expr_1.idf]
		intv_1 = self.intvs[expr_1.idf]

		'''
		1.  y = ((int) (exp(((float)e) / shr1) * shr2))
		'''
		
		maxExp = np.exp(-MIN)

		expr_2 = self.getTempVar()
		p_2 = self.get_expnt(maxExp)
		intv_2 = self.get_intv(p_2, maxExp, maxExp)

		shr1 = IR.Int(2 ** -p_1)
		shr2 = IR.Int(2 ** -p_2)

		expr_1_elt = IRUtil.addIndex(expr_1, [IRUtil.zero] * typ_1.dim)
		expr_2_elt = IRUtil.addIndex(expr_2, [IRUtil.zero] * typ_1.dim)

		cmd = IR.Assn(expr_2_elt, IRUtil.castToInt(IRUtil.mul(IR.Exp(IRUtil.div(IRUtil.castToFloat(expr_1_elt), shr1)), shr2)))
		
		cmd0 = IR.Comment('exp(' + expr_1.idf + ')')

		# extra
		#p = IR.PrintAsFloat(expr_1_elt, p_1)

		prog_exp = IR.Prog([cmd0, cmd])

		prog_2 = IRUtil.prog_merge(prog_1, prog_exp)
		self.decls[expr_2.idf] = typ_1
		self.expts[expr_2.idf] = p_2
		self.intvs[expr_2.idf] = intv_2

		return (prog_2, expr_2)

	# Note: We assume e<=0 for exp(e)
	def visitTableExp(self, node:AST.Func):

		(prog_1, expr_1) = self.visit(node.expr)

		# TODO: use MAX_VAL_EXP
		typ_1 = node.expr.type
		p_1 = self.expts[expr_1.idf]
		intv_1 = self.intvs[expr_1.idf]

		[m, M] = self.expRange
		[m_scale, M_scale] = [int(np.ldexp(m, -p_1)), int(np.ldexp(M, -p_1))]

		max = int(np.ldexp(M - m, -p_1))
		shl = self.getShl(max)
		
		input = self.getTempVar()
		[i, j] = self.getTempVars(2)
		expr_2 = self.getTempVar()

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
		table = self.getExpTable(p_1)

		p1 = self.get_expnt(1)
		p2 = self.get_expnt(abs(np.exp(-m)))

		[shr1, shr2] = self.get_shr_mul(p1, p2)

		expr_1_elt = IRUtil.addIndex(expr_1, [IRUtil.zero] * typ_1.dim)
		expr_2_elt = IRUtil.addIndex(expr_2, [IRUtil.zero] * typ_1.dim)

		cond = IRUtil.lt(IRUtil.negate(expr_1_elt), IR.Int(m_scale))
		
		cmd2 = IR.Assn(i, IR.Int(0))
		cmd3 = IR.Assn(j, IR.Int(0))

		cmd6 = IR.Assn(input, IRUtil.shl(IRUtil.sub(IRUtil.negate(expr_1_elt), IR.Int(m_scale)), shl))
		cmd7 = IR.Assn(i, IRUtil.bitAnd(IRUtil.shrUint(input, shrI), mask))
		cmd8 = IR.Assn(j, IRUtil.bitAnd(IRUtil.shrUint(input, shrJ), mask))
		
		cmd1 = IR.If(cond, [cmd2, cmd3], [cmd6, cmd7, cmd8])
		cmd10 = IR.Assn(expr_2_elt, IRUtil.mul(IRUtil.shrUint(IRUtil.addIndex(table[0], [i]), shr1),	IRUtil.shrUint(IRUtil.addIndex(table[1], [j]), shr2)))

		p_2 = self.get_expnt_exp(p1, shr1, p2, shr2)
		intv_2 = self.get_intv_exp(p_2, [-m_scale, -M_scale])
		
		cmd0 = IR.Comment('exp(' + expr_1.idf + ')')

		prog_exp = IR.Prog([cmd0, cmd1, cmd10])
		prog_2 = IRUtil.prog_merge(prog_1, prog_exp)
		
		self.decls[expr_2.idf] = typ_1
		self.decls.update(dict((var.idf, Type.Int()) for var in [input, i, j]))
		self.expts[expr_2.idf] = p_2
		self.intvs[expr_2.idf] = intv_2

		return (prog_2, expr_2)

	def get_expnt_exp(self, p1:int, shr1:int, p2:int, shr2:int):
		return (p1 + shr1) + (p2 + shr2)

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
		pRes = self.get_expnt(1)
		for i in range(alpha_count):
			num = i * 2 ** (alpha + p)
			exp = np.exp(-num)
			table[0][i] = int(np.ldexp(exp, -pRes))

		beta = alpha - b
		pRes = self.get_expnt(abs(np.exp(-m)))
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

	def visitArgMax(self, node:AST.Func):

		(prog_1, expr_1) = self.visit(node.expr)

		typ_1 = node.expr.type

		assert typ_1.dim == 2

		[I, J] = typ_1.shape

		idx = self.getTempVar()

		expr_1.inputVar = False

		cmd0 = IR.Comment('argmax(' + expr_1.idf + ')')

		funcCall = IR.FuncCall("ArgMax", {
								expr_1: "A",
								IR.Int(I): "I",
								IR.Int(J): "J",
								idx: "index"
								})

		prog = IR.Prog([cmd0, funcCall])
			
		prog_2 = IRUtil.prog_merge(prog_1, prog)
		
		expr_2 = idx
		self.decls[idx.idf] = Type.Int()

		return (prog_2, expr_2)

	def visitSgn(self, node:AST.Func):

		(prog_1, expr_1) = self.visit(node.expr)

		typ_1 = node.expr.type
		
		x = self.getTempVar()
		e = IRUtil.addIndex(expr_1, [IRUtil.zero] * typ_1.dim)

		cmd = IR.Assn(x, IRUtil.cond_zero(e, IRUtil.one, IRUtil.zero))

		prog_2 = IRUtil.prog_merge(prog_1, IR.Prog([cmd]))
		expr_2 = x
		
		self.decls.update(dict((var.idf, Type.Int()) for var in [x]))
		
		return (prog_2, expr_2)

	def visitTanh(self, node:AST.Func):

		(prog_1, expr_1) = self.visit(node.expr)

		typ_1 = node.expr.type
		[I, J] = typ_1.shape

		p = self.expts[expr_1.idf]

		# Scale tanh limit
		tanh_limit = int(np.ldexp(Common.tanh_limit, -p))
		assert tanh_limit < np.iinfo(IR.DataType.getIntClass()).max
		tanh_limit = IR.DataType.getInt(tanh_limit)

		tanh_limit_pos = IR.Int(tanh_limit)
		tanh_limit_neg = IR.Int(-tanh_limit)

		iters = self.getTempIterators(typ_1.dim)
		expr_1_ite = IRUtil.addIndex(expr_1, iters)

		expr_1.inputVar = False

		cmd0 = IR.Comment("tanh(" + expr_1.idf + ")")

		funcCall = IR.FuncCall("TanH", {
								expr_1: "A",
								IR.Int(I): "I",
								IR.Int(J): "J",
								IR.Int(tanh_limit): "threshold"
								})

		tanh_intv = self.get_intv(p, Common.tanh_limit, Common.tanh_limit)
		intv_1 = self.intvs[expr_1.idf]
		self.intvs[expr_1.idf] = self.updateTanhIntv(intv_1, tanh_intv)

		prog_2 = IRUtil.prog_merge(prog_1, IR.Prog([cmd0, funcCall]))
		expr_2 = expr_1
		
		return (prog_2, expr_2)

	def visitSum(self, node:AST.Sum):

		i_idf = node.name
		self.decls[i_idf] = Type.Int()

		(prog_1, expr_1) = self.visit(node.expr)

		# i_{st,ed}, typ_2, typ_1_all
		i_st, i_ed = node.start, node.end
		
		expr_2 = self.getTempVar()
		typ_2 = node.type
		
		'''
		expr_2
		i = 0
		for (j = 0; j < n; j++)
		  expr_1 = prog_1
		  expr_2 = expr_2 + expr_1
		  i++

		1.  for i in [0, C]:
		2.    expr_2[i] = expr_2[i] + shr(expr_1[i])
		'''

		i_var = IR.Var(i_idf)
		i_iter = self.getTempIterator()
		iters = self.getTempIterators(typ_2.dim)

		# p_2, intv_2, cmdl_sum, decls_sum
		(p_2, H_1, H_2) = self.get_expnt_sum(self.expts[expr_1.idf], i_ed - i_st)

		expr_1_elt = IRUtil.addIndex(expr_1, iters)
		expr_2_elt = IRUtil.addIndex(expr_2, iters)

		cmd1 = IR.Memset(expr_2, typ_2.size())
		cmd2 = IR.Assn(expr_2_elt, IRUtil.add(expr_2_elt, IRUtil.shr(expr_1_elt, H_1)))
		sum_loop = IRUtil.loop(typ_2.shape, iters, [cmd2])

		cmd_sum = \
			[cmd1,
			IR.Assn(i_var, IR.Int(i_st)),
			 IR.For(i_iter, 0, IRUtil.lt(i_iter, IR.Int(i_ed - i_st)),
				prog_1.cmd_l + sum_loop + \
				[IR.Assn(i_var, IRUtil.inc(i_var))])]

		intv_2 = self.get_intv_sum(self.intvs[expr_1.idf], i_ed - i_st)

		prog_2 = IR.Prog(cmd_sum)
		
		self.decls[expr_2.idf] = typ_2
		self.expts[expr_2.idf] = p_2
		self.intvs[expr_2.idf] = intv_2

		return (prog_2, expr_2)

	def visitCond(self, node:AST.Cond):

		(prog_1, expr_1) = self.visit(node.expr)

		(prog_2, expr_2) = self.visit(node.trueBlock)

		(prog_3, expr_3) = self.visit(node.falseBlock)

		typ_1 = node.expr.type
		typ_2 = node.trueBlock.type
		if Type.isInt(typ_1): expr_1_elt = expr_1
		else                      : expr_1_elt = IRUtil.addIndex(expr_1, [IRUtil.zero] * typ_1.dim)
		
		# e2,e3 : Int
		if Type.isInt(typ_2):
			prog_4 = IRUtil.prog_merge(prog_1, prog_2, prog_3)
			expr_4 = IRUtil.cond_zero(expr_1_elt, expr_2, expr_3)

		# e2,e3 : Tensor(), or Tensor(..)
		else:
			# decl fresh vars
			expr_4 = self.getTempVar()
			iters = self.getTempIterators(typ_2.dim)

			# p_4, intv_4
			(p_2, p_3) = (self.expts[expr_2.idf], self.expts[expr_3.idf])
			(intv_2, intv_3) = (self.intvs[expr_2.idf], self.intvs[expr_3.idf])
			(m_2, M_2) = intv_2
			(m_3, M_3) = intv_3
			if p_2 >= p_3: (shr_n_2, shr_n_3) = (0, p_2 - p_3)
			else:          (shr_n_2, shr_n_3) = (p_3 - p_2, 0)
			p_4 = max(p_2, p_3)
			intv_4 = (min(m_2 >> shr_n_2, m_3 >> shr_n_3),
					  max(M_2 >> shr_n_2, M_3 >> shr_n_3))
				
			# prog_assn
			expr_2_elt = IRUtil.addIndex(expr_2, iters)
			expr_3_elt = IRUtil.addIndex(expr_3, iters)
			expr_4_elt = IRUtil.addIndex(expr_4, iters)
			rhs = IRUtil.cond_zero(expr_1_elt,
							   IRUtil.shr(expr_2_elt, shr_n_2),
							   IRUtil.shr(expr_3_elt, shr_n_3))
			cmdl_assn = IRUtil.loop(typ_2.shape, iters, [IR.Assn(expr_4_elt, rhs)])
			prog_assn = IR.Prog(cmdl_assn)
			
			prog_4 = IRUtil.prog_merge(prog_1, prog_2, prog_3, prog_assn)
			
			self.decls[expr_4.idf] = typ_2
			self.expts[expr_4.idf] = p_4
			self.intvs[expr_4.idf] = intv_4

		return (prog_4, expr_4)

	def visitLet(self, node:AST.Let):

		(prog_1, expr_1) = self.visit(node.decl)
		typ_1 = node.decl.type
		idf = node.name

		# e1 : Int
		if Type.isInt(typ_1):
			self.decls[idf] = Type.Int()

			(prog_2, expr_2) = self.visit(node.expr)

			prog_assn = IR.Prog([IR.Assn(IR.Var(idf), expr_1)])
			prog_3 = IRUtil.prog_merge(prog_1, prog_assn, prog_2)

			return (prog_3, expr_2)
		# e1 : Tensor{(),(..)}
		else:
			self.expts[idf] = self.expts[expr_1.idf]
			self.intvs[idf] = self.intvs[expr_1.idf]

			if isinstance(node.decl, AST.Decl):
				self.VAR_IDF_INIT.append(idf)
				self.decls[idf] = node.decl.type
				expr_1.idf = idf
				expr_1.inputVar = True

			(prog_2, expr_2) = self.visit(node.expr)

			prog_2 = prog_2.subst(idf, expr_1)
			expr_2 = expr_2.subst(idf, expr_1)
			
			prog_3 = IRUtil.prog_merge(prog_1, prog_2)

			return (prog_3, expr_2)
