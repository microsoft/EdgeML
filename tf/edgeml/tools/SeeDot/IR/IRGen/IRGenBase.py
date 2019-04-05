# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import numpy as np
import operator

import AST.AST as AST
from AST.ASTVisitor import ASTVisitor
from Antlr.SeeDotParser  import SeeDotParser

import Common
from Util import *
import Type as Type

import IR.IR as IR
import IR.IRUtil as IRUtil

from Codegen.CodegenBase import CodegenBase

class IRGenBase(ASTVisitor):

	def __init__(self):
		
		self.profileLoaded = False

		if getMaxScale() == None:
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
			self.MAX_EXPNT_ALL = getMaxScale()

		self.expTables = {}

		# fresh vars
		self._var_cnt = 0
		self._iter_cnt = 0

		# idf of vars that need to be init'ed
		self.VAR_IDF_INIT = []

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

	# Set and retrieve state
	def get_arg(self, ctx):
		if ctx.parentCtx == None:
			decls = {}
			expts = {}
			intvs = {}
			cnsts = {}
		else:
			decls = ctx.parentCtx.decls
			expts = ctx.parentCtx.expts
			intvs = ctx.parentCtx.intvs
			cnsts = ctx.parentCtx.cnsts
		return (decls, expts, intvs, cnsts)
	
	def set_arg2(self, child, decls, expts, intvs, cnsts):
		child.decls = decls
		child.expts = expts
		child.intvs = intvs
		child.cnsts = cnsts

	def set_arg(self, child, parent):
		child.decls = parent.decls
		child.expts = parent.expts
		child.intvs = parent.intvs
		child.cnsts = parent.cnsts

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

	# Computing exponent and intervals
	def get_expnt(self, maxabs:float): # -> int
		return int(np.ceil(np.log2(maxabs) - np.log2((1 << (Common.wordLength - 2)) - 1)))

	# Takes range [r1, r2] and returns the interval scaled by p
	def get_intv(self, p:int, r1:float, r2:float): # -> int^2
		m = int(np.ldexp(r1, -p))
		M = int(np.ldexp(r2, -p))
		return (m, M)

	def get_shr_mul(self, p1, p2):
		shr = (Common.wordLength - 2) // 2
		return [shr, shr]

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
			
		return (p, (m,M), shr_n)

	#=== sum: expnt, intv, IR ===#
	def get_expnt_sum(self, p:int, n:int): # -> int^3
		H_tot = int(np.ceil(np.log2(n)))
		if p >= self.MAX_EXPNT_ALL:
			p_res = p
		else:
			p_res = min(p + H_tot, self.MAX_EXPNT_ALL)
		H_1 = p_res - p
		assert(H_1 >= 0)
		H_2 = H_tot - H_1
		assert(H_2 >= 0)
		return (p_res, H_1, H_2)

	def get_intv_sum(self, intv, n:int): # int^2 -> int^2
		max_abs = (1 << Common.wordLength - 2) - 1
		(m, M) = intv
		m = max(n * m, -max_abs)
		M = min(n * M,  max_abs)
		return (m,M)

	def compile_sum(self, N:int, H_1:int, H_2:int, src:IR.Var, dst:IR.Var, idx_to_prefix:bool):
		# -> cmd list * decl dict
		n_cur = self.getTempVar()
		itmp = self.getTempVar()
		[h,i] = self.getTempIterators(2)

		def get_cmdl_body(shr_n:int) -> IR.CmdList:
			assert(0 <= shr_n <= 1)
			# rhs_1 =
			#     itmp < (n_cur>>1)
			#   ?  ((src[2*i]+src[2*i+1])>>@shr_n@)
			#   : itmp == (n_cur>>1)
			#   ?  ((src[2*i] )>>@shr_n@)
			#   : 0 //src[i]
			src_i = IRUtil.addIndex(src, [i]                          , idx_to_prefix)
			src_2i = IRUtil.addIndex(src, [IRUtil.mul(IR.Int(2),i)], idx_to_prefix)
			src_2i1 = IRUtil.addIndex(src, [IRUtil.inc(IRUtil.mul(IR.Int(2),i))], idx_to_prefix)
			rhs_1 = IR.CExpr(IRUtil.lt(itmp, IRUtil.shrUint(n_cur, 1)),
							 IRUtil.shr(IRUtil.add(src_2i, src_2i1), shr_n),
					IR.CExpr(IRUtil.andd(IRUtil.eq(itmp, IRUtil.shrUint(n_cur, 1)),
									  IRUtil.eq(IR.IntBop(n_cur, IR.Op.Op['&'], IRUtil.one), IRUtil.one)),
							 IRUtil.shr(src_2i          , shr_n),
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

	def sparseSum(self, N:IR.Var, H_1:int, H_2:int, src:IR.Var, dst:IR.Var, idx_to_prefix:bool):
		# -> cmd list * decl dict
		n_cur = N
		nHalf = self.getTempVar()
		# shrCnt = self.getTempVar()
		# shrCur = self.getTempVar()
		shr = self.getTempVar()
		itmp = self.getTempVar()
		[h,i] = self.getTempIterators(2)

		def get_cmdl_body(shr_n:int) -> IR.CmdList:
			assert(0 <= shr_n <= 1)
			# rhs_1 =
			#     itmp < (n_cur>>1)
			#   ?  ((src[2*i]+src[2*i+1])>>@shr_n@)
			#   : itmp == (n_cur>>1)
			#   ?  ((src[2*i] )>>@shr_n@)
			#   : 0 //src[i]
			src_i = IRUtil.addIndex(src, [i]                          , idx_to_prefix)
			src_2i = IRUtil.addIndex(src, [IRUtil.mul(IR.Int(2),i)], idx_to_prefix)
			src_2i1 = IRUtil.addIndex(src, [IRUtil.inc(IRUtil.mul(IR.Int(2),i))], idx_to_prefix)
			rhs_1 = IR.CExpr(IRUtil.lt(itmp, IRUtil.shrUint(n_cur, 1)),
							 IRUtil.shr(IRUtil.add(src_2i, src_2i1), shr_n),
					IR.CExpr(IRUtil.andd(IRUtil.eq(itmp, IRUtil.shrUint(n_cur, 1)),
									  IRUtil.eq(IR.IntBop(n_cur, IR.Op.Op['&'], IRUtil.one), IRUtil.one)),
							 IRUtil.shr(src_2i          , shr_n),
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
			'''
			res = \
				[IR.Assn(itmp, IRUtil.zero)] + \
				IRUtil.loop([N // 2 + 1], [i],
					[IR.Assn(src_i, rhs_1),
					 IR.Assn(itmp, IRUtil.inc(itmp))]) + \
				[# IR.Assn(lhs_2, rhs_2),
				 IR.Assn(n_cur, IRUtil.shrUint(IRUtil.inc(n_cur), 1))]
			'''
			#IRUtil.loop([0], [i], [IR.Assn(src_i, rhs_1), IR.Assn(itmp,
			#IRUtil.inc(itmp))]) + \
			if shr_n == 1:
				shrCmd = [IRUtil.decCmd(shr)]
			else:
				shrCmd = []
			res = \
				[IR.Assn(itmp, IRUtil.zero)] + \
				[IR.For(i, 0, IRUtil.andd(IRUtil.lt(i,nHalf), IRUtil.gt(n_cur, IRUtil.one)), [IR.Assn(src_i, rhs_1), IR.Assn(itmp, IRUtil.inc(itmp))])] + \
				[IR.Assn(n_cur, IRUtil.shrUint(IRUtil.inc(n_cur), 1))] + shrCmd
			return res
			
		# cmdl_res =
		#   n_cur = @N@
		#   for h in [0,@H_1@): get_cmdl_body(1)
		#   for h in [0,@H_2@): get_cmdl_body(0)
		#   dst = src[0]
		cmdl_res = \
			[IR.Assn(nHalf, IRUtil.inc(IRUtil.shrUint(n_cur,1))), IR.Assn(shr, IR.Int(H_1))] + \
			IRUtil.loop([H_1], [h], get_cmdl_body(1)) + \
			IRUtil.loop([H_2], [h], get_cmdl_body(0)) + \
			[IR.Assn(dst, IRUtil.shrVar(IRUtil.addIndex(src, [IRUtil.zero], idx_to_prefix), shr))]
		decls_res = {n_cur.idf : Type.Int(),
					 nHalf.idf : Type.Int(),
					 itmp .idf : Type.Int(),
					 shr.idf : Type.Int(),
					}
		return (cmdl_res, decls_res)

	#=== exp: expnt, intv, IR ===#
	def _get_exp_table_shape(self):
		return [Common.wordLength // self.expB, 2 ** self.expB]
	
	def _get_exp_tables(self, p:int): # -> IR.Var list
		if p >= 0: p_str = 'p' + str(p)
		else   : p_str = 'n' + str(-p)
		return [IR.Var('EXPP' + p_str),
				IR.Var('EXPN' + p_str)]
	
	def _get_exp_expr(self, e_pos:IR.Expr, table:IR.Var): # -> (IR.CmdList, IR.Expr)
		[idx1_num, idx2_num] = self._get_exp_table_shape()

		# decl fresh vars, decls_res
		idx2_tmp = self.getTempVar()
		table_tmp = self.getTempVar()
		decls_res = { idx2_tmp .idf : Type.Int(),
					  table_tmp.idf : Type.Tensor([idx1_num]) }
		
		# cmdl_res =
		#   for i \in [0, idx1_num),
		#   table_tmp[idx1] = table[idx1][idx2],
		#     where idx1 = i, idx2 = (e_pos >> 8*i) & 255
		cmdl_res = []
		for i in range(idx1_num):
			idx1 = IR.Int(i)
			idx2 = IR.IntBop(IRUtil.shrUint(e_pos, self.expB * i),
							 IR.Op.Op['&'], IR.Int(idx2_num - 1))
			# NOTE: shrUint is used since e_pos >= 0
			cmdl_res += [IR.Assn(idx2_tmp, idx2)]
			rhs = IRUtil.add_idx_priv(IRUtil.addIndex(table, [idx1]), idx2_tmp, self.expB)
			lhs = IRUtil.addIndex(table_tmp, [idx1])
			cmdl_res += [IR.Assn(lhs, rhs)]

		# expr_res =
		#   ( ((table_tmp[0] >> 15) * (table_tmp[1] >> 15)) >> 15 ) *
		#   ( ((table_tmp[2] >> 15) * (table_tmp[3] >> 15)) >> 15 )
		expr_res = [IRUtil.addIndex(table_tmp, [IR.Int(i)]) for i in range(idx1_num)]
		shr_num = (Common.wordLength - 2) // 2
		mul_num = idx1_num
		while(mul_num > 1):
			for i in range(mul_num // 2):
				expr_res[i] = IRUtil.mul(IRUtil.shrUint(expr_res[2 * i], shr_num),
									 IRUtil.shrUint(expr_res[2 * i + 1], shr_num))
				# NOTE: shrUint is used since expr_res[.] >= 0
			mul_num //= 2

		return (cmdl_res, expr_res[0], decls_res)

	def get_expnt_exp(self, p:int, table_kind:str):
		assert(table_kind in ('pos', 'neg'))
		[idx1_num, idx2_num] = self._get_exp_table_shape()

		# consider mul's
		p_res = 0
		for i in range(idx1_num):
			if   table_kind == 'pos':
				max_arg = (2 ** (p + self.expB * i)) * (idx2_num - 1)
				# max_arg = min(max_arg, self.MAX_VAL_EXP) # TODO: use
				# MAX_VAL_EXP
			elif table_kind == 'neg':
				max_arg = (-2 ** (p + self.expB * i))
			p_res += self.get_expnt(np.exp(max_arg))

		# consider shr's
		shr_num = (Common.wordLength - 2) // 2
		if   Common.wordLength == 8: p_res += shr_num * (0)
		elif Common.wordLength == 16: p_res += shr_num * (2)
		elif Common.wordLength == 32: p_res += shr_num * (2 + 4)
		elif Common.wordLength == 64: p_res += shr_num * (2 + 4 + 8)
		else: assert(False)
		return p_res

	def get_intv_exp(self, p:int, intv): # int^2 -> int^2
		(m, M) = intv
		assert(m < np.ldexp(self.MAX_VAL_EXP, -p))
		M = min(M, np.ldexp(self.MAX_VAL_EXP, -p))
		return self.get_intv(p,
							 np.exp(np.ldexp(m, p)),
							 np.exp(np.ldexp(M, p)))

	def compile_exp(self, e:IR.Expr, p:int): # -> IR.CmdList * IR.Expr * decl dict
		assert(Common.wordLength % self.expB == 0)

		# table_{pos,neg}, typ_table
		[table_pos, table_neg] = self._get_exp_tables(p)
		typ_table = Type.Tensor(self._get_exp_table_shape())

		# NOTE: we assume e<=0 always.
		# e is always <=0
		if True:
			# {cmdl, expr, decls}_exp_neg
			(cmdl_exp_neg, expr_exp_neg, decls_exp_neg) = self._get_exp_expr(IRUtil.negate(e), table_neg)

			cmdl_res = cmdl_exp_neg
			expr_res = expr_exp_neg
			decls_res = {table_neg.idf : typ_table}
			decls_res.update(decls_exp_neg)

		# e can be >=0 or <0
		# WARNING: the below code is wrong (needs to be re-written).
		else:
			# expr_exp_{pos,neg}
			expnt_diff = self.get_expnt_exp(p, 'pos') - \
						   self.get_expnt_exp(p, 'neg')
			expr_exp_pos = self._get_exp_expr(e , table_pos)
			expr_exp_neg = self._get_exp_expr(IRUtil.negate(e), table_neg)
			expr_exp_neg = IRUtil.shrUint(expr_exp_neg, expnt_diff)
			# NOTE: shrUint is used since expr_exp_neg >= 0
			
			expr_res = IRUtil.cond_zero(e, expr_exp_pos, expr_exp_neg)
			decls_res = {table_pos.idf : typ_table,
						 table_neg.idf : typ_table}

		return (cmdl_res, expr_res, decls_res)

	#==========#
	#  visit* #
	#==========#
	# visit*: Context -> IR.Prog * IR.Expr * decl dict * expt dict * intv dict
	# * cnst dict
	# decl dict = (str -> Type.ExprType) dict
	# expt dict = (str -> int) dict
	# intv dict = (str -> int^2) dict
	# cnst dict = (str -> IR.UInt) dict

	def visitInt(self, node:AST.Int):
		(decls_0, expts_0, intvs_0, cnsts_0) = node.decls, node.expts, node.intvs, node.cnsts

		n = node.value

		prog_1 = IR.Prog([])
		expr_1 = IR.Int(n)
		return (prog_1, expr_1, decls_0, expts_0, intvs_0, cnsts_0)

	def visitFloat(self, node:AST.Float):
		(decls_0, expts_0, intvs_0, cnsts_0) = node.decls, node.expts, node.intvs, node.cnsts

		r = node.value
		p = self.get_expnt(abs(r))
		intv = self.get_intv(p, r, r)
		k = IR.DataType.getInt(np.ldexp(r, -p))

		prog_1 = IR.Prog([])
		expr_1 = self.getTempVar()
		decls_1 = copy_dict(decls_0, {expr_1.idf : node.type})
		expts_1 = copy_dict(expts_0, {expr_1.idf :       p})
		intvs_1 = copy_dict(intvs_0, {expr_1.idf :    intv})
		cnsts_1 = copy_dict(cnsts_0, {expr_1.idf :       k})

		return (prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_1)

	def visitId(self, node:AST.ID):
		(decls_0, expts_0, intvs_0, cnsts_0) = node.decls, node.expts, node.intvs, node.cnsts

		idf = node.name

		prog_1 = IR.Prog([])
		
		expr_1 = IR.Var(idf, inputVar = True if idf in self.VAR_IDF_INIT else False)

		return (prog_1, expr_1, decls_0, expts_0, intvs_0, cnsts_0)

	def visitDecl(self, node:AST.Decl):
		(decls_0, expts_0, intvs_0, cnsts_0) = node.decls, node.expts, node.intvs, node.cnsts

		r1, r2 = node.range
		p = self.get_expnt(max(abs(r1), abs(r2)))
		intv = self.get_intv(p, r1, r2)

		prog_1 = IR.Prog([])
		expr_1 = self.getTempVar()
		expr_1.inputVar = True

		#decls_1 = copy_dict(decls_0, {expr_1.idf : node.type})
		decls_1 = decls_0
		expts_1 = copy_dict(expts_0, {expr_1.idf :       p})
		intvs_1 = copy_dict(intvs_0, {expr_1.idf :    intv})

		return (prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_0)

	#=== re-structure ===#
	def visitTransp(self, node:AST.Transp):
		self.set_arg(node.expr, node)
		(prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)
		
		# decl fresh vars
		expr_2 = self.getTempVar()
		[i, j] = self.getTempIterators(2)

		# cmdl_for
		typ_2 = node.type

		cmd0 = IR.Comment(expr_1.idf + "^T")
		cmdl_for = IRUtil.loop(typ_2.shape, [i,j], 
			[IR.Assn(IRUtil.addIndex(expr_2, [i,j]),
				IRUtil.addIndex(expr_1, [j,i]))])

		# prog_for, p_2, intv_2
		prog_for = IR.Prog([cmd0] + cmdl_for)
		p_2 = expts_1[expr_1.idf]
		intv_2 = intvs_1[expr_1.idf]

		prog_2 = IRUtil.concatPrograms(prog_1, prog_for)
		decls_2 = copy_dict(decls_1, {expr_2.idf :  typ_2})
		expts_2 = copy_dict(expts_1, {expr_2.idf :    p_2})
		intvs_2 = copy_dict(intvs_1, {expr_2.idf : intv_2})

		return (prog_2, expr_2, decls_2, expts_2, intvs_2, cnsts_1)

	def visitReshape(self, node:AST.Reshape):

		self.set_arg(node.expr, node)
		(prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)

		# typ_{1,2}
		typ_1 = node.expr.type
		typ_2 = node.type

		# NOTE: the below handles only a special case (which appears in cnn).
		if True:
			assert((typ_1.dim == 1) or (typ_2.dim == 2 and typ_2.shape[0] == 1))
			
			# decl fresh vars
			expr_2 = self.getTempVar()
			if typ_1.dim == 1: typ_iters = typ_2
			else             : typ_iters = typ_1
			iters = self.getTempIterators(typ_iters.dim)
			
			# cmdl_for
			assert(typ_iters.dim == 4)
			idx = \
				IRUtil.add(iters[3],
				IRUtil.add(IRUtil.mul(iters[2], IR.Int(typ_iters.shape[3])),
				IRUtil.add(IRUtil.mul(iters[1], IR.Int(typ_iters.shape[3] * typ_iters.shape[2])),
					   IRUtil.mul(iters[0], IR.Int(typ_iters.shape[3] * typ_iters.shape[2] * typ_iters.shape[1])))))
			if typ_1.dim == 1:
				lhs = IRUtil.addIndex(expr_2, iters)
				rhs = IRUtil.addIndex(expr_1, [idx])
			else:
				lhs = IRUtil.addIndex(expr_2, [IRUtil.zero, idx])
				rhs = IRUtil.addIndex(expr_1, iters)
			cmdl_for = IRUtil.loop(typ_iters.shape, iters, [IR.Assn(lhs, rhs)])

			# prog_{decl,for}, p_2, intv_2
			prog_for = IR.Prog(cmdl_for)
			p_2 = expts_1[expr_1.idf]
			intv_2 = intvs_1[expr_1.idf]

			prog_2 = IRUtil.concatPrograms(prog_1, prog_for)
			decls_2 = copy_dict(decls_1, {expr_2.idf :  typ_2})
			expts_2 = copy_dict(expts_1, {expr_2.idf :    p_2})
			intvs_2 = copy_dict(intvs_1, {expr_2.idf : intv_2})

			return (prog_2, expr_2, decls_2, expts_2, intvs_2, cnsts_1)

		# NOTE: the below code handles general cases but is insecure (as
		# private values are used in array index).
		else:
			assert(False)
			# decl fresh vars
			expr_2 = self.getTempVar()
			carry = self.getTempVar()
			iters_2 = self.getTempIterators(typ_2.dim)
			iters_1 = self.getTempVars(typ_1.dim)

			# cmdl_decl
			cmdl_decl = [IR.Assn(itr, IRUtil.zero) for itr in iters_1]
			# cmdl_inc
			cmdl_inc = []
			for i in reversed(range(1,typ_1.dim)):
				itr_cur = iters_1[i]
				itr_prv = iters_1[i - 1]
				sz_cur = IR.Int(typ_1.shape[i])
				cmdl_inc = cmdl_inc + \
					[IR.Assn(carry  , IR.CExpr(IRUtil.eq(itr_cur, sz_cur), IRUtil.one         , IRUtil.zero)),
					 IR.Assn(itr_cur, IR.CExpr(IRUtil.eq(carry  , IRUtil.one), IRUtil.zero        , itr_cur)),
					 IR.Assn(itr_prv, IR.CExpr(IRUtil.eq(carry  , IRUtil.one), IRUtil.inc(itr_prv), itr_prv))]
			# cmdl_for
			itr_1_end = iters_1[typ_1.dim - 1]
			cmdl_for = IRUtil.loop(typ_2.shape, iters_2,
				[IR.Assn(IRUtil.addIndex(expr_2, iters_2),
						  IRUtil.addIndex(expr_1, iters_1)),
				 IR.Assn(itr_1_end, IRUtil.inc(itr_1_end))] + cmdl_inc)

			# prog_{decl,for}, p_2, intv_2
			prog_decl = IR.Prog(cmdl_decl)
			prog_for = IR.Prog(cmdl_for)
			p_2 = expts_1[expr_1.idf]
			intv_2 = intvs_1[expr_1.idf]

			prog_2 = IRUtil.concatPrograms(prog_1, prog_decl, prog_for)
			decls_2 = copy_dict(decls_1, {expr_2.idf :  typ_2})
			expts_2 = copy_dict(expts_1, {expr_2.idf :    p_2})
			intvs_2 = copy_dict(intvs_1, {expr_2.idf : intv_2})
			decls_2.update(dict((var.idf, Type.Int()) for var in tuple([carry] + iters_1)))

			return (prog_2, expr_2, decls_2, expts_2, intvs_2, cnsts_1)
	
	def visitMaxpool(self, node:AST.Maxpool):

		self.set_arg(node.expr, node)
		(prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)

		# decl fresh vars
		expr_2 = self.getTempVar()
		tmp = self.getTempVar()
		[i,j,k,l] = self.getTempIterators(4)
		[jd,kd] = self.getTempIterators(2)

		# cmdl_for
		M = node.dim
		def idx_1(e_jd:IR.Expr, e_kd:IR.Expr):
			return [i,
					IRUtil.add(IRUtil.mul(IR.Int(M), j), e_jd),
					IRUtil.add(IRUtil.mul(IR.Int(M), k), e_kd),
					l]
		# NOTE: we assume expr_1[..]>=0 always.
		#       (as Maxpool is used only in cnn, and Maxpool is used right
		#       after Relu)
		if True: rhs = IRUtil.max_uint(tmp, IRUtil.addIndex(expr_1, idx_1(jd, kd))) # if expr_1[..] is always >=0
		else   : rhs = IRUtil.max_sint(tmp, IRUtil.addIndex(expr_1, idx_1(jd, kd))) # if expr_1[..] can be <0
		cmdl_max = \
			[IR.Assn(tmp, IRUtil.addIndex(expr_1, idx_1(IRUtil.zero,IRUtil.zero)))] + \
			IRUtil.loop([M,M], [jd,kd], [IR.Assn(tmp, rhs)]) + \
			[IR.Assn(IRUtil.addIndex(expr_2, [i,j,k,l]), tmp)]
		typ_2 = node.type
		cmdl_for = IRUtil.loop(typ_2.shape, [i,j,k,l], cmdl_max)

		# prog_for, p_2
		prog_for = IR.Prog(cmdl_for)
		p_2 = expts_1[expr_1.idf]
		intv_2 = intvs_1[expr_1.idf]

		prog_2 = IRUtil.concatPrograms(prog_1, prog_for)
		decls_2 = copy_dict(decls_1, {expr_2.idf :  typ_2})
		expts_2 = copy_dict(expts_1, {expr_2.idf :    p_2})
		intvs_2 = copy_dict(intvs_1, {expr_2.idf : intv_2})
		decls_2.update({tmp.idf : Type.Int()})

		return (prog_2, expr_2, decls_2, expts_2, intvs_2, cnsts_1)
		
	def visitIndex(self, node:AST.Index):
		self.set_arg(node.expr, node)
		(prog_1, expr_1,  decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)
		self.set_arg2(node.index, decls_1, expts_1, intvs_1, cnsts_1)
		(prog_2, expr_2,  decls_2, expts_2, intvs_2, cnsts_2) = self.visit(node.index)

		prog_3 = IRUtil.concatPrograms(prog_1, prog_2)
		expr_3 = IRUtil.addIndex(expr_1, [expr_2])
#        out = StringIO(); sys.stdout = out; expr_2.print('c'); sys.stdout =
#        sys.__stdout__
#        expr_3 = IR.add_idx_str(expr_1, '[%s]' % out.getvalue())

		return (prog_3, expr_3, decls_2, expts_2, intvs_2, cnsts_2)

	#=== binary op ===#
	def visitUop(self, node:AST.Uop):   # {+,-}
		self.set_arg(node.expr, node)
		(prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)

		# op, typ_2
		op = node.op
		if     op == SeeDotParser.ADD: return (prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_1)
		assert(op == SeeDotParser.SUB)
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
			p_2 = expts_1[expr_1.idf]
			(m,M) = intvs_1[expr_1.idf]
			intv_2 = (-M,-m)

			prog_2 = IRUtil.concatPrograms(prog_1, prog_assn)
			decls_2 = copy_dict(decls_1, {expr_2.idf :  typ_2})
			expts_2 = copy_dict(expts_1, {expr_2.idf :    p_2})
			intvs_2 = copy_dict(intvs_1, {expr_2.idf : intv_2})

		return (prog_2, expr_2, decls_2, expts_2, intvs_2, cnsts_1)

	def visitBop1(self, node:AST.Bop1):
		op = node.op

		if    op == SeeDotParser.MUL:        return self.visitBopMul(node)
		elif  op == SeeDotParser.SPARSEMUL:  return self.visitBopSparseMul(node)
		elif  op == SeeDotParser.MULCIR:     return self.visitBopMulCir(node)
		elif  op == SeeDotParser.CONV:       return self.visitBopConv(node)
		elif  op == SeeDotParser.ADDCIR:     return self.visitBopAddOrSubCir(node)
		elif  op == SeeDotParser.SUBCIR:     return self.visitBopAddOrSubCir(node)
		else:                             assert(False)

	def visitBopMul(self, node:AST.Bop1):
		typ_1 = node.expr1.type
		typ_2 = node.expr2.type
		typ_3 = node.type

		if    Type.isInt(typ_3):  return self.visitBopMulInt(node)
		elif  typ_1.dim == 0:     return self.visitBopMul1DTensor(node)
		elif  typ_2.dim == 0:     return self.visitBopMul1DTensor(node)
		else:                     return self.visitBopMul2DTensor(node)

	def visitBopMulInt(self, node:AST.Bop1):

		self.set_arg(node.expr1, node)
		(prog_1, expr_1,  decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr1)
		self.set_arg2(node.expr2, decls_1, expts_1, intvs_1, cnsts_1)
		(prog_2, expr_2,  decls_2, expts_2, intvs_2, cnsts_2) = self.visit(node.expr2)

		prog_3 = IRUtil.concatPrograms(prog_1, prog_2)
		expr_3 = IRUtil.mul(expr_1, expr_2)
		decls_3 = decls_2
		expts_3 = expts_2
		intvs_3 = intvs_2

		return (prog_3, expr_3, decls_3, expts_3, intvs_3, cnsts_2)

	def visitBopMul1DTensor(self, node:AST.Bop1):

		self.set_arg(node.expr1, node)
		(prog_1, expr_1,  decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr1)
		self.set_arg2(node.expr2, decls_1, expts_1, intvs_1, cnsts_1)
		(prog_2, expr_2,  decls_2, expts_2, intvs_2, cnsts_2) = self.visit(node.expr2)

		typ_1 = node.expr1.type
		typ_2 = node.expr2.type
		typ_3 = node.type

		# decl fresh vars
		expr_3 = self.getTempVar()
		expr_1_shr = self.getTempVar()
		expr_2_shr = self.getTempVar()
		iters_1 = self.getTempIterators(typ_1.dim)
		iters_2 = self.getTempIterators(typ_2.dim)
		iters_3 = iters_1 if typ_1.dim > 0 else iters_2

		p1 = expts_1[expr_1.idf]
		p2 = expts_2[expr_2.idf]
		intv1 = intvs_1[expr_1.idf]
		intv2 = intvs_2[expr_2.idf]

		[shr1, shr2] = self.get_shr_mul(p1, p2)

		# cmdl_{shr1, shr2, mul}
		cmdl_shr1 = IRUtil.loop_shr(expr_1_shr, expr_1, typ_1.shape, iters_1, shr1)
		cmdl_shr2 = IRUtil.loop_shr(expr_2_shr, expr_2, typ_2.shape, iters_2, shr2)
		expr_1_shr_elt = IRUtil.addIndex(expr_1_shr, iters_1)
		expr_2_shr_elt = IRUtil.addIndex(expr_2_shr, iters_2)
		expr_3_elt = IRUtil.addIndex(expr_3, iters_3)
		rhs = IRUtil.mul(expr_1_shr_elt, expr_2_shr_elt)
		cmdl_mul = IRUtil.loop(typ_3.shape, iters_3, [IR.Assn(expr_3_elt, rhs)])

		# prog_assn, p_3, intv_3
		prog_assn = IR.Prog(cmdl_shr1 + cmdl_shr2 + cmdl_mul)
		p_3 = self.get_expnt_mul(p1, shr1, p2, shr2)
		intv_3 = self.get_intv_mul(intv1, shr1, intv2, shr2)
					
		prog_3 = IRUtil.concatPrograms(prog_1, prog_2, prog_assn)
		decls_3 = copy_dict(decls_2, {expr_3.idf :  typ_3})
		expts_3 = copy_dict(expts_2, {expr_3.idf :    p_3})
		intvs_3 = copy_dict(intvs_2, {expr_3.idf : intv_3})
		decls_3.update({expr_1_shr.idf : typ_1,
						expr_2_shr.idf : typ_2})

		return (prog_3, expr_3, decls_3, expts_3, intvs_3, cnsts_2)

	def visitBopMul2DTensor(self, node:AST.Bop1):

		self.set_arg(node.expr1, node)
		(prog_1, expr_1,  decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr1)
		self.set_arg2(node.expr2, decls_1, expts_1, intvs_1, cnsts_1)
		(prog_2, expr_2,  decls_2, expts_2, intvs_2, cnsts_2) = self.visit(node.expr2)

		typ_1 = node.expr1.type
		typ_2 = node.expr2.type
		typ_3 = node.type

		[I, J] = typ_1.shape
		[J, K] = typ_2.shape
		typ_mul = Type.Tensor([I,K,J])

		# decl fresh vars
		expr_3 = self.getTempVar()
		expr_1_shr = self.getTempVar()
		expr_2_shr = self.getTempVar()
		expr_mul = self.getTempVar()  # I*K*J
		[i,j,k] = self.getTempIterators(3)

		p1 = expts_1[expr_1.idf]
		p2 = expts_2[expr_2.idf]
		intv1 = intvs_1[expr_1.idf]
		intv2 = intvs_2[expr_2.idf]

		[shr1, shr2] = self.get_shr_mul(p1, p2)

		# cmdl_{shr1, shr2, mul}
		cmdl_shr1 = IRUtil.loop_shr(expr_1_shr, expr_1, [I,J], [i,j], shr1)
		cmdl_shr2 = IRUtil.loop_shr(expr_2_shr, expr_2, [J,K], [j,k], shr2)
		expr_1_shr_elt = IRUtil.addIndex(expr_1_shr, [i,j])
		expr_2_shr_elt = IRUtil.addIndex(expr_2_shr, [j,k])
		expr_mul_elt = IRUtil.addIndex(expr_mul  , [i,k,j])
		rhs = IRUtil.mul(expr_1_shr_elt, expr_2_shr_elt)
		cmdl_mul = IRUtil.loop([I,K,J], [i,k,j], [IR.Assn(expr_mul_elt, rhs)])
		p_mul = self.get_expnt_mul(p1, shr1, p2, shr2)
		intv_mul = self.get_intv_mul(intv1, shr1, intv2, shr2)

		# p_3, intv_3, cmdl_sum, decls_sum
		(p_3, H_1, H_2) = self.get_expnt_sum(p_mul, J)
		intv_3 = self.get_intv_sum(intv_mul, J)
		(cmdl_sum_body, decls_sum) = \
			self.compile_sum(J, H_1, H_2,
				IRUtil.addIndex(expr_mul, [i,k]),
				IRUtil.addIndex(expr_3  , [i,k]), False)
		cmdl_sum = IRUtil.loop([I,K], [i,k], cmdl_sum_body)
					
		# prog_assn
		prog_assn = IR.Prog(cmdl_shr1 + cmdl_shr2 + cmdl_mul + cmdl_sum)
					
		prog_3 = IRUtil.concatPrograms(prog_1, prog_2, prog_assn)
		decls_3 = copy_dict(decls_2, {expr_3.idf :  typ_3})
		expts_3 = copy_dict(expts_2, {expr_3.idf :    p_3})
		intvs_3 = copy_dict(intvs_2, {expr_3.idf : intv_3})
		decls_3.update({expr_1_shr.idf : typ_1,
						expr_2_shr.idf : typ_2,
						expr_mul  .idf : typ_mul})
		decls_3.update(decls_sum)

		return (prog_3, expr_3, decls_3, expts_3, intvs_3, cnsts_2)

	def visitBopMulCir(self, node:AST.Bop1):

		self.set_arg(node.expr1, node)
		(prog_1, expr_1,  decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr1)
		self.set_arg2(node.expr2, decls_1, expts_1, intvs_1, cnsts_1)
		(prog_2, expr_2,  decls_2, expts_2, intvs_2, cnsts_2) = self.visit(node.expr2)

		typ_1 = node.expr1.type
		typ_2 = node.expr2.type
		typ_3 = node.type

		# decl fresh vars
		expr_3 = self.getTempVar()
		expr_1_shr = self.getTempVar()
		expr_2_shr = self.getTempVar()
		iters = self.getTempIterators(typ_3.dim)

		p1 = expts_1[expr_1.idf]
		p2 = expts_2[expr_2.idf]
		intv1 = intvs_1[expr_1.idf]
		intv2 = intvs_2[expr_2.idf]

		[shr1, shr2] = self.get_shr_mul(p1, p2)

		# cmdl_*
		cmdl_shr1 = IRUtil.loop_shr(expr_1_shr, expr_1, typ_3.shape, iters, shr1)
		cmdl_shr2 = IRUtil.loop_shr(expr_2_shr, expr_2, typ_3.shape, iters, shr2)
		expr_1_shr_elt = IRUtil.addIndex(expr_1_shr, iters)
		expr_2_shr_elt = IRUtil.addIndex(expr_2_shr, iters)
		expr_3_elt = IRUtil.addIndex(expr_3    , iters)
		rhs = IRUtil.mul(expr_1_shr_elt, expr_2_shr_elt)
		cmdl_mul = IRUtil.loop(typ_3.shape, iters, [IR.Assn(expr_3_elt, rhs)])

		# prog_assn, p_3, intv_3
		prog_assn = IR.Prog(cmdl_shr1 + cmdl_shr2 + cmdl_mul)
		
		p_3 = self.get_expnt_mul(p1, shr1, p2, shr2)
		intv_3 = self.get_intv_mul(intv1, shr1, intv2, shr2)
		#p_3 = self.get_expnt_mul(expts_1[expr_1.idf], expts_2[expr_2.idf])
		#intv_3 = self.get_intv_mul(intvs_1[expr_1.idf], intvs_2[expr_2.idf])

		prog_3 = IRUtil.concatPrograms(prog_1, prog_2, prog_assn)
		decls_3 = copy_dict(decls_2, {expr_3.idf :  typ_3})
		expts_3 = copy_dict(expts_2, {expr_3.idf :    p_3})
		intvs_3 = copy_dict(intvs_2, {expr_3.idf : intv_3})
		decls_3.update({expr_1_shr.idf : typ_3,
						expr_2_shr.idf : typ_3})

		return (prog_3, expr_3, decls_3, expts_3, intvs_3, cnsts_2)

	def visitBopConv(self, node:AST.Bop1):

		self.set_arg(node.expr1, node)
		(prog_1, expr_1,  decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr1)
		self.set_arg2(node.expr2, decls_1, expts_1, intvs_1, cnsts_1)
		(prog_2, expr_2,  decls_2, expts_2, intvs_2, cnsts_2) = self.visit(node.expr2)

		typ_1 = node.expr1.type
		typ_2 = node.expr2.type
		typ_3 = node.type

		[N , H , W , CI] = typ_1.shape
		[HF, WF, CI, CO] = typ_2.shape

		# pad-related
		pad_h = HF - 1
		pad_top = pad_h // 2
		pad_w = WF - 1
		pad_left = pad_w // 2
		typ_1_pad = Type.Tensor([N, H + pad_h, W + pad_w, CI])
		typ_mul = Type.Tensor([N, H, W, CI, HF * WF])

		# decl fresh vars
		expr_3 = self.getTempVar()  # : typ_3
		expr_1_shr = self.getTempVar()  # : typ_1
		expr_1_pad = self.getTempVar()  # : typ_1_pad
		expr_2_shr = self.getTempVar()  # : typ_2
		expr_mul = self.getTempVar()  # : typ_mul
		[h_tmp, w_tmp, cnt] = self.getTempVars(3)  # : Int
		[n, h, w, ci] = self.getTempIterators(4)
		[hf, wf, co] = self.getTempIterators(3)

		p1 = expts_1[expr_1.idf]
		p2 = expts_2[expr_2.idf]
		intv1 = intvs_1[expr_1.idf]
		intv2 = intvs_2[expr_2.idf]

		[shr1, shr2] = self.get_shr_mul(p1, p2)

		# cmdl_shr
		cmdl_shr = \
			IRUtil.loop_shr(expr_1_shr, expr_1, [N ,H ,W ,CI], [n ,h ,w ,ci], shr1) + \
			IRUtil.loop_shr(expr_2_shr, expr_2, [HF,WF,CI,CO], [hf,wf,ci,co], shr2)

		# cmdl_pad
		src = IRUtil.addIndex(expr_1_shr, [n,
										IRUtil.sub(h, IR.Int(pad_top)),
										IRUtil.sub(w, IR.Int(pad_left)),
										ci])
		dst = IRUtil.addIndex(expr_1_pad, [n,h,w,ci])
		cond = IRUtil.andd(IRUtil.lt(IRUtil.sub(h_tmp,IR.Int(pad_top)), IR.Int(H)),
			IRUtil.lt(IRUtil.sub(w_tmp,IR.Int(pad_left)), IR.Int(W)))
		rhs = IR.CExpr(cond, src, IRUtil.zero)
		cmdl_pad = \
			IRUtil.loop([N], [n],
				[IR.Assn(h_tmp, IRUtil.zero)] + \
				IRUtil.loop([H], [h],
					[IR.Assn(w_tmp, IRUtil.zero)] + \
					IRUtil.loop([W], [w],
						IRUtil.loop([CI], [ci],
							[IR.Assn(dst, rhs)]) + \
					[IR.Assn(w_tmp, IRUtil.inc(w_tmp))]) + \
				[IR.Assn(h_tmp, IRUtil.inc(h_tmp))]))
			
		# cmdl_mul
		expr_1_pad_elt = IRUtil.addIndex(expr_1_pad, [n, IRUtil.add(h,hf), IRUtil.add(w,wf), ci])
		expr_2_shr_elt = IRUtil.addIndex(expr_2_shr, [hf, wf, ci, co])
		lhs = IRUtil.addIndex(expr_mul  , [n,  h,  w, co, cnt])
		rhs = IRUtil.mul(expr_1_pad_elt, expr_2_shr_elt)
		cmdl_mul = \
			IRUtil.loop([N,H,W,CO], [n,h,w,co],
				[IR.Assn(cnt, IRUtil.zero)] + \
				IRUtil.loop([HF,WF,CI], [hf,wf,ci],
					[IR.Assn(lhs, rhs),
						IR.Assn(cnt, IRUtil.inc(cnt))]))
		p_mul = self.get_expnt_mul(p1, shr1, p2, shr2)
		intv_mul = self.get_intv_mul(intv1, shr1, intv2, shr2)

		# p_3, intv_3, cmdl_sum, decls_sum
		(p_3, H_1, H_2) = self.get_expnt_sum(p_mul, HF * WF * CI)
		intv_3 = self.get_intv_sum(intv_mul, HF * WF * CI)
		(cmdl_sum_body, decls_sum) = \
			self.compile_sum(HF * WF, H_1, H_2,
				IRUtil.addIndex(expr_mul, [n,h,w,co]),
				IRUtil.addIndex(expr_3  , [n,h,w,co]), False)
		cmdl_sum = IRUtil.loop([N,H,W,CO], [n,h,w,co], cmdl_sum_body)

		# prog_assn
		prog_assn = IR.Prog(cmdl_shr + cmdl_pad + cmdl_mul + cmdl_sum)
					
		prog_3 = IRUtil.concatPrograms(prog_1, prog_2, prog_assn)
		decls_3 = copy_dict(decls_2, {expr_3.idf :  typ_3})
		expts_3 = copy_dict(expts_2, {expr_3.idf :    p_3})
		intvs_3 = copy_dict(intvs_2, {expr_3.idf : intv_3})
		decls_3.update({expr_1_shr.idf : typ_1,
						expr_1_pad.idf : typ_1_pad,
						expr_2_shr.idf : typ_2,
						expr_mul  .idf : typ_mul})
		decls_3.update(dict((var.idf, Type.Int()) for var in [h_tmp, w_tmp, cnt]))
		decls_3.update(decls_sum)

		return (prog_3, expr_3, decls_3, expts_3, intvs_3, cnsts_2)

	def visitBopAddOrSubCir(self, node:AST.Bop1):

		self.set_arg(node.expr1, node)
		(prog_1, expr_1,  decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr1)
		self.set_arg2(node.expr2, decls_1, expts_1, intvs_1, cnsts_1)
		(prog_2, expr_2,  decls_2, expts_2, intvs_2, cnsts_2) = self.visit(node.expr2)

		op = node.op
		typ_1 = node.expr1.type
		typ_2 = node.expr2.type
		typ_3 = node.type

		# decl fresh vars
		expr_3 = self.getTempVar()
		expr_1_shr = self.getTempVar()
		expr_2_shr = self.getTempVar()
		iters = self.getTempIterators(typ_3.dim)

		# p_3, intv_3, shr_n*
		if   op == SeeDotParser.ADDCIR: (op_ir, op_fn) = (IR.Op.Op['+'], operator.add)
		elif op == SeeDotParser.SUBCIR: (op_ir, op_fn) = (IR.Op.Op['-'], operator.sub)
		(p_3, intv_3, [shr_n1, shr_n2, shr_n3]) = self.get_expnt_intv_add(expts_1[expr_1.idf], expts_2[expr_2.idf],
										intvs_1[expr_1.idf], intvs_2[expr_2.idf], op_fn)
			
		# prog_assn
		expr_1_elt = IRUtil.addIndex(expr_1,  iters)
		expr_2_elt = IRUtil.addIndex(expr_2, [iters[-1]])
		expr_3_elt = IRUtil.addIndex(expr_3,  iters)
		rhs_1 = IRUtil.shr(expr_1_elt, shr_n1)
		rhs_2 = IRUtil.shr(expr_2_elt, shr_n2)
		rhs_3 = IR.IntBop(expr_1_shr, op_ir, expr_2_shr)
		rhs_4 = IRUtil.shr(expr_3_elt, shr_n3)
		cmdl_assn = IRUtil.loop(typ_3.shape, iters,
			[IR.Assn(expr_1_shr, rhs_1),
				IR.Assn(expr_2_shr, rhs_2),
				IR.Assn(expr_3_elt, rhs_3),
				IR.Assn(expr_3_elt, rhs_4)])

		comment = IR.Comment(expr_1.idf + " <" + op_ir.name + "> " + expr_2.idf)
		prog_assn = IR.Prog([comment] + cmdl_assn)
			
		prog_3 = IRUtil.concatPrograms(prog_1, prog_2, prog_assn)
		decls_3 = copy_dict(decls_2, {expr_3.idf :  typ_3})
		expts_3 = copy_dict(expts_2, {expr_3.idf :    p_3})
		intvs_3 = copy_dict(intvs_2, {expr_3.idf : intv_3})
		decls_3.update({expr_1_shr.idf : Type.Int(),
						expr_2_shr.idf : Type.Int()})

		return (prog_3, expr_3, decls_3, expts_3, intvs_3, cnsts_2)

	def visitBop2(self, node:AST.Bop2):  # {+, -}
		self.set_arg(node.expr1, node)
		(prog_1, expr_1,  decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr1)
		self.set_arg2(node.expr2, decls_1, expts_1, intvs_1, cnsts_1)
		(prog_2, expr_2,  decls_2, expts_2, intvs_2, cnsts_2) = self.visit(node.expr2)

		# op_ir, typ_3
		op = node.op
		if   op == SeeDotParser.ADD: (op_ir, op_fn) = (IR.Op.Op['+'], operator.add)
		elif op == SeeDotParser.SUB: (op_ir, op_fn) = (IR.Op.Op['-'], operator.sub)
		typ_3 = node.type

		# e : Int
		if Type.isInt(typ_3):
			prog_3 = IRUtil.concatPrograms(prog_1, prog_2)
			expr_3 = IR.IntBop(expr_1, op_ir, expr_2)
			decls_3 = decls_2
			expts_3 = expts_2
			intvs_3 = intvs_2

		# e : Tensor(), or Tensor(..)
		else:
			# decl fresh vars
			expr_3 = self.getTempVar()
			expr_1_shr = self.getTempVar()
			expr_2_shr = self.getTempVar()
			iters = self.getTempIterators(typ_3.dim)
			
			# p_3, intv_3, shr_n*
			(p_3, intv_3, [shr_n1, shr_n2, shr_n3]) = self.get_expnt_intv_add(expts_1[expr_1.idf], expts_2[expr_2.idf],
										  intvs_1[expr_1.idf], intvs_2[expr_2.idf], op_fn)

			# prog_assn
			expr_1_elt = IRUtil.addIndex(expr_1, iters)
			expr_2_elt = IRUtil.addIndex(expr_2, iters)
			expr_3_elt = IRUtil.addIndex(expr_3, iters)
			
			[tmp1, tmp2] = self.getTempVars(2)
			assn1 = IR.Assn(tmp1, expr_1_elt)
			assn2 = IR.Assn(tmp2, expr_2_elt)

			rhs_1 = IRUtil.shr(tmp1, shr_n1)
			rhs_2 = IRUtil.shr(tmp2, shr_n2)
			rhs_3 = IR.IntBop(expr_1_shr, op_ir, expr_2_shr)
			rhs_4 = IRUtil.shr(expr_3_elt, shr_n3)
			cmdl_assn = IRUtil.loop(typ_3.shape, iters,
				[assn1, assn2, IR.Assn(expr_1_shr, rhs_1),
				 IR.Assn(expr_2_shr, rhs_2),
				 IR.Assn(expr_3_elt, rhs_3),
				 IR.Assn(expr_3_elt, rhs_4)])

			cmd0 = IR.Comment(expr_1.idf + ' ' + op_ir.name + ' ' + expr_2.idf)

			if expr_2.idf == 'Min':
				iters = self.getTempIterators(typ_3.dim)
				p = IRUtil.print_loop(typ_3.shape, iters, [IR.PrintAsFloat(IRUtil.addIndex(expr_3, iters), p_3)])
				p = []
			else:
				p = []

			prog_assn = IR.Prog([cmd0] + cmdl_assn + p)

			prog_3 = IRUtil.concatPrograms(prog_1, prog_2, prog_assn)
			decls_3 = copy_dict(decls_2, {expr_3.idf :  typ_3})
			expts_3 = copy_dict(expts_2, {expr_3.idf :    p_3})
			intvs_3 = copy_dict(intvs_2, {expr_3.idf : intv_3})
			decls_3.update({expr_1_shr.idf : Type.Int(),
							expr_2_shr.idf : Type.Int(),
							tmp1.idf : Type.Int(),
							tmp2.idf : Type.Int()
							})

		return (prog_3, expr_3, decls_3, expts_3, intvs_3, cnsts_2)

	def visitFunc(self, node:AST.Func):
		op = node.op
		
		if    op == SeeDotParser.RELU:    return self.visitRelu(node)
		elif  op == SeeDotParser.EXP:     return self.visitExp(node)
		elif  op == SeeDotParser.ARGMAX:  return self.visitArgMax(node)
		elif  op == SeeDotParser.SGN:     return self.visitSgn(node)
		elif  op == SeeDotParser.TANH:    return self.visitTanh(node)
		else:                          assert False

	def visitRelu(self, node:AST.Func):

		self.set_arg(node.expr, node)
		(prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)

		typ_1 = node.expr.type

		# decl fresh vars
		expr_2 = self.getTempVar()
		iters = self.getTempIterators(typ_1.dim)

		# prog_for
		expr_1_elt = IRUtil.addIndex(expr_1, iters)
		expr_2_elt = IRUtil.addIndex(expr_2, iters)
		rhs = IRUtil.relu(expr_1_elt)
		cmdl_for = IRUtil.loop(typ_1.shape, iters, [IR.Assn(expr_2_elt, rhs)])
		prog_for = IR.Prog(cmdl_for)

		# p_2, intv_2
		p_2 = expts_1[expr_1.idf]
		(m,M) = intvs_1[expr_1.idf]
		intv_2 = (0,M)
			
		prog_2 = IRUtil.concatPrograms(prog_1, prog_for)
		decls_2 = copy_dict(decls_1, {expr_2.idf :  typ_1})
		expts_2 = copy_dict(expts_1, {expr_2.idf :    p_2})
		intvs_2 = copy_dict(intvs_1, {expr_2.idf : intv_2})

		return (prog_2, expr_2, decls_2, expts_2, intvs_2, cnsts_1)

	def visitExp(self, node:AST.Func):

		# Change shift right for multiplication
		assert False

		self.readProfileFile()

		self.set_arg(node.expr, node)
		(prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)

		typ_1 = node.expr.type

		# decl fresh vars
		expr_2 = self.getTempVar()
		''' iters  = self.getTempIterators(typ_1.dim) '''

		# p_2, intv_2
		p_1 = expts_1[expr_1.idf]  # TODO: use MAX_VAL_EXP
		intv_1 = intvs_1[expr_1.idf]  # TODO: use MAX_VAL_EXP
		# NOTE: we assume e<=0 for any e with exp(e).
		if True: p_2 = self.get_expnt_exp(p_1, 'neg')  # expr_1 is always <0
		else:    p_2 = self.get_expnt_exp(p_1, 'pos')  # expr_1 can be >=0 or <0
		intv_2 = self.get_intv_exp(p_2, intv_1)

		# prog_exp
		expr_1_elt = IRUtil.addIndex(expr_1, [IRUtil.zero] * typ_1.dim)
		expr_2_elt = IRUtil.addIndex(expr_2, [IRUtil.zero] * typ_1.dim)
		(cmdl_load, rhs, decls_load) = self.compile_exp(expr_1_elt, p_1)
		cmdl_assn = [IR.Assn(expr_2_elt, rhs)]
		prog_exp = IR.Prog(cmdl_load + cmdl_assn)
		'''
		# prog_for
		expr_1_elt = IRUtil.addIndex(expr_1, iters)
		expr_2_elt = IRUtil.addIndex(expr_2, iters)
		(rhs, decls_exp) = self.compile_exp(expr_1_elt, p_1)
		cmdl_for   = IR.loop(typ_1.shape, iters, [IR.Assn(expr_2_elt, rhs)])
		prog_for   = IR.Prog(cmdl_for)
		'''
			
		prog_2 = IRUtil.concatPrograms(prog_1, prog_exp)
		decls_2 = copy_dict(decls_1, {expr_2.idf :  typ_1})
		expts_2 = copy_dict(expts_1, {expr_2.idf :    p_2})
		intvs_2 = copy_dict(intvs_1, {expr_2.idf : intv_2})
		decls_2.update(decls_load)

		return (prog_2, expr_2, decls_2, expts_2, intvs_2, cnsts_1)

	def visitArgMax(self, node:AST.Func):

		self.set_arg(node.expr, node)
		(prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)

		typ_1 = node.expr.type

		# decl fresh vars
		curind = self.getTempVar()
		maxind = self.getTempVar()
		maxval = self.getTempVar()
		newmax = self.getTempVar()
		iters = self.getTempIterators(typ_1.dim)

		# prog_tot
		expr_1_fst = IRUtil.addIndex(expr_1, [IRUtil.zero] * len(iters))
		expr_1_elt = IRUtil.addIndex(expr_1, iters)
		cmdl_decl = \
			[IR.Assn(curind, IRUtil.zero),
				IR.Assn(maxind, IRUtil.zero),
				IR.Assn(maxval, expr_1_fst)]
		cmdl_assn = IRUtil.loop(typ_1.shape, iters,
			[IR.Assn(newmax, IRUtil.max_sint(maxval, expr_1_elt)),
				IR.Assn(maxind, IR.CExpr(IRUtil.eq(maxval, newmax), maxind, curind)),
				IR.Assn(maxval, newmax),
				IR.Assn(curind, IRUtil.inc(curind))])
		prog_tot = IR.Prog(cmdl_decl + cmdl_assn)
			
		prog_2 = IRUtil.concatPrograms(prog_1, prog_tot)
		expr_2 = maxind
		decls_2 = copy_dict(decls_1, dict((var.idf, Type.Int()) for var in [curind, maxind, maxval, newmax]))
		expts_2 = expts_1
		intvs_2 = intvs_1

		return (prog_2, expr_2, decls_2, expts_2, intvs_2, cnsts_1)

	def visitSgn(self, node:AST.Func):

		self.set_arg(node.expr, node)
		(prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)

		typ_1 = node.expr.type

		expr_1_elt = IRUtil.addIndex(expr_1, [IRUtil.zero] * typ_1.dim)
		prog_2 = prog_1
		expr_2 = IRUtil.cond_zero(expr_1_elt, IRUtil.one, IRUtil.zero) # TODO: it this correct?
		decls_2 = decls_1
		expts_2 = expts_1
		intvs_2 = intvs_1
		
		return (prog_2, expr_2, decls_2, expts_2, intvs_2, cnsts_1)

	def visitSum(self, node:AST.Sum):
		(decls_0, expts_0, intvs_0, cnsts_0) = node.decls, node.expts, node.intvs, node.cnsts
		i_idf = node.name
		decls_0 = copy_dict(decls_0, {i_idf : Type.Int()})
		self.set_arg2(node.expr, decls_0, expts_0, intvs_0, cnsts_0)
		(prog_1, expr_1,  decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)

		# i_{st,ed}, typ_2, typ_1_all
		i_st, i_ed = node.start, node.end
		typ_2 = node.type
		typ_1_all = Type.Tensor([i_ed - i_st] + typ_2.shape)
		
		# decl fresh vars
		expr_2 = self.getTempVar()  # : typ_2
		expr_1_all = self.getTempVar()  # : typ_1_all
		#i_var = IR.Var(i_idf)
												 # i_iter =
																						  # self.get_fresh_iter()
		iters = self.getTempIterators(typ_2.dim)

		# cmdl_for
		cmdl_for = []
		for i in range(i_st, i_ed):
			prog_1_subst = \
				prog_1.subst(i_idf, IR.Int(i))\
					  .subst(expr_1.idf, IRUtil.addIndex(expr_1_all, [IR.Int(i - i_st)]))
			cmdl_for += prog_1_subst.cmd_l
		'''
		prog_1 = prog_1.subst(expr_1.idf, IR.addIndex(expr_1_all, [i_iter]))
		cmdl_for =\
			[IR.Assn(i_var, IR.Int(i_st)),
			 IR.For(i_iter, 0, i_ed-i_st,
				prog_1.cmd_l +\
				[IR.Assn(i_var, IRUtil.inc(i_var))]
			 )]
		'''

		# p_2, intv_2, cmdl_sum, decls_sum
		(p_2, H_1, H_2) = self.get_expnt_sum(expts_1[expr_1.idf], i_ed - i_st)
		intv_2 = self.get_intv_sum(intvs_1[expr_1.idf], i_ed - i_st)
		(cmdl_sum_body, decls_sum) = \
			self.compile_sum(i_ed - i_st, H_1, H_2,
				IRUtil.addIndex(expr_1_all, iters),
				IRUtil.addIndex(expr_2    , iters), True)
		cmdl_sum = IRUtil.loop(typ_2.shape, iters, cmdl_sum_body)
		
		prog_2 = IR.Prog(cmdl_for + cmdl_sum)
		decls_2 = copy_dict(decls_1, {expr_2.idf :  typ_2})
		expts_2 = copy_dict(expts_1, {expr_2.idf :    p_2})
		intvs_2 = copy_dict(intvs_1, {expr_2.idf : intv_2})
		decls_2.update({expr_1_all.idf : typ_1_all})
		decls_2.update(decls_sum)

		return (prog_2, expr_2, decls_2, expts_2, intvs_2, cnsts_1)
		
	def visitCond(self, node:AST.Cond):
		self.set_arg(node.expr, node)
		(prog_1, expr_1,  decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)
		self.set_arg2(node.trueBlock, decls_1, expts_1, intvs_1, cnsts_1)
		(prog_2, expr_2,  decls_2, expts_2, intvs_2, cnsts_2) = self.visit(node.trueBlock)
		self.set_arg2(node.falseBlock, decls_2, expts_2, intvs_2, cnsts_2)
		(prog_3, expr_3,  decls_3, expts_3, intvs_3, cnsts_3) = self.visit(node.falseBlock)

		typ_1 = node.expr.type
		typ_2 = node.trueBlock.type
		if Type.isInt(typ_1): expr_1_elt = expr_1
		else                      : expr_1_elt = IRUtil.addIndex(expr_1, [IRUtil.zero] * typ_1.dim)
		
		# e2,e3 : Int
		if Type.isInt(typ_2):
			prog_4 = IRUtil.concatPrograms(prog_1, prog_2, prog_3)
			expr_4 = IRUtil.cond_zero(expr_1_elt, expr_2, expr_3)
			decls_4 = decls_3
			expts_4 = expts_3
			intvs_4 = intvs_3

		# e2,e3 : Tensor(), or Tensor(..)
		else:
			# decl fresh vars
			expr_4 = self.getTempVar()
			iters = self.getTempIterators(typ_2.dim)

			# p_4, intv_4
			(p_2,    p_3) = (expts_2[expr_2.idf], expts_3[expr_3.idf])
			(intv_2, intv_3) = (intvs_2[expr_2.idf], intvs_3[expr_3.idf])
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
			
			prog_4 = IRUtil.concatPrograms(prog_1, prog_2, prog_3, prog_assn)
			decls_4 = copy_dict(decls_3, {expr_4.idf :  typ_2})
			expts_4 = copy_dict(expts_3, {expr_4.idf :    p_4})
			intvs_4 = copy_dict(intvs_3, {expr_4.idf : intv_4})

		return (prog_4, expr_4, decls_4, expts_4, intvs_4, cnsts_3)

	def visitLet(self, node:AST.Let):
		self.set_arg(node.decl, node)
		(prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.decl)
		typ_1 = node.decl.type
		idf = node.name

		# e1 : Int
		if Type.isInt(typ_1):
			decls_1 = copy_dict(decls_1, {idf : Type.Int()})
			self.set_arg2(node.expr, decls_1, expts_1, intvs_1, cnsts_1)
			(prog_2, expr_2,  decls_2, expts_2, intvs_2, cnsts_2) = self.visit(node.expr)

			prog_assn = IR.Prog([IR.Assn(IR.Var(idf), expr_1)])
			prog_3 = IRUtil.concatPrograms(prog_1, prog_assn, prog_2)

			return (prog_3, expr_2, decls_2, expts_2, intvs_2, cnsts_2)
		# e1 : Tensor{(),(..)}
		else:
			expts_1 = copy_dict(expts_1, {idf : expts_1[expr_1.idf]})
			intvs_1 = copy_dict(intvs_1, {idf : intvs_1[expr_1.idf]})

			if isinstance(node.decl, AST.Decl):
				self.VAR_IDF_INIT.append(idf)
				decls_1[idf] = node.decl.type
				expr_1.idf = idf
				expr_1.inputVar = True

			self.set_arg2(node.expr, decls_1, expts_1, intvs_1, cnsts_1)
			(prog_2, expr_2,  decls_2, expts_2, intvs_2, cnsts_2) = self.visit(node.expr)

			prog_2 = prog_2.subst(idf, expr_1)
			expr_2 = expr_2.subst(idf, expr_1)
			prog_3 = IRUtil.concatPrograms(prog_1, prog_2)

			return (prog_3, expr_2, decls_2, expts_2, intvs_2, cnsts_2)

	#================
	# Print functions
	#================

	# Print the compiled code (IR)
	def print(self, writer, prog:IR.Prog, expr:IR.Expr,	decls:dict, expnts:dict, intvs:dict, cnsts:dict):
		self._out_prefix(writer, decls, cnsts)
		CodegenBase(writer).print(prog)
		#prog.print(writer)
		self._out_suffix(writer, expr, decls, expnts)

	def printVarDecls(self, writer, decls:dict):
		for decl in decls:
			typ_str = IR.DataType.getIntStr()
			idf_str = decl
			type = decls[decl]
			if Type.isInt(type): shape_str = ''
			elif Type.isTensor(type): shape_str = ''.join(['[' + str(n) + ']' for n in typeshape])
			writer.printf('%s %s%s;\n', typ_str, idf_str, shape_str, indent=True)
		writer.printf('\n')

	def printConstDecls(self, writer, cnsts:dict):
		for cnst in cnsts:
			var, num = cnst, cnsts[cnst]
			if np.iinfo(np.int16).min <= num <= np.iinfo(np.int16).max:
				writer.printf('%s = %d;\n', var, num, indent=True)
			elif np.iinfo(np.int32).min <= num <= np.iinfo(np.int32).max:
				writer.printf('%s = %dL;\n', var, num, indent=True)
			elif np.iinfo(np.int64).min <= num <= np.iinfo(np.int64).max:
				writer.printf('%s = %dLL;\n', var, num, indent=True)
			else:
				assert False
