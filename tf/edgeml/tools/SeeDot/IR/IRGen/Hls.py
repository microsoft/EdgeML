# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import numpy as np
import operator

import AST.AST as AST
from Antlr.SeeDotParser  import SeeDotParser

import Common

from IR.IRGen.IRGenBase import IRGenBase
import Type as Type
import IR.IR as IR
import IR.IRUtil as IRUtil
from Util import *
from Writer import Writer
import IR.IRGen.ResourceEstimation as REst

class Hls(IRGenBase):

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

	def visitReshape(self, node:AST.Reshape):

		self.set_arg(node.expr, node)
		(prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)

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
		p_2 = expts_1[expr_1.idf]
		intv_2 = intvs_1[expr_1.idf]

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
		prog_2 = IRUtil.concatPrograms(prog_1, reshape_prog)

		# Update context
		decls_2 = copy_dict(decls_1, {expr_2.idf :  typ_2})
		expts_2 = copy_dict(expts_1, {expr_2.idf :    p_2})
		intvs_2 = copy_dict(intvs_1, {expr_2.idf : intv_2})
		
		# Update declarations
		decls_2.update(dict((var.idf, Type.Int()) for var in iters_2))

		return (prog_2, expr_2, decls_2, expts_2, intvs_2, cnsts_1)

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
		expr_1_shr = self.getTempVar()
		expr_2_shr = self.getTempVar()
		sum = self.getTempVar()
		iters = self.getTempIterators(typ_3.dim)

		# p_3, intv_3, shr_n*
		if   op == SeeDotParser.ADDCIR: (op_ir, op_fn) = (IR.Op.Op['+'], operator.add)
		elif op == SeeDotParser.SUBCIR: (op_ir, op_fn) = (IR.Op.Op['-'], operator.sub)
		(p_3, intv_3, [shr_n1, shr_n2, shr_n3]) = self.get_expnt_intv_add(expts_1[expr_1.idf], expts_2[expr_2.idf],
										intvs_1[expr_1.idf], intvs_2[expr_2.idf], op_fn)
			
		# prog_assn
		expr_1_elt = IRUtil.addIndex(expr_1,  iters)
		expr_2_elt = IRUtil.addIndex(expr_2, [iters[-1]])
		rhs_1 = IRUtil.shr(expr_1_elt, shr_n1)
		rhs_2 = IRUtil.shr(expr_2_elt, shr_n2)
		rhs_3 = IR.IntBop(expr_1_shr, op_ir, expr_2_shr)
		rhs_4 = IRUtil.shr(sum, shr_n3)
		cmdl_assn = IRUtil.loop(typ_3.shape, iters,
			[IR.Assn(expr_1_shr, rhs_1),
				IR.Assn(expr_2_shr, rhs_2),
				IR.Assn(sum, rhs_3),
				IR.Assn(expr_1_elt, rhs_4)])

		comment = IR.Comment(expr_1.idf + " <" + op_ir.name + "> " + expr_2.idf)
		prog_assn = IR.Prog([comment] + cmdl_assn)
			
		prog_3 = IRUtil.concatPrograms(prog_1, prog_2, prog_assn)
		#decls_3 = copy_dict(decls_2, {expr_1.idf :  typ_3})
		decls_3 = decls_2
		expts_3 = copy_dict(expts_2, {expr_1.idf :    p_3})
		intvs_3 = copy_dict(intvs_2, {expr_1.idf : intv_3})
		decls_3.update({expr_1_shr.idf : Type.Int(),
						expr_2_shr.idf : Type.Int(),
						sum.idf : Type.Int()})

		return (prog_3, expr_1, decls_3, expts_3, intvs_3, cnsts_2)

	def visitRelu(self, node:AST.Func):

		self.set_arg(node.expr, node)
		(prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)

		'''
		relu(A)

		for i in 0:I:
		  for j in 0:J:
		    A[i][j] = (A[i][j] > 0)? A[i][j]: 0;
		'''

		typ_1 = node.expr.type

		iters = self.getTempIterators(typ_1.dim)

		# prog_for
		expr_1_elt = IRUtil.addIndex(expr_1, iters)
		rhs = IRUtil.relu(expr_1_elt)
		cmdl_for = IRUtil.loop(typ_1.shape, iters, [IR.Assn(expr_1_elt, rhs)])
		
		comment = IR.Comment("relu(" + expr_1.idf + ")")
		prog_for = IR.Prog([comment] + cmdl_for)

		# p_2, intv_2
		(m, M) = intvs_1[expr_1.idf]
		intv_2 = (0, M)
		
		prog_2 = IRUtil.concatPrograms(prog_1, prog_for)
		intvs_2 = copy_dict(intvs_1, {expr_1.idf : intv_2})

		return (prog_2, expr_1, decls_1, expts_1, intvs_2, cnsts_1)

	def visitMaxpool(self, node:AST.Maxpool):

		self.set_arg(node.expr, node)
		(prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)

		'''
		maxpool(A, F)

		loop1: for n in 0:N:
		         for h in 0:H:
		           for w in 0:W:
		             for c in 0:C:
		cmd2:          tmp = A[n][F*h][F*w][c];
		loop3:         for hf in 0:F:
		                 for wf in 0:F:
		cmd4:              a = A[n][F*h+hf][F*w+wf][c]
		cmd5:              tmp = (a > tmp)? a: tmp;
		cmd6:          B[n][h][w][c] = tmp
		'''

		typ_2 = node.type
		[N, H, W, C] = typ_2.shape
		F = node.dim

		# Compute scaling factor
		p_2 = expts_1[expr_1.idf]
		intv_2 = intvs_1[expr_1.idf]

		# Declare variables
		[_, tmp, a] = self.getTempVars(3)
		[n, h, w, c] = self.getTempIterators(4)
		[hf, wf] = self.getTempIterators(2)

		# Inner loop
		h_idx = IRUtil.mul(IR.Int(F), h)
		w_idx = IRUtil.mul(IR.Int(F), w)

		hf_idx = IRUtil.add(h_idx, hf)
		wf_idx = IRUtil.add(w_idx, wf)

		cmd4 = IR.Assn(a, IRUtil.addIndex(expr_1, [n, hf_idx, wf_idx, c]))
		cmd5 = IR.Assn(tmp, IRUtil.max(a, tmp))
		loop3 = IRUtil.loop([F, F], [hf, wf], [cmd4, cmd5])

		# Filter loop
		cmd2 = IR.Assn(tmp, IRUtil.addIndex(expr_1, [n, h_idx, w_idx, c]))
		cmd6 = IR.Assn(IRUtil.addIndex(expr_1, [n, h, w, c]), tmp)
		loop1 = IRUtil.loop([N, H, W, C], [n, h, w, c], [cmd2] + loop3 + [cmd6])

		# Finalize
		comment = IR.Comment("maxpool(" + expr_1.idf + ", " + str(F) + ")")
		prog_for = IR.Prog([comment] + loop1)
		prog_2 = IRUtil.concatPrograms(prog_1, prog_for)

		# Update declarations
		#decls_2 = copy_dict(decls_1, {expr_2.idf :  typ_2})
		decls_2 = decls_1
		expts_2 = copy_dict(expts_1, {expr_1.idf :    p_2})
		intvs_2 = copy_dict(intvs_1, {expr_1.idf : intv_2})
		
		decls_2.update({tmp.idf : Type.Int()})
		decls_2.update({a.idf : Type.Int()})

		return (prog_2, expr_1, decls_2, expts_2, intvs_2, cnsts_1)

	def visitBopConv(self, node:AST.Bop1):

		self.set_arg(node.expr1, node)
		(prog_1, expr_1,  decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr1)
		self.set_arg2(node.expr2, decls_1, expts_1, intvs_1, cnsts_1)
		(prog_2, expr_2,  decls_2, expts_2, intvs_2, cnsts_2) = self.visit(node.expr2)

		'''
		C = A # B

		cmd1:  for n in 0:N:
		cmd2:    for h in 0:H:
		cmd3:      for w in 0:W:
		cmd4:        for co in 0:CO:
		cmd5:          k = 0
		cmd6:          for hf in 0:HF:
		cmd7:            for wf in 0:WF:
		cmd8:              for ci in 0:CI:
		cmd9:                t1 = (((h+hf) < padH || (h+hf) > (padH+H)) ||
		                           ((w+wf) < padW || (w+wf) > (padW+W)))?
		                           0: A[n][h+hf][w+wf][co];
		cmd10:               t1 = t1 >> shr1;
		cmd11:               t2 = B[hf][wf][ci][co];
		cmd12:               t2 = t2 >> shr2;
		cmd13:               Sum[k] = t1 * t2;
		cmd14:               k += 1;
		cmd15:         C[n][h][w][co] = TreeSum(Sum);
		'''
		
		[N , H , W , CI] = node.expr1.type.shape
		[HF, WF, CI, CO] = node.expr2.type.shape

		typ_mul = Type.Tensor([HF * WF * CI])
		typ_3 = node.type

		# Compute padding
		padH = (HF - 1) // 2
		padW = (WF - 1) // 2

		# Declare variables
		[expr_3, sum] = self.getTempVars(2)
		[t1, t2, k] = self.getTempVars(3)
		[n, h, w, ci] = self.getTempIterators(4)
		[hf, wf, co] = self.getTempIterators(3)

		# Compute scale reductions and new scaling factors
		p1 = expts_1[expr_1.idf]
		p2 = expts_2[expr_2.idf]
		intv1 = intvs_1[expr_1.idf]
		intv2 = intvs_2[expr_2.idf]

		[shr1, shr2] = self.get_shr_mul(p1, p2)

		p_mul = self.get_expnt_mul(p1, shr1, p2, shr2)
		intv_mul = self.get_intv_mul(intv1, shr1, intv2, shr2)

		(p_3, H_1, H_2) = self.get_expnt_sum(p_mul, HF * WF * CI)
		intv_3 = self.get_intv_sum(intv_mul, HF * WF * CI)

		# Inner loop
		h_index = IRUtil.add(h, hf)
		w_index = IRUtil.add(w, wf)

		padH_limit = IRUtil.add(padH, H)
		padW_limit = IRUtil.add(padW, W)

		e1_idx = IRUtil.addIndex(expr_1, [n, IRUtil.sub(h_index, IR.Int(padH)), IRUtil.sub(w_index, IR.Int(padW)), ci])
		e2_idx = IRUtil.addIndex(expr_2, [hf, wf, ci, co])
		e3_idx = IRUtil.addIndex(expr_3, [n, h, w, co])

		cmd9_1 = IRUtil.orr(IRUtil.lt(h_index, IR.Int(padH)), IRUtil.gte(h_index, IR.Int(padH + H)))
		cmd9_2 = IRUtil.orr(IRUtil.lt(w_index, IR.Int(padW)), IRUtil.gte(w_index, IR.Int(padW + W)))
		cmd9_3 = IRUtil.orr(cmd9_1, cmd9_2)
		
		cmd9 = IR.Assn(t1, IR.CExpr(cmd9_3, IRUtil.zero, e1_idx))
		cmd10 = IR.Assn(t1, IRUtil.shr(t1, shr1))

		cmd11 = IR.Assn(t2, e2_idx)
		cmd12 = IR.Assn(t2, IRUtil.shr(t2, shr2))

		cmd13 = IR.Assn(IRUtil.addIndex(sum, [k]), IRUtil.mul(t1, t2))
		cmd14 = IRUtil.incCmd(k)

		loop6 = IRUtil.loop([HF, WF, CI], [hf, wf, ci], [cmd9, cmd10, cmd11, cmd12, cmd13, cmd14])

		# Tree sum
		cmd5 = IRUtil.initVarToZero(k)
		(cmd15, decls_sum) = self.compile_sum(HF * WF * CI, H_1, H_2, sum, e3_idx, False)
		
		# Outer loop
		loop1 = IRUtil.loop([N, H, W, CO], [n, h, w, co], [cmd5] + loop6 + cmd15)

		# Finalize
		comment = IR.Comment(expr_1.idf + ' # ' + expr_2.idf)
		prog_conv = IR.Prog([comment] + loop1)
		prog_3 = IRUtil.concatPrograms(prog_1, prog_2, prog_conv)
		
		# Update context for output variable
		decls_3 = copy_dict(decls_2, {expr_3.idf :  typ_3})
		expts_3 = copy_dict(expts_2, {expr_3.idf :    p_3})
		intvs_3 = copy_dict(intvs_2, {expr_3.idf : intv_3})
		
		# Update declarations
		decls_3.update({sum.idf : typ_mul})
		decls_3.update(dict((var.idf, Type.Int()) for var in [t1, t2, k]))
		decls_3.update(decls_sum)

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
		typ_mul = Type.Tensor([J])

		# decl fresh vars
		expr_3 = self.getTempVar()
		expr_1_shr = self.getTempVar()
		expr_2_shr = self.getTempVar()
		expr_mul = self.getTempVar()  # I*K*J
		[i, j, k] = self.getTempIterators(3)

		p1 = expts_1[expr_1.idf]
		p2 = expts_2[expr_2.idf]
		intv1 = intvs_1[expr_1.idf]
		intv2 = intvs_2[expr_2.idf]

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
		#resEst
		Res = REst.Mul2DTensorResCalc(I,J,K,H_1,H_2)
		Res = Res + Common.LUTCount

		cmdt = IRUtil.loop([K],[k],loop_1 + cmdl_sum_body, -1)	
		if(I > 2 or Res < Common.LUTUpperBound):
			cmdl_sum = IRUtil.loop([I], [i], cmdt,0)
		else:
			cmdl_sum = IRUtil.loop([I], [i], cmdt, -1) 
		# prog_assn
		cmd0 = IR.Comment(expr_1.idf + ' * ' + expr_2.idf)
		prog_assn = IR.Prog([cmd0] + cmdl_sum)
		
		#resEst
		Res = REst.Mul2DTensorResCalc(I,J,K,H_1,H_2)
		#Res = Res + Common.LUTCount
		prog_assn.resource = Res
		prog_3 = IRUtil.concatPrograms(prog_1, prog_2, prog_assn)


		Common.LUTCount += prog_3.resource
		decls_3 = copy_dict(decls_2, {expr_3.idf :  typ_3})
		expts_3 = copy_dict(expts_2, {expr_3.idf :    p_3})
		intvs_3 = copy_dict(intvs_2, {expr_3.idf : intv_3})
		decls_3.update({expr_1_shr.idf : Type.Int(),
						expr_2_shr.idf : Type.Int(),
						expr_mul  .idf : typ_mul,
						tmp1.idf : Type.Int(),
						tmp2.idf : Type.Int()
						})
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
		iters = self.getTempIterators(typ_3.dim)

		p1 = expts_1[expr_1.idf]
		p2 = expts_2[expr_2.idf]
		intv1 = intvs_1[expr_1.idf]
		intv2 = intvs_2[expr_2.idf]

		[shr1, shr2] = self.get_shr_mul(p1, p2)

		# cmdl_*
		'''
		e3 = (e1 >> shr1) * (e2 * shr2);
		'''
		#cmdl_shr1 = IRUtil.loop_shr(expr_1_shr, expr_1, typ_3.shape, iters, shr1)
		#cmdl_shr2 = IRUtil.loop_shr(expr_2_shr, expr_2, typ_3.shape, iters, shr2)
		expr_1_elt = IRUtil.addIndex(expr_1, iters)
		expr_2_elt = IRUtil.addIndex(expr_2, iters)
		expr_3_elt = IRUtil.addIndex(expr_3, iters)
		e1_shr = IRUtil.shr(expr_1_elt, shr1)
		e2_shr = IRUtil.shr(expr_2_elt, shr2)
		rhs = IRUtil.mul(e1_shr, e2_shr)
		#resEst
		Res = REst.MulCIRResCalc(typ_3.shape)

		if(typ_3.shape[0] > 1):
			cmdl_mul = IRUtil.loopNoUnroll(typ_3.shape, iters, [IR.Assn(expr_3_elt, rhs)])
		else:
			if(Res < Common.LUTUpperBound):
				cmdl_mul = IRUtil.loop(typ_3.shape, iters, [IR.Assn(expr_3_elt, rhs)],-1)
			else:
				cmdl_mul = IRUtil.loop(typ_3.shape, iters, [IR.Assn(expr_3_elt, rhs)], 0)

		# prog_assn, p_3, intv_3
		prog_assn = IR.Prog(cmdl_mul)
		
		p_3 = self.get_expnt_mul(p1, shr1, p2, shr2)
		intv_3 = self.get_intv_mul(intv1, shr1, intv2, shr2)
	
		prog_assn.resource = Res
		prog_3 = IRUtil.concatPrograms(prog_1, prog_2, prog_assn)

		decls_3 = copy_dict(decls_2, {expr_3.idf :  typ_3})
		expts_3 = copy_dict(expts_2, {expr_3.idf :    p_3})
		intvs_3 = copy_dict(intvs_2, {expr_3.idf : intv_3})

		return (prog_3, expr_3, decls_3, expts_3, intvs_3, cnsts_2)

	# C = A |*| B, where A(P,Q) and B(Q,1)
	def visitBopSparseMul(self, node:AST.Bop1):

		self.set_arg(node.expr1, node)
		(prog_1, expr_1,  decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr1)
		self.set_arg2(node.expr2, decls_1, expts_1, intvs_1, cnsts_1)
		(prog_2, expr_2,  decls_2, expts_2, intvs_2, cnsts_2) = self.visit(node.expr2)

		[P, Q] = node.expr1.type.shape
		[Q, R] = node.expr2.type.shape
		assert R == 1

		# Initialize C
		expr_3 = self.getTempVar()
		typ_3 = node.type
		worker_type = Type.Tensor([getNumWorkers()])# worker type for decls
		ZX_t_type = Type.Tensor([getNumWorkers(),P])# ZX_t type for decls; notice that I use 'P' for changing shape to [numWorkers, d]
		p1, p2 = expts_1[expr_1.idf], expts_2[expr_2.idf]
		intv1, intv2 = intvs_1[expr_1.idf], intvs_2[expr_2.idf]

		[shr1, shr2] = self.get_shr_mul(p1, p2)

		p_mul = self.get_expnt_mul(p1, shr1, p2, shr2)
		intv_mul = self.get_intv_mul(intv1, shr1, intv2, shr2)

		(p_3, H_1, H_2) = self.get_expnt_sum(p_mul, Q)
		intv_3 = self.get_intv_sum(intv_mul, Q)
			
		#for use by converter in CustomTypes.sv
		Common.H_1 = H_1
		Common.shr1 = shr1
		Common.shr2 = shr2
	

		'''
		Decls: i_idx, i_val, j, a, b, t

		1.   zero(C)
		2.   i_idx = 0
		3.   i_val = 0
		4.   for i in [0, Q-1]
		5.     b = B[i][0]
		6.     b = shr(b)
		7.     j = Aidx[i_idx]
		8.     while(j != 0)
		9.       a = Aval[i_val]
		10.      a = shr(a)
		11.      t = a * b
		12.      t = shr(t)
		13.      c[j-1] = c[j-1] + t
		
		14.      i_idx++
		15.      i_val++
		16.      j = Aidx[i_idx]
		17.    i_idx++
		'''

		# decl fresh vars
		[i_idx, i_val, j, a, b, t] = self.getTempVars(6)
		i = self.getTempIterator()

		worker = self.getTempIterator()
		i = self.getTempIterator()

		ZX_t_ex = self.getTempVar()
		ZX_t_ex = IR.Var(expr_1.idf[0] + 'X_t_ex',ZX_t_ex.idx, inputVar=True)

		Aidx = IR.Var(expr_1.idf[0] + 'idx', expr_1.idx, inputVar = True)
		Aval = IR.Var(expr_1.idf[0] + 'val', expr_1.idx, inputVar = True)
		cmd0 = IR.Comment(expr_1.idf + ' |*| ' + expr_2.idf)
		
		cmdp0 = IR.Pragmas("#ifndef __SYNTHESIS__", vital=1)

		memInd = self.getTempIterator()
		#cmd1 = IR.Memset(expr_3, typ_3.size())
		cmd1a = IR.Assn(IRUtil.addIndex(expr_3, [memInd, IR.Int(0)]), IR.Int(0)) 
		cmd1 = IR.For(memInd, 0, IRUtil.lt(memInd,IR.Int(typ_3.size())), [cmd1a], fac=0)  
		cmd2 = IRUtil.initVarToZero(i_idx)
		cmd3 = IRUtil.initVarToZero(i_val)

		cmd5 = IR.Assn(b, IRUtil.addIndex(expr_2, [i, IR.Int(0)]))
		cmd6 = IR.Assn(b, IRUtil.shr(b, shr2))
		cmd7 = IR.Assn(j, IRUtil.addIndex(Aidx, [i_idx]))

		cmd9 = IR.Assn(a, IRUtil.addIndex(Aval, [i_val]))
		cmd10 = IR.Assn(a, IRUtil.shr(a, shr1))
		cmd11 = IR.Assn(t, IRUtil.mul(a, b))
		cmd12 = IR.Assn(t, IRUtil.shr(t, H_1))

		minus = IRUtil.sub(j, IR.Int(1))
		c_idx = IRUtil.addIndex(expr_3, [minus, IR.Int(0)])
		cmd13 = IR.Assn(c_idx, IRUtil.add(c_idx, t))

		cmd14 = IRUtil.incCmd(i_idx)
		cmd15 = IRUtil.incCmd(i_val)
		cmd16 = IR.Assn(j, IRUtil.addIndex(Aidx, [i_idx]))

		cmd17 = IRUtil.incCmd(i_idx)

		loop8 = IR.While(IRUtil.neq(j, IR.Int(0)), [cmd9, cmd10, cmd11, cmd12, cmd13, cmd14, cmd15, cmd16])
		loop4 = IRUtil.loop([Q], [i], [cmd5, cmd6, cmd7] + [loop8] + [cmd17])

		prog = IR.Prog([cmd0, cmd1, cmdp0, cmd2, cmd3] + loop4)
		prog_3 = IRUtil.concatPrograms(prog_1, prog_2, prog)
		
		
		cmdp1 = IR.Pragmas("#else", vital=1)

		cmdv0 = IR.Pragmas("#pragma HLS RESOURCE variable=Zx_t_ex core=ROM_1P_BRAM")
		#memset ZX using for-loops
		cmdv1 = IR.Assn(IRUtil.addIndex(expr_3, [i,IR.Int(0)]), IR.Int(0))
		cmdv2 = IRUtil.loop([typ_3.size()], [i], [cmdv1], factor=0)
		
		done = IR.Var('doneSpMV' , inputVar = True)
		cmdv3 = IR.While(IRUtil.eq(done,IR.Int(0)),[])
		cmdv4 = IR.Comment(" acuumulate results")
		cmdv5 = IRUtil.add(IRUtil.addIndex(expr_3, [i,IR.Int(0)]), IRUtil.addIndex(ZX_t_ex, [worker, i]))
		cmdv6 = IR.Assn(IRUtil.addIndex(expr_3, [i, IR.Int(0)]),cmdv5)
		loopAcc1 = IRUtil.loop([typ_3.size(),getNumWorkers()], [i, worker], [cmdv6], 0)
		cmdv7 = IR.Pragmas("#endif",vital=1)
		list_t = [cmdp1] + cmdv2 + [cmdv3] + [cmdv4] + loopAcc1 + [cmdv7]
		prog_6 = IR.Prog(list_t)

		
		prog_3 = IRUtil.concatPrograms(prog_3, prog_6)

		#resource Estimation
		Res = REst.SparseMulResCalc(getNumWorkers())
		Res = Res + prog_1.resource + prog_2.resource
		prog_3.resource = Res


		decls_3 = copy_dict(decls_2, {expr_3.idf :  typ_3})
		
		
		expts_3 = copy_dict(expts_2, {expr_3.idf :    p_3})
		expts_3 = copy_dict(expts_3, {b.idf :    p_3})
		intvs_3 = copy_dict(intvs_2, {expr_3.idf : intv_3})
		decls_3.update({i_idx.idf : Type.Int(),
						i_val.idf : Type.Int(),
						j.idf : Type.Int(),
						a.idf : Type.Int(),
						b.idf : Type.Int(),
						t.idf : Type.Int()
						})


		

		return (prog_3, expr_3, decls_3, expts_3, intvs_3, cnsts_2)

	# C = A |*| B, where A(P,Q) and B(Q,1)
	def visitBopSparseMulOld1(self, node:AST.Bop1):
		self.set_arg(node.expr1, node)
		(prog_1, expr_1,  decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr1)
		self.set_arg2(node.expr2, decls_1, expts_1, intvs_1, cnsts_1)
		(prog_2, expr_2,  decls_2, expts_2, intvs_2, cnsts_2) = self.visit(node.expr2)
		
		[P, Q] = node.expr1.type.shape
		[Q, R] = node.expr2.type.shape
		assert R == 1

		# Initialize C
		expr_3 = self.getTempVar()
		typ_3 = node.type
		worker_type = Type.Tensor([getNumWorkers()])# worker type for decls
		ZX_t_type = Type.Tensor([getNumWorkers(),P])# ZX_t type for decls; notice that I use 'P' for changing shape to [numWorkers, d]
		p1, p2 = expts_1[expr_1.idf], expts_2[expr_2.idf]
		intv1, intv2 = intvs_1[expr_1.idf], intvs_2[expr_2.idf]
		[shr1, shr2] = self.get_shr_mul(p1, p2)
		p_mul = self.get_expnt_mul(p1, shr1, p2, shr2)
		intv_mul = self.get_intv_mul(intv1, shr1, intv2, shr2)
		(p_3, H_1, H_2) = self.get_expnt_sum(p_mul, Q)

		intv_3 = self.get_intv_sum(intv_mul, Q)
		
		'''
		Decls: i_idx, i_val, j, a, b, t

		1.   zero(C)
		2.   i_idx = 0
		3.   i_val = 0
		4.   for i in [0, Q-1]
		5.     b = B[i][0]
		6.     b = shr(b)
		7.     j = Aidx[i_idx]
		8.     while(j != 0)
		9.       a = Aval[i_val]
		10.      a = shr(a)
		11.      t = a * b
		12.      t = shr(t)
		13.      c[j-1] = c[j-1] + t
		
		14.      i_idx++
		15.      i_val++
		16.      j = Aidx[i_idx]
		17.    i_idx++
		'''
		
		
	
		# decl fresh vars
		[i_idx, i_val, j, a, b, t] = self.getTempVars(6)
		worker = self.getTempIterator()
		i = self.getTempIterator()


		cmdp0 = IR.Pragmas("#ifndef __SYNTHESIS__")
		#parallel sparse matrix mul for FPGA
		cmdc0 = IR.Comment(" experimenting with worker threads" "\n//" + expr_1.idf + ' |*| ' + expr_2.idf + " with worker threads")
		job = self.getTempIterator()
		Aidx_t,Aval_t, index_t, ite_t, offset_t, mulAssn, offsetAssn, AvalAssn, XAssn, ZXAssn, offsetAssn, X_t = [],[],[],[],[],[],[],[],[],[],[],[]
		ZX_t = self.getTempVar()
		ZX_t = IR.Var(expr_1.idf[0] + 'X_t',ZX_t.idx)
		ZX_t_ex = self.getTempVar()
		ZX_t_ex = IR.Var(expr_1.idf[0] + 'X_t_ex',ZX_t_ex.idx, inputVar=True)


		#memset ZX_t using for-loops
		cmd1a = IR.Assn(IRUtil.addIndex(ZX_t, [worker,i]), IR.Int(0))# add idx to tmp5 to paralellize
		cmd1 = IRUtil.loop([getNumWorkers(),typ_3.size()], [worker,i], [cmd1a], -1)

		#memset ZX using for-loops
		cmd2a = IR.Assn(IRUtil.addIndex(expr_3, [i,IR.Int(0)]), IR.Int(0))
		cmd2 = IRUtil.loop([typ_3.size()], [i], [cmd2a], -1)

		#decls
		for thread in range(0,getNumWorkers()):
			Aidx_t.append(IR.Var(expr_1.idf[0] + 'idx_t' + str(thread),  expr_1.idx, inputVar = True))
			Aval_t.append(IR.Var(expr_1.idf[0] + 'val_t' + str(thread), expr_1.idx, inputVar = True))
			X_t.append(IR.Var(expr_2.idf[0] + str(thread), expr_2.idx, inputVar = True))
			index_t.append(self.getTempVar())
			ite_t.append(self.getTempVar())
			mulAssn.append(self.getTempVar())
			offsetAssn.append(self.getTempVar())
			AvalAssn.append(self.getTempVar())
			XAssn.append(self.getTempVar())
			ZXAssn.append(self.getTempVar())

		offset_t = IR.Var('offset' , expr_1.idx, inputVar = True)
		maxJobSize_t = IR.Var('maxJobSize',inputVar = True)
		maxJobLen_t = IR.Var('maxJobLen',inputVar = True)
		#initialize
		cmd0a ,cmd0b, cmd0c,cmd0d , cmd0e ,cmd0f ,cmd0g ,cmd0h ,cmd0i ,cmd0j ,cmd0k ,cmd0l ,cmd0m ,cmd0n ,cmd0o = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
	
		#initialize job iterator 'ite'
		for thread in range(0,getNumWorkers()):
			cmd0a.append(IR.Assn(ite_t[thread], IR.Int(0)))
		
		#worker job cmds
		for thread in range(0,getNumWorkers()):
			cmd0b.append(IR.Assn(index_t[thread], IRUtil.addIndex(Aidx_t[thread], [job])))
			cmd0c.append(IR.Assn(offsetAssn[thread], IRUtil.addIndex(offset_t,[IR.Int(thread)])))
			cmd0d.append(IR.Assn(offsetAssn[thread],IRUtil.add(offsetAssn[thread], ite_t[thread])))
			cmd0e.append(IR.Assn(XAssn[thread],IRUtil.addIndex(X_t[thread],[offsetAssn[thread]])))
			cmd0f.append(IR.Assn(XAssn[thread],  IRUtil.shr(XAssn[thread], shr2))) #shr
			cmd0g.append(IR.Assn(AvalAssn[thread],IRUtil.addIndex(Aval_t[thread], [job])))
			cmd0h.append(IR.Assn(AvalAssn[thread],IRUtil.shr(AvalAssn[thread], shr1))) #shr
			cmd0i.append(IR.Assn(mulAssn[thread], IRUtil.mul(XAssn[thread],AvalAssn[thread])))
			cmd0j.append(IR.Assn(mulAssn[thread], IRUtil.shr(mulAssn[thread], H_1))) #shr
			minus = IRUtil.sub(index_t[thread],IR.Int(1))
			cmd0k.append(IR.Assn(ZXAssn[thread],IRUtil.add(IRUtil.addIndex(ZX_t,[IR.Int(thread),minus]),mulAssn[thread])))
			cmd0l.append(IR.Assn(IRUtil.addIndex(ZX_t,[IR.Int(thread),minus]),ZXAssn[thread]))
			sum = IRUtil.add(ite_t[thread],IR.Int(1))
			cmd0m.append(IR.Assn(ite_t[thread],sum))
			cond = IRUtil.andd(IRUtil.cmp_neq(index_t[thread], IR.Int(0))  ,  IRUtil.cmp_lt(ite_t[thread],maxJobSize_t))
			cond1 = IRUtil.cmp_lt(ite_t[thread],maxJobSize_t)
			cmd0o.append(IR.If(cond1, [cmd0m[thread]],[]))
			cmd0n.append(IR.If(cond, [cmd0l[thread]], [cmd0o[thread]]))

		

		
		cmdt3 = IRUtil.cmp_lt(job,IRUtil.mul(maxJobSize_t, maxJobLen_t))
		cmd0 = IR.ForNoUnroll(job,0, cmdt3, cmd0b + cmd0c + cmd0d + cmd0e + cmd0f + cmd0g + cmd0h + cmd0i + cmd0j + cmd0k + cmd0n)

		cmdc1 = IR.Comment(" experimenting done with worker threads")
		
		prog = IR.Prog([cmdp0] + [cmdc0] + cmd1 + cmd2 + cmd0a + [cmd0] + [cmdc1])
		prog_3 = IRUtil.concatPrograms(prog_1, prog_2, prog)

		#accumulate results
		cmdc2 = IR.Comment(" acuumulate results")
		prog_4 = IR.Prog([cmdc2])
		prog_3 = IRUtil.concatPrograms(prog_3,prog_4)
		cmd18 = IRUtil.add(IRUtil.addIndex(expr_3, [i,IR.Int(0)]), IRUtil.addIndex(ZX_t, [worker, i]))
		cmd19 = IR.Assn(IRUtil.addIndex(expr_3, [i, IR.Int(0)]),cmd18)
		loopAcc = IRUtil.loopUnroll([typ_3.size(),getNumWorkers()], [i, worker], [cmd19])
		cmdp1 = IR.Pragmas("#else")
		prog_4 = IR.Prog(loopAcc + [cmdp1])
		
		cmdv0 = IR.Pragmas("#pragma HLS RESOURCE variable=SpMV_t core=ROM_1P_BRAM")
		#memset ZX using for-loops
		cmdv1 = IR.Assn(IRUtil.addIndex(expr_3, [i,IR.Int(0)]), IR.Int(0))
		cmdv2 = IRUtil.loopUnroll([typ_3.size()], [i], [cmdv1])
		
		done = IR.Var('doneSpMV' , inputVar = True)
		cmdv3 = IR.While(IRUtil.cmp_eq(done,IR.Int(0)),[])
		cmdv4 = IR.Comment(" acuumulate results")
		cmdv5 = IRUtil.add(IRUtil.addIndex(expr_3, [i,IR.Int(0)]), IRUtil.addIndex(ZX_t_ex, [worker, i]))
		cmdv6 = IR.Assn(IRUtil.addIndex(expr_3, [i, IR.Int(0)]),cmdv5)
		loopAcc1 = IRUtil.loopUnroll([typ_3.size(),getNumWorkers()], [i, worker], [cmdv6])
		cmdv7 = IR.Pragmas("#endif")
		list_t = [cmdv0] + cmdv2 + [cmdv3] + [cmdv4] + loopAcc1 + [cmdv7]
		prog_6 = IR.Prog(list_t)

		
		prog_3 = IRUtil.concatPrograms(prog_3,prog_4, prog_6)

		#resource Estimation
		Res = REst.SparseMulResCalc(getNumWorkers())
		Res = Res + prog_1.resource + prog_2.resource
		prog_3.resource = Res


		decls_3 = copy_dict(decls_2, {expr_3.idf :  typ_3})
		
		
		expts_3 = copy_dict(expts_2, {expr_3.idf :    p_3})
		expts_3 = copy_dict(expts_3, {b.idf :    p_3})
		intvs_3 = copy_dict(intvs_2, {expr_3.idf : intv_3})
		decls_3.update({i_idx.idf : worker_type,
						i_val.idf : worker_type,
						j.idf : worker_type,
						a.idf : worker_type,
						ZX_t.idf : ZX_t_type,
						#ZX_t_ex.idf : ZX_t_type,
						b.idf : worker_type,
						t.idf : worker_type
						})
		for thread in range(0,getNumWorkers()):
			decls_3.update({index_t[thread].idf : Type.Int(),
							ite_t[thread].idf : Type.Int(),
							mulAssn[thread].idf : Type.Int(),
							offsetAssn[thread].idf : Type.Int(),
							XAssn[thread].idf : Type.Int(),
							#done.idf : Type.Int(),
							ZXAssn[thread].idf : Type.Int(),
							AvalAssn[thread].idf : Type.Int()
							})

		

		return (prog_3, expr_3, decls_3, expts_3, intvs_3, cnsts_2)
	
	def visitBopSparseMulOld(self, node:AST.Bop1):

		assert False

		self.set_arg(node.expr1, node)
		(prog_1, expr_1,  decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr1)
		self.set_arg2(node.expr2, decls_1, expts_1, intvs_1, cnsts_1)
		(prog_2, expr_2,  decls_2, expts_2, intvs_2, cnsts_2) = self.visit(node.expr2)

		# op, typ_{1,2,3}
		op = node.op
		typ_1 = node.expr1.type
		typ_2 = node.expr2.type
		typ_3 = node.type

		[I, J] = typ_1.shape
		[J, K] = typ_2.shape
		typ_mul = Type.Tensor([J])

		'''
		1.  Ai_idx = 0;
		2.  Aval_idx = 0;
		3.  for each i in [0, I-1]
		4.    Ai_start = Ai_idx;
		5.    Aval_start = Aval_idx;
		6.    for each k in [0, K-1]
		7.      Ai_idx = Ai_start;
		8.      Aval_idx = Aval_start;
		9.      n = 0;
		10.     j = Ai[Ai_idx];
		11.     while(j > 0)
		12.       x = shr(Aval[Aval_idx]);
		13.       y = shr(B[j][k]);
		14.       T[n] = x * y;
		15.       n++;
		16.       Aval_idx++;
		17.       Ai_idx++;
		18.       j = Ai[Ai_idx];
		19.     Ai_idx++;
		20.     TreeSum(T,n); C[i][k] = T[0]
		'''

		# decl fresh vars
		expr_3 = self.getTempVar()
		[i,k] = self.getTempIterators(2)
		j = self.getTempVar()

		p1 = expts_1[expr_1.idf]
		p2 = expts_2[expr_2.idf]
		intv1 = intvs_1[expr_1.idf]
		intv2 = intvs_2[expr_2.idf]

		[shr1, shr2] = self.get_shr_mul(p1, p2)

		Ai = IR.Var(expr_1.idf + 'idx', expr_1.idx)
					
		[Ai_idx, Aval_idx] = self.getTempVars(2)
		[Ai_start, Aval_start] = self.getTempVars(2)

		[x, y] = self.getTempVars(2)
		T = self.getTempVar()

		n = self.getTempVar()

		cmd1 = IR.Assn(Ai_idx, IRUtil.zero)
		cmd2 = IR.Assn(Aval_idx, IRUtil.zero)

		cmd4 = IR.Assn(Ai_start, Ai_idx)
		cmd5 = IR.Assn(Aval_start, Aval_idx)

		cmd7 = IR.Assn(Ai_idx, Ai_start)
		cmd8 = IR.Assn(Aval_idx, Aval_start)
					
		cmd9 = IR.Assn(n, IRUtil.zero)
		cmd10 = IR.Assn(j, IRUtil.addIndex(Ai, [Ai_idx]))

		# cmdl_{shr1, shr2, mul}
		cmd12 = IR.Assn(x, IRUtil.shr(IRUtil.addIndex(expr_1, [Aval_idx]), shr1))
		cmd13 = IR.Assn(y, IRUtil.shr(IRUtil.addIndex(expr_2, [j,k]), shr2))
					
		cmd14 = IR.Assn(IRUtil.addIndex(T, [n]), IRUtil.mul(x, y))
		cmd15 = IRUtil.incCmd(n)

		cmd16 = IRUtil.incCmd(Aval_idx)

		cmd17 = IRUtil.incCmd(Ai_idx)
		cmd18 = IR.Assn(j, IRUtil.addIndex(Ai, [Ai_idx]))

		cmd19 = IRUtil.incCmd(Ai_idx)

		loop11 = [IR.While(IRUtil.cmp_gte(j, IRUtil.zero), [cmd12, cmd13, cmd14, cmd15, cmd16, cmd17, cmd18])]

		# loop_1 = IRUtil.loop([J], [j], [cmd_1, cmd_2, cmd_3])

		p_mul = self.get_expnt_mul(p1, shr1, p2, shr2)
		intv_mul = self.get_intv_mul(intv1, shr1, intv2, shr2)

		# p_3, intv_3, cmdl_sum, decls_sum
		(p_3, H_1, H_2) = self.get_expnt_sum(p_mul, J)
		intv_3 = self.get_intv_sum(intv_mul, J)
		(sparseSumCmds, decls_sum) = \
			self.sparseSum(n, H_1, H_2,
				T,
				IRUtil.addIndex(expr_3, [i,k]), False)
					
		loop6 = IRUtil.loop([K], [k], [cmd7, cmd8, cmd9, cmd10] + loop11 + [cmd19] + sparseSumCmds)

		loop3 = IRUtil.loop([I], [i], [cmd4, cmd5] + loop6)

		prog = ([cmd1, cmd2] + loop3)
					
		prog_3 = IRUtil.concatPrograms(prog)
		decls_3 = copy_dict(decls_2, {expr_3.idf :  typ_3})
		expts_3 = copy_dict(expts_2, {expr_3.idf :    p_3})
		intvs_3 = copy_dict(intvs_2, {expr_3.idf : intv_3})
		decls_3.update({x.idf : Type.Int(),
						y.idf : Type.Int(),
						n.idf : Type.Int(),
						j.idf : Type.Int(),
						Ai_idx.idf : Type.Int(),
						Aval_idx.idf : Type.Int(),
						Ai_start.idf : Type.Int(),
						Aval_start.idf : Type.Int(),
						T.idf : typ_mul})
		decls_3.update(decls_sum)

		return (prog_3, expr_3, decls_3, expts_3, intvs_3, cnsts_2)

	def visitArgMax(self, node:AST.Func):
		
		self.set_arg(node.expr, node)
		(prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)

		typ_1 = node.expr.type

		'''
		1.  j = 0
		2.  idx = 0
		3.  max = A[0]
		4.  for (...)
		5.    if ( max < A[i] )
		6.      idx = j
		7.      max = A[i]
		8.    j++
		'''

		idx = self.getTempVar()
		max = self.getTempVar()
		j = self.getTempVar()
		iters = self.getTempIterators(typ_1.dim)
		#resource Estimation
		Res = REst.ArgMaxResCalc()
		cmd1 = IR.Assn(j, IRUtil.zero)
		cmd2 = IR.Assn(idx, IRUtil.zero)
		cmd3 = IR.Assn(max, IRUtil.addIndex(expr_1, [IRUtil.zero] * len(iters)))

		cmd6 = IR.Assn(idx, j)
		cmd7 = IR.Assn(max, IRUtil.addIndex(expr_1, iters))
		
		if5 = IR.If(IRUtil.lt(max, IRUtil.addIndex(expr_1, iters)), [cmd6, cmd7], [])

		cmd8 = IRUtil.incCmd(j)
		if(typ_1.shape[0] > 1 or Res > Common.LUTUpperBound):
			for4 = IRUtil.loop(typ_1.shape, iters, [if5, cmd8], 0)
		else:
			for4 = IRUtil.loop(typ_1.shape, iters, [if5, cmd8], -1)

		prog = IR.Prog([cmd1, cmd2, cmd3] + for4)
		prog.resource = Res

		prog_2 = IRUtil.concatPrograms(prog_1, prog)
		expr_2 = idx
		decls_2 = copy_dict(decls_1, dict((var.idf, Type.Int()) for var in [j, idx, max]))
		expts_2 = expts_1
		intvs_2 = intvs_1


		


		return (prog_2, expr_2, decls_2, expts_2, intvs_2, cnsts_1)
	
	def visitTanh(self, node:AST.Func):
		self.set_arg(node.expr, node)
		(prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)

		typ_1 = node.expr.type
		[I, J] = typ_1.shape

		p = expts_1[expr_1.idf]

		# Scale tanh limit
		tanh_limit = int(np.ldexp(Common.tanh_limit, -p))
		assert tanh_limit < np.iinfo(IR.DataType.getIntClass()).max
		tanh_limit = IR.DataType.getInt(tanh_limit)

		tanh_limit_pos = IR.Int(tanh_limit)
		tanh_limit_neg = IR.Int(-tanh_limit)

		#resource Estimation
		Res = REst.TanhResCalc()

		iters = self.getTempIterators(typ_1.dim)
		expr_1_ite = IRUtil.addIndex(expr_1, iters)

		cmd0 = IR.Comment("tanh(" + expr_1.idf + ")")
		if(I > 1 or Res > Common.LUTUpperBound):
			loop = IRUtil.loop([I, J], iters,
					[IR.Assn(expr_1_ite, IR.CExpr(IRUtil.cmp_gte(expr_1_ite, tanh_limit_pos), tanh_limit_pos, IR.CExpr(IRUtil.cmp_lte(expr_1_ite, tanh_limit_neg), tanh_limit_neg, expr_1_ite)))], 0)
		else:
			loop = IRUtil.loop([I, J], iters,
					[IR.Assn(expr_1_ite, IR.CExpr(IRUtil.gte(expr_1_ite, tanh_limit_pos), tanh_limit_pos, IR.CExpr(IRUtil.lte(expr_1_ite, tanh_limit_neg), tanh_limit_neg, expr_1_ite)))], -1)
		tanh_intv = self.get_intv(p, Common.tanh_limit, Common.tanh_limit)
		intv_1 = intvs_1[expr_1.idf]
		intv_2 = self.updateTanhIntv(intv_1, tanh_intv)
		intvs_2 = copy_dict(intvs_1, {expr_1.idf: intv_2})

		
		prog_2 = IRUtil.concatPrograms(prog_1, IR.Prog([cmd0] + loop,Res))
		
		expr_2 = expr_1
		decls_2 = decls_1
		expts_2 = expts_1
		
		return (prog_2, expr_2, decls_2, expts_2, intvs_2, cnsts_1)

	def updateTanhIntv(self, intv_1, intv_tanh):
		m_e, M_e = intv_1
		m_t, M_t = intv_tanh
		return min(m_e, m_t), min(M_e, M_t)

	# sgn(e)
	def visitSgn(self, node:AST.Func):

		self.set_arg(node.expr, node)
		(prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)

		typ_1 = node.expr.type

		# x = e > 0?  1: 0;
		
		x = self.getTempVar()
		e = IRUtil.addIndex(expr_1, [IRUtil.zero] * typ_1.dim)

		cmd = IR.Assn(x, IRUtil.cond_zero(e, IRUtil.one, IRUtil.zero))
		#cmd = IR.Assn(x, IR.CExpr(IRUtil.cmp_gt(e,IRUtil.zero), IRUtil.one,
		#IR.CExpr(IRUtil.cmp_eq(e, IRUtil.zero), IRUtil.zero, IRUtil.negone)))
		
		#resource Estimation
		Res = REst.SgnResCalc()

		prog_2 = IRUtil.concatPrograms(prog_1, IR.Prog([cmd], Res))
		expr_2 = x
		decls_2 = copy_dict(decls_1, dict((var.idf, Type.Int()) for var in [x]))
		expts_2 = expts_1
		intvs_2 = intvs_1

		return (prog_2, expr_2, decls_2, expts_2, intvs_2, cnsts_1)

	def visitDecl(self, node:AST.Decl):
		(decls_0, expts_0, intvs_0, cnsts_0) = node.decls, node.expts, node.intvs, node.cnsts

		#idf = node.name
		#self.VAR_IDF_INIT.append(idf)
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
		(prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)

		typ_1 = node.expr.type
		p_1 = expts_1[expr_1.idf]
		intv_1 = intvs_1[expr_1.idf]

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

		prog_2 = IRUtil.concatPrograms(prog_1, prog_exp)
		decls_2 = copy_dict(decls_1, {expr_2.idf : typ_1})
		expts_2 = copy_dict(expts_1, {expr_2.idf : p_2})
		intvs_2 = copy_dict(intvs_1, {expr_2.idf : intv_2})

		return (prog_2, expr_2, decls_2, expts_2, intvs_2, cnsts_1)

	# Note: We assume e<=0 for exp(e)
	def visitTableExp(self, node:AST.Func):

		self.set_arg(node.expr, node)
		(prog_1, expr_1, decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)

		# TODO: use MAX_VAL_EXP
		typ_1 = node.expr.type
		p_1 = expts_1[expr_1.idf]
		intv_1 = intvs_1[expr_1.idf]

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
		prog_2 = IRUtil.concatPrograms(prog_1, prog_exp)
		decls_2 = copy_dict(decls_1, {expr_2.idf : typ_1})
		decls_2.update(dict((var.idf, Type.Int()) for var in [input, i, j]))
		expts_2 = copy_dict(expts_1, {expr_2.idf : p_2})
		intvs_2 = copy_dict(intvs_1, {expr_2.idf : intv_2})

		Res = REst.ExpResCalc()
		Res = Res + prog_1.resource 
		prog_2.resource = prog_2.resource + Res

		return (prog_2, expr_2, decls_2, expts_2, intvs_2, cnsts_1)

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

	def visitSum(self, node:AST.Sum):
		(decls_0, expts_0, intvs_0, cnsts_0) = node.decls, node.expts, node.intvs, node.cnsts
		i_idf = node.name
		decls_0 = copy_dict(decls_0, {i_idf : Type.Int()})
		self.set_arg2(node.expr, decls_0, expts_0, intvs_0, cnsts_0)
		(prog_1, expr_1,  decls_1, expts_1, intvs_1, cnsts_1) = self.visit(node.expr)

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
		
		#resource Estimation
		Res = REst.SumResCalc(prog_1.resource, i_ed - i_st)


		i_var = IR.Var(i_idf)
		i_iter = self.getTempIterator()
		iters = self.getTempIterators(typ_2.dim)

		# p_2, intv_2, cmdl_sum, decls_sum
		(p_2, H_1, H_2) = self.get_expnt_sum(expts_1[expr_1.idf], i_ed - i_st)

		expr_1_elt = IRUtil.addIndex(expr_1, iters)
		expr_2_elt = IRUtil.addIndex(expr_2, iters)

		
		memInd = self.getTempIterator()
		cmd1a = IR.Assn(IRUtil.addIndex(expr_2, [memInd,IR.Int(0)]), IR.Int(0))
		cmd1 = IR.For(memInd, 0, IRUtil.lt(memInd, IR.Int(0)), [cmd1a], fac=-1)
		#cmd1 = IR.Memset(expr_2, typ_2.size())

		#calc unroll factor
		fac = 1
		Res = REst.SumResCalc(prog_1.resource, fac)
		while(Res < Common.LUTUpperBound):
			fac = fac + 1
			Res = REst.SumResCalc(prog_1.resource, fac)
		Res = REst.SumResCalc(prog_1.resource, fac - 1)
		fac = fac - 1

		cmd2 = IR.Assn(expr_2_elt, IRUtil.add(expr_2_elt, IRUtil.shr(expr_1_elt, H_1)))
		sum_loop = IRUtil.loop(typ_2.shape, iters, [cmd2], factor=-1)

		cmd_l1 = IRUtil.loop([i_ed - i_st],[i_iter], prog_1.cmd_l + sum_loop + [IR.Assn(i_var, IRUtil.inc(i_var))], fac)

		cmd_sum = \
			[cmd1,\
			 IR.Assn(i_var, IR.Int(i_st)),] + cmd_l1	

		#cmd_sum = \
		#	[cmd1,\
		#		IR.Assn(i_var, IR.Int(i_st)),
		#	 IR.ForUnroll(i_iter, 0, IRUtil.cmp_lt(i_iter, IR.Int(i_ed - i_st)),
		#		prog_1.cmd_l + sum_loop + \
		#		[IR.Assn(i_var, IRUtil.inc(i_var))],6 )]
		

		intv_2 = self.get_intv_sum(intvs_1[expr_1.idf], i_ed - i_st)

		prog_2 = IR.Prog(cmd_sum)
		decls_2 = copy_dict(decls_1, {expr_2.idf :  typ_2})
		expts_2 = copy_dict(expts_1, {expr_2.idf :    p_2})
		intvs_2 = copy_dict(intvs_1, {expr_2.idf : intv_2})

		prog_2.resource = prog_2.resource + Res
		return (prog_2, expr_2, decls_2, expts_2, intvs_2, cnsts_1)

	def visitSumOld(self, node:AST.Sum):
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
		
		i_var = IR.Var(i_idf)
		i_iter = self.getTempIterator()
		
		iters = self.getTempIterators(typ_2.dim)

		# cmdl_for
		'''
		cmdl_for = []
		for i in range(i_st, i_ed):
			prog_1_subst = \
				prog_1.subst(i_idf, IR.Int(i))\
					  .subst(expr_1.idf, IRUtil.addIndex(expr_1_all, [IR.Int(i - i_st)]))
			cmdl_for += prog_1_subst.cmd_l
		'''

		prog_1 = prog_1.subst(expr_1.idf, IRUtil.addIndex(expr_1_all, [i_iter]))
		cmdl_for = \
			[IR.Assn(i_var, IR.Int(i_st)),
			 IR.For(i_iter, 0, IRUtil.cmp_lt(i_iter, IR.Int(i_ed - i_st)),
				prog_1.cmd_l + \
				[IR.Assn(i_var, IRUtil.inc(i_var))])]

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
		# Remove the previous temp variable
		decls_2.pop(expr_1.idf)
		expts_2 = copy_dict(expts_1, {expr_2.idf :    p_2})
		intvs_2 = copy_dict(intvs_1, {expr_2.idf : intv_2})
		decls_2.update({expr_1_all.idf : typ_1_all})
		decls_2.update(decls_sum)

		return (prog_2, expr_2, decls_2, expts_2, intvs_2, cnsts_1)
