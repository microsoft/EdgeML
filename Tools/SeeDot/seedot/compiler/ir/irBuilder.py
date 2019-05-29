# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import operator
import os

from seedot.compiler.antlr.seedotParser import seedotParser as SeeDotParser

import seedot.compiler.ast.ast as AST
from seedot.compiler.ast.astVisitor import ASTVisitor

import seedot.compiler.ir.ir as IR
import seedot.compiler.ir.irUtil as IRUtil

import seedot.common as Common
import seedot.compiler.type as Type
from seedot.util import *


class IRBuilder(ASTVisitor):

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

            [min_all, max_all] = data[0]

            self.MAX_SCALE = self.getScale(max(abs(min_all), abs(max_all)))
        else:
            self.MAX_SCALE = getMaxScale()

        self.expTables = {}

        # Counter for temp variables
        self.counter_var = 0
        self.counter_iter = 0

        # idf of vars that need to be init'ed
        self.globalVars = []

        # Global variables
        self.decls = {}
        self.scales = {}
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

        [min_exp, max_exp] = data[1]
        #[min_exp, max_exp] = [0.022, 15.012]

        expB = getExpBitLength()

        # Data for computing exp
        self.expRange = [min_exp, max_exp]
        self.expB = expB
        self.expTableShape = [2, 2 ** self.expB]

        self.MAX_VAL_EXP = max_exp

    def visitInt(self, node: AST.Int):
        val = node.value

        prog = IR.Prog([])
        expr = IR.Int(val)

        return (prog, expr)

    def visitFloat(self, node: AST.Float):
        val = node.value
        scale = self.getScale(abs(val))
        intv = self.getInterval(scale, val, val)
        val_int = IR.DataType.getInt(np.ldexp(val, -scale))

        prog = IR.Prog([])
        expr = self.getTempVar()

        self.decls[expr.idf] = node.type
        self.scales[expr.idf] = scale
        self.intvs[expr.idf] = intv
        self.cnsts[expr.idf] = val_int

        return (prog, expr)

    def visitId(self, node: AST.ID):
        idf = node.name

        prog = IR.Prog([])

        expr = IR.Var(idf, inputVar=True if idf in self.globalVars else False)

        return (prog, expr)

    def visitDecl(self, node: AST.Decl):
        minVal, maxVal = node.range

        scale = self.getScale(max(abs(minVal), abs(maxVal)))
        intv = self.getInterval(scale, minVal, maxVal)

        prog = IR.Prog([])
        expr = self.getTempVar()
        expr.inputVar = True

        self.scales[expr.idf] = scale
        self.intvs[expr.idf] = intv

        return (prog, expr)

    # out = in ^ T
    def visitTransp(self, node: AST.Transp):

        (prog_in, expr_in) = self.visit(node.expr)

        expr_out = self.getTempVar()

        type_out = node.type
        [I, J] = type_out.shape

        scale_out = self.scales[expr_in.idf]
        intv_out = self.intvs[expr_in.idf]

        expr_in.inputVar = False
        expr_out.inputVar = False

        cmd0 = IR.Comment(expr_in.idf + "^T")

        funcCall = IR.FuncCall("Transpose", {
            expr_in: "A",
            expr_out: "B",
            IR.Int(I): "I",
            IR.Int(J): "J"
        })

        prog_transp = IR.Prog([cmd0, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_transp)

        self.decls[expr_out.idf] = type_out
        self.scales[expr_out.idf] = scale_out
        self.intvs[expr_out.idf] = intv_out

        return (prog_out, expr_out)

    # out = reshape(in, shape, order)
    def visitReshape(self, node: AST.Reshape):

        (prog_in, expr_in) = self.visit(node.expr)

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

        type_in = node.expr.type
        type_out = node.type

        # Compute scaling factors
        scale_out = self.scales[expr_in.idf]
        intv_out = self.intvs[expr_in.idf]

        # Declare variables
        expr_out = self.getTempVar()

        iters_in = self.getTempIterators(type_in.dim)
        iters_out = self.getTempVars(type_out.dim)

        # Initialize to 0
        cmd1 = [IR.Assn(var, IRUtil.zero) for var in iters_out]

        # Incrementing the first index
        first_iter = iters_out[0]
        cmd4 = IRUtil.incCmd(first_iter)

        # Incrementing other indices using a loop
        cmd5 = [cmd4]
        for i in range(1, type_out.dim):
            curr_iter = iters_out[i]
            curr_size = IR.Int(type_out.shape[i])
            cmd5 = [IRUtil.incCmd(curr_iter), IR.If(IRUtil.eq(curr_iter, curr_size), [
                IRUtil.initVarToZero(curr_iter)] + cmd5)]

        # Outer loop
        loopShape = []
        loopIters = []
        for order in node.order:
            order = order - 1
            loopShape.append(type_in.shape[order])
            loopIters.append(iters_in[order])

        loop2 = IRUtil.loop(loopShape, loopIters, [IR.Assn(IRUtil.addIndex(
            expr_out, iters_out), IRUtil.addIndex(expr_in, iters_in))] + cmd5)

        # Finalize
        comment = IR.Comment(
            "reshape(" + expr_in.idf + ", " + ', '.join(str(e) for e in type_out.shape) + ")")
        prog_reshape = IR.Prog([comment] + cmd1 + loop2)

        prog_out = IRUtil.concatPrograms(prog_in, prog_reshape)

        # Update context
        self.decls[expr_out.idf] = type_out
        self.scales[expr_out.idf] = scale_out
        self.intvs[expr_out.idf] = intv_out

        # Update declarations
        self.decls.update(dict((var.idf, Type.Int()) for var in iters_out))

        return (prog_out, expr_out)

    # out = maxpool(in, stride)
    def visitMaxpool(self, node: AST.Maxpool):

        (prog_in, expr_in) = self.visit(node.expr)

        type_out = node.type
        stride = node.dim

        # Compute scaling factor
        scale_out = self.scales[expr_in.idf]
        intv_out = self.intvs[expr_in.idf]

        # Declare variables
        expr_out = self.getTempVar()

        [N, H, W, C] = node.expr.type.shape

        expr_in.inputVar = False
        expr_out.inputVar = False

        cmd0 = IR.Comment("maxpool(" + expr_in.idf + ", " + str(stride) + ")")

        funcCall = IR.FuncCall("Maxpool", {
            expr_in: "A",
            expr_out: "B",
            IR.Int(N): "N",
            IR.Int(H): "H",
            IR.Int(W): "W",
            IR.Int(C): "C",
            IR.Int(stride): "stride"
        })

        prog_maxpool = IR.Prog([cmd0, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_maxpool)

        # Update declarations
        self.decls[expr_out.idf] = type_out
        self.scales[expr_out.idf] = scale_out
        self.intvs[expr_out.idf] = intv_out

        return (prog_out, expr_out)

    # out = in[index]
    def visitIndex(self, node: AST.Index):

        (prog_in, expr_in) = self.visit(node.expr)
        (prog_idx, expr_idx) = self.visit(node.index)

        prog_out = IRUtil.concatPrograms(prog_in, prog_idx)
        expr_out = IRUtil.addIndex(expr_in, [expr_idx])

        return (prog_out, expr_out)

    # out = func(in_A, in_B, in_C, ...)
    def visitFuncCall(self, node: AST.FuncCall):
        # Assumes that the output of the uninterpretted function call is stored in one of the arguments
        # Also assumes that the scale of the output is equal to the scale of
        # the first argument

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

        str = [expr.idf for expr in exprs]
        cmd0 = IR.Comment(node.name + '(' + ', '.join(str) + ')')

        funcCall = IR.FuncCall(node.name, args)

        prog_funcCall = IR.Prog([cmd0, funcCall])

        prog_out = IRUtil.concatPrograms(prog_out, prog_funcCall)

        self.decls[expr_out.idf] = node.type
        self.scales[expr_out.idf] = self.scales[exprs[0].idf]
        self.intvs[expr_out.idf] = self.intvs[exprs[0].idf]

        return (prog_out, expr_out)

    # out = +- in
    def visitUop(self, node: AST.Uop):

        (prog_in, expr_in) = self.visit(node.expr)

        op = node.op

        if op == SeeDotParser.ADD:
            return (prog_in, expr_in)
        assert op == SeeDotParser.SUB

        type_out = node.type

        # e : Int
        if Type.isInt(type_out):
            prog_out = prog_in
            expr_out = IRUtil.negate(expr_in)

        # e: Tensor(), or Tensor(..)
        else:
            expr_out = self.getTempVar()
            iters = self.getTempIterators(type_out.dim)

            scale_out = self.scales[expr_in.idf]
            (m, M) = self.intvs[expr_in.idf]
            intv_out = (-M, -m)

            expr_in_idx = IRUtil.addIndex(expr_in, iters)
            expr_out_idx = IRUtil.addIndex(expr_out, iters)
            rhs = IRUtil.negate(expr_in_idx)
            loop = IRUtil.loop(type_out.shape, iters, [
                               IR.Assn(expr_out_idx, rhs)])
            prog_uop = IR.Prog(loop)

            prog_out = IRUtil.concatPrograms(prog_in, prog_uop)

            self.decls[expr_out.idf] = type_out
            self.scales[expr_out.idf] = scale_out
            self.intvs[expr_out.idf] = intv_out

        return (prog_out, expr_out)

    # out = in_A op in_B
    def visitBop1(self, node: AST.Bop1):
        op = node.op

        if op == SeeDotParser.MUL:
            return self.visitBopMul(node)
        elif op == SeeDotParser.SPARSEMUL:
            return self.visitBopSparseMul(node)
        elif op == SeeDotParser.MULCIR:
            return self.visitBopMulCir(node)
        elif op == SeeDotParser.CONV:
            return self.visitBopConv(node)
        elif op == SeeDotParser.ADDCIR:
            return self.visitBopAddOrSubCir(node)
        elif op == SeeDotParser.SUBCIR:
            return self.visitBopAddOrSubCir(node)
        else:
            assert False

    # out = in_A * in_B
    def visitBopMul(self, node: AST.Bop1):
        type_in_A = node.expr1.type
        type_in_B = node.expr2.type
        type_out = node.type

        if Type.isInt(type_out):
            return self.visitBopMulInt(node)
        elif type_in_A.dim == 0:
            return self.visitBopMul1DTensor(node)
        elif type_in_B.dim == 0:
            return self.visitBopMul1DTensor(node)
        else:
            return self.visitBopMul2DTensor(node)

    # out = in_A * in_B
    def visitBopMulInt(self, node: AST.Bop1):

        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B)
        expr_out = IRUtil.mul(expr_in_A, expr_in_B)

        return (prog_out, expr_out)

    # out = in_A * in_B
    def visitBopMul1DTensor(self, node: AST.Bop1):

        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        type_in_A, type_in_B = node.expr1.type, node.expr2.type
        type_out = node.type

        expr_out = self.getTempVar()

        scale_in_A, scale_in_B = self.scales[
            expr_in_A.idf], self.scales[expr_in_B.idf]
        intv_in_A, intv_in_B = self.intvs[
            expr_in_A.idf], self.intvs[expr_in_B.idf]

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

        cmd0 = IR.Comment(expr_in_A.idf + ' * ' + expr_in_B.idf)

        funcCall = IR.FuncCall("ScalarMul", {
            a: "A",
            b: "B",
            expr_out: "C",
            IR.Int(I): "I",
            IR.Int(J): "J",
            shr_a: "shr1",
            shr_b: "shr2"
        })

        prog_mul = IR.Prog([cmd0, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_mul)

        self.decls[expr_out.idf] = type_out
        self.scales[expr_out.idf] = scale_out
        self.intvs[expr_out.idf] = intv_out

        return (prog_out, expr_out)

    # out = in_A * in_B
    def visitBopMul2DTensor(self, node: AST.Bop1):

        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        expr_treeSum = self.getTempVar()
        expr_out = self.getTempVar()

        # Compute scales
        scale_in_A, scale_in_B = self.scales[
            expr_in_A.idf], self.scales[expr_in_B.idf]
        intv_in_A, intv_in_B = self.intvs[
            expr_in_A.idf], self.intvs[expr_in_B.idf]

        [shr_A, shr_B] = self.getShrForMul(scale_in_A, scale_in_B)

        type_in_A, type_in_B = node.expr1.type, node.expr2.type
        type_out = node.type

        [I, J] = type_in_A.shape
        [J, K] = type_in_B.shape
        type_treeSum = Type.Tensor([J])

        scale_treeSum = self.getScaleForMul(
            scale_in_A, shr_A, scale_in_B, shr_B)
        intv_treeSum = self.getIntvervalForMul(
            intv_in_A, shr_A, intv_in_B, shr_B)

        (scale_out, height_shr, height_noshr) = self.getScaleForTreeSum(
            scale_treeSum, J)
        intv_out = self.getIntervalForTreeSum(intv_treeSum, J)

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

        cmd0 = IR.Comment(expr_in_A.idf + ' * ' + expr_in_B.idf)

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

        prog_mul = IR.Prog([cmd0, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_mul)

        self.decls[expr_out.idf] = type_out
        self.scales[expr_out.idf] = scale_out
        self.intvs[expr_out.idf] = intv_out

        self.decls[expr_treeSum.idf] = type_treeSum

        return (prog_out, expr_out)

    # out = in_A |*| in_B
    def visitBopSparseMul(self, node: AST.Bop1):

        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        [P, Q] = node.expr1.type.shape
        [Q, R] = node.expr2.type.shape
        assert R == 1

        expr_out = self.getTempVar()
        type_out = node.type

        scale_in_A, scale_in_B = self.scales[
            expr_in_A.idf], self.scales[expr_in_B.idf]
        intv_in_A, intv_in_B = self.intvs[
            expr_in_A.idf], self.intvs[expr_in_B.idf]

        [shr_A, shr_B] = self.getShrForMul(scale_in_A, scale_in_B)

        scale_treeSum = self.getScaleForMul(
            scale_in_A, shr_A, scale_in_B, shr_B)
        intv_treeSum = self.getIntvervalForMul(
            intv_in_A, shr_A, intv_in_B, shr_B)

        (scale_out, height_shr, height_noshr) = self.getScaleForTreeSum(
            scale_treeSum, Q)
        intv_out = self.getIntervalForTreeSum(intv_treeSum, Q)

        in_A_idx = IR.Var(expr_in_A.idf[0] +
                          'idx', expr_in_A.idx, inputVar=True)
        in_A_val = IR.Var(expr_in_A.idf[0] +
                          'val', expr_in_A.idx, inputVar=True)

        shr_A = self.formatShr(shr_A)
        shr_B = self.formatShr(shr_B)
        height_shr = self.formatShr(height_shr)

        in_A_idx.inputVar = False
        in_A_val.inputVar = False
        expr_in_B.inputVar = False
        expr_out.inputVar = False

        cmd0 = IR.Comment(expr_in_A.idf + ' |*| ' + expr_in_B.idf)
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

        prog_mul = IR.Prog([cmd0, cmd1, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_mul)

        self.decls[expr_out.idf] = type_out
        self.scales[expr_out.idf] = scale_out
        self.intvs[expr_out.idf] = intv_out

        # Length of Aidx and Aval hard coded to 100
        # This is safe as it will be ignored in the generated code
        self.decls.update({in_A_idx.idf: Type.Tensor([100]),
                           in_A_val.idf: Type.Tensor([100]),
                           })
        self.globalVars.append(in_A_idx.idf)
        self.globalVars.append(in_A_val.idf)

        return (prog_out, expr_out)

    # out = in_A <*> in_B
    def visitBopMulCir(self, node: AST.Bop1):

        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        type_in_A, type_in_B = node.expr1.type, node.expr2.type
        type_out = node.type

        expr_out = self.getTempVar()

        assert type_out.dim == 2

        [I, J] = type_out.shape

        scale_in_A, scale_in_B = self.scales[
            expr_in_A.idf], self.scales[expr_in_B.idf]
        intv_in_A, intv_in_B = self.intvs[
            expr_in_A.idf], self.intvs[expr_in_B.idf]

        [shr_A, shr_B] = self.getShrForMul(scale_in_A, scale_in_B)

        scale_out = self.getScaleForMul(scale_in_A, shr_A, scale_in_B, shr_B)
        intv_out = self.getIntvervalForMul(intv_in_A, shr_A, intv_in_B, shr_B)

        shr_A = self.formatShr(shr_A)
        shr_B = self.formatShr(shr_B)

        expr_in_A.inputVar = False
        expr_in_B.inputVar = False
        expr_out.inputVar = False

        cmd0 = IR.Comment(expr_in_A.idf + ' <*> ' + expr_in_B.idf)

        funcCall = IR.FuncCall("MulCir", {
            expr_in_A: "A",
            expr_in_B: "B",
            expr_out: "C",
            IR.Int(I): "I",
            IR.Int(J): "J",
            shr_A: "shrA",
            shr_B: "shrB"
        })

        prog_mul = IR.Prog([cmd0, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_mul)

        self.decls[expr_out.idf] = type_out
        self.scales[expr_out.idf] = scale_out
        self.intvs[expr_out.idf] = intv_out

        return (prog_out, expr_out)

    # out = in_A # in_B
    def visitBopConv(self, node: AST.Bop1):

        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        [N, H, W, CI] = node.expr1.type.shape
        [HF, WF, CI, CO] = node.expr2.type.shape

        type_treeSum = Type.Tensor([HF * WF * CI])
        type_out = node.type

        # Compute padding
        padH = (HF - 1) // 2
        padW = (WF - 1) // 2

        # Declare variables
        [expr_treeSum, expr_out] = self.getTempVars(2)

        # Compute scale reductions and new scaling factors
        scale_in_A, scale_in_B = self.scales[
            expr_in_A.idf], self.scales[expr_in_B.idf]
        intv_in_A, intv_in_B = self.intvs[
            expr_in_A.idf], self.intvs[expr_in_B.idf]

        [shr_A, shr_B] = self.getShrForMul(scale_in_A, scale_in_B)

        scale_treeSum = self.getScaleForMul(
            scale_in_A, shr_A, scale_in_B, shr_B)
        intv_treeSum = self.getIntvervalForMul(
            intv_in_A, shr_A, intv_in_B, shr_B)

        (scale_out, height_shr, height_noshr) = self.getScaleForTreeSum(
            scale_treeSum, HF * WF * CI)
        intv_out = self.getIntervalForTreeSum(intv_treeSum, HF * WF * CI)

        shr_A = self.formatShr(shr_A)
        shr_B = self.formatShr(shr_B)

        expr_in_A.inputVar = False
        expr_in_B.inputVar = False
        expr_out.inputVar = False

        cmd0 = IR.Comment(expr_in_A.idf + ' # ' + expr_in_B.idf)

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

        prog_conv = IR.Prog([cmd0, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_conv)

        # Update context for output variable
        self.decls[expr_out.idf] = type_out
        self.scales[expr_out.idf] = scale_out
        self.intvs[expr_out.idf] = intv_out

        # Update declarations
        self.decls[expr_treeSum.idf] = type_treeSum

        return (prog_out, expr_out)

    # out = in_A <+-> in_B
    def visitBopAddOrSubCir(self, node: AST.Bop1):

        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        op = node.op
        type_in_A, type_in_B = node.expr1.type, node.expr2.type
        type_out = node.type

        if op == SeeDotParser.ADDCIR:
            (op_ir, op_fn) = (IR.Op.Op['+'], operator.add)
            add = True
        elif op == SeeDotParser.SUBCIR:
            (op_ir, op_fn) = (IR.Op.Op['-'], operator.sub)
            add = False

        assert add == True

        scale_in_A, scale_in_B = self.scales[
            expr_in_A.idf], self.scales[expr_in_B.idf]
        intv_in_A, intv_in_B = self.intvs[
            expr_in_A.idf], self.intvs[expr_in_B.idf]

        (scale_out, intv_out, [shr_A, shr_B, shr_out]) = self.getScaleAndIntervalForAdd(
            scale_in_A, scale_in_B, intv_in_A, intv_in_B, op_fn)

        shr_A = self.formatShr(shr_A)
        shr_B = self.formatShr(shr_B)
        shr_out = self.formatShr(shr_out)

        expr_in_A.inputVar = False
        expr_in_B.inputVar = False

        cmd0 = IR.Comment(expr_in_A.idf + " <" +
                          op_ir.name + "> " + expr_in_B.idf)

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
                IR.Bool(True): "add"
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
                IR.Bool(True): "add"
            })
        else:
            assert False

        prog_cir = IR.Prog([cmd0, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_cir)

        self.scales[expr_in_A.idf] = scale_out
        self.intvs[expr_in_A.idf] = intv_out

        return (prog_out, expr_in_A)

    # out = in_A 'op' in_B
    def visitBop2(self, node: AST.Bop2):

        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        op = node.op
        if op == SeeDotParser.ADD:
            (op_ir, op_fn) = (IR.Op.Op['+'], operator.add)
            funcName = "MatAdd"
        elif op == SeeDotParser.SUB:
            (op_ir, op_fn) = (IR.Op.Op['-'], operator.sub)
            funcName = "MatSub"

        type_out = node.type

        # e : Int
        if Type.isInt(type_out):
            prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B)
            expr_out = IR.IntBop(expr_in_A, op_ir, expr_in_B)

        # e : Tensor(), or Tensor(..)
        else:
            expr_out = self.getTempVar()

            scale_in_A, scale_in_B = self.scales[
                expr_in_A.idf], self.scales[expr_in_B.idf]
            intv_in_A, intv_in_B = self.intvs[
                expr_in_A.idf], self.intvs[expr_in_B.idf]

            (scale_out, intv_out, [shr_A, shr_B, shr_out]) = self.getScaleAndIntervalForAdd(
                scale_in_A, scale_in_B, intv_in_A, intv_in_B, op_fn)

            assert type_out.dim == 2

            [I, J] = type_out.shape

            shr_A = self.formatShr(shr_A)
            shr_B = self.formatShr(shr_B)
            shr_out = self.formatShr(shr_out)

            expr_in_A.inputVar = False
            expr_in_B.inputVar = False
            expr_out.inputVar = False

            cmd0 = IR.Comment(expr_in_A.idf + ' ' +
                              op_ir.name + ' ' + expr_in_B.idf)

            funcCall = IR.FuncCall(funcName, {
                expr_in_A: "A",
                expr_in_B: "B",
                expr_out: "C",
                IR.Int(I): "I",
                IR.Int(J): "J",
                shr_A: "shrA",
                shr_B: "shrB",
                shr_out: "shrC"
            })

            prog_bop = IR.Prog([cmd0, funcCall])

            prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_bop)

            self.decls[expr_out.idf] = type_out
            self.scales[expr_out.idf] = scale_out
            self.intvs[expr_out.idf] = intv_out

        return (prog_out, expr_out)

    # out = func(in)
    def visitFunc(self, node: AST.Func):
        op = node.op

        if op == SeeDotParser.RELU:
            return self.visitRelu(node)
        elif op == SeeDotParser.EXP:
            return self.visitExp(node)
        elif op == SeeDotParser.ARGMAX:
            return self.visitArgMax(node)
        elif op == SeeDotParser.SGN:
            return self.visitSgn(node)
        elif op == SeeDotParser.TANH:
            return self.visitTanh(node)
        else:
            assert False

    # out = relu(in)
    def visitRelu(self, node: AST.Func):

        (prog_in, expr_in) = self.visit(node.expr)

        type_out = node.expr.type

        (m, M) = self.intvs[expr_in.idf]
        if m < 0:
            m = 0
        if M < 0:
            M = 0
        intv_out = (m, M)

        expr_in.inputVar = False

        cmd0 = IR.Comment("relu(" + expr_in.idf + ")")

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
            assert False

        prog_relu = IR.Prog([cmd0, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_relu)

        self.intvs[expr_in.idf] = intv_out

        return (prog_out, expr_in)

    # out = exp(in)
    def visitExp(self, node: AST.Func):

        self.readProfileFile()

        if useMathExp():
            return self.visitMathExp(node)
        elif useTableExp():
            return self.visitTableExp(node)
        else:
            assert False

    # Note: We assume e<=0 for exp(e)
    def visitMathExp(self, node: AST.Func):

        # Tunable parameter
        MIN = 0.1

        (prog_in, expr_in) = self.visit(node.expr)

        type_in = node.expr.type

        scale_in = self.scales[expr_in.idf]
        intv_in = self.intvs[expr_in.idf]

        '''
		1.  y = ((int) (exp(((float)e) / shr1) * shr2))
		'''

        maxExp = np.exp(-MIN)

        expr_out = self.getTempVar()

        scale_out = self.getScale(maxExp)
        intv_out = self.getInterval(scale_out, maxExp, maxExp)

        shr1 = IR.Int(2 ** -scale_in)
        shr2 = IR.Int(2 ** -scale_out)

        expr_in_idx = IRUtil.addIndex(expr_in, [IRUtil.zero] * type_in.dim)
        expr_out_idx = IRUtil.addIndex(expr_out, [IRUtil.zero] * type_in.dim)

        cmd0 = IR.Comment('exp(' + expr_in.idf + ')')

        cmd_assn = IR.Assn(expr_out_idx, IRUtil.castToInt(IRUtil.mul(
            IR.Exp(IRUtil.div(IRUtil.castToFloat(expr_in_idx), shr1)), shr2)))

        prog_exp = IR.Prog([cmd0, cmd_assn])

        prog_out = IRUtil.concatPrograms(prog_in, prog_exp)

        self.decls[expr_out.idf] = type_in
        self.scales[expr_out.idf] = scale_out
        self.intvs[expr_out.idf] = intv_out

        return (prog_out, expr_out)

    # Note: We assume e<=0 for exp(e)
    def visitTableExp(self, node: AST.Func):

        (prog_in, expr_in) = self.visit(node.expr)

        # TODO: use MAX_VAL_EXP
        type_in = node.expr.type

        scale_in = self.scales[expr_in.idf]
        intv_in = self.intvs[expr_in.idf]

        [m, M] = self.expRange
        [m_scale, M_scale] = [
            int(np.ldexp(m, -scale_in)), int(np.ldexp(M, -scale_in))]

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

        cmd6 = IR.Assn(input, IRUtil.shl(IRUtil.sub(
            IRUtil.negate(expr_1_elt), IR.Int(m_scale)), shl))
        cmd7 = IR.Assn(i, IRUtil.bitAnd(IRUtil.shrUint(input, shrI), mask))
        cmd8 = IR.Assn(j, IRUtil.bitAnd(IRUtil.shrUint(input, shrJ), mask))

        cmd1 = IR.If(cond, [cmd2, cmd3], [cmd6, cmd7, cmd8])
        cmd10 = IR.Assn(expr_2_elt, IRUtil.mul(IRUtil.shrUint(IRUtil.addIndex(
            table[0], [i]), shr1), IRUtil.shrUint(IRUtil.addIndex(table[1], [j]), shr2)))

        scale_out = self.getScaleForExp(scale1, shr1, scale2, shr2)
        intv_out = self.getIntervalForExp(scale_out, [-m_scale, -M_scale])

        cmd0 = IR.Comment('exp(' + expr_in.idf + ')')

        prog_exp = IR.Prog([cmd0, cmd1, cmd10])

        prog_out = IRUtil.concatPrograms(prog_in, prog_exp)

        self.decls[expr_out.idf] = type_in
        self.scales[expr_out.idf] = scale_out
        self.intvs[expr_out.idf] = intv_out

        self.decls.update(dict((var.idf, Type.Int()) for var in [input, i, j]))

        return (prog_out, expr_out)

    def getShl(self, n: int):
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

        table = [[0 for _ in range(alpha_count)], [
            0 for _ in range(beta_count)]]

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

        tableVar = [IR.Var('EXP' + str(abs(p)) + 'A', inputVar=True),
                    IR.Var('EXP' + str(abs(p)) + 'B', inputVar=True)]

        return [table, tableVar]

    def getAlphaCount(self, max, shl):
        mask = 2 ** self.expB - 1
        shr = Common.wordLength - shl - self.expB
        return ((max >> shr) & mask) + 1

    # out = argmax(in)
    def visitArgMax(self, node: AST.Func):

        (prog_in, expr_in) = self.visit(node.expr)

        type_out = node.expr.type

        assert type_out.dim == 2

        [I, J] = type_out.shape

        expr_out = self.getTempVar()

        expr_in.inputVar = False

        cmd0 = IR.Comment('argmax(' + expr_in.idf + ')')

        funcCall = IR.FuncCall("ArgMax", {
            expr_in: "A",
            IR.Int(I): "I",
            IR.Int(J): "J",
            expr_out: "index"
        })

        prog_argmax = IR.Prog([cmd0, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_argmax)

        self.decls[expr_out.idf] = Type.Int()

        return (prog_out, expr_out)

    # out = sgn(in)
    def visitSgn(self, node: AST.Func):

        (prog_in, expr_in) = self.visit(node.expr)

        expr_out = self.getTempVar()
        type_in = node.expr.type

        expr_in_idx = IRUtil.addIndex(expr_in, [IRUtil.zero] * type_in.dim)

        cmd0 = IR.Comment('sgn(' + expr_in.idf + ')')
        cmd1 = IR.Assn(expr_out, IRUtil.cond_zero(
            expr_in_idx, IRUtil.one, IRUtil.zero))

        prog_sgn = IR.Prog([cmd0, cmd1])

        prog_out = IRUtil.concatPrograms(prog_in, prog_sgn)

        self.decls[expr_out.idf] = Type.Int()

        return (prog_out, expr_out)

    # out = tanh(in)
    def visitTanh(self, node: AST.Func):

        (prog_in, expr_in) = self.visit(node.expr)

        type_in = node.expr.type
        [I, J] = type_in.shape

        scale_in = self.scales[expr_in.idf]
        intv_in = self.intvs[expr_in.idf]

        # Scale tanh limit
        tanh_limit = int(np.ldexp(Common.tanh_limit, -scale_in))
        assert tanh_limit < np.iinfo(IR.DataType.getIntClass()).max
        tanh_limit = IR.DataType.getInt(tanh_limit)

        tanh_intv = self.getInterval(
            scale_in, Common.tanh_limit, Common.tanh_limit)
        intv_out = self.updateTanhIntv(intv_in, tanh_intv)

        expr_in.inputVar = False

        cmd0 = IR.Comment("tanh(" + expr_in.idf + ")")

        funcCall = IR.FuncCall("TanH", {
            expr_in: "A",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(tanh_limit): "threshold"
        })

        prog_tanh = IR.Prog([cmd0, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_tanh)

        self.intvs[expr_in.idf] = intv_out
        expr_out = expr_in

        return (prog_out, expr_out)

    # out = $x[start:end] in
    def visitSum(self, node: AST.Sum):
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
        self.decls[var_idf] = Type.Int()

        (prog_in, expr_in) = self.visit(node.expr)

        start, end = node.start, node.end

        expr_out = self.getTempVar()
        type_out = node.type

        var = IR.Var(var_idf)
        var_iter = self.getTempIterator()
        iters = self.getTempIterators(type_out.dim)

        (scale_out, height_shr, height_noshr) = self.getScaleForTreeSum(
            self.scales[expr_in.idf], end - start)
        intv_out = self.getIntervalForTreeSum(
            self.intvs[expr_in.idf], end - start)

        # Tree sum to sum output of each iteration
        expr_in_idx = IRUtil.addIndex(expr_in, iters)
        expr_out_idx = IRUtil.addIndex(expr_out, iters)

        cmd1 = IR.Memset(expr_out, type_out.size())
        cmd2 = IR.Assn(expr_out_idx, IRUtil.add(
            expr_out_idx, IRUtil.shr(expr_in_idx, height_shr)))
        treeSum = IRUtil.loop(type_out.shape, iters, [cmd2])

        # Final program to sum output of each iteration
        prog_sum = [cmd1,
                    IR.Assn(var, IR.Int(start)),
                    IR.For(var_iter, 0, IRUtil.lt(var_iter, IR.Int(end - start)),
                           prog_in.cmd_l + treeSum +
                           [IR.Assn(var, IRUtil.inc(var))])]

        prog_out = IR.Prog(prog_sum)

        self.decls[expr_out.idf] = type_out
        self.scales[expr_out.idf] = scale_out
        self.intvs[expr_out.idf] = intv_out

        return (prog_out, expr_out)

    # out = in_cond > 0? in_A: in_B
    def visitCond(self, node: AST.Cond):

        (prog_in_cond, expr_in_cond) = self.visit(node.expr)

        (prog_in_A, expr_in_A) = self.visit(node.trueBlock)

        (prog_in_B, expr_in_B) = self.visit(node.falseBlock)

        type_in_cond = node.expr.type
        type_in_A = node.trueBlock.type

        if Type.isInt(type_in_cond):
            expr_in_cond_idx = expr_in_cond
        else:
            expr_in_cond_idx = IRUtil.addIndex(
                expr_in_cond, [IRUtil.zero] * type_in_cond.dim)

        # e2,e3 : Int
        if Type.isInt(type_in_A):
            # TODO: Update the scale and intv of expr_out based on in_A and
            # in_B
            prog_out = IRUtil.concatPrograms(
                prog_in_cond, prog_in_A, prog_in_B)
            expr_out = IRUtil.cond_zero(expr_in_cond_idx, expr_in_A, expr_in_B)

        # e2,e3 : Tensor(), or Tensor(..)
        else:
            expr_out = self.getTempVar()
            iters = self.getTempIterators(type_in_A.dim)

            scale_in_A, scale_in_B = self.scales[
                expr_in_A.idf], self.scales[expr_in_B.idf]
            intv_in_A, intv_in_B = self.intvs[
                expr_in_A.idf], self.intvs[expr_in_B.idf]
            m_2, M_2 = intv_in_A
            m_3, M_3 = intv_in_B

            if scale_in_A >= scale_in_B:
                shr_n_2, shr_n_3 = 0, scale_in_A - scale_in_B
            else:
                shr_n_2, shr_n_3 = scale_in_B - scale_in_A, 0

            scale_out = max(scale_in_A, scale_in_B)
            intv_out = (min(m_2 >> shr_n_2, m_3 >> shr_n_3),
                        max(M_2 >> shr_n_2, M_3 >> shr_n_3))

            # prog_assn
            expr_in_A_idx = IRUtil.addIndex(expr_in_A, iters)
            expr_in_B_idx = IRUtil.addIndex(expr_in_B, iters)
            expr_out_idx = IRUtil.addIndex(expr_out, iters)
            rhs = IRUtil.cond_zero(expr_in_cond_idx,
                                   IRUtil.shr(expr_in_A_idx, shr_n_2),
                                   IRUtil.shr(expr_in_B_idx, shr_n_3))
            cmdl_assn = IRUtil.loop(type_in_A.shape, iters, [
                                    IR.Assn(expr_out_idx, rhs)])
            prog_cond = IR.Prog(cmdl_assn)

            prog_out = IRUtil.concatPrograms(
                prog_in_cond, prog_in_A, prog_in_B, prog_cond)

            self.decls[expr_out.idf] = type_in_A
            self.scales[expr_out.idf] = scale_out
            self.intvs[expr_out.idf] = intv_out

        return (prog_out, expr_out)

    # let idf = decl 'in' in
    def visitLet(self, node: AST.Let):

        (prog_decl, expr_decl) = self.visit(node.decl)
        type_decl = node.decl.type

        idf = node.name

        # e1 : Int
        if Type.isInt(type_decl):
            self.decls[idf] = Type.Int()

            (prog_in, expr_in) = self.visit(node.expr)

            cmd = IR.Assn(IR.Var(idf), expr_decl)
            prog_let = IR.Prog([cmd])

            prog_out = IRUtil.concatPrograms(prog_decl, prog_let, prog_in)

            return (prog_out, expr_in)

        # e1 : Tensor{(),(..)}
        else:
            self.scales[idf] = self.scales[expr_decl.idf]
            self.intvs[idf] = self.intvs[expr_decl.idf]

            if isinstance(node.decl, AST.Decl):
                self.globalVars.append(idf)
                self.decls[idf] = node.decl.type
                expr_decl.idf = idf
                expr_decl.inputVar = True

            (prog_in, expr_in) = self.visit(node.expr)

            prog_in = prog_in.subst(idf, expr_decl)
            expr_in = expr_in.subst(idf, expr_decl)

            prog_out = IRUtil.concatPrograms(prog_decl, prog_in)

            return (prog_out, expr_in)

    # Computing exponent and intervals
    def getScale(self, maxabs: float):  # -> int
        return int(np.ceil(np.log2(maxabs) - np.log2((1 << (Common.wordLength - 2)) - 1)))

    # Takes range [r1, r2] and returns the interval scaled by p
    def getInterval(self, p: int, r1: float, r2: float):
        return (int(np.ldexp(r1, -p)), int(np.ldexp(r2, -p)))

    def getScaleForMul(self, p1: int, shr1: int, p2: int, shr2: int) -> int:
        return (p1 + shr1) + (p2 + shr2)

    # int^2 * int^2 -> int^2
    def getIntvervalForMul(self, intv_1, shr1: int, intv_2, shr2: int):
        (m_1, M_1) = intv_1
        (m_1, M_1) = (m_1 >> shr1, M_1 >> shr1)

        (m_2, M_2) = intv_2
        (m_2, M_2) = (m_2 >> shr2, M_2 >> shr2)

        m = min([m_1 * m_2, m_1 * M_2, M_1 * m_2, M_1 * M_2])
        M = max([m_1 * m_2, m_1 * M_2, M_1 * m_2, M_1 * M_2])

        return (m, M)

    def getScaleForTreeSum(self, p: int, n: int):
        H_tot = int(np.ceil(np.log2(n)))
        if p >= self.MAX_SCALE:
            p_res = p
        else:
            p_res = min(p + H_tot, self.MAX_SCALE)
        H_1 = p_res - p
        assert H_1 >= 0
        H_2 = H_tot - H_1
        assert H_2 >= 0
        return (p_res, H_1, H_2)

    def getIntervalForTreeSum(self, intv, n: int):
        max_abs = (1 << Common.wordLength - 2) - 1
        (m, M) = intv
        m = max(n * m, -max_abs)
        M = min(n * M,  max_abs)
        return (m, M)

    def getScaleAndIntervalForAdd(self, p_1: int, p_2: int, intv_1, intv_2, op_fn):
        (m_1, M_1) = intv_1
        (m_2, M_2) = intv_2

        if p_1 >= p_2:
            shr_n = [0, p_1 - p_2, 0]
            p = p_1
        else:
            shr_n = [p_2 - p_1, 0, 0]
            p = p_2
        m = op_fn(m_1 >> shr_n[0], m_2 >> shr_n[1])
        M = op_fn(M_1 >> shr_n[0], M_2 >> shr_n[1])

        if max(abs(m), abs(M)) >= (1 << (Common.wordLength - 2)) and p < self.MAX_SCALE:
            shr_n[2] = 1
            p += 1
        max_abs = (1 << Common.wordLength - 2) - 1
        m = max(m >> shr_n[2], -max_abs)
        M = min(M >> shr_n[2],  max_abs)

        return (p, (m, M), shr_n)

    def getScaleForExp(self, p1: int, shr1: int, p2: int, shr2: int):
        return (p1 + shr1) + (p2 + shr2)

    def getIntervalForExp(self, p: int, intv):  # int^2 -> int^2
        (m, M) = intv
        assert m < np.ldexp(self.MAX_VAL_EXP, -p)
        M = min(M, np.ldexp(self.MAX_VAL_EXP, -p))
        return self.getInterval(p, np.exp(np.ldexp(m, p)), np.exp(np.ldexp(M, p)))

    def getShrForMul(self, p1, p2):
        shr = (Common.wordLength - 2) // 2
        pRes = (p1 + shr) + (p2 + shr)
        if pRes < self.MAX_SCALE:
            return [shr, shr]
        else:
            save = abs(abs(pRes) - abs(self.MAX_SCALE))
            save1 = save // 2
            save2 = save - save1
            shr1 = max(shr - save1, 0)
            shr2 = max(shr - save2, 0)
            return [shr1, shr2]

    def updateTanhIntv(self, intv_1, intv_tanh):
        m_e, M_e = intv_1
        m_t, M_t = intv_tanh
        return min(m_e, m_t), min(M_e, M_t)

    # Variable and iterators creation
    def getTempVars(self, n: int):
        return [self.getTempVar() for i in range(n)]

    def getTempVar(self):
        var = IR.Var('tmp' + str(self.counter_var))
        self.counter_var += 1
        return var

    def getTempIterators(self, n: int):
        return [self.getTempIterator() for i in range(n)]

    def getTempIterator(self):
        var = IR.Var('i' + str(self.counter_iter))
        self.counter_iter += 1
        return var

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
