# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from functools import reduce
import operator

from seedot.compiler.antlr.seedotParser import seedotParser as SeeDotParser

import seedot.compiler.ast.ast as AST
from seedot.compiler.ast.astVisitor import ASTVisitor


class Type:
    pass


class Int(Type):
    pass


class Tensor(Type):

    def __init__(self, shape: list):
        self.shape = shape
        self.dim = len(shape)

    def size(self):
        return reduce(operator.mul, self.shape, 1)

    # Tensor without any dimension (float) or a tensor with all dimensions
    # equal to 1
    def isShapeOne(self):
        return self.dim == 0 or self.size() == 1


def isInt(type: Type):
    return isinstance(type, Int)


def isTensor(type: Type):
    return isinstance(type, Tensor)


def isEqual(type1: Type, type2: Type):
    if isInt(type1) and isInt(type2):
        return True
    elif isTensor(type1) and isTensor(type2):
        if type1.dim != type2.dim:
            return False
        return type1.shape == type2.shape
    else:
        assert False


class InferType(ASTVisitor):

    def visitInt(self, node: AST.Int):
        node.type = Int()
        return node.type

    # Float is represented as a tensor with 0 dimension
    def visitFloat(self, node: AST.Float):
        node.type = Tensor([])
        return node.type

    def visitId(self, node: AST.ID):
        node.type = node.gamma[node.name]
        return node.type

    def visitDecl(self, node: AST.Decl):
        node.type = Tensor(node.shape)
        return node.type

    # Matrix transpose
    def visitTransp(self, node: AST.Transp):
        node.expr.gamma = dict(node.gamma)
        exprType = self.visit(node.expr)

        assert isTensor(exprType) and exprType.dim == 2

        [m, n] = exprType.shape
        node.type = Tensor([n, m])

        return node.type

    # Reshape the tensor with custom dimensions
    def visitReshape(self, node: AST.Reshape):
        node.expr.gamma = dict(node.gamma)
        exprType = self.visit(node.expr)

        assert isTensor(exprType) and exprType.dim >= 1

        # Reshape is valid if the total number of elements remain same after
        # reshape
        assert reduce(operator.mul, exprType.shape, 1) == reduce(
            operator.mul, node.shape, 1)
        node.type = Tensor(node.shape)

        return node.type

    # Reduces the shape of a tensor by choosing the maximum from a filter
    def visitMaxpool(self, node: AST.Maxpool):
        node.expr.gamma = dict(node.gamma)
        exprType = self.visit(node.expr)

        [n1, n2, n3, n4] = exprType.shape

        # Implementation only performs maxpool over a 4D input
        assert isTensor(exprType) and exprType.dim == 4

        # Implementation needs node.dim to exactly divide matrix dimensions
        assert n2 % node.dim == 0 and n3 % node.dim == 0

        shape = [n1, n2 // node.dim, n3 // node.dim, n4]
        node.type = Tensor(shape)

        return node.type

    # Indexing a tensor
    def visitIndex(self, node: AST.Index):
        node.expr.gamma = dict(node.gamma)
        exprType = self.visit(node.expr)

        assert isTensor(exprType) and exprType.dim >= 1

        node.index.gamma = dict(node.gamma)
        indexType = self.visit(node.index)

        assert isInt(indexType)

        shape = exprType.shape[1:]
        node.type = Tensor(shape)

        return node.type

    # Currently assuming that the type of each expr is same
    def visitFuncCall(self, node: AST.FuncCall):
        type = None
        for expr in node.exprList:
            expr.gamma = dict(node.gamma)
            currType = self.visit(expr)

            if type != None:
                assert isEqual(type, currType)
            else:
                type = currType

        node.type = type

        return node.type

    def visitUop(self, node: AST.Uop):
        node.expr.gamma = dict(node.gamma)
        node.type = self.visit(node.expr)
        return node.type

    # e BINOP f
    def visitBop1(self, node: AST.Bop1):
        node.expr1.gamma = dict(node.gamma)
        eType = self.visit(node.expr1)

        node.expr2.gamma = dict(node.gamma)
        fType = self.visit(node.expr2)

        if node.op == SeeDotParser.MUL or node.op == SeeDotParser.SPARSEMUL:
            return self.visitBopMul(node, eType, fType)
        elif node.op == SeeDotParser.ADDCIR or node.op == SeeDotParser.SUBCIR:
            return self.visitBopAddOrSubCir(node, eType, fType)
        elif node.op == SeeDotParser.MULCIR:
            return self.visitBopMulCir(node, eType, fType)
        elif node.op == SeeDotParser.CONV:
            return self.visitBopConv(node, eType, fType)
        else:
            assert False

    # e * f OR e |*| f
    def visitBopMul(self, node: AST.Bop1, eType: Type, fType: Type):
        if isInt(eType) and isInt(fType):
            node.type = Int()
        elif isTensor(eType) and isTensor(fType):
            # Tensor() * Tensor(...)
            if eType.dim == 0:
                node.type = fType
            elif fType.dim == 0:
                node.type = eType

            # Tensor(...) * Tensor(...)
            else:
                assert eType.dim == 2 and fType.dim == 2

                [n1, n2] = eType.shape
                [n3, n4] = fType.shape
                assert n2 == n3

                node.type = Tensor([n1, n4])
        else:
            assert False

        return node.type

    # e <+> f OR e <-> f
    def visitBopAddOrSubCir(self, node: AST.Bop1, eType: Type, fType: Type):
        assert isTensor(eType) and isTensor(fType)
        assert eType.dim >= fType.dim
        assert fType.dim == 1
        assert eType.shape[-1] == fType.shape[-1]

        shape = eType.shape
        node.type = Tensor(shape)
        return node.type

    # e <*> f - Point-wise multiplication
    def visitBopMulCir(self, node: AST.Bop1, eType: Type, fType: Type):
        assert isTensor(eType) and isTensor(fType)
        assert eType.dim >= 1
        assert eType.shape == fType.shape

        node.type = eType
        return node.type

    # e # f
    def visitBopConv(self, node: AST.Bop1, eType: Type, fType: Type):
        assert isTensor(eType) and isTensor(fType)
        assert eType.dim == 4 and fType.dim == 4

        # Implementation does Conv on 4D input on 4D filter
        # Input is padded with 0s to ensure that the output dimension of the
        # matrix is same as the input
        [n, h, w, cin] = eType.shape
        [hf, wf, cin_, cout] = fType.shape
        assert cin == cin_

        shape = [n, h, w, cout]
        node.type = Tensor(shape)
        return node.type

    # e + f OR e - f
    def visitBop2(self, node: AST.Bop2):
        node.expr1.gamma = dict(node.gamma)
        eType = self.visit(node.expr1)

        node.expr2.gamma = dict(node.gamma)
        fType = self.visit(node.expr2)

        if isInt(eType) and isInt(fType):
            pass
        elif isTensor(eType) and isTensor(fType):
            assert eType.shape == fType.shape
        else:
            assert False

        node.type = eType
        return node.type

    def visitFunc(self, node: AST.Func):
        node.expr.gamma = dict(node.gamma)
        eType = self.visit(node.expr)

        # relu(e)
        if node.op == SeeDotParser.RELU:
            assert isTensor(eType) and eType.dim >= 1
            node.type = eType

        # exp(e)
        elif node.op == SeeDotParser.EXP:
            # Currently supports exp() on a tensor with single element
            assert isTensor(eType) and eType.isShapeOne()
            node.type = eType

        # argmax(e)
        elif node.op == SeeDotParser.ARGMAX:
            assert isTensor(eType) and eType.dim >= 1
            node.type = Int()

        # sgn(e)
        elif node.op == SeeDotParser.SGN:
            assert isTensor(eType) and eType.isShapeOne()
            node.type = Int()

        # tanh(e)
        elif node.op == SeeDotParser.TANH:
            assert isTensor(eType) and eType.dim == 2
            node.type = eType

        else:
            assert False

        return node.type

    # $(x=[1:5]) e
    def visitSum(self, node: AST.Sum):
        node.expr.gamma = dict(node.gamma)
        node.expr.gamma[node.name] = Int()
        eType = self.visit(node.expr)

        assert isTensor(eType)
        node.type = eType

        return node.type

    # e >= 0?  f : g
    def visitCond(self, node: AST.Cond):
        node.expr.gamma = dict(node.gamma)
        eType = self.visit(node.expr)

        node.trueBlock.gamma = dict(node.gamma)
        fType = self.visit(node.trueBlock)

        node.falseBlock.gamma = dict(node.gamma)
        gType = self.visit(node.falseBlock)

        assert isInt(eType) or (isTensor(eType) and eType.isShapeOne())
        assert (isInt(fType) and isInt(gType)) or (isTensor(fType)
                                                   and isTensor(gType) and fType.shape == gType.shape)

        node.type = fType
        return node.type

    # Let x = e in f
    def visitLet(self, node: AST.Let):
        node.decl.gamma = dict(node.gamma)
        eType = self.visit(node.decl)

        node.expr.gamma = dict(node.gamma)
        node.expr.gamma[node.name] = eType
        fType = self.visit(node.expr)

        node.type = fType
        return node.type
