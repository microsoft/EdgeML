# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from functools import reduce
import operator

import seedot.compiler.antlr.seedotParser as seedotParser

import seedot.compiler.ast.ast as ast
import seedot.compiler.ast.astVisitor as astVisitor

import numpy as np

class Type:
    pass


class Int(Type):

    def isShapeOne(self):
        return True


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


class InferType(astVisitor.ASTVisitor):

    def __init__(self):
        self.mutableVars = []

    def visitInt(self, node: ast.Int):
        node.type = Int()
        return node.type

    # Float is represented as a tensor with 0 dimension
    def visitFloat(self, node: ast.Float):
        node.type = Tensor([])
        return node.type

    def visitId(self, node: ast.ID):
        node.type = node.gamma[node.name]
        return node.type

    def visitDecl(self, node: ast.Decl):
        node.type = Tensor(node.shape)
        return node.type

    def visitInit(self, node: ast.Init):
        node.type = Tensor(node.shape)
        return node.type

    # Matrix transpose
    def visitTransp(self, node: ast.Transp):
        node.expr.gamma = dict(node.gamma)
        exprType = self.visit(node.expr)

        assert isTensor(exprType) and exprType.dim == 2

        [m, n] = exprType.shape
        node.type = Tensor([n, m])

        return node.type

    def visitSplice(self, node: ast.Splice):
        node.expr.gamma = dict(node.gamma)
        exprType = self.visit(node.expr)

        assert isTensor(exprType) and exprType.dim >= 1
        # For splicing to be valid, the number of dimensions in input variable should match the 
        # indices provided
        assert exprType.dim == len(node.sizes)
        # For splicing to be valid, all target dimensions must be lesser than the input variable
        assert np.all(np.array(exprType.shape) >= np.array(node.sizes))
        for var in node.vars:
            var.gamma = dict(node.gamma)
        assert np.all([self.visit(var).isShapeOne for var in node.vars])
        node.type = Tensor(node.sizes)

        return node.type

    # currently not visited while type checking
    def visitLeftSplice(self, node: ast.LeftSplice):
        node.expr.gamma = dict(node.gamma)
        exprType = self.visit(node.expr)

        assert isTensor(exprType) and exprType.dim >= 1
        # For splicing to be valid, the number of dimensions in input variable should match the 
        # indices provided
        assert exprType.dim == len(node.sizes)
        # For splicing to be valid, all target dimensions must be lesser than the input variable
        assert np.all(np.array(exprType.shape) >= np.array(node.sizes))
        for var in node.vars:
            var.gamma = dict(node.gamma)
        assert np.all([self.visit(var).isShapeOne for var in node.vars])
        node.type = Tensor(node.sizes)

        return node.type    

    # Reshape the tensor with custom dimensions
    def visitReshape(self, node: ast.Reshape):
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
    def visitMaxpool(self, node: ast.Maxpool):
        node.expr.gamma = dict(node.gamma)
        exprType = self.visit(node.expr)

        [n1, H, W, n4] = exprType.shape

        # Implementation only performs maxpool over a 4D input
        assert isTensor(exprType) and exprType.dim == 4

        FH = node.kernelSize[0]
        FW = node.kernelSize[1]
        HPADL = node.padding[0]
        HPADR = node.padding[1]
        WPADL = node.padding[2]    
        WPADR = node.padding[3]

        assert HPADL == HPADR == WPADL == WPADR == 0, "Non zero paddings not supported currently"

        outH =  ((H + HPADL + HPADR - FH)//node.stride[0]) + 1
        outW = ((W + WPADL + WPADR - FW)//node.stride[1]) + 1

        shape = [n1, outH, outW, n4]
        node.type = Tensor(shape)

        return node.type    

    # Indexing a tensor
    def visitIndex(self, node: ast.Index):
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
    def visitFuncCall(self, node: ast.FuncCall):
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

    def visitUop(self, node: ast.Uop):
        node.expr.gamma = dict(node.gamma)
        node.type = self.visit(node.expr)
        return node.type

    # e BINOP f
    def visitBop1(self, node: ast.Bop1):
        node.expr1.gamma = dict(node.gamma)
        eType = self.visit(node.expr1)

        node.expr2.gamma = dict(node.gamma)
        fType = self.visit(node.expr2)

        if node.op == seedotParser.seedotParser.MUL or node.op == seedotParser.seedotParser.SPARSEMUL:
            return self.visitBopMul(node, eType, fType)
        elif node.op == seedotParser.seedotParser.ADDCIR or node.op == seedotParser.seedotParser.SUBCIR:
            return self.visitBopAddOrSubCir(node, eType, fType)
        elif node.op == seedotParser.seedotParser.MULCIR:
            return self.visitBopMulCir(node, eType, fType)
        elif node.op == seedotParser.seedotParser.CONV:
            return self.visitBopConv(node, eType, fType)
        else:
            assert False

    # e * f OR e |*| f
    def visitBopMul(self, node: ast.Bop1, eType: Type, fType: Type):
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
    def visitBopAddOrSubCir(self, node: ast.Bop1, eType: Type, fType: Type):
        assert isTensor(eType) and isTensor(fType)       
        assert eType.dim >= fType.dim
        assert fType.dim == 1
        assert eType.shape[-1] == fType.shape[-1]

        shape = eType.shape
        node.type = Tensor(shape)
        return node.type

    # e <*> f - Point-wise multiplication
    def visitBopMulCir(self, node: ast.Bop1, eType: Type, fType: Type):
        assert isTensor(eType) and isTensor(fType)
        assert eType.dim >= 1
        assert eType.shape == fType.shape

        node.type = eType
        return node.type

    # e # f
    def visitBopConv(self, node: ast.Bop1, eType: Type, fType: Type):
        assert isTensor(eType) and isTensor(fType)
        assert eType.dim == 4 and fType.dim == 4

        # Implementation does Conv on 4D input on 4D filter
        # Input is padded with 0s to ensure that the output dimension of the
        # matrix is same as the input
        [n, h, w, cin] = eType.shape
        [_, _, cin_, cout] = fType.shape
        assert cin == cin_

        shape = [n, h, w, cout]
        node.type = Tensor(shape)
        return node.type

    # c = mbconv(a, filters, weights, biases, <params>)
    def visitMbconv(self, node: ast.MBConv):
        node.expr1.gamma = dict(node.gamma)
        eType = self.visit(node.expr1)

        assert eType.dim == 4
        [N, H, W, Cin] = eType.shape

        node.exprF1.gamma = dict(node.gamma)
        F1Type = self.visit(node.exprF1)
        node.exprF2.gamma = dict(node.gamma)
        F2Type = self.visit(node.exprF2)
        node.exprF3.gamma = dict(node.gamma)
        F3Type = self.visit(node.exprF3)

        node.exprW1.gamma = dict(node.gamma)
        W1Type = self.visit(node.exprW1)
        node.exprW2.gamma = dict(node.gamma)
        W2Type = self.visit(node.exprW2)
        node.exprW3.gamma = dict(node.gamma)
        W3Type = self.visit(node.exprW3)

        node.exprB1.gamma = dict(node.gamma)
        B1Type = self.visit(node.exprB1)
        node.exprB2.gamma = dict(node.gamma)
        B2Type = self.visit(node.exprB2)
        node.exprB3.gamma = dict(node.gamma)
        B3Type = self.visit(node.exprB3)

        assert F1Type.dim == F2Type.dim == F3Type.dim == 5
        assert W1Type.dim == W2Type.dim == W3Type.dim == 1
        assert B1Type.dim == B2Type.dim == B3Type.dim == 1

        [F1g, F1hf, F1wf, F1cin, F1cout] = F1Type.shape
        [F2g, F2hf, F2wf, F2cin, F2cout] = F2Type.shape
        [F3g, F3hf, F3wf, F3cin, F3cout] = F3Type.shape

        [W1] = W1Type.shape
        [W2] = W2Type.shape
        [W3] = W3Type.shape

        [B1] = B1Type.shape
        [B2] = B2Type.shape
        [B3] = B3Type.shape

        assert F1g == F3g == 1, "first and third filters must have only 1 group"
        assert F2cin == F2cout == 1, "second filter must be depthwise separable with equal input and output size"
        assert F1hf == F1wf == F3hf == F3wf == 1, "first and third filters must be 1x1 convolutions"

        assert F2hf % 2 == F2wf % 2 == 1, "odd filter size necessary for second filter"

        for i in range(0,4): 
            assert node.padding[i] >= 0, "Padding cannot be negative"
        assert node.stride[0] > 0 and node.stride[1] > 0, "Stride must be positive"

        assert Cin == F1cin, "incompatible size of first filter"

        assert F1cout == F2g, "first filter's output channels must match number of groups of second filter"
        assert F1cout == F3cin, "first filter's output channels must match third filters input channel"

        Hout = (H + node.padding[0] + node.padding[1] - F2hf) // node.stride[0] + 1
        Wout = (W + node.padding[2] + node.padding[3] - F2wf) // node.stride[1] + 1

        node.type = Tensor([N, Hout, Wout, F3cout])
        return node.type

    # c = conv(a, b, <params>)
    def visitConvolution(self, node: ast.Convolution):
        node.expr1.gamma = dict(node.gamma)
        eType = self.visit(node.expr1)

        assert eType.dim == 4
        [n, h, w, cin] = eType.shape

        node.expr2.gamma = dict(node.gamma)
        fType = self.visit(node.expr2)

        assert fType.dim == 5
        [g, hf, wf, cin_, cout] = fType.shape

        assert cin_ * g == cin
        assert g == node.groups

        assert hf % 2 == wf % 2 == 1, "Odd filter sizes supported"

        for i in range(0,4): 
            assert node.padding[i] >= 0, "Padding cannot be negative"
        assert node.stride[0] > 0 and node.stride[1] > 0, "Stride must be positive"
        assert node.dilation[0] > 0 and node.dilation[1] > 0, "Dilation must be positive"

        hout = (h + node.padding[0] + node.padding[1] - node.dilation[0] * (hf - 1) - 1) // node.stride[0] + 1
        wout = (w + node.padding[2] + node.padding[3] - node.dilation[1] * (wf - 1) - 1) // node.stride[1] + 1
        shape = [n, hout, wout, g * cout]

        node.type = Tensor(shape)
        return node.type
        

    # e + f OR e - f
    def visitBop2(self, node: ast.Bop2):
        node.expr1.gamma = dict(node.gamma)
        eType = self.visit(node.expr1)

        node.expr2.gamma = dict(node.gamma)
        fType = self.visit(node.expr2)

        if isInt(eType) and isInt(fType):
            node.type = eType
        elif isTensor(eType) and isTensor(fType):
            if eType.dim == 0:
                node.type = fType
            elif fType.dim == 0:
                node.type = eType
            else:
                assert eType.shape == fType.shape
                node.type = eType
        else:
            assert False

        return node.type

    def visitFunc(self, node: ast.Func):
        node.expr.gamma = dict(node.gamma)
        eType = self.visit(node.expr)

        # relu(e)
        if node.op == seedotParser.seedotParser.RELU:
            assert isTensor(eType) and eType.dim >= 1
            node.type = eType

        # relu(e)
        elif node.op == seedotParser.seedotParser.RELU6:
            assert isTensor(eType) and eType.dim >= 1
            node.type = eType

        # exp(e)
        elif node.op == seedotParser.seedotParser.EXP:
            # Currently supports exp() on a tensor with single element
            assert isTensor(eType) and eType.isShapeOne()
            node.type = eType

        # argmax(e)
        elif node.op == seedotParser.seedotParser.ARGMAX:
            assert isTensor(eType) and eType.dim >= 1
            node.type = Int()

        # sgn(e)
        elif node.op == seedotParser.seedotParser.SGN:
            assert isTensor(eType) and eType.isShapeOne()
            node.type = Int()

        # tanh(e)
        elif node.op == seedotParser.seedotParser.TANH:
            assert isTensor(eType) and eType.dim == 2
            node.type = eType

        # sigmoid(e)
        elif node.op == seedotParser.seedotParser.SIGMOID:
            assert isTensor(eType) and eType.dim == 2
            node.type = eType

        elif node.op == seedotParser.seedotParser.NORMALISEL2:
            assert isTensor(eType) and eType.dim == 4   
            node.type = eType
        
        else:
            assert False

        return node.type

    # $(x=[1:5]) e
    def visitSum(self, node: ast.Sum):
        assert node.name not in node.gamma, "%s defined more than once" % (
            node.name)

        node.expr.gamma = dict(node.gamma)
        node.expr.gamma[node.name] = Int()
        eType = self.visit(node.expr)

        assert isTensor(eType)
        node.type = eType

        return node.type

    # loop(x=[1:5]) e
    def visitLoop(self, node: ast.Loop):
        assert node.name not in node.gamma, "%s defined more than once" % (
            node.name)

        node.mutableVar.gamma = dict(node.gamma)
        self.visit(node.mutableVar)

        self.mutableVars.append(node.mutableVar.name)
        assert isinstance(node.mutableVar, ast.ID)

        node.expr.gamma = dict(node.gamma)
        node.expr.gamma[node.name] = Int()
        eType = self.visit(node.expr)

        assert isTensor(eType)
        node.type = eType

        return node.type

    # e >= 0?  f : g
    def visitCond(self, node: ast.Cond):
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
    def visitLet(self, node: ast.Let):
        node.decl.gamma = dict(node.gamma)
        eType = self.visit(node.decl)

        if node.name not in self.mutableVars and node.leftSplice is None:
            assert node.name not in node.gamma, "%s defined more than once" % (
                node.name)

        node.expr.gamma = dict(node.gamma)

        if not node.leftSplice: 
            node.expr.gamma[node.name] = eType
                
        fType = self.visit(node.expr)
        node.type = fType
        return node.type

    # Reverse a tensor along given axis
    def visitReverse(self, node: ast.Reverse):
        node.expr.gamma = dict(node.gamma)
        exprType = self.visit(node.expr)

        assert isTensor(exprType) and exprType.dim >= 1
        node.type = Tensor(exprType.shape)

        return node.type     
