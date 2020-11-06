# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''
This file contains the definitions of various nodes in the Abstract Syntax Tree (AST).
For a given program, the nodes of the AST is created based on the operators present in the input program.
'''


class ASTNode:
    mtdKeyTFOpName = "TFOpName"
    mtdKeyTFNodeName = "TFNodeName"

    def __init__(self):
        self.printLevel = 0
        self.gamma = {}


class Int(ASTNode):

    def __init__(self, value):
        super().__init__()
        self.value = value


class Float(ASTNode):

    def __init__(self, value):
        super().__init__()
        self.value = value


class ID(ASTNode):

    def __init__(self, name: str):
        super().__init__()
        self.name = name


class Decl(ASTNode):

    def __init__(self, shape: list, range: tuple):
        super().__init__()
        self.shape = shape
        self.range = range


class Init(ASTNode):
    def __init__(self, shape: list, value: Float):
        super().__init__()
        self.shape = shape
        self.value = value


class Transp(ASTNode):

    def __init__(self, expr):
        super().__init__()
        self.expr = expr


class Splice(ASTNode):

    def __init__(self, expr, vars, sizes):
        super().__init__()
        self.expr = expr
        self.vars = vars
        self.sizes = sizes

class LeftSplice(ASTNode):

    def __init__(self, expr, vars, sizes):
        super().__init__()
        self.expr = expr
        self.vars = vars
        self.sizes = sizes

class Reshape(ASTNode):

    def __init__(self, expr, shape, order):
        super().__init__()
        self.expr = expr
        self.shape = shape
        self.order = order


class Maxpool(ASTNode):

    def __init__(self, expr, kernelSize: list, padding: list, stride: list):
        super().__init__()
        self.expr = expr
        self.kernelSize = kernelSize
        self.padding = padding
        self.stride = stride


class Index(ASTNode):

    def __init__(self, expr, index):
        super().__init__()
        self.expr = expr
        self.index = index


class FuncCall(ASTNode):

    def __init__(self, name: str, exprList):
        super().__init__()
        self.name = name
        self.exprList = exprList


class Uop(ASTNode):

    def __init__(self, op, expr):
        super().__init__()
        self.op = op
        self.expr = expr


class Bop1(ASTNode):

    def __init__(self, expr1, op, expr2):
        super().__init__()
        self.expr1 = expr1
        self.op = op
        self.expr2 = expr2


class Bop2(ASTNode):

    def __init__(self, expr1, op, expr2):
        super().__init__()
        self.expr1 = expr1
        self.op = op
        self.expr2 = expr2


class MBConv(ASTNode):

    def __init__(self, expr1, exprF1, exprW1, exprB1, exprF2, exprW2, exprB2, exprF3, exprW3, exprB3, stride, padding):
        self.expr1 = expr1
        self.exprF1 = exprF1
        self.exprW1 = exprW1
        self.exprB1 = exprB1
        self.exprF2 = exprF2
        self.exprW2 = exprW2
        self.exprB2 = exprB2
        self.exprF3 = exprF3
        self.exprW3 = exprW3
        self.exprB3 = exprB3
        self.stride = stride
        self.padding = padding
        

class Convolution(ASTNode):

    def __init__(self, expr1, expr2, stride, padding, dilation, groups):
        super().__init__()
        self.expr1 = expr1
        self.expr2 = expr2
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups


class Func(ASTNode):

    def __init__(self, op, expr):
        super().__init__()
        self.op = op
        self.expr = expr


class Sum(ASTNode):

    def __init__(self, name, start, end, expr):
        super().__init__()
        self.name = name
        self.start = start
        self.end = end
        self.expr = expr


class Loop(ASTNode):

    def __init__(self, name, start, end, mutableVar, expr):
        super().__init__()
        self.name = name
        self.start = start
        self.end = end
        self.mutableVar = mutableVar
        self.expr = expr


class Cond(ASTNode):

    def __init__(self, expr, num, trueBlock, falseBlock):
        super().__init__()
        self.expr = expr
        self.num = num
        self.trueBlock = trueBlock
        self.falseBlock = falseBlock


class Let(ASTNode):

    def __init__(self, name, decl, expr, leftSplice = None):
        super().__init__()
        self.name = name
        self.decl = decl
        self.expr = expr
        self.leftSplice = leftSplice

class Reverse(ASTNode):

    def __init__(self, expr, axis):
        super().__init__()
        self.expr = expr
        self.axis = axis
