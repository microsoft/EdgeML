# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''
This file contains the definitions of various nodes in the Abstract Syntax Tree (AST).
For a given program, the nodes of the AST is created based on the operators present in the input program.
'''


class ASTNode:

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


class Transp(ASTNode):

    def __init__(self, expr):
        super().__init__()
        self.expr = expr


class Reshape(ASTNode):

    def __init__(self, expr, shape, order):
        super().__init__()
        self.expr = expr
        self.shape = shape
        self.order = order


class Maxpool(ASTNode):

    def __init__(self, expr, dim: int):
        super().__init__()
        self.expr = expr
        self.dim = dim


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


class Cond(ASTNode):

    def __init__(self, expr, num, trueBlock, falseBlock):
        super().__init__()
        self.expr = expr
        self.num = num
        self.trueBlock = trueBlock
        self.falseBlock = falseBlock


class Let(ASTNode):

    def __init__(self, name, decl, expr):
        super().__init__()
        self.name = name
        self.decl = decl
        self.expr = expr
