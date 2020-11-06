# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from seedot.compiler.antlr.seedotParser import seedotParser as SeeDotParser

import seedot.compiler.ast.ast as AST
from seedot.compiler.ast.astVisitor import ASTVisitor


class Param:

    def __init__(self, name, shape, range):
        self.name = name
        self.shape = shape
        self.range = range

        self.sparse = False


class ParamsBuilder(ASTVisitor):

    def __init__(self):
        self.params = {}

    def visitInt(self, node: AST.Int):
        pass

    def visitFloat(self, node: AST.Float):
        pass

    def visitId(self, node: AST.ID):
        pass

    def visitDecl(self, node: AST.Decl):
        return node.shape, node.range

    def visitSplice(self, node: AST.Splice):
        self.visit(node.expr)
        for var in node.vars:
            self.visit(var)

    def visitInit(self, node: AST.Init):
        pass

    def visitTransp(self, node: AST.Transp):
        self.visit(node.expr)

    def visitReshape(self, node: AST.Reshape):
        self.visit(node.expr)

    def visitMaxpool(self, node: AST.Maxpool):
        self.visit(node.expr)

    def visitReverse(self, node: AST.Reverse):
        self.visit(node.expr)    

    def visitIndex(self, node: AST.Index):
        self.visit(node.expr)
        self.visit(node.index)

    def visitFuncCall(self, node: AST.Index):
        for expr in node.exprList:
            self.visit(expr)

    def visitUop(self, node: AST.Uop):
        self.visit(node.expr)

    def visitBop1(self, node: AST.Bop1):
        self.visit(node.expr1)
        self.visit(node.expr2)

        if node.op == SeeDotParser.SPARSEMUL:
            if isinstance(node.expr1, AST.ID) and node.expr1.name in self.params:
                param = self.params[node.expr1.name]
                param.sparse = True

    def visitBop2(self, node: AST.Bop2):
        self.visit(node.expr1)
        self.visit(node.expr2)

    def visitMbconv(self, node: AST.MBConv):
        self.visit(node.expr1)
        self.visit(node.exprF1)
        self.visit(node.exprW1)
        self.visit(node.exprB1)
        self.visit(node.exprF2)
        self.visit(node.exprW2)
        self.visit(node.exprB2)
        self.visit(node.exprF3)
        self.visit(node.exprW3)
        self.visit(node.exprB3)

    def visitConvolution(self, node: AST.Convolution):
        self.visit(node.expr1)
        self.visit(node.expr2)

    def visitFunc(self, node: AST.Func):
        self.visit(node.expr)

    def visitSum(self, node: AST.Sum):
        self.visit(node.expr)

    def visitLoop(self, node: AST.Loop):
        self.visit(node.expr)

    def visitCond(self, node: AST.Cond):
        self.visit(node.expr)
        self.visit(node.trueBlock)
        self.visit(node.falseBlock)

    def visitLet(self, node: AST.Let):
        if isinstance(node.decl, AST.Decl) and node.name != 'X':
            shape, range = self.visit(node.decl)

            param = Param(node.name, shape, range)
            self.params[node.name] = param
        else:
            self.visit(node.decl)
        self.visit(node.expr)
