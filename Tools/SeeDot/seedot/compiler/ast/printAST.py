# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''
PrintAST can be used to print the generated AST for a given SeeDot program.
'''

from seedot.compiler.antlr.seedotParser import seedotParser as SeeDotParser

import seedot.compiler.ast.ast as AST
from seedot.compiler.ast.astVisitor import ASTVisitor

indent = "  "


class PrintAST(ASTVisitor):

    def visitInt(self, node: AST.Int):
        print(indent * node.printLevel, node.value)

    def visitFloat(self, node: AST.Float):
        print(indent * node.printLevel, node.value)

    def visitId(self, node: AST.ID):
        print(indent * node.printLevel, node.name)

    def visitDecl(self, node: AST.Decl):
        print(indent * node.printLevel, node.shape, "in", node.range)

    def visitTransp(self, node: AST.Transp):
        node.expr.printLevel = node.printLevel + 1
        self.visit(node.expr)
        print(indent * node.printLevel, "^T")

    def visitReshape(self, node: AST.Reshape):
        node.expr.printLevel = node.printLevel + 1
        print(indent * node.printLevel, "reshape")
        self.visit(node.expr)
        print(indent * node.printLevel, node.shape, "order", node.order)

    def visitMaxpool(self, node: AST.Maxpool):
        node.expr.printLevel = node.printLevel + 1
        print(indent * node.printLevel, "maxpool")
        self.visit(node.expr)
        print(indent * node.printLevel, node.dim)

    def visitIndex(self, node: AST.Index):
        node.expr.printLevel = node.index.printLevel = node.printLevel + 1
        self.visit(node.expr)
        print(indent * node.printLevel, "[")
        self.visit(node.index)
        print(indent * node.printLevel, "]")

    def visitFuncCall(self, node: AST.Index):
        for expr in node.exprList:
            expr.printLevel = node.printLevel + 1
        print(indent * node.printLevel, node.name + " (")
        for expr in node.exprList:
            self.visit(expr)
            print(indent * node.printLevel, ",")
        print(indent * node.printLevel, ")")

    def visitUop(self, node: AST.Uop):
        node.expr.printLevel = node.printLevel + 1
        print(indent * node.printLevel, SeeDotParser.literalNames[node.op])
        self.visit(node.expr)

    def visitBop1(self, node: AST.Bop1):
        node.expr1.printLevel = node.expr2.printLevel = node.printLevel + 1
        self.visit(node.expr1)
        print(indent * node.printLevel, SeeDotParser.literalNames[node.op])
        self.visit(node.expr2)

    def visitBop2(self, node: AST.Bop2):
        node.expr1.printLevel = node.expr2.printLevel = node.printLevel + 1
        self.visit(node.expr1)
        print(indent * node.printLevel, SeeDotParser.literalNames[node.op])
        self.visit(node.expr2)

    def visitFunc(self, node: AST.Func):
        print(indent * node.printLevel, SeeDotParser.literalNames[node.op])
        node.expr.printLevel = node.printLevel + 1
        self.visit(node.expr)

    def visitSum(self, node: AST.Sum):
        print(indent * node.printLevel, node.name,
              str(node.start), str(node.end))
        node.expr.printLevel = node.printLevel + 1
        self.visit(node.expr)

    def visitCond(self, node: AST.Cond):
        node.expr.printLevel = node.trueBlock.printLevel = node.falseBlock.printLevel = node.printLevel + 1
        self.visit(node.expr)
        print(indent * node.printLevel, ">", str(node.num))
        self.visit(node.trueBlock)
        print(indent * node.printLevel, ":")
        self.visit(node.falseBlock)

    def visitLet(self, node: AST.Let):
        node.decl.printLevel = node.expr.printLevel = node.printLevel + 1
        print(indent * node.printLevel, "let", node.name, "=")
        self.visit(node.decl)
        print(indent * node.printLevel, "in")
        self.visit(node.expr)
