# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''
ASTBuilder creates the AST for a given SeeDot program.
'''

from seedot.compiler.antlr.seedotParser import seedotParser as SeeDotParser
from seedot.compiler.antlr.seedotVisitor import seedotVisitor as SeeDotVisitor

import seedot.compiler.ast.ast as AST


class ASTBuilder(SeeDotVisitor):

    def visitInt(self, ctx: SeeDotParser.IntContext):
        value = int(ctx.IntConst().getText())
        return AST.Int(value)

    def visitFloat(self, ctx: SeeDotParser.FloatContext):
        value = float(ctx.FloatConst().getText())
        return AST.Float(value)

    def visitId(self, ctx: SeeDotParser.IdContext):
        name = ctx.Id().getText()
        return AST.ID(name)

    def visitDecl(self, ctx: SeeDotParser.DeclContext):
        shape = [int(IntConst.getText())
                 for IntConst in ctx.intConstList().IntConst()]
        range = float(ctx.FloatConst(0).getText()), float(
            ctx.FloatConst(1).getText())
        return AST.Decl(shape, range)

    def visitTransp(self, ctx: SeeDotParser.TranspContext):
        expr = self.visit(ctx.expr())
        return AST.Transp(expr)

    def visitReshape(self, ctx: SeeDotParser.ReshapeContext):
        expr = self.visit(ctx.expr())
        shape = [int(IntConst.getText())
                 for IntConst in ctx.intConstList(0).IntConst()]
        order = [int(IntConst.getText())
                 for IntConst in ctx.intConstList(1).IntConst()]
        return AST.Reshape(expr, shape, order)

    def visitMaxpool(self, ctx: SeeDotParser.MaxpoolContext):
        expr = self.visit(ctx.expr())
        dim = int(ctx.IntConst().getText())
        return AST.Maxpool(expr, dim)

    def visitIndex(self, ctx: SeeDotParser.IndexContext):
        expr = self.visit(ctx.expr(0))
        index = self.visit(ctx.expr(1))
        return AST.Index(expr, index)

    def visitFuncCall(self, ctx: SeeDotParser.FuncCallContext):
        name = ctx.Id().getText()
        exprList = [self.visit(expr) for expr in ctx.expr()]
        return AST.FuncCall(name, exprList)

    def visitUop(self, ctx: SeeDotParser.UopContext):
        op = ctx.addOp().getChild(0).symbol.type
        expr = self.visit(ctx.expr())
        return AST.Uop(op, expr)

    def visitBop1(self, ctx: SeeDotParser.Bop1Context):
        expr1 = self.visit(ctx.expr(0))
        op = ctx.binOp().getChild(0).symbol.type
        expr2 = self.visit(ctx.expr(1))
        return AST.Bop1(expr1, op, expr2)

    def visitBop2(self, ctx: SeeDotParser.Bop2Context):
        expr1 = self.visit(ctx.expr(0))
        op = ctx.addOp().getChild(0).symbol.type
        expr2 = self.visit(ctx.expr(1))
        return AST.Bop2(expr1, op, expr2)

    def visitFunc(self, ctx: SeeDotParser.FuncContext):
        op = ctx.specialFunc().getChild(0).symbol.type
        expr = self.visit(ctx.expr())
        return AST.Func(op, expr)

    def visitSum(self, ctx: SeeDotParser.SumContext):
        name = ctx.Id().getText()
        start = int(ctx.IntConst(0).getText())
        end = int(ctx.IntConst(1).getText())
        expr = self.visit(ctx.expr())
        return AST.Sum(name, start, end, expr)

    def visitCond(self, ctx: SeeDotParser.CondContext):
        expr = self.visit(ctx.expr(0))
        # Currently Cond node is used only to check sign of the expression
        # Hence the rhs of the conditional is supposed to be zero
        assert ctx.IntConst().getText() == '0'
        num = 0
        trueBlock = self.visit(ctx.expr(1))
        falseBlock = self.visit(ctx.expr(2))
        return AST.Cond(expr, num, trueBlock, falseBlock)

    def visitLet(self, ctx: SeeDotParser.LetContext):
        name = ctx.Id().getText()
        decl = self.visit(ctx.expr(0))
        expr = self.visit(ctx.expr(1))
        return AST.Let(name, decl, expr)

    def visitParen(self, ctx: SeeDotParser.ParenContext):
        expr = self.visit(ctx.expr())
        return expr
