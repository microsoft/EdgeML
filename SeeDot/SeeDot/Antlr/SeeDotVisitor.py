# Generated from SeeDot.g4 by ANTLR 4.7
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .SeeDotParser import SeeDotParser
else:
    from SeeDotParser import SeeDotParser

# This class defines a complete generic visitor for a parse tree produced by SeeDotParser.

class SeeDotVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by SeeDotParser#bop1.
    def visitBop1(self, ctx:SeeDotParser.Bop1Context):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#bop2.
    def visitBop2(self, ctx:SeeDotParser.Bop2Context):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#decl.
    def visitDecl(self, ctx:SeeDotParser.DeclContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#index.
    def visitIndex(self, ctx:SeeDotParser.IndexContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#sum.
    def visitSum(self, ctx:SeeDotParser.SumContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#reshape.
    def visitReshape(self, ctx:SeeDotParser.ReshapeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#float.
    def visitFloat(self, ctx:SeeDotParser.FloatContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#cond.
    def visitCond(self, ctx:SeeDotParser.CondContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#int.
    def visitInt(self, ctx:SeeDotParser.IntContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#transp.
    def visitTransp(self, ctx:SeeDotParser.TranspContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#paren.
    def visitParen(self, ctx:SeeDotParser.ParenContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#func.
    def visitFunc(self, ctx:SeeDotParser.FuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#loop.
    def visitLoop(self, ctx:SeeDotParser.LoopContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#uop.
    def visitUop(self, ctx:SeeDotParser.UopContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#let.
    def visitLet(self, ctx:SeeDotParser.LetContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#id.
    def visitId(self, ctx:SeeDotParser.IdContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#funcCall.
    def visitFuncCall(self, ctx:SeeDotParser.FuncCallContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#maxpool.
    def visitMaxpool(self, ctx:SeeDotParser.MaxpoolContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#addOp.
    def visitAddOp(self, ctx:SeeDotParser.AddOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#binOp.
    def visitBinOp(self, ctx:SeeDotParser.BinOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#specialFunc.
    def visitSpecialFunc(self, ctx:SeeDotParser.SpecialFuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SeeDotParser#intConstList.
    def visitIntConstList(self, ctx:SeeDotParser.IntConstListContext):
        return self.visitChildren(ctx)



del SeeDotParser