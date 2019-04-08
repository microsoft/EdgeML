# Generated from seedot.g4 by ANTLR 4.7
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .seedotParser import seedotParser
else:
    from seedotParser import seedotParser

# This class defines a complete generic visitor for a parse tree produced by seedotParser.

class seedotVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by seedotParser#bop1.
    def visitBop1(self, ctx:seedotParser.Bop1Context):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#bop2.
    def visitBop2(self, ctx:seedotParser.Bop2Context):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#decl.
    def visitDecl(self, ctx:seedotParser.DeclContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#index.
    def visitIndex(self, ctx:seedotParser.IndexContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#sum.
    def visitSum(self, ctx:seedotParser.SumContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#reshape.
    def visitReshape(self, ctx:seedotParser.ReshapeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#float.
    def visitFloat(self, ctx:seedotParser.FloatContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#cond.
    def visitCond(self, ctx:seedotParser.CondContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#int.
    def visitInt(self, ctx:seedotParser.IntContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#transp.
    def visitTransp(self, ctx:seedotParser.TranspContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#paren.
    def visitParen(self, ctx:seedotParser.ParenContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#func.
    def visitFunc(self, ctx:seedotParser.FuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#uop.
    def visitUop(self, ctx:seedotParser.UopContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#let.
    def visitLet(self, ctx:seedotParser.LetContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#id.
    def visitId(self, ctx:seedotParser.IdContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#funcCall.
    def visitFuncCall(self, ctx:seedotParser.FuncCallContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#maxpool.
    def visitMaxpool(self, ctx:seedotParser.MaxpoolContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#addOp.
    def visitAddOp(self, ctx:seedotParser.AddOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#binOp.
    def visitBinOp(self, ctx:seedotParser.BinOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#specialFunc.
    def visitSpecialFunc(self, ctx:seedotParser.SpecialFuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by seedotParser#intConstList.
    def visitIntConstList(self, ctx:seedotParser.IntConstListContext):
        return self.visitChildren(ctx)



del seedotParser