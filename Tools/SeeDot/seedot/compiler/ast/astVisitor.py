# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''
ASTVisitor contains the dynamic dispatcher used by printAST.py
'''

import seedot.compiler.ast.ast as AST


class ASTVisitor:

    def visit(self, node):
        if isinstance(node, AST.Int):
            return self.visitInt(node)
        elif isinstance(node, AST.Float):
            return self.visitFloat(node)
        elif isinstance(node, AST.ID):
            return self.visitId(node)
        elif isinstance(node, AST.Decl):
            return self.visitDecl(node)
        elif isinstance(node, AST.Transp):
            return self.visitTransp(node)
        elif isinstance(node, AST.Reshape):
            return self.visitReshape(node)
        elif isinstance(node, AST.Maxpool):
            return self.visitMaxpool(node)
        elif isinstance(node, AST.Index):
            return self.visitIndex(node)
        elif isinstance(node, AST.FuncCall):
            return self.visitFuncCall(node)
        elif isinstance(node, AST.Uop):
            return self.visitUop(node)
        elif isinstance(node, AST.Bop1):
            return self.visitBop1(node)
        elif isinstance(node, AST.Bop2):
            return self.visitBop2(node)
        elif isinstance(node, AST.Func):
            return self.visitFunc(node)
        elif isinstance(node, AST.Sum):
            return self.visitSum(node)
        elif isinstance(node, AST.Cond):
            return self.visitCond(node)
        elif isinstance(node, AST.Let):
            return self.visitLet(node)
        else:
            assert False
