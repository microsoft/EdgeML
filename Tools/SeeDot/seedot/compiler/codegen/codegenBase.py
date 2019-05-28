# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''
CodegenBase has print functions for the IR classes defined in IR.py
'''

import numpy as np

import seedot.compiler.ir.ir as IR

import seedot.common as Common
import seedot.compiler.type as Type
from seedot.util import *


class CodegenBase:

    def __init__(self, writer):
        self.out = writer

    def printOp(self, ir):
        self.out.printf('%s', ir.name)

    def printInt(self, ir):
        if np.iinfo(np.int16).min <= ir.n <= np.iinfo(np.int16).max:
            self.out.printf('%d', ir.n)
        elif np.iinfo(np.int32).min <= ir.n <= np.iinfo(np.int32).max:
            self.out.printf('%dL', ir.n)
        elif np.iinfo(np.int64).min <= ir.n <= np.iinfo(np.int64).max:
            self.out.printf('%dLL', ir.n)
        else:
            assert False

    def printVar(self, ir):
        self.out.printf('%s', ir.idf)
        for e in ir.idx:
            self.out.printf('[')
            self.print(e)
            self.out.printf(']')

    def printBool(self, ir):
        self.out.printf({True: 'true', False: 'false'}[ir.b])

    def printIntUop(self, ir):
        self.out.printf('(')
        self.print(ir.op)
        self.print(ir.e)
        self.out.printf(')')

    def printIntBop(self, ir):
        self.out.printf('(')
        self.print(ir.e1)
        self.out.printf(' ')
        self.print(ir.op)
        self.out.printf(' ')
        self.print(ir.e2)
        self.out.printf(')')

    def printBoolUop(self, ir):
        self.out.printf('(')
        self.print(ir.op)
        self.print(ir.e)
        self.out.printf(')')

    def printBoolBop(self, ir):
        self.out.printf('(')
        self.print(ir.e1)
        self.out.printf(' ')
        self.print(ir.op)
        self.out.printf(' ')
        self.print(ir.e2)
        self.out.printf(')')

    def printBoolCop(self, ir):
        self.out.printf('(')
        self.print(ir.e1)
        self.out.printf(' ')
        self.print(ir.op)
        self.out.printf(' ')
        self.print(ir.e2)
        self.out.printf(')')

    def printCExpr(self, ir):
        self.out.printf('(')
        self.print(ir.cond)
        self.out.printf(' ? ')
        self.print(ir.et)
        self.out.printf(' : ')
        self.print(ir.ef)
        self.out.printf(')')

    def printExp(self, ir):
        self.out.printf('(exp(')
        self.print(ir.e)
        self.out.printf('))')

    def printTypeCast(self, ir):
        self.out.printf('(')
        self.out.printf('(' + ir.type + ')')
        self.print(ir.expr)
        self.out.printf(')')

    def printAssn(self, ir):
        self.out.printf('', indent=True)
        self.print(ir.var)
        self.out.printf(' = ')
        self.print(ir.e)
        self.out.printf(';\n')

    def printIf(self, ir):
        self.out.printf('if (', indent=True)
        self.print(ir.cond)
        self.out.printf(') {\n')

        self.out.increaseIndent()
        for cmd in ir.trueCmds:
            self.print(cmd)
        self.out.decreaseIndent()

        if len(ir.falseCmds) == 0:
            self.out.printf('}\n', indent=True)
            return

        self.out.printf('} else {\n', indent=True)

        self.out.increaseIndent()
        for cmd in ir.falseCmds:
            self.print(cmd)
        self.out.decreaseIndent()

        self.out.printf('}\n', indent=True)

    def printFor(self, ir):
        self.printForHeader(ir)
        self.out.increaseIndent()
        for cmd in ir.cmd_l:
            self.print(cmd)
        self.out.decreaseIndent()
        self.out.printf('}\n', indent=True)

    def printForHeader(self, ir):
        self.out.printf('for (%s ', IR.DataType.getIntStr(), indent=True)
        self.print(ir.var)
        self.out.printf(' = %d; ', ir.st)
        self.print(ir.cond)
        self.out.printf('; ')
        self.print(ir.var)
        self.out.printf('++) {\n')

    def printWhile(self, ir):
        self.out.printf('while (', indent=True)
        self.print(ir.expr)
        self.out.printf(') {\n')
        self.out.increaseIndent()
        for cmd in ir.cmds:
            self.print(cmd)
        self.out.decreaseIndent()
        self.out.printf('}\n', indent=True)

    def printFuncCall(self, ir):
        self.out.printf("%s(" % ir.name, indent=True)
        keys = list(ir.argList)
        for i in range(len(keys)):
            arg = keys[i]
            if isinstance(arg, IR.Var) and arg.idf in self.decls.keys() and not arg.idf == 'X':
                type = self.decls[arg.idf]
                if isinstance(type, Type.Tensor):
                    if type.dim == 0:
                        x = -1
                    else:
                        x = type.dim - len(arg.idx)
                else:
                    x = -1
            else:
                x = 0
            if x != 0:
                self.out.printf("&")
            self.print(arg)
            if x != 0 and x != -1:
                self.out.printf("[0]" * x)
            if i != len(keys) - 1:
                self.out.printf(", ")
        self.out.printf(");\n\n")

    def printMemset(self, ir):
        self.out.printf('memset(', indent=True)
        self.print(ir.e)
        self.out.printf(', 0, sizeof(%s) * %d);\n' %
                        (IR.DataType.getIntStr(), ir.len))

    def printPrint(self, ir):
        self.out.printf('cout << ', indent=True)
        self.print(ir.expr)
        self.out.printf(' << endl;\n')

    def printPrintAsFloat(self, ir):
        self.out.printf('cout << ((float)(', indent=True)
        self.print(ir.expr)
        self.out.printf(')) * ' + str(2 ** ir.expnt) + ' << "";\n')

    def printPragmas(self, ir):
        if ir.vital == 1:
            self.out.printf('\n')
            self.out.printf(ir.msg + '\n', indent=True)

    def printComment(self, ir):
        self.out.printf('\n')
        self.out.printf('// ' + ir.msg + '\n', indent=True)

    def printProg(self, ir):
        for cmd in ir.cmd_l:
            self.print(cmd)

    def print(self, ir):
        if isinstance(ir, IR.Int):
            return self.printInt(ir)
        elif isinstance(ir, IR.Var):
            return self.printVar(ir)
        elif isinstance(ir, IR.Bool):
            return self.printBool(ir)
        elif isinstance(ir, IR.IntUop):
            return self.printIntUop(ir)
        elif isinstance(ir, IR.IntBop):
            return self.printIntBop(ir)
        elif isinstance(ir, IR.BoolUop):
            return self.printBoolUop(ir)
        elif isinstance(ir, IR.BoolBop):
            return self.printBoolBop(ir)
        elif isinstance(ir, IR.BoolCop):
            return self.printBoolCop(ir)
        elif isinstance(ir, IR.CExpr):
            return self.printCExpr(ir)
        elif isinstance(ir, IR.Exp):
            return self.printExp(ir)
        elif isinstance(ir, IR.TypeCast):
            return self.printTypeCast(ir)
        elif isinstance(ir, IR.Assn):
            return self.printAssn(ir)
        elif isinstance(ir, IR.If):
            return self.printIf(ir)
        elif isinstance(ir, IR.For):
            return self.printFor(ir)
        elif isinstance(ir, IR.While):
            return self.printWhile(ir)
        elif isinstance(ir, IR.FuncCall):
            return self.printFuncCall(ir)
        elif isinstance(ir, IR.Memset):
            return self.printMemset(ir)
        elif isinstance(ir, IR.Print):
            return self.printPrint(ir)
        elif isinstance(ir, IR.PrintAsFloat):
            return self.printPrintAsFloat(ir)
        elif isinstance(ir, IR.Comment):
            return self.printComment(ir)
        elif isinstance(ir, IR.Prog):
            return self.printProg(ir)
        elif isinstance(ir, IR.Op.Op):
            return self.printOp(ir)
        else:
            assert False

    def printAll(self, prog: IR.Prog, expr: IR.Expr):
        self.printPrefix()
        self.print(prog)
        self.printSuffix(expr)

    def printVarDecls(self):
        for decl in self.decls:
            if decl in self.globalVars:
                continue
            typ_str = IR.DataType.getIntStr()
            idf_str = decl
            type = self.decls[decl]
            if Type.isInt(type):
                shape_str = ''
            elif Type.isTensor(type):
                shape_str = ''.join(['[' + str(n) + ']' for n in type.shape])
            self.out.printf('%s %s%s;\n', typ_str, idf_str,
                            shape_str, indent=True)
        self.out.printf('\n')

    def printConstDecls(self):
        for cnst in self.cnsts:
            var, num = cnst, self.cnsts[cnst]
            if np.iinfo(np.int16).min <= num <= np.iinfo(np.int16).max:
                self.out.printf('%s = %d;\n', var, num, indent=True)
            elif np.iinfo(np.int32).min <= num <= np.iinfo(np.int32).max:
                self.out.printf('%s = %dL;\n', var, num, indent=True)
            elif np.iinfo(np.int64).min <= num <= np.iinfo(np.int64).max:
                self.out.printf('%s = %dLL;\n', var, num, indent=True)
            else:
                assert False
