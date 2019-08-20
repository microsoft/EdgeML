# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from enum import Enum
import numpy as np

import seedot.common as Common
from seedot.util import *


class Op:
    Op = Enum('Op', '+ - * / << >> & | ^ ~ ! && || < <= > >= == !=')
    Op.print = lambda self, writer: writer.printf('%s', self.name)
    Op.op_list = lambda op_str: list(map(lambda x: Op.Op[x], op_str.split()))


class Expr:
    pass


class IntExpr(Expr):
    pass


class BoolExpr(Expr):
    pass


class Int(IntExpr):

    @staticmethod
    def max():
        return DataType.getMax()

    @staticmethod
    def negMax():
        return DataType.getNegMax()

    def __init__(self, n: int):
        self.n = DataType.getInt(n)

    def subst(self, from_idf: str, to_e: Expr):
        return Int(self.n)


class Var(IntExpr):

    def __init__(self, idf: str, idx: list=[], inputVar=False):
        self.idf = idf
        self.idx = idx
        self.inputVar = inputVar

    def subst(self, from_idf: str, to_e: Expr):
        idx_new = list(map(lambda e: e.subst(from_idf, to_e), self.idx))
        if self.idf != from_idf:
            return Var(self.idf, idx_new, self.inputVar)
        else:
            if isinstance(to_e, Var):
                return Var(to_e.idf, to_e.idx + idx_new, to_e.inputVar and self.inputVar)
            elif isinstance(to_e, Int):
                return to_e
            else:
                assert False


class Bool(BoolExpr):

    def __init__(self, b: bool):
        self.b = b

    def subst(self, from_idf: str, to_e: Expr):
        return Bool(self.b)


class IntUop(IntExpr):

    def __init__(self, op: Op.Op, e: IntExpr):
        assert op in Op.Op.op_list('- ~')
        self.op = op
        self.e = e

    def subst(self, from_idf: str, to_e: Expr):
        return IntUop(self.op, self.e.subst(from_idf, to_e))


class IntBop(IntExpr):

    def __init__(self, e1: IntExpr, op: Op.Op, e2: IntExpr):
        assert op in Op.Op.op_list('+ - * / << >> & | ^')
        self.e1 = e1
        self.op = op
        self.e2 = e2

    def subst(self, from_idf: str, to_e: Expr):
        return IntBop(self.e1.subst(from_idf, to_e), self.op, self.e2.subst(from_idf, to_e))


class BoolUop(BoolExpr):

    def __init__(self, op: Op.Op, e: BoolExpr):
        assert op in Op.Op.op_list('')
        self.op = op
        self.e = e

    def subst(self, from_idf: str, to_e: Expr):
        return BoolUop(self.op, self.e.subst(from_idf, to_e))


class BoolBop(BoolExpr):

    def __init__(self, e1: BoolExpr, op: Op.Op, e2: BoolExpr):
        assert op in Op.Op.op_list('&& ||')
        self.e1 = e1
        self.op = op
        self.e2 = e2

    def subst(self, from_idf: str, to_e: Expr):
        return BoolBop(self.e1.subst(from_idf, to_e), self.op, self.e2.subst(from_idf, to_e))


class BoolCop(BoolExpr):

    def __init__(self, e1: IntExpr, op: Op.Op, e2: IntExpr):
        assert op in Op.Op.op_list('< <= > >= == !=')
        self.e1 = e1
        self.op = op
        self.e2 = e2

    def subst(self, from_idf: str, to_e: Expr):
        return BoolCop(self.e1.subst(from_idf, to_e), self.op, self.e2.subst(from_idf, to_e))


class CExpr(Expr):

    def __init__(self, cond: BoolExpr, et: Expr, ef: Expr):
        self.cond = cond
        self.et = et
        self.ef = ef

    def subst(self, from_idf: str, to_e: Expr):
        return CExpr(self.cond.subst(from_idf, to_e), self.et.subst(from_idf, to_e), self.ef.subst(from_idf, to_e))


class Exp(IntExpr):

    def __init__(self, e: IntExpr):
        self.e = e

    def subst(self, from_idf: str, to_e: Expr):
        return Exp(self.e.subst(from_idf, to_e))


class TypeCast(IntExpr):

    def __init__(self, type, expr: Expr):
        self.type = type
        self.expr = expr

    def subst(self, from_idf: str, to_e: Expr):
        return TypeCast(self.type, self.expr.subst(from_idf, to_e))


class Cmd:
    pass


class CmdList:
    pass


class Assn(Cmd):

    def __init__(self, var: Var, e: Expr):
        self.var = var
        self.e = e

    def subst(self, from_idf: str, to_e: Expr):
        return Assn(self.var.subst(from_idf, to_e), self.e.subst(from_idf, to_e))


class If(Cmd):

    def __init__(self, cond: Expr, trueCmds: CmdList, falseCmds: CmdList=[]):
        self.cond = cond
        self.trueCmds = trueCmds
        self.falseCmds = falseCmds

    def subst(self, from_idf: str, to_e: Expr):
        trueCmdsNew = list(
            map(lambda cmd: cmd.subst(from_idf, to_e), self.trueCmds))
        falseCmdsNew = list(
            map(lambda cmd: cmd.subst(from_idf, to_e), self.falseCmds))
        return If(self.cond.subst(from_idf, to_e), trueCmdsNew, falseCmdsNew)


class For(Cmd):

    def __init__(self, var: Var, st: int, cond: Expr, cmd_l: CmdList, fac=0):
        self.var = var
        self.st = DataType.getInt(st)
        self.cond = cond
        self.cmd_l = cmd_l
        self.factor = fac

    def subst(self, from_idf: str, to_e: Expr):
        cmd_l_new = list(
            map(lambda cmd: cmd.subst(from_idf, to_e), self.cmd_l))
        return For(self.var, self.st, self.cond.subst(from_idf, to_e), cmd_l_new, self.factor)


class While(Cmd):

    def __init__(self, expr: BoolExpr, cmds: CmdList):
        self.expr = expr
        self.cmds = cmds

    def subst(self, from_idf: str, to_e: Expr):
        cmds_new = list(map(lambda cmd: cmd.subst(from_idf, to_e), self.cmds))
        return While(self.expr.subst(from_idf, to_e), cmds_new)


class FuncCall(Cmd):

    def __init__(self, name, argList):
        self.name = name
        self.argList = argList

    def subst(self, from_idf: str, to_e: Expr):
        argList_new = dict(
            map(lambda cmd: (cmd[0].subst(from_idf, to_e), cmd[1]), self.argList.items()))
        return FuncCall(self.name, argList_new)


class Memset(Cmd):

    def __init__(self, e: Var, len: int, dim=1, lens=[]):
        self.e = e
        self.len = len
        self.dim = dim
        self.lens = lens

    def subst(self, from_idf: str, to_e: Expr):
        return Memset(self.e.subst(from_idf, to_e), self.len)


class Print(Cmd):

    def __init__(self, expr: Expr):
        self.expr = expr

    def subst(self, from_idf: str, to_e: Expr):
        return Print(self.expr.subst(from_idf, to_e))


class PrintAsFloat(Cmd):

    def __init__(self, expr: Expr, expnt: int):
        self.expr = expr
        self.expnt = expnt

    def subst(self, from_idf: str, to_e: Expr):
        return PrintAsFloat(self.expr.subst(from_idf, to_e), self.expnt)


class Comment(Cmd):

    def __init__(self, msg):
        self.msg = msg

    def subst(self, from_idf: str, to_e: Expr):
        return Comment(self.msg)


class Prog:

    def __init__(self, cmd_l: CmdList, resource=0):
        self.cmd_l = cmd_l
        self.resource = resource

    def subst(self, from_idf: str, to_e: Expr):
        cmd_l_new = list(
            map(lambda cmd: cmd.subst(from_idf, to_e), self.cmd_l))
        return Prog(cmd_l_new, self.resource)


class DataType:
    intType = {Common.Target.Arduino: {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64},
               Common.Target.X86: {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
               }
    intStr = {Common.Target.Arduino: 'MYINT',
              Common.Target.X86: 'MYINT'
              }
    floatStr = "float"

    @staticmethod
    def getInt(x: int):
        target = getTarget()
        wordLen = Common.wordLength
        return DataType.intType[target][wordLen](x)

    @staticmethod
    def getIntClass():
        target = getTarget()
        wordLen = Common.wordLength
        return DataType.intType[target][wordLen]

    @staticmethod
    def getIntStr():
        target = getTarget()
        return DataType.intStr[target]

    @staticmethod
    def getFloatStr():
        return DataType.floatStr

    @staticmethod
    def getMax():
        intClass = DataType.getIntClass()
        return intClass(np.iinfo(intClass).max)

    @staticmethod
    def getNegMax():
        intClass = DataType.getIntClass()
        return intClass(np.iinfo(intClass).min)
