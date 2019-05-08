# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np

from IR.IR import *

import Common
from Util import *

def init():
	global zero, one, negone, negmax

	zero = Int(0)
	one = Int(1)
	negone = Int(-1)
	negmax = Int.negMax()

def add(e1:Expr, e2:Expr) -> Expr: return IntBop(e1, Op.Op['+'], e2)
def sub(e1:Expr, e2:Expr) -> Expr: return IntBop(e1, Op.Op['-'], e2)
def mul(e1:Expr, e2:Expr) -> Expr: return IntBop(e1, Op.Op['*'], e2)
def div(e1:Expr, e2:Expr) -> Expr: return IntBop(e1, Op.Op['/'], e2)

def inc(e:Expr) -> Expr: return add(e, one)
def dec(e:Expr) -> Expr: return sub(e, one)

def andd(e1:Expr, e2:Expr) -> Expr: return BoolBop(e1, Op.Op['&&'], e2)
def orr(e1:Expr, e2:Expr) -> Expr: return BoolBop(e1, Op.Op['||'], e2)

def eq(e1:Expr, e2:Expr) -> Expr: return BoolCop(e1, Op.Op['=='], e2)
def neq(e1:Expr, e2:Expr) -> Expr: return BoolCop(e1, Op.Op['!='], e2)
def lt(e1:Expr, e2:Expr) -> Expr: return BoolCop(e1, Op.Op['<'],  e2)
def lte(e1:Expr, e2:Expr) -> Expr: return BoolCop(e1, Op.Op['<='], e2)
def gt(e1:Expr, e2:Expr) -> Expr: return BoolCop(e1, Op.Op['>'],  e2)
def gte(e1:Expr, e2:Expr) -> Expr: return BoolCop(e1, Op.Op['>='], e2)

def bitAnd(e1:Expr, e2:Expr) -> Expr: return IntBop(e1, Op.Op['&'], e2)

def max(e1:Expr, e2:Expr) -> Expr:
	return CExpr(BoolCop(e1, Op.Op['>'], e2), e1, e2)

def max_uint(e1:Expr, e2:Expr) -> Expr:
	return max(e1, e2)

def max_sint(e1:Expr, e2:Expr) -> Expr:
	return cond_zero(e1, cond_zero(e2, max_uint(e1, e2), e1), cond_zero(e2, e2, max_uint(e1, e2)))

def negate(e:Expr) -> Expr:
	return IntUop(Op.Op['-'], e)

def shl(e:Expr, n:int) -> Expr:
	assert n >= 0
	if n == 0: return e
	return IntBop(e, Op.Op['<<'], Int(n))

def shrUint(e:Expr, n:int) -> Expr:
	assert n >= 0
	if n == 0: return e
	return IntBop(e, Op.Op['>>'], Int(n))

def shr(e:Expr, n:int) -> Expr:
	assert n >= 0
	if n == 0: return e
	
	if getShrType() == "shr":
		return cond_zero(e, IntBop(e, Op.Op['>>'], Int(n)), IntUop(Op.Op['-'], IntBop(IntUop(Op.Op['-'], e), Op.Op['>>'], Int(n))))
	elif getShrType() == "shr+":
		mask = Int((2 ** n) - 1)
		return cond_zero(e, IntBop(e, Op.Op['>>'], Int(n)), IntBop(IntBop(e, Op.Op['+'], mask), Op.Op['>>'], Int(n)))
	elif getShrType() == "div":
		intVar = Int(2 ** n)
		if intVar.n == 0: return zero
		return div(e, intVar)
	elif getShrType() == "negate":
		return cond_zero(e, IntBop(e, Op.Op['>>'], Int(n)), IntBop(IntBop(IntBop(e, Op.Op['^'], negone), Op.Op['>>'], Int(n)), Op.Op['^'], negone))
	else:
		assert False

def shrVar(e:Expr, n:Var) -> Expr:
	# Using "shr" version
	return cond_zero(e, IntBop(e, Op.Op['>>'], n), IntUop(Op.Op['-'], IntBop(IntUop(Op.Op['-'], e), Op.Op['>>'], n)))

def castToInt(e:Expr):
	return TypeCast(DataType.getIntStr(), e)

def castToFloat(e:Expr):
	return TypeCast(DataType.getFloatStr(), e)

def addIndex(var:Var, indices:list, prefix:bool=False) -> Var:
	if prefix == False:
		return Var(var.idf, var.idx + indices, var.inputVar)
	else:
		return Var(var.idf, indices + var.idx, var.inputVar)

def cond_zero(e:Expr, et:Expr, ef:Expr) -> Expr:
	return CExpr(BoolCop(e, Op.Op['>'], zero), et, ef)

def relu(e:Expr): return cond_zero(e, e, zero)

def loop_shr(lhs:Expr, rhs:Expr, shape:list, iters:list, n:int) -> CmdList:
	lhs_elt = addIndex(lhs, iters)
	rhs_elt = addIndex(rhs, iters)
	return loop(shape, iters, [Assn(lhs_elt, shr(rhs_elt, n))])

def initVarToZero(e:Expr) -> Cmd: return Assn(e, Int(0))

def incCmd(e:Var) -> Cmd: return Assn(e, inc(e))
def decCmd(e:Var) -> Cmd: return Assn(e, dec(e))

def concatPrograms(*prog_l, resource=0):
	cmd_l = flatten([prog.cmd_l for prog in prog_l])
	Res = 0
	for x in prog_l:
		Res = Res + x.resource
	return Prog(cmd_l, resource=Res)

# multiplexer
def add_idx_priv(var:Var, e:Expr, n:int, offset:int=0) -> Expr:
	assert n >= 1
	# add_idx_priv_ifelse(var, e, n, offset)
	add_idx_priv_ops(var, e, n, offset)

def add_idx_priv_ifelse(var:Var, e:Expr, n:int, offset:int=0) -> Expr:
	'''
	Using if-else:
	--------------
	for n=3:
	if e & 100 == 0:
	  if e & 010 == 0:
	    if e & 001 == 0: var[000]
	    else: var[001]
	  else:
	    if e & 001 == 0: var[010]
	    else: var[011]
	else: ...
	'''
	mask = 1 << (n - 1)
	expr_cmp = eq(IntBop(e, Op.Op['&'], Int(mask)), zero)
	if n == 1:
		return CExpr(expr_cmp, addIndex(var, [Int(offset + 0)]), addIndex(var, [Int(offset + mask)]))
	else:
		return CExpr(expr_cmp, add_idx_priv_ifelse(var, e, n - 1, offset + 0), add_idx_priv_ifelse(var, e, n - 1, offset + mask))

def add_idx_priv_ops(var:Var, e:Expr, n:int, offset:int=0) -> Expr:
	'''
	Using *, +:
	-----------
	for n = 2:
	(1-(e&10)>>1) * ((1-(e&01)>>0)*var[00] + ((e&01)>>0)*var[01]) +
	( (e&10)>>1) * ((1-(e&01)>>0)*var[10] + ((e&01)>>0)*var[11])
	'''
	mask = 1 << (n - 1)
	expr_1 = shrUint(IntBop(e, Op.Op['&'], Int(mask)), n - 1)
	expr_0 = sub(one, expr_1)
	if n == 1:
		return add(mul(expr_0, addIndex(var, [Int(offset + 0)])), mul(expr_1, addIndex(var, [Int(offset + mask)])))
	else:
		return add(mul(expr_0, add_idx_priv_ops(var, e, n - 1, offset + 0)), mul(expr_1, add_idx_priv_ops(var, e, n - 1, offset + mask)))

# iteration
def loop(shape:list, iters:list, cmdl_body:CmdList, factor=0) -> CmdList:
	cmdl_for = cmdl_body
	for i in reversed(range(len(shape))):
		cmdl_for = [For(iters[i], 0, lt(iters[i], Int(shape[i])), cmdl_for, factor)]
	return cmdl_for

def print_loop(shape:list, iters:list, cmdl_body:CmdList, factor=0) -> CmdList:
	cmdl_for = cmdl_body
	for i in reversed(range(len(shape))):
		cmdl_for = [For(iters[i], 0, lt(iters[i], Int(shape[i])), cmdl_for, factor), Print(Var('""'))]
	return cmdl_for
