# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import Common

def HresCalc(h1, h2):
	Lh1Val = [0, 300, 688, 1464, 1978]
	Lh2Val = [0, 86, 387, 903, 1144]
	h1val, h2val = 0, 0
	if(h1 == 0):
		h1val = 0
	elif(h1 <= 4):
		h1val = (Lh1Val[h1] + 141)
	elif(h1 > 4):
		h1val = (Lh1Val[4] + (63 * (h1 - 4)) + 141)

	if(h2 == 0):
		h2val = 0
	elif(h2 <= 4):
		h2val = (Lh2Val[h2] + 141)
	elif(h2 > 4):
		h2val = (Lh2Val[4] + 141)

	return h1val + h2val

def KresCalc(x, xh, Lh):
	Lk = Lh + (2000 * (x - 1))
	LhVal = [0,63, 126, 193, 215, 315]
	return Lk
	
def IresCalc(x, Lkh):
	if(x == 0):
		return 0
	else:
		return (Lkh + 200 * (x - 1))

def AddSubResCalc(x):
	return 82 * x

def SparseMulResCalc(workers):
	return 464 * workers

def SumResCalc(x, fac):
	res = 0
	if(fac == 1):
		res = x
	else:
		res = x + ((x * 0.9) * (fac - 1))
	return res

def ExpResCalc():
	return 219

def TransposeResCalc():
	return 200

def ArgMaxResCalc():
	return 150

def TanhResCalc():
	return 330

def SgnResCalc():
	return 30

def ReluResCalc():
	return 30

def MulCIRResCalc(x):
	if(x == 1):
		return 15
	else:
		return 80

#resource calc of MatMul
def Mul2DTensorResCalc(I,J,K,H1,H2):
	Hval = HresCalc(H1, H2)
	Kval = KresCalc(K, H1, Hval)
	Ival = IresCalc(I, Kval)

	return Ival
	
