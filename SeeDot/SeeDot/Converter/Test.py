# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import math
import numpy as np

def getScale(maxabs:float):
	return int(np.ceil(np.log2(maxabs) - np.log2((1 << (16 - 2)) - 1)))

e = getScale(0.000001)

def printUspsRNN():
	ite = 16
	print("let a0 = (XX[0] * W) in")
	print("let c0 = a0 in")
	print("let g0 = sigmoid(c0 + Bg) in")
	print("let h0 = tanh(c0 + Bh) in")
	print("let H0 = (zeta * (1.0 - g0) + nu) <*> h0 in ")
	print("")
	for i in range(1, ite):
		print("let a%d = (XX[%d] * W) in" % (i, i))
		print("let b%d = (H%d * U) in" % (i, i-1))
		print("let c%d = a%d + b%d in" % (i, i, i))
		print("let g%d = sigmoid(c%d + Bg) in" % (i, i))
		print("let h%d = tanh(c%d + Bh) in" % (i, i))
		print("let H%d = (g%d <*> H%d) + (zeta * (1.0 - g%d) + nu) <*> h%d in " % (i, i, i-1, i, i))
		print("\n")
	print("let score = (H%d * FC) + FCbias in" % (ite-1))
	print("argmax(score)")

def printDsaRNN():
	ite = 125
	print("let a0 = (XX[0] * W1) * W2 in")
	print("let c0 = a0 in")
	print("let g0 = sigmoid(c0 + Bg) in")
	print("let h0 = tanh(c0 + Bh) in")
	print("let H0 = (zeta * (1.0 - g0) + nu) <*> h0 in ")
	print("")
	for i in range(1, ite):
		print("let a%d = (XX[%d] * W1) * W2 in" % (i, i))
		print("let b%d = (H%d * U1) * U2 in" % (i, i-1))
		print("let c%d = a%d + b%d in" % (i, i, i))
		print("let g%d = sigmoid(c%d + Bg) in" % (i, i))
		print("let h%d = tanh(c%d + Bh) in" % (i, i))
		print("let H%d = (g%d <*> H%d) + (zeta * (1.0 - g%d) + nu) <*> h%d in " % (i, i, i-1, i, i))
		print("\n")
	print("let score = (H%d * FC) + FCbias in" % (ite-1))
	print("argmax(score)")

def printSpectakomRNN():
	ite = 7
	print("let a0 = (XX[0] * W1) * W2 in")
	print("let c0 = a0 in")
	print("let g0 = sigmoid(c0 + Bg) in")
	print("let h0 = tanh(c0 + Bh) in")
	print("let H0 = (zeta * (1.0 - g0) + nu) <*> h0 in ")
	print("")
	for i in range(1, ite):
		print("let a%d = (XX[%d] * W1) * W2 in" % (i, i))
		print("let b%d = (H%d * U1) * U2 in" % (i, i-1))
		print("let c%d = a%d + b%d in" % (i, i, i))
		print("let g%d = sigmoid(c%d + Bg) in" % (i, i))
		print("let h%d = tanh(c%d + Bh) in" % (i, i))
		print("let H%d = (g%d <*> H%d) + (zeta * (1.0 - g%d) + nu) <*> h%d in " % (i, i, i-1, i, i))
		print("\n")
	print("let score = ((H%d * FC1) * FC2) + FCBias in" % (ite-1))
	print("argmax(score)")

def treeSum(tmp, length, height_shr, height_noshr):

	count = length
	depth = 0
	shr = True

	while depth < (height_shr + height_noshr):
		if depth >= height_shr:
			shr = False

		for p in range(int(length / 2) + 1):
			if p < (count >> 1):
				sum = tmp[2 * p] + tmp[(2 * p) + 1]
			elif (p == (count >> 1)) and ((count & 1) == 1):
				sum = tmp[2 * p]
			else:
				sum = 0

			if shr:
				tmp[p] = sum / 2
			else:
				tmp[p] = sum
		count = (count + 1) >> 1

		depth += 1

		print(tmp)

	return tmp[0]

def treeSumNew(tmp, count, height_shr, height_noshr):
	if count == 1:
		return tmp[0]

	shr = True

	for depth in range(height_shr + height_noshr):
		if depth >= height_shr:
			shr = False

		for p in range(count // 2):
			sum = tmp[2 * p] + tmp[(2 * p) + 1]

			if shr:
				tmp[p] = sum / 2
			else:
				tmp[p] = sum

		if count % 2 == 1:
			index = count // 2 + 1
			if shr:
				tmp[index - 1] = tmp[count - 1] / 2
			else:
				tmp[index - 1] = tmp[count - 1]

			tmp[index - 1 + 1] = 0
		else:
			tmp[count // 2] = 0
		
		count = (count + 1) >> 1

		print(tmp)

	return tmp[0]

#printUspsRNN()

#printDsaRNN()

#printSpectakomRNN()

def treeSumTest():
	#tmp = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
	tmp = [0.1]
	tmpNew = list(tmp)
	
	sum = treeSum(tmp, 1, 1, 0)
	print(sum)
	
	print('\n\n')
	
	sum = treeSumNew(tmpNew, 1, 0, 1)
	print(sum)
	

e = getScale(0.15637416)
e_int = np.ldexp(0.15637416, 16)

print(e)
print(e_int)

x = np.int16(10248)
print(x)
