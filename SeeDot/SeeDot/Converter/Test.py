# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import math
import numpy as np

def getExpnt(max, B):
	# 16 is B
	return int(np.ceil(np.log2(max) - np.log2((1 << (B - 2)) - 1)))

e = getExpnt(0.8, 8)

a = [20, 10, 30]
b = [10, 20, 30]

print(a == b)
