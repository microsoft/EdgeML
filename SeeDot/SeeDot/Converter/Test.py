# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import math
import numpy as np

def getScale(maxabs:float):
	return int(np.ceil(np.log2(maxabs) - np.log2((1 << (16 - 2)) - 1)))

e = getScale(0.000001)

print(e)

x = None

y = [x]

print(y)
