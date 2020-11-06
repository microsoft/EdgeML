
'''

Authors: Shubham Ugare.

Copyright:
Copyright (c) 2018 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

import numpy.random
import numpy as np
import os, sys
import onnx
from onnx import helper
import math
from onnx import numpy_helper

import seedot.compiler.ONNX.common as common

#refactor this later. Also part of paramBuilder file.
class Param:

    def __init__(self, name, shape, range):
        self.name = name
        self.shape = shape
        self.range = range

        self.sparse = False

# shift to common
def get_range(np_array):
	return (np.min(np_array), np.max(np_array))      

def getParams(file_path):	

	model = onnx.load(file_path)
	graph_def = model.graph
	
	model_name_to_val_dict = { init_vals.name: numpy_helper.to_array(init_vals).tolist() for init_vals in model.graph.initializer}

	paramList = []

	for init_vals in model.graph.initializer:
		name = 	init_vals.name	
		shape = numpy_helper.to_array(init_vals).shape
		range = get_range(numpy_helper.to_array(init_vals))
		param = Param(name, shape, range)
		param.data = numpy_helper.to_array(init_vals).reshape((1,-1)).tolist()
		paramList.append(param) 

	return paramList	

def preprocess_batch_normalization(graph_def, model_name_to_val_dict):
	# set names to graph nodes if not present
	for node in graph_def.node: 
		node.name = node.output[0]
		# Update the batch normalization scale and B
		# so that mean and var are not required
		if(node.op_type == 'BatchNormalization'):
			# scale
			gamma = model_name_to_val_dict[node.input[1]]
			# B
			beta = model_name_to_val_dict[node.input[2]]
			mean = model_name_to_val_dict[node.input[3]]
			var = model_name_to_val_dict[node.input[4]]
			for i in range(len(gamma)):
				rsigma = 1/math.sqrt(var[i]+1e-5)
				gamma[i] = gamma[i]*rsigma
				beta[i] = beta[i]-gamma[i]*mean[i]	
				mean[i] = 0
				var[i] = 1-1e-5

	# Just testing if the correct values are put			
	model_name_to_val_dict2 = {}
	for init_vals in graph_def.initializer:
		# TODO: Remove float_data
		model_name_to_val_dict2[init_vals.name] = init_vals.float_data		
	for node in graph_def.node: 
		node.name = node.output[0]
		if(node.op_type == 'BatchNormalization'):
			mean = model_name_to_val_dict[node.input[3]]
			for val in mean:
				assert(val == 0)

if __name__ == "__main__":
	main()											