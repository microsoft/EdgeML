# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy.random
import numpy as np
import os, sys
import onnx
from onnx import helper
import math
from onnx import numpy_helper

import seedot.compiler.ONNX.common as common


#TODO: Refactor this later. Also part of paramBuilder file.
class Param:

    def __init__(self, name, shape, range):
        self.name = name
        self.shape = shape
        self.range = range
        self.sparse = False


#TODO: Shift to common.py
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
	# Set names to graph nodes if not present.
	for node in graph_def.node:
		node.name = node.output[0]
		# Update the batch normalization scale and B
		# so that mean and var are not required.
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

	# Just testing if the correct values are set.
	model_name_to_val_dict2 = {}
	for init_vals in graph_def.initializer:
		# TODO: Remove float_data.
		model_name_to_val_dict2[init_vals.name] = init_vals.float_data
	for node in graph_def.node:
		node.name = node.output[0]
		if(node.op_type == 'BatchNormalization'):
			mean = model_name_to_val_dict[node.input[3]]
			for val in mean:
				assert(val == 0)

if __name__ == "__main__":
	main()
