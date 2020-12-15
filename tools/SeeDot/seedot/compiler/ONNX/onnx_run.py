# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import onnxruntime
import common
import os, sys
import onnx
from onnx import helper

# First read the ONNX file.
def get_onnx_output(model, input, intermediate_node=None):
	sess = onnxruntime.InferenceSession(file_path)

	x = input
	x = x.astype(np.float32)

	input_name = model.graph.input[0].name

	if (intermediate_node != None):
		intermediate_layer_value_info = helper.ValueInfoProto()
		intermediate_layer_value_info.name = sys.argv[2]
		model.graph.output.extend([intermediate_layer_value_info])
		onnx.save(model, file_path + '_1')
		sess = onnxruntime.InferenceSession(file_path + '_1')
		pred = sess.run([intermediate_layer_value_info.name], {input_name: x})
		return pred

	pred = sess.run(None, {input_name: x})
	return pred
