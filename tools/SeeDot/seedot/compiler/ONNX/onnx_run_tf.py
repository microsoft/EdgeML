
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

'''
onnx_run is faster but may not support all operations
onnx_run_tf uses tensorflow backend to run the inference
'''

import numpy as np
import common
import os, sys
import onnx
from onnx import helper
from onnx_tf.backend import prepare
from onnx import TensorProto

def main():
	# First read the ONNX file
	if (len(sys.argv) < 2):
		print("TF python file unspecified.", file=sys.stderr)
		exit(1)

	file_name = sys.argv[1]
	file_path = 'models/' + file_name
	model_name = file_name[:-5] # name without the '.onnx' extension
	model = onnx.load(file_path)
	model = preprocess_for_tf(model)

	x = np.load('debug/' + model_name + '/' + model_name + '_input.npy')
	x = x.astype(np.float32)

	input_name = model.graph.input[0].name
	output_name = model.graph.output[0].name

	if (len(sys.argv) > 2):
		intermediate_layer_value_info = helper.ValueInfoProto()
		intermediate_layer_value_info_name = 'tf_' + sys.argv[2]
		intermediate_layer_value_info = helper.make_tensor_value_info(intermediate_layer_value_info_name, TensorProto.FLOAT, [])
		model.graph.output.extend([intermediate_layer_value_info])
		output = prepare(model).run(x) 
		pred = getattr(output, intermediate_layer_value_info_name)
		np.save('debug/' + model_name + '/' + model_name + '_debug', pred)
		with open('debug/onnx_debug.txt', 'w') as f:
			f.write(common.numpy_float_array_to_float_val_str(pred))
		print("Saving the onnx runtime intermediate output for " + intermediate_layer_value_info.name)
		exit() 

	output = prepare(model).run(x) 
	pred = getattr(output, output_name)
	np.save('debug/' + model_name + '/' + model_name + '_output', pred)
	with open('debug/onnx_output.txt', 'w') as f:
			f.write(common.numpy_float_array_to_float_val_str(pred))
	output_dims = common.proto_val_to_dimension_tuple(model.graph.output[0])
	print("Saving the onnx runtime output of dimension " + str(output_dims))

def preprocess_for_tf(model):
	for init_vals in model.graph.initializer:
		init_vals.name = 'tf_' + init_vals.name

	for inp in model.graph.input:
		inp.name = 'tf_' + inp.name

	for op in model.graph.output:
		op.name = 'tf_' + op.name

	for node in model.graph.node:
		node.name = 'tf_' + node.name
		for i in range(len(node.input)):
			node.input[i] = 'tf_' + node.input[i]
		for i in range(len(node.output)):
			node.output[i] = 'tf_' + node.output[i]	
	return model

if __name__ == "__main__":
	main()				
