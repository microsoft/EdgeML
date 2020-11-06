
'''

Authors: Shubham Ugare.

Copyright:
Copyright (c) 2020 Microsoft Research
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

import numpy as np
import onnxruntime
import common
import os, sys
import onnx
from onnx import helper

# First read the ONNX file
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
