
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
import numpy
import os
import _pickle as pickle
import re

def proto_val_to_dimension_tuple(proto_val):
	return tuple([dim.dim_value for dim in proto_val.type.tensor_type.shape.dim])

def numpy_float_array_to_fixed_point_val_str(input_array, scale):
	cnt = 0
	chunk = ''
	for val in numpy.nditer(input_array):
		val = int(val*(2**scale))
		chunk += str(val) + '\n'
		cnt += 1
	return (chunk, cnt)	

def numpy_float_array_to_float_val_str(input_array):
	chunk = ''
	for val in numpy.nditer(input_array):
		chunk += str(val) + '\n'
	return chunk		

def write_debug_info(node_name_to_out_var_dict):
	if not os.path.exists('debug'):
		os.makedirs('debug')	

	with open('debug/onnx_seedot_name_map.pkl', 'wb') as f:
		pickle.dump(node_name_to_out_var_dict, f)	

	with open('debug/onnx_seedot_name_map.txt', 'w') as f:
		for val in node_name_to_out_var_dict:
			f.write(val + '   ' + node_name_to_out_var_dict[val] + '\n')


def merge_name_map():
	onnx_seedot_name_map = pickle.load(open('debug/onnx_seedot_name_map.pkl', 'rb'))
	seedot_ezpc_name_map = pickle.load(open('debug/seedot_ezpc_name_map.pkl', 'rb'))

	with open('debug/onnx_ezpc_name_map.txt', 'w') as f:
		for val in onnx_seedot_name_map:
			f.write(val + '   ' + seedot_ezpc_name_map[onnx_seedot_name_map[val]])	

def get_seedot_name_from_onnx_name(onnx_name):
	onnx_seedot_name_map = pickle.load(open('debug/onnx_seedot_name_map.pkl', 'rb'))
	print(onnx_seedot_name_map[onnx_name])

def parse_output(scale):
	f = open('debug/cpp_output_raw.txt', 'r')
	g = open('debug/cpp_output.txt', 'w')
	chunk = ''
	for line in f:	
		if line.rstrip().replace('-','0').isdigit():
			val = float(line.rstrip())
			val = val/(2**scale)
			chunk += str(val) + '\n'
	g.write(chunk)
	g.close()

def extract_txt_to_numpy_array(file):
	f = open(file, 'r')
	op = [float(line.rstrip()) for line in f]
	f.close()
	return numpy.array(op, dtype=numpy.float32)

def match_debug(decimal=4):
	a = extract_txt_to_numpy_array('debug/onnx_debug.txt')
	b = extract_txt_to_numpy_array('debug/cpp_output.txt')
	numpy.testing.assert_almost_equal(a, b, decimal)	

def match_output(decimal=4):
	a = extract_txt_to_numpy_array('debug/onnx_output.txt')
	b = extract_txt_to_numpy_array('debug/cpp_output.txt')
	numpy.testing.assert_almost_equal(a, b, decimal)		
		
def add_openmp_threading_to_convolution(file):
	with open(file, 'r+') as f:
		newfilename = file[:-5]+'1.cpp'
		g = open(newfilename, 'w')
		content = f.read()
		content1 =  re.sub('void Conv3D\(.*','\g<0> \n #pragma omp parallel for collapse(5) ', content)
		content2 =  re.sub('void ConvTranspose3D\(.*','\g<0> \n #pragma omp parallel for collapse(5) ', content1)
		g.write(content2)
		g.close()

