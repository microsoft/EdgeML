
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

import os, sys

#Add SeeDot directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SeeDot')) 

# For this warning: https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import _pickle as pickle
import onnx
import onnx.shape_inference
from onnx.helper import make_tensor_value_info
from onnx import TensorProto

import seedot.compiler.ast.ast as AST
from seedot.compiler.ast.printAST import PrintAST 
from seedot.compiler.ast.mtdAST import MtdAST
from seedot.compiler.ONNX.ONNXNodesAST import ONNXNodesAST

import numpy
import seedot.compiler.ONNX.common as common

import numpy as np
np.set_printoptions(threshold=np.inf)

DEBUG = False
out_var_prefix = "J"

def process_input_variables(program, innermost_let_ast_node, node_name_to_out_var_dict, out_var_count, mtdAST, graph_def, value_info):
	node = graph_def.input[0]
	curAst = ONNXNodesAST.Input(node, value_info, node_name_to_out_var_dict)
	mtdForCurAST = {AST.ASTNode.mtdKeyTFOpName : 'Input',
						AST.ASTNode.mtdKeyTFNodeName : node.name}
	cur_out_var_ast_node = AST.ID(node.name)	

	if program:
		assert(type(innermost_let_ast_node) is AST.Let)
		newNode = AST.Let(node.name, curAst, cur_out_var_ast_node)
		# mtdAST.visit(newNode, mtdForCurAST)
		# Updating the innermost Let AST node and the expression for previous Let Node 
		innermost_let_ast_node.expr = newNode
		innermost_let_ast_node = newNode
	else:
		innermost_let_ast_node = AST.Let(node.name, curAst, cur_out_var_ast_node)
		# mtdAST.visit(innermost_let_ast_node, mtdForCurAST)
		innermost_let_ast_node.depth = 0
		program = innermost_let_ast_node

	node_name_to_out_var_dict[node.name] = node.name

	for node in graph_def.initializer:
		if(DEBUG):
			print("Node information")
			print(node)	
	
		curAst = ONNXNodesAST.Input(node, value_info, node_name_to_out_var_dict, node)
		mtdForCurAST = {AST.ASTNode.mtdKeyTFOpName : 'Input',
							AST.ASTNode.mtdKeyTFNodeName : node.name}
		if (curAst is None):
			continue		
	
		cur_out_var_ast_node = AST.ID(node.name)	

		if program:
			assert(type(innermost_let_ast_node) is AST.Let)
			newNode = AST.Let(node.name, curAst, cur_out_var_ast_node)
			# mtdAST.visit(newNode, mtdForCurAST)
			# Updating the innermost Let AST node and the expression for previous Let Node 
			innermost_let_ast_node.expr = newNode
			innermost_let_ast_node = newNode
		else:
			innermost_let_ast_node = AST.Let(node.name, curAst, cur_out_var_ast_node)
			# mtdAST.visit(innermost_let_ast_node, mtdForCurAST)
			innermost_let_ast_node.depth = 0
			program = innermost_let_ast_node
	
		node_name_to_out_var_dict[node.name] = node.name
	return (program, innermost_let_ast_node, out_var_count)	

def process_onnx_nodes(innermost_let_ast_node, node_name_to_out_var_dict, out_var_count, mtdAST, graph_def, value_info):	
	for node in graph_def.node:
		if(DEBUG):
			print("Node information")
			print(node)	

		print("Processing " + node.op_type + "\n")	

		func = getattr(ONNXNodesAST, node.op_type) 
		(innermost_let_ast_node, out_var_count) = func(node, value_info, node_name_to_out_var_dict, innermost_let_ast_node, out_var_count, mtdAST)					

		assert(type(innermost_let_ast_node) is AST.Let)

def get_seedot_ast(file_path):
	sys.setrecursionlimit(10000)
	print(os.getcwd())

	# load the model and extract the graph		
	model = onnx.load(file_path)
	graph_def = model.graph

	# print(model.graph.value_info)
	# Before shape inference (model.graph.value_info) should have shapes of all the variables and constants 
	model.graph.value_info.append(make_tensor_value_info(model.graph.input[0].name, TensorProto.FLOAT, common.proto_val_to_dimension_tuple(model.graph.input[0])))
	model.graph.value_info.append(make_tensor_value_info(model.graph.output[0].name, TensorProto.FLOAT, common.proto_val_to_dimension_tuple(model.graph.output[0])))

	# print(model.graph.value_info)

	for init_vals in model.graph.initializer:
		model.graph.value_info.append(make_tensor_value_info(init_vals.name, TensorProto.FLOAT, tuple(init_vals.dims)))	

	if(DEBUG):	
		print("Shape inference *****************")
		print(model.graph.value_info)

	inferred_model = onnx.shape_inference.infer_shapes(model)
	
	if(DEBUG):	
		print("Printing shape ******************")
		print(inferred_model.graph.value_info)
		print("Done ******************")

	# value_info: dictionary of name -> (type, dimension tuple)
	value_info = {}
	for val in inferred_model.graph.value_info:
		value_info[val.name] = (val.type.tensor_type.elem_type, common.proto_val_to_dimension_tuple(val))

	# Iterate through the ONNX graph nodes and translate them to SeeDot AST nodes	
	program = None
	innermost_let_ast_node = None
	node_name_to_out_var_dict = {}
	out_var_count = 0
	mtdAST = MtdAST()

	(program, innermost_let_ast_node, out_var_count) = process_input_variables(program, innermost_let_ast_node, node_name_to_out_var_dict, out_var_count, mtdAST, graph_def, value_info)

	process_onnx_nodes(innermost_let_ast_node, node_name_to_out_var_dict, out_var_count, mtdAST, graph_def, value_info)

	PrintAST().visit(program)	
	
	common.write_debug_info(node_name_to_out_var_dict)
	return program			