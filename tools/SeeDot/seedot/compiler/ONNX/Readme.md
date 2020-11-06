# Introduction
This part of the code compiles the onnx model to SeeDot AST. 

A model name must be provided to the `compile.sh` script and the model must be placed in `./models` directory 
The script can be run with `./compile.sh model_name.onnx` command on the command line

1) The script calls `onnx_run.py` to generate a random input of size matching the input size of the model. `onnx_run.py` further runs the model using `onnxruntime` and stores the output result as a `numpy` array. The input is stored as `model_name_input.npy` and the output is stored as `model_name_output.npy`

2) Then it runs `process_onnx.py`. This python code combines `model_name_input.npy` and the values of other variables stored in the model to generate a `model_name_input.h` file which is later fed to the final code as input. `model_name_input.h` has all the values stored as fixed-point integers using the value of scale in the script. 

3) Then it runs `onnx inference` to calculate the input and output size for each onnx node. and it parses the onnx model using `OnnxNodesAST.py` and creates a `SeeDot` AST which is stored as `model_name.pkl` (using pickle)

4) The `compile.sh` script further converts the SeeDot AST to EzPC code and the `EzPC` code is finally converted to the `CPP` program. This CPP program is compiled and ran with the given input. The output is stored as `debug/cpp_output_raw.txt`. Again, using the same scale this raw output is converted to the floating-point output and stored in `debug/cpp_output.txt` for easier manual comparison with the original onnx output. 

# Debugging and Logging
Since debugging the code is an arduous task, several things are logged in the following files

To log the values of specific variables, the script can be run in debug mode using `./compile.sh model_name.onnx name_of_onnx_node`

`onnx_seedot_name_map.txt` It stores a map from onnx names to SeeDot names of variables

`seedot_ezpc_name_map.txt` It stores a map from SeeDot names to EzPC names of variables

`onnx_ezpc_name_map.txt` The above two maps are combined to create a map that shows the mapping from onnx names to ezpc/cpp names

`cpp_output_raw.txt` It contains the raw output after running the final code. In case if the script is run on `debug` mode with a debug name specified then the output has the values of the selected debug variable instead of the final variable. 

`cpp_output.txt` The above file is parsed and converted into a format where all fixed point integer values are converted to the easily readable floating format. As earlier in the case of `debug` mode the output contains the value of debug variable.

`onnx_debug.txt` In the debug mode this file contains the value of selected onnx node computed using onnx runtime.

`onnx_output.txt` This file contains the value of output computed using onnx runtime. 

`seedot_ast.txt` output of process_onnx.py is logged in this. It includes the seedot ast generated.

`seedot_to_ezpc_output.txt` output of seedot compilation to ezpc is logged in this. 

# Dependency
Other than EzPC dependencies 
`onnx` 
`onnxruntime`

# Testing
python3 -m unittest 


