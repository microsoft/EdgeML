#!/bin/bash

# Authors: Shubham Ugare.

# Copyright:
# Copyright (c) 2018 Microsoft Research
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This script will 
# 1) compile the ONNX model to SeeDot AST 
# 2) Compile the SeeDot AST to ezpc
# 3) Convert the ezpc code to cpp and then run it on the given dataset

# Any subsequent(*) commands which fail will cause the shell script to exit immediately
set -e

modelName=$1
debugOnnxNode=$2

EzPCDir="../../EzPC"
ONNX_dir="../../Athos/ONNXCompiler"	
data_dir="debug/"${modelName} 
BITLEN="64"
SCALINGFACTOR="24"
COMPILATIONTARGET="CPP"
ezpcOutputFullFileName=${modelName}'.ezpc'
compilationTargetLower=$(echo "$COMPILATIONTARGET" | awk '{print tolower($0)}')
compilationTargetHigher=$(echo "$COMPILATIONTARGET" | awk '{print toupper($0)}')
finalCodeOutputFileName=${modelName}'0.cpp'
finalCodeOutputFileName1=${modelName}'1.cpp'
inputFileName=${modelName}'_input.h'
seedotASTName=${modelName}'.pkl'

# modelname_input.npy and modelname_output.npy
onnxInputFileName=${modelName}'_input.npy'
onnxOutputFileName=${modelName}'_output.npy'

GREEN='\033[0;32m'
NC='\033[0m' # No Color

mkdir -p debug
mkdir -p ${data_dir}

# Generating input may take time, hence skip if already generated
if [ -f  ${data_dir}"/"${inputFileName} ]; then 
	echo -e "${GREEN}$inputFileName already exist, skipping process_onnx${NC}"
else 
	echo "Starting to gemerate random input"
	python3 "create_input.py" ${modelName}'.onnx' $SCALINGFACTOR
	echo -e "${GREEN}Finished generating input${NC}"
fi 	

echo "Starting onnx run"
# can use either 'onnx_run_tf' or 'onnx_run'
# onnx_run is faster and has lesser dependencies 
# but may not support all operations
python3 "onnx_run.py" ${modelName}'.onnx' ${debugOnnxNode} > "debug/log_onnx_run.txt"
echo -e "${GREEN}Finished onnx run${NC}"

echo "Starting process_onnx"
echo "output of process_onnx and the resultant seedot ast are logged in debug/seedot_ast.txt"
python3 "process_onnx.py" ${modelName}'.onnx' > "debug/seedot_ast.txt"
echo -e "${GREEN}Finished process_onnx${NC}"

echo "Starting seedot to ezpc compilation"
echo "output is logged in debug/seedot_to_ezpc_output.txt"

if [ -z "$debugOnnxNode" ]; then 
	python3 ../SeeDot/SeeDot.py -p $seedotASTName --astFile ${data_dir}"/"$seedotASTName --outputFileName ${data_dir}"/"${ezpcOutputFullFileName} --consSF ${SCALINGFACTOR} --bitlen "$BITLEN" > "debug/seedot_to_ezpc_output.txt"
else 	
	debugSeedotNode=$(python3 -c "import common; common.get_seedot_name_from_onnx_name(\"${debugOnnxNode}\")")
	echo "${debugSeedotNode} is the corresponding SeeDot name"
	python3 ../SeeDot/SeeDot.py -p $seedotASTName --astFile ${data_dir}"/"$seedotASTName --outputFileName ${data_dir}"/"${ezpcOutputFullFileName} --consSF ${SCALINGFACTOR} --debugVar ${debugSeedotNode} --bitlen "$BITLEN" > "debug/seedot_to_ezpc_output.txt"
fi 
echo -e "${GREEN}Finished seedot to ezpc compilation${NC}"

python3 -c 'import common; common.merge_name_map()'


cat "../TFEzPCLibrary/Library${BITLEN}_cpp.ezpc" "../TFEzPCLibrary/Library${BITLEN}_common.ezpc" ${data_dir}"/"${ezpcOutputFullFileName} > temp
mv temp "$ezpcOutputFullFileName"

mv "$ezpcOutputFullFileName" "$EzPCDir/EzPC"
cd "$EzPCDir/EzPC"
eval `opam config env`

echo "Starting with ezpc to cpp compilation"
./ezpc.sh "$ezpcOutputFullFileName" --bitlen "$BITLEN" --codegen "$compilationTargetHigher" --disable-tac
echo -e "${GREEN}Finished ezpc to cpp compilation ${NC}"

# deleting the generated files
mv "$finalCodeOutputFileName" "$ONNX_dir"
DIREZPC="${EzPCDir}/EzPC/${modelName}"
for file in "$DIREZPC"*
do
  rm "${file}"
done

if [ "$compilationTargetLower" == "cpp" ]; then
	cd "$ONNX_dir"
	mv "$finalCodeOutputFileName" "$data_dir"

	echo "Adding openmp threading instructions to the 3d convolutions"
	python3 -c "import common; common.add_openmp_threading_to_convolution('${data_dir}"/"${finalCodeOutputFileName}')"

	echo "compiling generated cpp code"
	g++ -O3 -g -w -fopenmp ${data_dir}"/"${finalCodeOutputFileName1} -o ${data_dir}"/"${modelName}".out"
	echo -e "${GREEN}compiling done ${NC}"
	rm -f "debug/cpp_output_raw.txt" || true
	echo "running the final code"	
	eval './'${data_dir}'/'${modelName}'.out' < ${data_dir}'/'${inputFileName} > "debug/cpp_output_raw.txt"
	python3 -c "import common; common.parse_output(${SCALINGFACTOR})"
	echo -e "${GREEN}All operations done. ${NC}"
fi
