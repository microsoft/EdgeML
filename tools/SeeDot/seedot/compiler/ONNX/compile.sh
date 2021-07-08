#!/bin/bash

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# This script will
# 1) Compile the ONNX model to SeeDot AST.
# 2) Compile the SeeDot AST to EzPC.
# 3) Convert the EzPC code to CPP and then run it on the given dataset.

# Any subsequent(*) commands which fail will cause the shell script to exit immediately.
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

# Generating input may take time, hence skip if already generated.
if [ -f  ${data_dir}"/"${inputFileName} ]; then
	echo -e "${GREEN}$inputFileName already exist, skipping process_onnx${NC}"
else
	echo "Starting to generate random input"
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
echo "Output of process_onnx and the resultant SeeDot AST are logged in debug/seedot_ast.txt"
python3 "process_onnx.py" ${modelName}'.onnx' > "debug/seedot_ast.txt"
echo -e "${GREEN}Finished process_onnx${NC}"

echo "Starting SeeDot to EzPC compilation"
echo "Output is logged in debug/seedot_to_ezpc_output.txt"

if [ -z "$debugOnnxNode" ]; then
	python3 ../SeeDot/SeeDot.py -p $seedotASTName --astFile ${data_dir}"/"$seedotASTName --outputFileName ${data_dir}"/"${ezpcOutputFullFileName} --consSF ${SCALINGFACTOR} --bitlen "$BITLEN" > "debug/seedot_to_ezpc_output.txt"
else
	debugSeedotNode=$(python3 -c "import common; common.get_seedot_name_from_onnx_name(\"${debugOnnxNode}\")")
	echo "${debugSeedotNode} is the corresponding SeeDot name"
	python3 ../SeeDot/SeeDot.py -p $seedotASTName --astFile ${data_dir}"/"$seedotASTName --outputFileName ${data_dir}"/"${ezpcOutputFullFileName} --consSF ${SCALINGFACTOR} --debugVar ${debugSeedotNode} --bitlen "$BITLEN" > "debug/seedot_to_ezpc_output.txt"
fi
echo -e "${GREEN}Finished SeeDot to EzPC compilation${NC}"

python3 -c 'import common; common.merge_name_map()'

cat "../TFEzPCLibrary/Library${BITLEN}_cpp.ezpc" "../TFEzPCLibrary/Library${BITLEN}_common.ezpc" ${data_dir}"/"${ezpcOutputFullFileName} > temp
mv temp "$ezpcOutputFullFileName"

mv "$ezpcOutputFullFileName" "$EzPCDir/EzPC"
cd "$EzPCDir/EzPC"
eval `opam config env`

echo "Starting with EzPC to CPP compilation"
./ezpc.sh "$ezpcOutputFullFileName" --bitlen "$BITLEN" --codegen "$compilationTargetHigher" --disable-tac
echo -e "${GREEN}Finished EzPC to CPP compilation ${NC}"

# Deleting the generated files.
mv "$finalCodeOutputFileName" "$ONNX_dir"
DIREZPC="${EzPCDir}/EzPC/${modelName}"
for file in "$DIREZPC"*
do
  rm "${file}"
done

if [ "$compilationTargetLower" == "cpp" ]; then
	cd "$ONNX_dir"
	mv "$finalCodeOutputFileName" "$data_dir"

	echo "Adding OpenMP threading instructions to the 3D Convolutions"
	python3 -c "import common; common.add_openmp_threading_to_convolution('${data_dir}"/"${finalCodeOutputFileName}')"

	echo "Compiling generated CPP code"
	g++ -O3 -g -w -fopenmp ${data_dir}"/"${finalCodeOutputFileName1} -o ${data_dir}"/"${modelName}".out"
	echo -e "${GREEN}compiling done ${NC}"
	rm -f "debug/cpp_output_raw.txt" || true
	echo "Running the final code"
	eval './'${data_dir}'/'${modelName}'.out' < ${data_dir}'/'${inputFileName} > "debug/cpp_output_raw.txt"
	python3 -c "import common; common.parse_output(${SCALINGFACTOR})"
	echo -e "${GREEN}All operations done. ${NC}"
fi
