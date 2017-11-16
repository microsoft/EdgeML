# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# Have a look at README_BONSAI_OSS.md for details on how to setup the directory to run this script.


########################################################
# Input-output parameters
########################################################

input_dir="./usps10"
input_format="-f 0"

########################################################
# Data-dependent parameters
########################################################

ntrain="-nT 7291"
ntest="-nE 2007"
num_features="-F 256"
num_labels="-C 10"

########################################################
# Bonsai hyper-parameters (optional)
########################################################

projection_dimension="-P 28"
tree_depth="-D 3"
sigma="-S 1"
reg_W="-lW 0.001"
reg_V="-lV 0.001"
reg_Theta="-lT 0.001"
reg_Z="-lZ 0.0001"
sparse_W="-sW 0.3"
sparse_V="-sV 0.3"
sparse_Theta="-sT 0.62"
sparse_Z="-sZ 0.2"

########################################################
# Bonsai optimization hyper-parameters (optional)
########################################################

batch_factor="-B 1"
iters="-I 100"

########################################################
# execute Bonsai
########################################################

#gdb=" gdb --args"
executable="./BonsaiTrain"
command=$gdb" "$executable" "$input_format" "$num_features" "$num_labels" "$ntrain" "$ntest" "$projection_dimension" "$tree_depth" "$sigma" "$reg_W" "$reg_Z" "$reg_Theta" "$reg_V" "$sparse_Z" "$sparse_Theta" "$sparse_V" "$sparse_W" "$batch_factor" "$iters" "$input_dir
echo "Running Bonsai train with following command: "
echo $command
echo ""
exec $command
