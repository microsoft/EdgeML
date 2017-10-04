# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# Have a look at README.md and README_PROTONN_OSS.md for details on how to setup the data directory to run this script.

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/mkl/lib/intel64_lin/:./

########################################################
# Input-output parameters
########################################################

predefined_model="-P 0"
problem_format="-C 2"
train_file="-I Data/Wiki10/train.txt"
validation_file="-V Data/Wiki10/test.txt"
output_dir="-O Data/Wiki10/ProtoNNResults"
input_format="-F 0"



########################################################
# Data-dependent parameters
########################################################

ntrain="-r 14146"
nvalidation="-e 6616"
original_dimension="-D 101938"
num_labels="-l 30938" 



########################################################
# ProtoNN hyper-parameters (required)
########################################################

projection_dimension="-d 200"
#prototypes="-k 20"
prototypes="-m 2000"



########################################################
# ProtoNN hyper-parameters (optional)
########################################################

lambda_W="-W 0.4"
lambda_Z="-Z 0.005"
lambda_B="-B 0.4"
gammaNumerator="-g 1.0"
normalization="-N 2"
seed="-R 42"



########################################################
# ProtoNN optimization hyper-parameters (optional)
########################################################

batch_size="-b 1024"
iters="-T 100"
epochs="-E 1"



########################################################
# execute ProtoNN
########################################################

#gdb=" gdb --args" 
executable="./ProtoNN"
command=$gdb" "$executable" "$predefined_model" "$seed" "$problem_format" "$train_file" "$validation_file" "$output_dir" "$input_format" "$ntrain" "$nvalidation" "$original_dimension" "$projection_dimension" "$num_labels" "$prototypes" "$lambda_W" "$lambda_Z" "$lambda_B" "$gammaNumerator" "$batch_size" "$iters" "$epochs" "$normalization
echo "Running ProtoNN with following command: "
echo $command
echo ""
exec $command
