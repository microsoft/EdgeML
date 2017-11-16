# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# Have a look at README_BONSAI_OSS.md for details on how to setup the directory to run this script.


########################################################
# Input-output parameters
########################################################

input_dir="-D ./usps10"
input_format="-f 0"
model_dir="-M current_model" # Note: The model_dir has to be changed as Model naming is based on timestamp so required to be changed by the user

########################################################
# Data-dependent parameters
########################################################

ntest="-N 2007"

########################################################
# execute Bonsai
########################################################

#gdb=" gdb --args"
executable="./BonsaiPredict"
command=$gdb" "$executable" "$input_format" "$ntest" "$input_dir" "$model_dir
echo "Running Bonsai with following command: "
echo $command
echo ""
exec $command
