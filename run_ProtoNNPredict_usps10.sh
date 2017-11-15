# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# Have a look at README.md and README_PROTONN_OSS.md for details on how to setup the data directory to run this script.



test_file="-I usps10/test.txt"
model_file="-M usps10/ProtoNNResults/ProtoNNTrainer_pd_15_protPerClass_0_prot_200_spW_1.000000_spZ_1.000000_spB_1.000000_gammaNumer_1.000000_normal_3_seed_42_bs_1024_it_20_ep_20/model"
normalization_file="-n usps10/ProtoNNResults/ProtoNNTrainer_pd_15_protPerClass_0_prot_200_spW_1.000000_spZ_1.000000_spB_1.000000_gammaNumer_1.000000_normal_3_seed_42_bs_1024_it_20_ep_20/minMaxParams"
output_dir="-O usps10/ProtoNNResults"
input_format="-F 0"
ntest="-e 2007"
#batch_size="-b 1024"


########################################################
# execute ProtoNNPredict
########################################################

#gdb=" gdb --args" 
executable="./ProtoNNPredict"
command=$gdb" "$executable" "$test_file" "$model_file" "$output_dir" "$normalization_file" "$input_format" "$ntest" "$batch_size" "
echo "Running ProtoNNPredict with following command: "
echo $command
echo ""
exec $command

