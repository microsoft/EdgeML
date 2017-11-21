@ECHO OFF
REM # Copyright (c) Microsoft Corporation. All rights reserved.
REM # Licensed under the MIT license.

REM # Have a look at README.md and README_PROTONN_OSS.md for details on how to setup the data directory to run this script.



SET test_file=-I usps10/test.txt
SET model_file=-M usps10/ProtoNNResults/ProtoNNTrainer_pd_15_protPerClass_0_prot_200_spW_1.000000_spZ_1.000000_spB_1.000000_gammaNumer_1.000000_normal_3_seed_42_bs_1024_it_20_ep_20/model
SET normalization_file=-n usps10/ProtoNNResults/ProtoNNTrainer_pd_15_protPerClass_0_prot_200_spW_1.000000_spZ_1.000000_spB_1.000000_gammaNumer_1.000000_normal_3_seed_42_bs_1024_it_20_ep_20/minMaxParams
SET output_dir=-O usps10\ProtoNNResults
SET input_format=-F 0
SET ntest=-e 2007
REM SET batch_size=-b 1024


REM ########################################################
REM # execute ProtoNNPredict
REM ########################################################

SET executable=./ProtoNNPredict
SET command=%executable% %test_file% %model_file% %output_dir% %normalization_file% %input_format% %ntest% %batch_size%
@ECHO ON
echo Running ProtoNNPredict with following command:
echo %command%
START /B %command%
pause

