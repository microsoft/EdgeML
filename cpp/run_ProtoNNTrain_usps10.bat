@ECHO OFF
REM # Copyright (c) Microsoft Corporation. All rights reserved.
REM # Licensed under the MIT license.

REM # Have a look at README.md and README_PROTONN_OSS.md for details on how to setup the data directory to run this script.


REM ########################################################
REM # Input-output parameters
REM ########################################################

SET predefined_model=-P 0
SET problem_format=-C 1
SET train_file=-I usps10/train.txt
SET validation_file=-V usps10/test.txt
SET output_dir=-O usps10\ProtoNNResults
SET input_format=-F 0

REM ########################################################
REM # Data-dependent parameters
REM ########################################################

SET ntrain=-r 7291
SET ntest=-e 2007
SET original_dimension=-D 256
SET num_labels=-l 10

REM ########################################################
REM # ProtoNN hyper-parameters (required)
REM ########################################################

SET projection_dimension=-d 15
REM #prototypes=-k 20
SET prototypes=-m 200


REM ########################################################
REM # ProtoNN hyper-parameters (optional)
REM ########################################################

SET lambda_W=-W 1.0
SET lambda_Z=-Z 1.0
SET lambda_B=-B 1.0
SET gammaNumerator=-g 1.0
SET normalization=-N 0
SET seed=-R 42


REM ########################################################
REM # ProtoNN optimization hyper-parameters (optional)
REM ########################################################

SET batch_size=-b 1024
SET iters=-T 20
SET epochs=-E 20

REM ########################################################
REM # execute ProtoNN
REM ########################################################

SET executable=./ProtoNNTrain
SET command=%executable% %predefined_model% %seed% %problem_format% %train_file% %validation_file% %output_dir% %model_dir% %input_format% %ntrain% %validation% %original_dimension% %projection_dimension% %num_labels% %prototypes% %lambda_W% %lambda_Z% %lambda_B% %gammaNumerator% %batch_size% %iters% %epochs% %normalization%
@ECHO ON
ECHO Running ProtoNN with following command: 
ECHO %command%
START /B %command%
pause