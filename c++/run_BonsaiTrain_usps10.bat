@ECHO OFF
REM # Copyright (c) Microsoft Corporation. All rights reserved.
REM # Licensed under the MIT license.

REM # Have a look at README_BONSAI_OSS.md for details on how to setup the directory to run this script.


REM ########################################################
REM # Input-output parameters
REM ########################################################

SET input_dir=./usps10
SET input_format=-f 0

REM ########################################################
REM # Data-dependent parameters
REM ########################################################

SET ntrain=-nT 7291
SET ntest=-nE 2007
SET num_features=-F 256
SET num_labels=-C 10

REM ########################################################
REM # Bonsai hyper-parameters (optional)
REM ########################################################

SET projection_dimension=-P 28
SET tree_depth=-D 3
SET sigma=-S 1
SET reg_W=-lW 0.001
SET reg_V=-lV 0.001
SET reg_Theta=-lT 0.001
SET reg_Z=-lZ 0.0001
SET sparse_W=-sW 0.3
SET sparse_V=-sV 0.3
SET sparse_Theta=-sT 0.62
SET sparse_Z=-sZ 0.2

REM ########################################################
REM # Bonsai optimization hyper-parameters (optional)
REM ########################################################

SET batch_factor=-B 1
SET iters=-I 100

REM ########################################################
REM # execute Bonsai
REM ########################################################

SET executable=./BonsaiTrain
SET command=%executable% %input_format% %num_features% %num_labels% %ntrain% %ntest% %projection_dimension% %tree_depth% %sigma% %reg_W% %reg_Z% %reg_Theta% %reg_V% %sparse_Z% %sparse_Theta% %sparse_V% %sparse_W% %batch_factor% %iters% %input_dir%
@ECHO ON
ECHO Running Bonsai with following command: 
ECHO %command%
START /B %command%
pause
