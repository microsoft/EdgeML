@ECHO OFF
REM # Copyright (c) Microsoft Corporation. All rights reserved.
REM # Licensed under the MIT license.

REM # Have a look at README_BONSAI_OSS.md for details on how to setup the directory to run this script.


REM ########################################################
REM # Input-output parameters
REM ########################################################

SET input_dir=-D ./usps10
SET input_format=-f 0

REM # Note: The model_dir has to be changed as Model naming is based on timestamp so required to be changed by the user
SET model_dir=-M ./usps10/BonsaiResults/23_44_23_15_11

REM ########################################################
REM # Data-dependent parameters
REM ########################################################

SET ntest=-N 2007

REM ########################################################
REM # execute Bonsai
REM ########################################################

SET executable=./BonsaiPredict
SET command=%executable% %input_format% %ntest% %input_dir% %model_dir%
@ECHO ON
ECHO Running Bonsai Predict with following command: 
ECHO %command%
START /B %command%
pause