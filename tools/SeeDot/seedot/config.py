# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''
This file stores the configuration parameters for SeeDot's run. 
It also contains the various classes for CLI arguments.  
'''

# Target word length. Currently set to match the word length of Arduino (2 bytes).
wordLength = 16
availableBitwidths = [8, 16, 32]

# Range of max scale factor used for exploration.
# In the old SeeDot (PLDI'19), this explores across the maxscale parameter.
# In the new SeeDot (OOPSLA'20), this explores across the scale of the input variable 'X'.
maxScaleRange = 0, -wordLength

# TanH approximation limit. Used by old SeeDot (PLDI'19).
tanhLimit = 1.0

# MSBuild location
# Edit the path if not present at the following location.
msbuildPathOptions = [r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe",
                      r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\MSBuild\Current\Bin\MSBuild.exe",
                      r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\MSBuild\Current\Bin\MSBuild.exe"
                      ]

# IMPORTANT NOTE: Unsupported configuration (ddsEnabled = False and vbwEnabled = True).

# Enable data-driven scale computation. Turning this to False reverts the compiler to old verion (PLDI'19).
ddsEnabled = True

# Enable variable bit-width code generation. Setting this to false results in a code which uses mostly 16 bits.
vbwEnabled = True

# For exponential activation functions, turning this on restricts the range of the values taken by TanH and Sigmoid,
# which results in a better scale assignment (OOPSLA'20 Section 5.4).
functionReducedProfiling = True

# Turning this to True restricts function profiling to those datapoints whose extrema are not in highest 10%, which
# can make scales more precise but risk running into overflows for some cases.
trimHighestDecile = False

# If true, then during exploration, a higher scale would be preferred if two or more codes have similar accuracy results.
higherOffsetBias = True

# If true, the exploration iterates between Stage III and IV of the exploration to attempt to reach a global optimum.
fixedPointVbwIteration = False

# Number of offsets tried out for each variable (except X, for which 9 are tried) when they are demoted to 8 bits one at a time.
offsetsPerDemotedVariable = 3

# For a classification algorithm, fixed point code can have this much drop in accuracy compared to floating point code. Not used in regression algorithms.
permittedClassificationAccuracyLoss = 2.0

# For a regression algorithm, fixed point code can have this much more numerical loss compared to floating point code. Not used in classification algorithms.
permittedRegressionNumericalLossMargin = 90.0

# Minimum version of gcc required for compilation
min_gcc_version = 8

# Following classes are used sanity checks for arguments passed to the compiler, to prevent unexpected arguments being passed.
# These lists should be updated as the compiler is expanded to multiple algorithms and datasets.


class Metric:
    accuracy = "acc"
    disagreements = "disagree"
    reducedDisagreements = "red_disagree"
    default = [reducedDisagreements]
    all = [accuracy, disagreements, reducedDisagreements]


class Algo:
    bonsai = "bonsai"
    lenet = "lenet"
    protonn = "protonn"
    fastgrnn = "fastgrnn"
    rnnpool = "rnnpool"
    mbconv = "mbconv"
    test = "test"
    default = [fastgrnn]
    all = [bonsai, lenet, protonn, fastgrnn, rnnpool, mbconv, test]


class Encoding:
    fixed = "fixed"
    floatt = "float"
    default = [fixed]
    all = [floatt, fixed]


class DatasetType:
    training = "training"
    testing = "testing"
    default = testing
    all = [training, testing]


class ProblemType:
    classification = "classification"
    regression = "regression"
    default = classification
    all = [classification, regression]


class Target:
    arduino = "arduino"
    x86 = "x86"
    m3 = "m3"
    default = x86
    all = [arduino, x86, m3]


class Source:
    seedot = "seedot"
    tf = "tf"
    onnx = "onnx"
    default = seedot
    all = [seedot, tf, onnx]


class Log:
    error = "error"
    critical = "critical"
    warning = "warning"
    info = "info"
    debug = "debug"
    default = error
    all = [error, critical, warning, info, debug]
