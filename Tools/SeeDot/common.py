# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# Target word length. Currently set to match the word length of Arduino (2
# bytes)
wordLength = 16

inputFileType = "npy"

# Range of max scale factor used for exploration
maxScaleRange = 0, -wordLength

# tanh approximation limit
tanh_limit = 1.0

# MSBuild location
# Edit the path if not present at the following location
msbuildPathOptions = [r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\MSBuild.exe",
                      r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\MSBuild\15.0\Bin\MSBuild.exe",
                      r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\MSBuild.exe"
                      ]


class Algo:
    Bonsai = "bonsai"
    Protonn = "protonn"
    Default = [Bonsai, Protonn]
    All = [Bonsai, Protonn]


class Version:
    Fixed = "fixed"
    Float = "float"
    All = [Fixed, Float]


class DatasetType:
    Training = "training"
    Testing = "testing"
    Default = Testing
    All = [Training, Testing]


class Target:
    Arduino = "arduino"
    X86 = "x86"
    Default = Arduino
    All = [Arduino, X86]
