# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# Target word length. Currently set to match the word length of Arduino (2 bytes)
wordLength = 16

# Range of max scale factor used for exploration
maxScaleRange = 0, -wordLength

# tanh approximation limit
tanh_limit = 1.0

# LUT Upper bound for Artix-7
LUTUpperBound = 0.9 * 20800
LUTCount = 0

# MSBuild location
# Edit the path if not present at the following location
msbuildPath = r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\MSBuild\15.0\Bin\MSBuild.exe"
vivadoInstallPath = r"C:\Xilinx\Vivado\2018.2\bin"

class Algo:
	Bonsai = "bonsai"
	Lenet = "lenet"
	Protonn = "protonn"
	Default = [Bonsai, Protonn]
	All = [Bonsai, Lenet, Protonn]

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
	Hls = "hls"
	Verilog = "verilog"
	X86 = "x86"
	Default = Arduino
	All = [Arduino, Hls, Verilog, X86]
