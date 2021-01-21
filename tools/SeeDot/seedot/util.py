# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import numpy as np
import platform

import seedot.config as config
import logging

'''
This code contains the Configuration information to control
the exploration of SeeDot.
Additionally it contains the helper utility functions that are used by 
the entire program. 
'''


class Config:

    expBigLength = 6
        # This parameter is used if exp below is set to "table".
        # Please refer to old SeeDot (PLDI'19) paper Section 5.3.1.
        # In the fixed point mode, the top 2 * expBigLength bits are used to compute e^x, the remaining bits are discarded.
    exp = "new table"  # "table" "math" "new table"
        # This parameter controls the type of exponentiation to be used in the fixed point code.
        # "table": Uses the method used in old SeeDot (PLDI '19).
        # "new table": Uses the method used in new SeeDot (OOPSLA '20).
        # "math": Uses floating point implementation of math.h.
    codegen = "funcCall"  # "funcCall"
        # Type of codegen: as a sequence of library function calls or inlined (no longer supported).
    debug = False
        # Enable debug mode of the generated fixed-point/floating-point C++ inference code.
        # This should be always set to False.
    saturateArithmetic = False
        # Enable saturating arithmetic in the generated fixed-point code.
    fastApproximations = False
        # Enable fast approximations in the generated fixed-point code, like:
        #   -> In multiplication, truncate first and multiply instead of multiply first in higher bitwidth and then truncate.
        #   -> Similarly in multiplication-like functions convolution, hadamard product etc.
    x86MemoryOptimize = True
        # Enable memory optimization in the generated fixed-point code in x86, arduino or m3 codegen.
    memoryLimit = 200000
        # The maximum memory present on the target device. Used if memory optimizations are enabled in the target codegen.
    largeVariableLimit = 50000
        # Any variable with more elements than this are prioritized for demotion to 8 bits.
    defragmentEnabled = False
        # Enable defragmentation. Currently not supported, so must be kept to False.
    faceDetectionHacks = False
        # Quick fixes for face detection model. This parameter must always be kept to False apart from debugging purposes.


def isSaturate():
    return Config.saturateArithmetic

def isfastApprox():
    return Config.fastApproximations

def windows():
    return platform.system() == "Windows"

def linux():
    return platform.system() == "Linux"

def getAlgo():
    return Config.algo

def setAlgo(algo: str):
    Config.algo = algo

def getEncoding():
    return Config.encoding

def setEncoding(encoding: str):
    Config.encoding = encoding

def forFixed():
    return Config.encoding == config.Encoding.fixed

def forFloat():
    return Config.encoding == config.Encoding.floatt

def getTarget():
    return Config.target

def setTarget(target: str):
    Config.target = target

def forArduino():
    return Config.target == config.Target.arduino

def forM3():
    return Config.target == config.Target.m3

def forHls():
    return Config.target == config.Target.Hls

def forVerilog():
    return Config.target == config.Target.Verilog

def forX86():
    return Config.target == config.Target.x86

def getProfileLogFile():
    return Config.profileLogFile

def setProfileLogFile(file):
    Config.profileLogFile = file

def getExpBitLength():
    return Config.expBigLength

def getMaxScale():
    return Config.maxScale

def setMaxScale(x: int):
    Config.maxScale = x

def getShrType():
    # "shr" "shr+" "div" "negate"
    return "div"

def useMathExp():
    return Config.exp == "math"

def useTableExp():
    return Config.exp == "table"

def useNewTableExp():
    return Config.exp == "new table"

def genFuncCalls():
    return Config.codegen == "funcCall"

def debugMode():
    return Config.debug

def copy_dict(dict_src: dict, diff={}):
    dict_res = dict(dict_src)
    dict_res.update(diff)
    return dict_res

# set number of workers for FPGA sparseMUL
def setNumWorkers(WorkerThreads):
    Config.numWorkers = WorkerThreads

def getNumWorkers():
    return Config.numWorkers

# z = [y1,y2,..] = [[x1,..], [x2,..], ..] --> [x1,.., x2,.., ..]
def flatten(z: list):
    return [x for y in z for x in y]

def computeScalingFactor(val):
    '''
    The scale computation algorithm is different while generating function calls and while generating inline code.
    The inline code generation uses an extra padding bit for each parameter and is less precise.
    The scales computed while generating function calls uses all bits.
    '''
    if genFuncCalls():
        return computeScalingFactorForFuncCalls(val)
    else:
        return computeScalingFactorForInlineCodegen(val)

def computeScalingFactorForFuncCalls(val):
    l = np.log2(val)
    if int(l) == l:
        c = l + 1
    else:
        c = np.ceil(l)
    return -int((config.wordLength - 1) - c)

def computeScalingFactorForInlineCodegen(val):
    return int(np.ceil(np.log2(val) - np.log2((1 << (config.wordLength - 2)) - 1)))


# Logging Section
def getLogger():
    log =  logging.getLogger()    
    return log
