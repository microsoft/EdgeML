# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import platform

import seedot.config as config


class Config:
    expBigLength = 6
    exp = "new table"  # "table" "math" "new table"
    codegen = "funcCall"  # "funcCall" "inline"
    debug = False
    debugCompiler = True
    saturateArithmetic = False
    fastApproximations = False
    x86MemoryOptimize = True
    defragmentEnabled = False
    faceDetectionHacks = False # quick fix for face detection model

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


def getVersion():
    return Config.version


def setVersion(version: str):
    Config.version = version


def forFixed():
    return Config.version == config.Version.fixed


def forFloat():
    return Config.version == config.Version.floatt


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


def debugCompiler():
    return Config.debugCompiler


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
    The scales computed while generating fucntion calls uses all bits.
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
