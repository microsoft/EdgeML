# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''
Arduino backend handles the automatic Arduino sketch generation.
It adds the appropriate header files to the sketch and makes it easy to 'compile and upload' the sketch to a device.
Most of the routines in the base class CodegenBase are unchanged.
'''

import numpy as np
import os

from seedot.compiler.codegen.codegenBase import CodegenBase

import seedot.compiler.ir.ir as IR
import seedot.compiler.ir.irUtil as IRUtil

import seedot.compiler.type as Type
from seedot.util import *
from seedot.writer import Writer

import functools
import operator


class Arduino(CodegenBase):

    def __init__(self, outputDir, decls, localDecls, scales, intvs, cnsts, expTables, globalVars, internalVars, floatConstants, substitutions, demotedVarsOffsets, varsForBitwidth, varLiveIntervals, notScratch, coLocatedVariables):
        outputFile = os.path.join(outputDir, "predict.cpp")
        self.outputDir = outputDir
        self.out = Writer(outputFile)

        self.decls = decls
        self.localDecls = localDecls
        self.scales = scales
        self.intvs = intvs
        self.cnsts = cnsts
        self.expTables = expTables
        self.globalVars = globalVars
        self.internalVars = internalVars
        self.floatConstants = floatConstants

        self.demotedVarsOffsets = demotedVarsOffsets
        self.varsForBitwidth = varsForBitwidth

        self.varLiveIntervals = varLiveIntervals
        self.scratchSubs = {}
        self.notScratch = notScratch

        self.numberOfMemoryMaps = 0
        self.currentMemMap = 0
        self.defragmentationInstructions = []
        self.defragmentationParameters = []
        self.coLocatedVariables = dict(coLocatedVariables)

    def printCompilerConfig(self):
        configFile = os.path.join(self.outputDir, "compileConfig.h")
        with open(configFile, "w") as file:
            file.write("// The datatype of the fixed-point representation is specified below\n")
            file.write("#define INT%d\n" % config.wordLength)
            if forFloat():
                file.write("#define XFLOAT\n")
            else:
                if config.vbwEnabled:
                    file.write("#define XINT%d\n" % self.varsForBitwidth['X'])
                else:
                    file.write("#define XINT%d\n" % config.wordLength)


    def printPrefix(self):
        self.printCompilerConfig()

        self.printArduinoIncludes()

        self.printExpTables()

        self.printArduinoHeader()

        self.computeScratchLocationsFirstFitPriority()

        self.printVarDecls()

        self.printConstDecls()

        self.out.printf('\n')

    def printArduinoIncludes(self):
        self.out.printf('#include <Arduino.h>\n\n', indent=True)
        self.out.printf('#include "config.h"\n', indent=True)
        self.out.printf('#include "predict.h"\n', indent=True)
        self.out.printf('#include "library.h"\n', indent=True)
        self.out.printf('#include "model.h"\n\n', indent=True)
        self.out.printf('using namespace model;\n\n', indent=True)

    # Dumps the generated look-up table for computing exponentials.
    def printExpTables(self):
        for exp, [table, [tableVarA, tableVarB]] in self.expTables.items():
            self.printExpTable(table[0], tableVarA)
            self.printExpTable(table[1], tableVarB)
            self.out.printf('\n')

    def printExpTable(self, table_row, var):
        self.out.printf('const PROGMEM MYINT %s[%d] = {\n' % (
            var.idf, len(table_row)), indent=True)
        self.out.increaseIndent()
        self.out.printf('', indent=True)
        for i in range(len(table_row)):
            self.out.printf('%d, ' % table_row[i])
        self.out.decreaseIndent()
        self.out.printf('\n};\n')

    def printArduinoHeader(self):
        self.out.printf('int predict() {\n', indent=True)
        self.out.increaseIndent()

    # Generate the appropriate return experssion
    # If integer, return the integer
    # If tensor of size 0, convert the fixed-point integer to float and return the float value.
    # If tensor of size >0, convert the tensor to fixed-point integer, print
    # it to the serial port, and return void.
    def printSuffix(self, expr: IR.Expr):
        self.out.printf('\n')

        type = self.decls[expr.idf]

        if Type.isInt(type):
            self.out.printf('return ', indent=True)
            self.print(expr)
            self.out.printf(';\n')
        elif Type.isTensor(type):
            idfr = expr.idf
            exponent = self.scales[expr.idf]
            num = 2 ** exponent

            if type.dim == 0:
                self.out.printf('Serial.println(', indent=True)
                self.out.printf('float(' + idfr + ')*' + str(num))
                self.out.printf(', 6);\n')
            else:
                iters = []
                for i in range(type.dim):
                    s = chr(ord('i') + i)
                    tempVar = IR.Var(s)
                    iters.append(tempVar)
                expr_1 = IRUtil.addIndex(expr, iters)
                cmds = IRUtil.loop(type.shape, iters, [
                                   IR.PrintAsFloat(expr_1, exponent)])
                self.print(IR.Prog(cmds))
        else:
            assert False

        self.out.decreaseIndent()
        self.out.printf('}\n', indent=True)

        self.out.close()
    '''
    Below functions are overriding their corresponding definitions in codegenBase.py.
    These function have arduino-specific print functions.
    '''

    # Print the variable with pragmas
    def printVar(self, ir):
        if ir.inputVar:
            if config.wordLength == 8:
                self.out.printf('((MYINT) pgm_read_byte_near(&')
            if config.wordLength == 16:
                self.out.printf('((MYINT) pgm_read_word_near(&')
            elif config.wordLength == 32:
                self.out.printf('((MYINT) pgm_read_dword_near(&')
            else:
                assert False
        self.out.printf('%s', ir.idf)
        for e in ir.idx:
            self.out.printf('[')
            self.print(e)
            self.out.printf(']')
        if ir.inputVar:
            self.out.printf('))')

    def printFor(self, ir):
        self.printForHeader(ir)
        self.out.increaseIndent()
        varToLiveRange = []
        for var in ir.varDecls.keys():
            size = np.prod(self.localDecls[var].shape)
            varToLiveRange.append((self.varLiveIntervals[var], var, size, self.varsForBitwidth[var]))
        varToLiveRange.sort()
        usedSpaceMap = {}
        totalScratchSize = -1
        listOfDimensions = []
        for ([_,_], var, size, atomSize) in varToLiveRange:
            listOfDimensions.append(size)
        mode = (lambda x: np.bincount(x).argmax())(listOfDimensions) if len(listOfDimensions) > 0 else None
        for ([startIns, endIns], var, size, atomSize) in varToLiveRange:
            if var in self.notScratch:
                continue
            spaceNeeded = size * atomSize // 8
            varsToKill = []
            for activeVar in usedSpaceMap.keys():
                endingIns = usedSpaceMap[activeVar][0]
                if endingIns < startIns:
                    varsToKill.append(activeVar)
            for tbk in varsToKill:
                del usedSpaceMap[tbk]
            i = 0
            if spaceNeeded >= mode:
                blockSize = int(2**np.ceil(np.log2(spaceNeeded / mode))) * mode
            else:
                blockSize = mode / int(2**np.floor(np.log2(mode // spaceNeeded)))
            breakOutOfWhile = True
            while True:
                potentialStart = int(blockSize * i)
                potentialEnd = int(blockSize * (i+1)) - 1
                for activeVar in usedSpaceMap.keys():
                    (locationOccupiedStart, locationOccupiedEnd) = usedSpaceMap[activeVar][1]
                    if not (locationOccupiedStart > potentialEnd or locationOccupiedEnd < potentialStart):
                        i += 1
                        breakOutOfWhile = False
                        break
                    else:
                        breakOutOfWhile = True
                        continue
                if breakOutOfWhile:
                    break
                
            usedSpaceMap[var] = (endIns, (potentialStart, potentialEnd))
            totalScratchSize = max(totalScratchSize, potentialEnd)
            self.scratchSubs[var] = potentialStart
        self.out.printf("char scratch[%d];\n"%(totalScratchSize+1), indent=True)
        self.printLocalVarDecls(ir)
        for cmd in ir.cmd_l:
            self.print(cmd)
        self.out.decreaseIndent()
        self.out.printf('}\n', indent=True)

    # The variable X is used to define the data point.
    # It is either read from the serial port or from the device's memory based on the operating mode.
    # The getIntFeature() function reads the appropriate value of X based on the mode.
    def printAssn(self, ir):
        if isinstance(ir.e, IR.Var) and ir.e.idf == "X":
            self.out.printf("", indent=True)
            self.print(ir.var)
            indices = [index.idf for index in ir.e.idx]
            sizes = self.localDecls[ir.e.idf].shape if ir.e.idf in self.localDecls else self.decls[ir.e.idf].shape
            assert len(indices) == len(sizes), "Illegal state"
            prod = functools.reduce(operator.mul, sizes)
            dereferenceString = ""
            for i in range(len(indices)):
                prod = prod // sizes[i]
                dereferenceString += ("%s * %d + " % (indices[i], prod))
            dereferenceString = dereferenceString[:-3]
            if forFixed():
                self.out.printf(" = getIntFeature(%s);\n"%(dereferenceString))
            else:
                self.out.printf(" = getFloatFeature(%s);\n"%(dereferenceString))
        else:
            super().printAssn(ir)

    def printFuncCall(self, ir):
        self.out.printf("{\n", indent=True)
        self.out.increaseIndent()
        self.printLocalVarDecls(ir)
        self.out.printf("%s(" % ir.name, indent=True)
        keys = list(ir.argList)
        for i in range(len(keys)):
            arg = keys[i]

            # Do not print the 'X' variable as it will be read from the getIntFeature() function.
            if isinstance(arg, IR.Var) and arg.idf == 'X':
                continue

            # The value of x in the below code is the number of special characters (& and []) around the variable in the function call.
            # This number depends on the shape of the variable.
            # Example: A[10][10] is written as &A[0][0]. The value of x in this case is 2.
            # x is 0 for constants
            # x is -1 for integer variables where only & is printed and not []
            if isinstance(arg, IR.Var) and (arg.idf in self.decls.keys() or arg.idf in self.localDecls.keys()) and not arg.idf == 'X':
                type = self.decls[arg.idf] if arg.idf in self.decls else self.localDecls[arg.idf]
                if isinstance(type, Type.Tensor):
                    if type.dim == 0:
                        x = -1
                    else:
                        x = type.dim - len(arg.idx)
                else:
                    x = -1
            else:
                x = 0

            if forFixed():
                typeCast = "(int%d_t*)" % self.varsForBitwidth[arg.idf] if x > 0 else ""
                self.out.printf(typeCast)

            
            if not (isinstance(arg, IR.Var) and arg.idf in self.scratchSubs):
                if x != 0:
                    self.out.printf("&")
                self.print(arg)

                if x != 0 and x != -1:
                    self.out.printf("[0]" * x)
            else:
                self.out.printf("(scratch + %d)"%(self.scratchSubs[arg.idf]))
            if i != len(keys) - 1:
                self.out.printf(", ")

        self.out.printf(");\n")
        self.out.decreaseIndent()
        self.out.printf("}\n", indent=True)

    def printPrint(self, ir):
        self.out.printf('Serial.println(', indent=True)
        self.print(ir.expr)
        self.out.printf(');\n')

    def printPrintAsFloat(self, ir):
        self.out.printf('Serial.println(float(', indent=True)
        self.print(ir.expr)
        self.out.printf(') * ' + str(2 ** ir.expnt) + ', 6);')
