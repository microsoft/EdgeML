# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''
CodegenBase has print functions for the IR classes defined in IR.py
'''

import numpy as np
import multiprocessing as mp
import subprocess
import operator
import re
import os

import seedot.compiler.ir.ir as IR
import seedot.compiler.ir.irUtil as IRUtil
import seedot.compiler.codegen.dlx.scripts.dlxInputGen as DLXInputGen

import seedot.util as Util

import seedot.config as Common
import seedot.compiler.type as Type
from seedot.util import *

from bokeh.plotting import figure, output_file, show

class CodegenBase:

    def __init__(self, writer):
        self.out = writer

    def printOp(self, ir):
        self.out.printf('%s', ir.name)

    def printInt(self, ir):
        if np.iinfo(np.int16).min <= ir.n <= np.iinfo(np.int16).max:
            self.out.printf('%d', ir.n)
        elif np.iinfo(np.int32).min <= ir.n <= np.iinfo(np.int32).max:
            self.out.printf('%dL', ir.n)
        elif np.iinfo(np.int64).min <= ir.n <= np.iinfo(np.int64).max:
            self.out.printf('%dLL', ir.n)
        else:
            assert False

    def printFloat(self, ir):
        self.out.printf('%ff', ir.n)

    def printVar(self, ir, isPointer=False):
        if config.vbwEnabled and forFixed():
            if hasattr(self, "varsForBitwidth"):
                if Config.x86MemoryOptimize:
                    if hasattr(self, 'scratchSubs'):
                        if self.numberOfMemoryMaps in self.scratchSubs and ir.idf in self.scratchSubs[self.numberOfMemoryMaps]:
                            type = self.decls[ir.idf]
                            offset = self.scratchSubs[self.numberOfMemoryMaps][ir.idf]
                            if Type.isTensor(type):
                                resIndex = ' '
                                remSize = np.prod(type.shape)
                                if forM3():
                                    typeCast = "(Q%d_T*)" % (self.varsForBitwidth[ir.idf] - 1)
                                    if isPointer:
                                        self.out.printf("(scratch + %d +" % (offset))
                                    else:
                                        self.out.printf("*(%s(&(scratch[%d + " % (typeCast, offset))
                                else:
                                    typeCast = "(int%d_t&)" % self.varsForBitwidth[ir.idf]
                                    if isPointer:
                                        self.out.printf("(scratch + %d +" % (offset))
                                    else:
                                        self.out.printf("%s(scratch[%d + " % (typeCast, offset))
                                self.out.printf("%d * ("% (self.varsForBitwidth[ir.idf] // 8))
                                for i in range(type.dim):
                                    if i >= len(ir.idx):
                                        break
                                    remSize = remSize // type.shape[i]
                                    self.print(ir.idx[i])
                                    self.out.printf("*%d" % remSize)
                                    self.out.printf("+")
                                self.out.printf("0")
                                if forM3():
                                    if isPointer:
                                        self.out.printf("))")
                                    else:
                                        self.out.printf(")])))")
                                else:
                                    if isPointer:
                                        self.out.printf("))")
                                    else:
                                        self.out.printf(")])")
                                return
                            else:
                                pass
                        else:
                            pass
                    else:
                        assert False, "Illegal state, scratchSubs variable should be present if memory optimisation enabled"

                if ir.idf in self.varsForBitwidth and ir.idf[:3] == "tmp" and ir.idf in self.decls:
                    self.out.printf("%s_%d", ir.idf, self.varsForBitwidth[ir.idf])
                else:
                    self.out.printf("%s", ir.idf)
            else:
                assert False, "Illegal state, codegenBase must have variable bitwidth info for VBW mode"
        else:
            self.out.printf("%s", ir.idf)
        for e in ir.idx:
            self.out.printf('[')
            self.print(e)
            self.out.printf(']')

    def printBool(self, ir):
        self.out.printf({True: 'true', False: 'false'}[ir.b])

    def printIntUop(self, ir):
        self.out.printf('(')
        self.print(ir.op)
        self.print(ir.e)
        self.out.printf(')')

    def printIntBop(self, ir):
        self.out.printf('(')
        self.print(ir.e1)
        self.out.printf(' ')
        self.print(ir.op)
        self.out.printf(' ')
        self.print(ir.e2)
        self.out.printf(')')

    def printBoolUop(self, ir):
        self.out.printf('(')
        self.print(ir.op)
        self.print(ir.e)
        self.out.printf(')')

    def printBoolBop(self, ir):
        self.out.printf('(')
        self.print(ir.e1)
        self.out.printf(' ')
        self.print(ir.op)
        self.out.printf(' ')
        self.print(ir.e2)
        self.out.printf(')')

    def printBoolCop(self, ir):
        self.out.printf('(')
        self.print(ir.e1)
        self.out.printf(' ')
        self.print(ir.op)
        self.out.printf(' ')
        self.print(ir.e2)
        self.out.printf(')')

    def printCExpr(self, ir):
        self.out.printf('(')
        self.print(ir.cond)
        self.out.printf(' ? ')
        self.print(ir.et)
        self.out.printf(' : ')
        self.print(ir.ef)
        self.out.printf(')')

    def printExp(self, ir):
        self.out.printf('(exp(')
        self.print(ir.e)
        self.out.printf('))')

    def printTypeCast(self, ir):
        self.out.printf('(')
        self.out.printf('(' + ir.type + ')')
        self.print(ir.expr)
        self.out.printf(')')

    def printAssn(self, ir):
        self.out.printf('', indent=True)
        self.print(ir.var)
        self.out.printf(' = ')
        self.print(ir.e)
        self.out.printf(';\n')

    def printIf(self, ir):
        self.out.printf('if (', indent=True)
        self.print(ir.cond)
        self.out.printf(') {\n')

        self.out.increaseIndent()
        for cmd in ir.trueCmds:
            self.print(cmd)
        self.out.decreaseIndent()

        if len(ir.falseCmds) == 0:
            self.out.printf('}\n', indent=True)
            return

        self.out.printf('} else {\n', indent=True)

        self.out.increaseIndent()
        for cmd in ir.falseCmds:
            self.print(cmd)
        self.out.decreaseIndent()

        self.out.printf('}\n', indent=True)

    def printFor(self, ir):
        self.printForHeader(ir)
        self.out.increaseIndent()
        self.printLocalVarDecls(ir)
        for cmd in ir.cmd_l:
            self.print(cmd)
        self.out.decreaseIndent()
        self.out.printf('}\n', indent=True)

    def printForHeader(self, ir):
        self.out.printf('for (%s ', "int", indent=True) #Loop counter must be int16 else indices can overflow
        self.print(ir.var)
        self.out.printf(' = %d; ', ir.st)
        self.print(ir.cond)
        self.out.printf('; ')
        self.print(ir.var)
        self.out.printf('++) {\n') #TODO: What if --?

    def printWhile(self, ir):
        self.out.printf('while (', indent=True)
        self.print(ir.expr)
        self.out.printf(') {\n')
        self.out.increaseIndent()
        for cmd in ir.cmds:
            self.print(cmd)
        self.out.decreaseIndent()
        self.out.printf('}\n', indent=True)

    def printFuncCall(self, ir):
        self.out.printf("{\n", indent=True)
        self.out.increaseIndent()
        self.printLocalVarDecls(ir)
        self.out.printf("%s(" % ir.name, indent=True)
        keys = list(ir.argList)
        for i in range(len(keys)):
            arg = keys[i]
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
            if x != 0:
                self.out.printf("&")
            self.print(arg)
            if x != 0 and x != -1:
                self.out.printf("[0]" * x)
            if i != len(keys) - 1:
                self.out.printf(", ")
        self.out.printf(");\n")
        self.out.decreaseIndent()
        self.out.printf("}\n", indent=True)

    def printMemset(self, ir):
        self.out.printf('memset(', indent=True)
        if Config.x86MemoryOptimize and forFixed() and forX86() and self.numberOfMemoryMaps in self.scratchSubs:
            self.out.printf("(scratch + %d)", self.scratchSubs[self.numberOfMemoryMaps][ir.e.idf])
        else:
            self.print(ir.e)
        typ_str = "MYINT"
        if config.vbwEnabled:
            if hasattr(self, 'varsForBitwidth'):
                typ_str = ("int%d_t" % (self.varsForBitwidth[ir.e.idf])) if ir.e.idf in self.varsForBitwidth else typ_str
            else:
                assert False, "Illegal state, VBW mode but no variable information present"
        self.out.printf(', 0, sizeof(%s) * %d);\n' %
                        ("float" if forFloat() else typ_str, ir.len))

    def printMemcpy(self, ir):
        def printFlattenedIndices(indices, shape):
            remSize = np.prod(shape)
            for i in range(len(shape)):
                remSize //= shape[i]
                self.out.printf("%d*(", remSize)
                self.print(indices[i])
                self.out.printf(")")
                if i + 1 < len(shape):
                    self.out.printf("+")
        typ_str = "MYINT"
        if config.vbwEnabled:
            if hasattr(self, 'varsForBitwidth'):
                # Note ir.to and ir.start are constrained to have the same bitwidth
                typ_str = ("int%d_t" % (self.varsForBitwidth[ir.to.idf])) if ir.to.idf in self.varsForBitwidth else typ_str
            else:
                assert False, "Illegal state, VBW mode but no variable information present"
        typ_str = "float" if forFloat() else typ_str
        self.out.printf('memcpy(', indent=True)
        if Config.x86MemoryOptimize and forFixed() and self.numberOfMemoryMaps in self.scratchSubs:
            for (a, b, c, d) in [(ir.to.idf, ir.toIndex, 0, ir.to.idx), (ir.start.idf, ir.startIndex, 1, ir.start.idx)]:
                self.out.printf("((scratch + %d + sizeof(%s)*(", self.scratchSubs[self.numberOfMemoryMaps][a], typ_str)
                toIndexed = IRUtil.addIndex(IR.Var(""), b)
                if len(d + b) == 0:
                    self.out.printf("0")
                elif len(d + b) == len(self.decls[a].shape):
                    printFlattenedIndices(d + b, self.decls[a].shape)
                else:
                    assert False, "Illegal state, number of offsets to memcpy should be 0 or match the original tensor dimensions"
                self.out.printf(")))")
                if c == 0:
                    self.out.printf(", ")
        else:
            toIndexed = IRUtil.addIndex(IR.Var(ir.to.idf), ir.to.idx + ir.toIndex)
            startIndexed = IRUtil.addIndex(IR.Var(ir.start.idf), ir.start.idx + ir.startIndex)
            self.out.printf("&")
            self.print(toIndexed)
            self.out.printf(", &")
            self.print(startIndexed)
        self.out.printf(', sizeof(%s) * %d);\n' % (typ_str, ir.length))

    def printPrint(self, ir):
        self.out.printf('cout << ', indent=True)
        self.print(ir.expr)
        self.out.printf(' << endl;\n')

    def printPrintAsFloat(self, ir):
        self.out.printf('cout << ((float)(', indent=True)
        self.print(ir.expr)
        self.out.printf(')) * ' + str(2 ** ir.expnt) + ' << "";\n')

    def printPragmas(self, ir):
        if ir.vital == 1:
            self.out.printf('\n')
            self.out.printf(ir.msg + '\n', indent=True)

    def printComment(self, ir):
        self.out.printf('\n')
        if ir.instructionId is not None:
            self.out.printf('// ' + ('Instruction: %d ::: '%ir.instructionId) + ir.msg + '\n', indent=True)
        else:
            self.out.printf('// ' + ir.msg + '\n', indent=True)

    def printProg(self, ir):
        for cmd in ir.cmd_l:
            self.print(cmd)

    def print(self, ir):
        if isinstance(ir, IR.Int):
            return self.printInt(ir)
        elif isinstance(ir, IR.Float):
            return self.printFloat(ir)
        elif isinstance(ir, IR.Var):
            return self.printVar(ir)
        elif isinstance(ir, IR.Bool):
            return self.printBool(ir)
        elif isinstance(ir, IR.IntUop):
            return self.printIntUop(ir)
        elif isinstance(ir, IR.IntBop):
            return self.printIntBop(ir)
        elif isinstance(ir, IR.BoolUop):
            return self.printBoolUop(ir)
        elif isinstance(ir, IR.BoolBop):
            return self.printBoolBop(ir)
        elif isinstance(ir, IR.BoolCop):
            return self.printBoolCop(ir)
        elif isinstance(ir, IR.CExpr):
            return self.printCExpr(ir)
        elif isinstance(ir, IR.Exp):
            return self.printExp(ir)
        elif isinstance(ir, IR.TypeCast):
            return self.printTypeCast(ir)
        elif isinstance(ir, IR.Assn):
            return self.printAssn(ir)
        elif isinstance(ir, IR.If):
            return self.printIf(ir)
        elif isinstance(ir, IR.For):
            return self.printFor(ir)
        elif isinstance(ir, IR.While):
            return self.printWhile(ir)
        elif isinstance(ir, IR.FuncCall):
            return self.printFuncCall(ir)
        elif isinstance(ir, IR.Memset):
            return self.printMemset(ir)
        elif isinstance(ir, IR.Memcpy):
            return self.printMemcpy(ir)
        elif isinstance(ir, IR.Print):
            return self.printPrint(ir)
        elif isinstance(ir, IR.PrintAsFloat):
            return self.printPrintAsFloat(ir)
        elif isinstance(ir, IR.Comment):
            return self.printComment(ir)
        elif isinstance(ir, IR.Prog):
            return self.printProg(ir)
        elif isinstance(ir, IR.Op.Op):
            return self.printOp(ir)
        elif isinstance(ir, IR.String):
            return self.out.printf('\"%s\"', ir.s.idf)
        else:
            assert False

    def printAll(self, prog: IR.Prog, expr: IR.Expr):
        self.printPrefix()
        self.print(prog)
        self.printSuffix(expr)

    def printVarDecls(self):
        for decl in self.decls:
            if decl in self.globalVars:
                continue

            if forFloat() and decl not in self.internalVars:
                typ_str = IR.DataType.getFloatStr()
            else:
                typ_str = IR.DataType.getIntStr()
                if config.vbwEnabled:
                    if hasattr(self, 'varsForBitwidth'):
                        typ_str = ("int%d_t" % (self.varsForBitwidth[decl])) if decl in self.varsForBitwidth else typ_str
                    else:
                        assert False, "VBW enabled but bitwidth info missing"

            idf_str = decl
            type = self.decls[decl]
            if Type.isInt(type):
                shape_str = ''
            elif Type.isTensor(type):
                shape_str = ''.join(['[' + str(n) + ']' for n in type.shape])
            self.out.printf('%s %s%s;\n', typ_str, idf_str,
                            shape_str, indent=True)
        self.out.printf('\n')

    def printConstDecls(self):
        for cnst in self.cnsts:
            var, num = cnst, self.cnsts[cnst]

            if forFloat() and var in self.floatConstants:
                self.out.printf('%s = %f;\n', var,
                                self.floatConstants[var], indent=True)
            else:
                if config.vbwEnabled and var in self.varsForBitwidth.keys() and (forX86() or forM3()):
                    if np.iinfo(np.int16).min <= num <= np.iinfo(np.int16).max:
                        self.out.printf('%s_%d = %d;\n', var, self.varsForBitwidth[var], num, indent=True)
                    elif np.iinfo(np.int32).min <= num <= np.iinfo(np.int32).max:
                        self.out.printf('%s_%d = %dL;\n', var, self.varsForBitwidth[var], num, indent=True)
                    elif np.iinfo(np.int64).min <= num <= np.iinfo(np.int64).max:
                        self.out.printf('%s_%d = %dLL;\n', var, self.varsForBitwidth[var], num, indent=True)
                    else:
                        assert False
                else:
                    if np.iinfo(np.int16).min <= num <= np.iinfo(np.int16).max:
                        self.out.printf('%s = %d;\n', var, num, indent=True)
                    elif np.iinfo(np.int32).min <= num <= np.iinfo(np.int32).max:
                        self.out.printf('%s = %dL;\n', var, num, indent=True)
                    elif np.iinfo(np.int64).min <= num <= np.iinfo(np.int64).max:
                        self.out.printf('%s = %dLL;\n', var, num, indent=True)
                    else:
                        assert False
    
    def printLocalVarDecls(self, ir):
        for var in ir.varDecls.keys():
            if forFloat() and var not in self.internalVars:
                typ_str = IR.DataType.getFloatStr()
            else:
                typ_str = IR.DataType.getIntStr()
                if config.vbwEnabled:
                    if hasattr(self, 'varsForBitwidth'):
                        typ_str = ("int%d_t" % (self.varsForBitwidth[var])) if var in self.varsForBitwidth else typ_str
                    else:
                        assert False, "VBW enabled but bitwidth info missing"
            idf_str = var
            type = ir.varDecls[var]
            if Type.isInt(type):
                shape_str = ''
            elif Type.isTensor(type):
                shape_str = ''.join(['[' + str(n) + ']' for n in type.shape])
            self.out.printf('%s %s%s;\n', typ_str, idf_str,
                            shape_str, indent=True)

    def computeScratchLocations(self):
        if not Config.x86MemoryOptimize or forFloat():
            return
        else:
            varToLiveRange, decls = self.preProcessRawMemData()
            def sortkey(a):
                return (a[0][0], -a[0][1], -(a[2]*a[3])//8)
            varToLiveRange.sort(key=sortkey)
            usedSpaceMap = {}
            totalScratchSize = -1
            listOfDimensions = []
            for ([_,_], var, size, atomSize) in varToLiveRange:
                listOfDimensions.append(size)
            mode = 75 #(lambda x: np.bincount(x).argmax())(listOfDimensions) if len(listOfDimensions) > 0 else None
            plot = figure(plot_width=1000, plot_height=1000)
            x = []
            y = []
            w = []
            h = []
            c = []
            visualisation = []
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
                # if Config.faceDetectionHacks and var in ['tmp252', 'tmp253', 'tmp364', 'tmp367']: #quick fix for face detection
                #     i = 153600 // blockSize
                # if Config.faceDetectionHacks and var in ['tmp249']:
                #     i = 172800 // blockSize
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
                
                if False: #Config.defragmentEnabled and potentialStart + spaceNeeded > 200000:
                    usedSpaceMap = self.defragmentMemory(usedSpaceMap, var, spaceNeeded, endIns, mode)
                else:
                    if True: #Config.defragmentEnabled:
                        # if Config.faceDetectionHacks and var in ['tmp391', 'tmp405', 'tmp410', 'tmp412']:
                        #     potentialStart = potentialEnd - spaceNeeded + 1
                        #     usedSpaceMap[var] = (endIns, (potentialEnd - spaceNeeded + 1, potentialEnd))
                        # elif Config.faceDetectionHacks and var in ['tmp392', 'tmp406']:
                        #     potentialEnd = 96000
                        #     potentialStart = potentialEnd - spaceNeeded + 1
                        #     usedSpaceMap[var] = (endIns, (potentialStart, potentialEnd))
                        # else:
                            usedSpaceMap[var] = (endIns, (potentialStart, potentialStart + spaceNeeded - 1))
                    else:
                        usedSpaceMap[var] = (endIns, (potentialStart, potentialEnd))
                    totalScratchSize = max(totalScratchSize, potentialStart + spaceNeeded - 1)
                    if self.numberOfMemoryMaps not in self.scratchSubs.keys():
                        self.scratchSubs[self.numberOfMemoryMaps] = {}
                    self.scratchSubs[self.numberOfMemoryMaps][var] = potentialStart
                    varf = var
                    if not Config.faceDetectionHacks:
                        if varf in self.coLocatedVariables:
                            varf = self.coLocatedVariables[varf]
                            self.scratchSubs[self.numberOfMemoryMaps][varf] = potentialStart
                x.append((endIns + 1 + startIns) / 2)
                w.append(endIns - startIns + 1)
                y.append((usedSpaceMap[var][1][0] + usedSpaceMap[var][1][1]) / 20000)
                h.append((usedSpaceMap[var][1][1] - usedSpaceMap[var][1][0]) / 10000)
                c.append("#" + ''.join([str(int(i)) for i in 10*np.random.rand(6)]))
                visualisation.append((startIns, var, endIns, usedSpaceMap[var][1][0], usedSpaceMap[var][1][1]))
            plot.rect(x=x, y=y, width=w, height=h, color=c, width_units="data", height_units="data")
            # show(plot)
            self.out.printf("char scratch[%d];\n"%(totalScratchSize+1), indent=True)
            self.out.printf("/* %s */"%(str(self.scratchSubs)))

    def computeScratchLocationsFirstFit(self):
        if not Config.x86MemoryOptimize or forFloat():
            return
        else:
            varToLiveRange, decls = self.preProcessRawMemData()
            def sortkey(a):
                return (a[0][0], -a[0][1], -(a[2]*a[3])//8)
            varToLiveRange.sort(key=sortkey)
            freeSpace = {0:-1}
            freeSpaceRev = {-1:0}
            usedSpaceMap = {}
            totalScratchSize = -1
            listOfDimensions = []
            for ([_,_], var, size, atomSize) in varToLiveRange:
                listOfDimensions.append(size)
            #mode = 75 #(lambda x: np.bincount(x).argmax())(listOfDimensions) if len(listOfDimensions) > 0 else None
            plot = figure(plot_width=1000, plot_height=1000)
            x = []
            y = []
            w = []
            h = []
            c = []
            visualisation = []
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
                    (st, en) = usedSpaceMap[tbk][1]
                    en += 1
                    freeSpace[st] = en
                    freeSpaceRev[en] = st
                    if en in freeSpace.keys():
                        freeSpace[st] = freeSpace[en]
                        freeSpaceRev[freeSpace[st]] = st
                        del freeSpace[en]
                        del freeSpaceRev[en]
                    if st in freeSpaceRev.keys():
                        freeSpaceRev[freeSpace[st]] = freeSpaceRev[st]
                        freeSpace[freeSpaceRev[st]] = freeSpace[st]
                        del freeSpace[st]
                        del freeSpaceRev[st]
                    del usedSpaceMap[tbk]
                i = 0
                potentialStart = -1
                potentialEnd = -1
                for start in sorted(freeSpace.keys()):
                    end = freeSpace[start]
                    if end - start >= spaceNeeded or end == -1:
                        potentialStart = start
                        potentialEnd = potentialStart + spaceNeeded - 1
                        break
                    else:
                        continue
               
                if False: #Config.defragmentEnabled and potentialStart + spaceNeeded > 200000:
                    pass
                    #usedSpaceMap = self.defragmentMemory(usedSpaceMap, var, spaceNeeded, endIns, mode)
                else:
                    usedSpaceMap[var] = (endIns, (potentialStart, potentialEnd))
                    freeSpaceEnd = freeSpace[potentialStart]
                    del freeSpace[potentialStart]
                    if potentialEnd + 1 != freeSpaceEnd:
                        freeSpace[potentialEnd + 1] = freeSpaceEnd
                    freeSpaceRev[freeSpaceEnd] = potentialEnd + 1
                    if freeSpaceEnd == potentialEnd + 1:
                        del freeSpaceRev[freeSpaceEnd]
                    totalScratchSize = max(totalScratchSize, potentialEnd)
                    if self.numberOfMemoryMaps not in self.scratchSubs.keys():
                        self.scratchSubs[self.numberOfMemoryMaps] = {}
                    self.scratchSubs[self.numberOfMemoryMaps][var] = potentialStart
                    varf = var
                    if not Config.faceDetectionHacks:
                        if varf in self.coLocatedVariables:
                            varf = self.coLocatedVariables[varf]
                            self.scratchSubs[self.numberOfMemoryMaps][varf] = potentialStart
                x.append((endIns + 1 + startIns) / 2)
                w.append(endIns - startIns + 1)
                y.append((usedSpaceMap[var][1][0] + usedSpaceMap[var][1][1]) / 20000)
                h.append((usedSpaceMap[var][1][1] - usedSpaceMap[var][1][0]) / 10000)
                c.append("#" + ''.join([str(int(i)) for i in 10*np.random.rand(6)]))
                visualisation.append((startIns, var, endIns, usedSpaceMap[var][1][0], usedSpaceMap[var][1][1]))
            plot.rect(x=x, y=y, width=w, height=h, color=c, width_units="data", height_units="data")
            # show(plot)
            self.out.printf("char scratch[%d];\n"%(totalScratchSize+1), indent=True)
            self.out.printf("/* %s */"%(str(self.scratchSubs)))

    def computeScratchLocationsFirstFitPriority(self):
        if not Config.x86MemoryOptimize or forFloat():
            return
        else:
            varToLiveRange, decls = self.preProcessRawMemData()
            def sortkey(a):
                return (a[0][0], -a[0][1], -(a[2]*a[3])//8)
            varToLiveRange.sort(key=sortkey)
            freeSpace = {0:-1}
            freeSpaceRev = {-1:0}
            usedSpaceMap = {}
            totalScratchSize = -1
            listOfDimensions = []
            for ([_,_], var, size, atomSize) in varToLiveRange:
                listOfDimensions.append(size)
            #mode = 75 #(lambda x: np.bincount(x).argmax())(listOfDimensions) if len(listOfDimensions) > 0 else None
            priorityMargin = 19200
            plot = figure(plot_width=1000, plot_height=1000)
            x = []
            y = []
            w = []
            h = []
            c = []
            visualisation = []
            i = 0
            for i in range(len(varToLiveRange)):
                ([startIns, endIns], var, size, atomSize) = varToLiveRange[i]
                if var in self.notScratch:
                    continue
                spaceNeeded = size * atomSize // 8 #256 * np.ceil(size * atomSize // 8 /256)
                varsToKill = []
                for activeVar in usedSpaceMap.keys():
                    endingIns = usedSpaceMap[activeVar][0]
                    if endingIns < startIns:
                        varsToKill.append(activeVar)
                for tbk in varsToKill:
                    (st, en) = usedSpaceMap[tbk][1]
                    en += 1
                    freeSpace[st] = en
                    freeSpaceRev[en] = st
                    if en in freeSpace.keys():
                        freeSpace[st] = freeSpace[en]
                        freeSpaceRev[freeSpace[st]] = st
                        del freeSpace[en]
                        del freeSpaceRev[en]
                    if st in freeSpaceRev.keys():
                        freeSpaceRev[freeSpace[st]] = freeSpaceRev[st]
                        freeSpace[freeSpaceRev[st]] = freeSpace[st]
                        del freeSpace[st]
                        del freeSpaceRev[st]
                    del usedSpaceMap[tbk]
                potentialStart = -1
                potentialEnd = -1
                offset = 0
                for j in range(i+1, len(varToLiveRange)):
                    ([startIns_, endIns_], var_, size_, atomSize_) = varToLiveRange[j]
                    if var_ in self.notScratch:
                        continue
                    if startIns_ > endIns:
                        break
                    spaceNeeded_ = (size_ * atomSize_) // 8
                    if spaceNeeded_ >= priorityMargin and spaceNeeded < priorityMargin:
                    #if spaceNeeded_ > spaceNeeded or (spaceNeeded_ == spaceNeeded and spaceNeeded < priorityMargin and (endIns_ - startIns_ > endIns - startIns)):
                        offset = max(offset, spaceNeeded_)
                
                if offset not in freeSpace.keys() and offset > 0:
                    j = 0
                    for key in sorted(freeSpace.keys()):
                        j = key
                        if freeSpace[key] > offset:
                            break
                    if key < offset:
                        st = j
                        en = freeSpace[j]
                        freeSpace[st] = offset
                        freeSpace[offset] = en
                        freeSpaceRev[en] = offset
                        freeSpaceRev[offset] = st
                    

                for start in sorted(freeSpace.keys()):
                    if start < offset:
                        continue
                    end = freeSpace[start]
                    if end - start >= spaceNeeded or end == -1:
                        potentialStart = start
                        potentialEnd = potentialStart + spaceNeeded - 1
                        break
                    else:
                        continue
               
                if False: #Config.defragmentEnabled and potentialStart + spaceNeeded > 200000:
                    pass
                    #usedSpaceMap = self.defragmentMemory(usedSpaceMap, var, spaceNeeded, endIns, mode)
                else:
                    usedSpaceMap[var] = (endIns, (potentialStart, potentialEnd))
                    freeSpaceEnd = freeSpace[potentialStart]
                    del freeSpace[potentialStart]
                    if potentialEnd + 1 != freeSpaceEnd:
                        freeSpace[potentialEnd + 1] = freeSpaceEnd
                    freeSpaceRev[freeSpaceEnd] = potentialEnd + 1
                    if freeSpaceEnd == potentialEnd + 1:
                        del freeSpaceRev[freeSpaceEnd]
                    totalScratchSize = max(totalScratchSize, potentialEnd)
                    if self.numberOfMemoryMaps not in self.scratchSubs.keys():
                        self.scratchSubs[self.numberOfMemoryMaps] = {}
                    self.scratchSubs[self.numberOfMemoryMaps][var] = potentialStart
                    varf = var
                    if not Config.faceDetectionHacks:
                        if varf in self.coLocatedVariables:
                            varf = self.coLocatedVariables[varf]
                            self.scratchSubs[self.numberOfMemoryMaps][varf] = potentialStart
                x.append((endIns + 1 + startIns) / 2)
                w.append(endIns - startIns + 1)
                y.append((usedSpaceMap[var][1][0] + usedSpaceMap[var][1][1]) / 20000)
                h.append((usedSpaceMap[var][1][1] - usedSpaceMap[var][1][0]) / 10000)
                c.append("#" + ''.join([str(int(j)) for j in 10*np.random.rand(6)]))
                visualisation.append((startIns, var, endIns, usedSpaceMap[var][1][0], usedSpaceMap[var][1][1]))
            plot.rect(x=x, y=y, width=w, height=h, color=c, width_units="data", height_units="data")
            if not forX86():
                show(plot)
            self.out.printf("char scratch[%d];\n"%(totalScratchSize+1), indent=True)
            self.out.printf("/* %s */"%(str(self.scratchSubs)))

    def computeScratchLocationsDLX(self):
        assert not Config.faceDetectionHacks, "Please turn off Config.faceDetectionHacks flag to use DLX"
        if not Config.x86MemoryOptimize or forFloat():
            return
        else:
            varToLiveRange, decls = self.preProcessRawMemData()
            def sortkey(a):
                return (a[2]*a[3], (a[0][1]-a[0][0])*a[2]*a[3])
            varToLiveRange.sort(key=sortkey, reverse=True)
            memAlloc = [(l * m // 8, i, j) for ([i, j], k, l, m) in varToLiveRange if k not in self.notScratch]
            varOrderAndSize = [(k, l * m // 8) for ([i, j], k, l, m) in varToLiveRange if k not in self.notScratch]
            maxAllowedMemUsage = 200000
            timeout = 60
            bestCaseMemUsage = DLXInputGen.generateDLXInput(memAlloc, 1, 0, True)
            if maxAllowedMemUsage < bestCaseMemUsage:
                assert False, "Cannot fit the code within stipulated memory limit of %d" % maxAllowedMemUsage
            alignment = 1               
            dlxDumpFilesDirectory = os.path.join('seedot', 'compiler', 'codegen', 'dlx')
            dlxInputDumpDirectory = os.path.join(dlxDumpFilesDirectory, 'dlx.input')
            dlxOutputDumpDirectory = os.path.join(dlxDumpFilesDirectory, 'dlx.output')
            dlxErrorDumpDirectory = os.path.join(dlxDumpFilesDirectory, 'dlx.error')
            def getBestAlignment(mA, align, maxMem, memUsage, f):
                while True:
                    align *= 2
                    if f(DLXInputGen.generateDLXInput(mA, align, maxMem, True, None), memUsage):
                        continue
                    else:
                        return align // 2
            alignment = getBestAlignment(memAlloc, alignment, 0, bestCaseMemUsage, operator.eq)
            optimalInputGenSuccess = True
            p = mp.Process(target=DLXInputGen.generateDLXInput, args=(memAlloc, alignment, 0, False, dlxInputDumpDirectory))
            p.start()
            p.join(timeout)
            if p.is_alive():
                p.terminate()
                p.join()
                optimalInputGenSuccess = False
                print("Timeout while generating DLX input files for optimal memory usage, attempting to fit variables within %d bytes" % maxAllowedMemUsage)
                alignment = getBestAlignment(memAlloc, alignment, 0, maxAllowedMemUsage, operator.le)
                p = mp.Process(target=DLXInputGen.generateDLXInput, args=(memAlloc, alignment, maxAllowedMemUsage, False, dlxInputDumpDirectory))
                p.start()
                p.join(timeout)
                if p.is_alive():
                    p.terminate()
                    p.join()
                    assert False, "Timeout while generating DLX input files for maximum allowed memory usage, ABORT"
            if Util.windows():
                exeFile = os.path.join(dlxDumpFilesDirectory, "bin", "dlx.exe")
            else:
                exeFile = os.path.join("./%s" % dlxDumpFilesDirectory, "bin", "dlx")
            with open(dlxInputDumpDirectory) as fin, open(dlxOutputDumpDirectory, 'w') as fout, open(dlxErrorDumpDirectory, 'w') as ferr:
                try:
                    process = subprocess.call([exeFile], stdin=fin, stdout=fout, stderr=ferr, timeout=timeout)
                except subprocess.TimeoutExpired:
                    print("Memory Allocator Program Timed out.")
            if not self.checkDlxSuccess(dlxErrorDumpDirectory):
                if not optimalInputGenSuccess:
                    assert False, "Unable to allocate variables within %d bytes. ABORT" % maxAllowedMemUsage
                else:
                    alignment = getBestAlignment(memAlloc, alignment, 0, maxAllowedMemUsage, operator.le)
                    p = mp.Process(target=DLXInputGen.generateDLXInput, args=(memAlloc, alignment, maxAllowedMemUsage, False, dlxInputDumpDirectory))
                    p.start()
                    p.join(timeout)
                    if p.is_alive():
                        p.terminate()
                        p.join()
                        assert False, "Timeout while generating DLX input files for maximum allowed memory usage, ABORT"
                    with open(dlxInputDumpDirectory) as fin, open(dlxOutputDumpDirectory, 'w') as fout, open(dlxErrorDumpDirectory, 'w') as ferr:
                        try:
                            process = subprocess.call([exeFile], stdin=fin, stdout=fout, stderr=ferr, timeout=timeout)
                        except subprocess.TimeoutExpired:
                            print("Memory Allocator Program Timed out.")
                    if not self.checkDlxSuccess(dlxErrorDumpDirectory):
                        assert False, "Unable to allovate variables within %d bytes. ABORT" % maxAllowedMemUsage
            totalScratchSize = self.readDlxAllocation(dlxOutputDumpDirectory, alignment, varOrderAndSize)
            self.out.printf("char scratch[%d];\n"%(totalScratchSize), indent=True)
            self.out.printf("/* %s */"%(str(self.scratchSubs)))
            
    def preProcessRawMemData(self):
        varToLiveRange = []
        todelete = []
        decls = dict(self.decls)
        for var in decls.keys():
            if var in todelete:
                continue
            if var not in self.varLiveIntervals:
                todelete.append(var)
                continue
            if hasattr(self, 'floatConstants'):
                if var in self.floatConstants:
                    todelete.append(var)
                    continue
            if hasattr(self, 'intConstants'):
                if var in self.intConstants:
                    todelete.append(var)
                    continue
            if hasattr(self, 'internalVars'):
                if var in self.internalVars:
                    todelete.append(var)
                    continue
            size = np.prod(decls[var].shape)
            if not Config.faceDetectionHacks:
                varf = var
                while varf in self.coLocatedVariables:
                    variableToBeRemoved = self.coLocatedVariables[varf]
                    if self.varLiveIntervals[var][1] == self.varLiveIntervals[variableToBeRemoved][0]:
                        self.varLiveIntervals[var][1] = self.varLiveIntervals[variableToBeRemoved][1]
                        todelete.append(variableToBeRemoved)
                    else:
                        del self.coLocatedVariables[varf]
                        break
                    varf = variableToBeRemoved
            else:
                if var in self.coLocatedVariables.keys():
                    inp = var
                    out = self.coLocatedVariables[var]
                    if self.varLiveIntervals[inp][1] == self.varLiveIntervals[out][0]:
                        self.varLiveIntervals[inp][1] -= 1
            varToLiveRange.append((self.varLiveIntervals[var], var, size, self.varsForBitwidth[var]))
        for var in todelete:
            del decls[var]
        return varToLiveRange, decls
    
    def checkDlxSuccess(self, errorFile):
        found = False
        try:
            with open(errorFile) as ferr:
                line = ferr.readline()
                if line[:5] == "Found":
                    found = True
        except:
            pass
        return found
        
    def readDlxAllocation(self, outputfile, alignment, varOrderAndSize):
        patternRegex = re.compile(r'v(\d*).l(\d*)')
        memUsage = 0
        with open(outputfile) as fout:
            lines = fout.readlines()
            for line in lines:
                digs = patternRegex.search(line)
                varId = int(digs.group(1))
                loc = int(digs.group(2))
                varName = varOrderAndSize[varId][0]
                if self.numberOfMemoryMaps not in self.scratchSubs.keys():
                    self.scratchSubs[self.numberOfMemoryMaps] = {}
                self.scratchSubs[self.numberOfMemoryMaps][varName] = loc * alignment
                varf = varName
                if varf in self.coLocatedVariables:
                    varf = self.coLocatedVariables[varf]
                    self.scratchSubs[self.numberOfMemoryMaps][varf] = loc * alignment
                memUsage = max(memUsage, loc * alignment + varOrderAndSize[varId][1])
        return memUsage