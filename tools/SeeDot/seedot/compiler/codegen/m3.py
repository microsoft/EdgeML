# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import os

from seedot.compiler.codegen.codegenBase import CodegenBase

import seedot.compiler.ir.ir as IR
import seedot.compiler.ir.irUtil as IRUtil

import seedot.compiler.type as Type
from seedot.util import *
from seedot.writer import Writer

import time

class M3(CodegenBase):

    def __init__(self, outputDir, decls, localDecls, scales, intvs, cnsts, expTables, globalVars, internalVars, floatConstants, substitutions, demotedVarsOffsets, varsForBitwidth, varLiveIntervals, notScratch, coLocatedVariables):
        self.outputDir = outputDir
        cppFile = os.path.join(
            self.outputDir, "predict.c")
        
        self.out = Writer(cppFile)

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
        self.notScratch = notScratch
        self.scratchSubs = {}

        self.numberOfMemoryMaps = 0
        self.currentMemMap = 0
        self.defragmentationInstructions = []
        self.defragmentationParameters = []
        self.coLocatedVariables = dict(coLocatedVariables)

    def printPrefix(self):

        self.printCincludes()

        # self.printVarDecls()

        self.printCHeader()

        self.computeScratchLocationsFirstFitPriority()

        self.printVarDecls(globalVarDecl=False)

        self.printConstDecls()

        self.out.printf('\n')

    def printCincludes(self):
        self.out.printf('#include <math.h>\n', indent=True)
        self.out.printf('#include <stdbool.h>\n\n', indent=True)
        self.out.printf('#include "quantized_datatypes.h"\n', indent=True)
        self.out.printf('#include "quantized_utils.h"\n', indent=True)
        self.out.printf('#include "quantized_mbconv.h"\n', indent=True)
        self.out.printf('#include "model.h"\n', indent=True)
    

    def printCHeader(self):
        if forFloat():
            func = "Float"
            type = "float"
        else:
            func = "Fixed"
            type = "Q15_T"
        if forFloat():
            self.out.printf('void seedot%s(%s **X, float* res) {\n' % (func, type), indent=True)
        else: 
            self.out.printf('void seedot%s%s(%s **X, Q31_T* res) {\n' % (func, "", type), indent=True)
        self.out.increaseIndent()

    def printVarDecls(self, globalVarDecl=True):
        for decl in self.decls:
            if decl in self.globalVars:
                continue

            if decl in self.scratchSubs[self.numberOfMemoryMaps]:
                continue

            if forFloat() and decl not in self.internalVars:
                typ_str = IR.DataType.getFloatStr()
            elif forFixed() and decl not in self.internalVars:
                if config.vbwEnabled and decl not in self.internalVars:
                    bw = self.varsForBitwidth.get(decl, config.wordLength)
                    typ_str = "Q%d_T" % (bw - 1)
                else:
                    typ_str = IR.DataType.getIntStr()
            else:
                typ_str = "Q31_T"

            idf_str = decl
            type = self.decls[decl]
            if Type.isInt(type):
                shape_str = ''
            elif Type.isTensor(type):
                shape_str = ''.join(['[' + str(n) + ']' for n in type.shape])

            if not config.vbwEnabled:
                self.out.printf('%s %s%s;\n', typ_str, idf_str, shape_str, indent=True)
            else:
                if forFixed() and idf_str in self.varsForBitwidth and idf_str[:3] == "tmp":
                    if globalVarDecl:
                        for bw in config.availableBitwidths:
                            self.out.printf("Q%d_T vars_%s::%s_%d%s;\n", bw - 1, getVersion(), idf_str, bw, shape_str, indent=True)
                    else:
                        self.out.printf("Q%d_T %s_%d%s;\n", self.varsForBitwidth[idf_str] - 1, idf_str, bw, shape_str, indent=True)
                else:
                    if globalVarDecl:
                        self.out.printf("%s vars_%s::%s%s;\n", typ_str, getVersion(), idf_str, shape_str, indent=True)
                    else:
                        self.out.printf("%s %s%s;\n", typ_str, idf_str, shape_str, indent=True)

        self.out.printf('\n')


    def printSuffix(self, expr: IR.Expr):
        self.out.printf('\n')

        type = self.decls[expr.idf]

        if Type.isInt(type) or (Type.isTensor(type) and type.dim == 0):
            self.out.printf('res[0] = ', indent = True)
            self.print(expr)
            self.out.printf(';\n')
        elif Type.isTensor(type):
            idfr = expr.idf
            iters = []
            resIndex = ''
            remSize = np.prod(type.shape)
            for i in range(type.dim):
                s = chr(ord('i') + i)
                remSize = remSize // type.shape[i]
                resIndex += str(s) + '*' + str(remSize) + '+'
                tempVar = IR.Var(s)
                iters.append(tempVar)
            resIndex = resIndex[:-1]
            expr_1 = IRUtil.addIndex(expr, iters)
            cmds = IRUtil.loop(type.shape, iters, [
                IR.Assn(IRUtil.addIndex(IR.Var('res'), [IR.Var(resIndex)]), IRUtil.addIndex(expr, iters))
            ])
            self.print(IR.Prog(cmds))
        else:
            assert False, "Illegal type of program output"

        self.out.decreaseIndent()
        self.out.printf('}\n', indent=True)

        self.out.close()

    def printForHeader(self, ir):
        self.out.printf('for (%s ', "S_ITER_T", indent=True) #Loop counter must be int16 else indices can overflow
        self.print(ir.var)
        self.out.printf(' = %d; ', ir.st)
        self.print(ir.cond)
        self.out.printf('; ')
        self.print(ir.var)
        self.out.printf('++) {\n') #TODO: What if --?

    def printMemset(self, ir):
        self.out.printf('memset(', indent=True)
        if Config.x86MemoryOptimize and forFixed() and self.numberOfMemoryMaps in self.scratchSubs:
            self.out.printf("(scratch + %d)", self.scratchSubs[self.numberOfMemoryMaps][ir.e.idf])
        else:
            self.print(ir.e)
        typ_str = "Q15_T"
        if config.vbwEnabled:
            if hasattr(self, 'varsForBitwidth'):
                typ_str = ("Q%d_T" % (self.varsForBitwidth[ir.e.idf] - 1)) if ir.e.idf in self.varsForBitwidth else typ_str
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
        typ_str = "Q15_T"
        if config.vbwEnabled:
            if hasattr(self, 'varsForBitwidth'):
                # Note ir.to and ir.start are constrained to have the same bitwidth
                typ_str = ("Q%d_T" % (self.varsForBitwidth[ir.to.idf] - 1)) if ir.to.idf in self.varsForBitwidth else typ_str
            else:
                assert False, "Illegal state, VBW mode but no variable information present"
        typ_str = "float" if forFloat() else typ_str
        self.out.printf('memcpy(', indent=True)
        if Config.x86MemoryOptimize and forFixed() and self.numberOfMemoryMaps in self.scratchSubs:
            for (a, b, c) in [(ir.to.idf, ir.toIndex, 0), (ir.start.idf, ir.startIndex, 1)]:
                self.out.printf("((scratch + %d + sizeof(%s)*(", self.scratchSubs[self.numberOfMemoryMaps][a], typ_str)
                toIndexed = IRUtil.addIndex(IR.Var(""), b)
                if len(b) == 0:
                    self.out.printf("0")
                elif len(b) == len(self.decls[a].shape):
                    printFlattenedIndices(b, self.decls[a].shape)
                else:
                    assert False, "Illegal state, number of offsets to memcpy should be 0 or match the original tensor dimensions"
                self.out.printf(")))")
                if c == 0:
                    self.out.printf(", ")
        else:
            toIndexed = IRUtil.addIndex(IR.Var(ir.to.idf), ir.toIndex)
            startIndexed = IRUtil.addIndex(IR.Var(ir.start.idf), ir.startIndex)
            self.out.printf("&")
            self.print(toIndexed)
            self.out.printf(", &")
            self.print(startIndexed)
        self.out.printf(', sizeof(%s) * %d);\n' % (typ_str, ir.length))


    def printFuncCall(self, ir):
        if forFloat():
            super().printFuncCall(ir)
        else:
            self.printLocalVarDecls(ir)
            name, args = self.translateToC(ir.name, ir.argList)
            self.out.printf("%s(" % name, indent=True)
            keys = list(args)
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

                # if forFixed():
                #     typeCast = ("(Q%d_T*)" % (self.varsForBitwidth[arg.idf] - 1)) if x > 0 else ""
                #     self.out.printf(typeCast)
                
                if not (isinstance(arg, IR.Var) and arg.idf in self.scratchSubs[self.currentMemMap]):
                    if x != 0:
                        self.out.printf("&")
                    self.print(arg)
                    if x != 0 and x != -1:
                        self.out.printf("[0]" * x)
                else:
                    self.out.printf("(scratch + %d)"%(self.scratchSubs[self.currentMemMap][arg.idf]))
                if i != len(keys) - 1:
                    self.out.printf(", ")
            self.out.printf(");\n")

    def translateToC(self, varName, argList):
        varName = varName.replace('<', ' ').replace('>', '').replace(',', '')
        varName = varName.split(' ')
        name = varName[0]
        bitwidths = []
        if name[:7] == "Sigmoid" or name[:4] == "TanH":
            pass
        else:
            for bws in varName[1:]:
                bitwidths.append(int((bws[3:])[:-2]))
        revArgList = {}
        for key, value in argList.items():
            revArgList[value] = key

        assert forFixed(), "Only fixed point code for M3 supported"
        assert config.vbwEnabled, "Function calls for VBW mode only supported on M3"
        
        # Type checking has already been done so no exhaustive checks here
        if name[:-2] == "MatAdd" or name == "MatSub":   #MatAddNC MatAddCN MatAddCC MatAddNN
            shapeB = self.decls[revArgList["B"].idf].shape
            if shapeB[1] == 1:  
                op = "add" if name[3:6] == "Add" else "sub"
                assert bitwidths[0] == bitwidths[1] == bitwidths[3]
                if op == "add":
                    assert bitwidths[0] == 16, "Not Implemented for M3"
                funcName = "q%d_v_%s" % (bitwidths[0] - 1, op)
                scret = revArgList["shrC"].n * revArgList["demote"].n
                args = {
                    revArgList["A"] : "vec1",
                    revArgList["B"] : "vec2",
                    revArgList["I"] : "len",
                    revArgList["C"] : "ret",
                    revArgList["shrA"]: "scvec1",
                    revArgList["shrB"]: "scvec2",
                    revArgList["shrC"]: "scret",
                    revArgList["demote"]: "demote"
                } if op == "add" else {
                    revArgList["A"] : "vec1",
                    revArgList["B"] : "vec2",
                    revArgList["J"] : "len",
                    revArgList["C"] : "ret",
                    revArgList["shrA"]: "scvec1",
                    revArgList["shrB"]: "scvec2",
                    IR.Int(scret) : "scret"
                }
                return funcName, args
            else:
                assert False, "Not Implemented for M3"
        elif name[:-1] == "MatAdd": #MatAdd4
            assert bitwidths[0] == bitwidths[1] == bitwidths[3]
            funcName = "q%d_t_add" % (bitwidths[0] - 1)
            scret = revArgList["shrC"].n * revArgList["demote"].n
            args = {
                revArgList["A"] : "ten1",
                revArgList["B"] : "ten2",
                revArgList["N"] : "nbatches",
                revArgList["H"] : "nrows",
                revArgList["W"] : "ncols",
                revArgList["C"] : "nchannels",
                revArgList["X"] : "ret",
                revArgList["shrA"] : "scten1",
                revArgList["shrB"] : "scten2",
                IR.Int(scret) : "scret"
            }
            return funcName, args
        elif name[:6] == "MatAdd" and name[6:-1] == "BroadCast": #MatAddBroadCastA MatAddBroadCastB
            if name[-1] == "A":
                shapeVec = self.decls[revArgList["B"].idf].shape
                vec = revArgList["B"]
                scalar = revArgList["A"]
                scvec = revArgList["shrB"]
                scscalar = revArgList["shrA"]
            elif name[-1] == "B":
                shapeVec = self.decls[revArgList["A"].idf].shape
                vec = revArgList["A"]
                scalar = revArgList["B"]
                scvec = revArgList["shrA"]
                scscalar = revArgList["shrB"]
            else:
                assert False, "Illegal State"
            if shapeVec[1] == 1:
                assert bitwidths[0] == bitwidths[1] == bitwidths[3] == 16
                funcName = "q15_v_scalar_add"
                scret = revArgList["shrC"].n * revArgList["demote"].n
                args = {
                    IR.Int(self.cnsts[scalar.idf]) : "scalar", #scalar : "scalar",
                    vec : "vec",
                    revArgList["I"] : "len",
                    revArgList["C"] : "ret",
                    scscalar : "scscalar",
                    scvec : "scvec",
                    IR.Int(scret) : "scret"
                }
                return funcName, args
            else:
                assert False, "Not implemented for M3"
        elif name[:6] == "MatSub" and name[6:-1] == "BroadCast":            
            assert bitwidths[0] == bitwidths[1] == bitwidths[3] == 16, "Not implemented on M3"
            scret = revArgList["shrC"].n * revArgList["demote"].n
            if name[-1] == "A":
                shapeB = self.decls[revArgList["B"].idf].shape
                assert shapeB[1] == 1
                funcName = "q15_v_scalar_sub"
                args = {
                    IR.Int(self.cnsts[revArgList["A"].idf]) : "scalar", #revArgList["A"] : "scalar",
                    revArgList["B"] : "vec",
                    revArgList["I"] : "len",
                    revArgList["C"] : "ret",
                    revArgList["shrA"] : "scscalar",
                    revArgList["shrB"] : "scvec",
                    IR.Int(scret) : "scret"
                }   
            elif name[-1] == "B":
                assert False, "Not implemented on M3"
            return funcName, args
        elif name[:-2] == "AddOrSubCir": #AddOrSubCir2D AddOrSubCir4D
            addOrSub = "add" if revArgList["add"].b else "sub"
            bwA = bitwidths[0]
            bwB = bitwidths[1]
            bwX = bitwidths[3]
            dim = 2 if name[-2:] == "2D" else 4
            scret = revArgList["shrC"].n * revArgList["demote"].n
            if dim == 2:
                assert False, "Not implemented for M3"
            else:
                args = {
                    revArgList["A"] : "mat",
                    revArgList["B"] : "vec",
                    revArgList["N"] : "nbatches",
                    revArgList["H"] : "nrows",
                    revArgList["W"] : "ncols",
                    revArgList["C"] : "nchannels",
                    revArgList["X"] : "ret",
                    revArgList["shrA"] : "scmat",
                    revArgList["shrB"] : "scvec",
                    IR.Int(scret) : "scret"
                }
                if bwA == bwB == bwX == 16:
                    bwString = "q15_t"
                elif bwA == bwX == 8 and bwB == 16 and addOrSub == "add":
                    bwString = "q7xq15_q7_t"
                else:
                    assert False, "Not implemented for M3"
            return ("%s_%s_vec" % (bwString, addOrSub)), args
        elif name == "MulCir": #MulCir
            shapeB = self.decls[revArgList["B"].idf].shape
            if shapeB[1] == 1:  
                assert bitwidths[0] == bitwidths[1] == bitwidths[3]
                funcName = "q%d_v_hadamard" % (bitwidths[0] - 1)
                scvec2 = revArgList["shrB"].n * revArgList["demote"].n
                args = {
                    revArgList["A"] : "vec1",
                    revArgList["B"] : "vec2",
                    revArgList["I"] : "len",
                    revArgList["C"] : "ret",
                    revArgList["shrA"]: "scvec1",
                    IR.Int(scvec2) : "scvec2"
                }
                return funcName, args
            else:
                assert False, "Not Implemented for M3"
        elif name[:7] == "Sigmoid": #Sigmoid SigmoidNew16
            shapeA = self.decls[revArgList["A"].idf].shape
            use_tables = useNewTableExp() or useMathExp()
            if shapeA[1] == 1:
                assert self.varsForBitwidth[revArgList["A"].idf] == 16
                funcName = "q15_v_sigmoid"
                args = {
                    revArgList["A"] : "vec",
                    revArgList["I"] : "len",
                    revArgList["B"] : "ret",
                    revArgList.get("div", IR.Int(0)) : "div",
                    revArgList.get("add", IR.Int(0)) : "add",
                    revArgList.get("sigmoid_limit", IR.Int(0)) : "sigmoid_limit",
                    revArgList.get("scale_in", IR.Int(0)) : "scale_in",
                    revArgList.get("scale_out", IR.Int(0)) : "scale_out",
                    IR.Bool(use_tables) : "use_tables"
                }
                return funcName, args
            else:
                assert False, "Not Implemented for M3"
        elif name[:4] == "TanH": #TanH TanHNew16
            shapeA = self.decls[revArgList["A"].idf].shape
            use_tables = useNewTableExp() or useMathExp()
            if shapeA[1] == 1:
                assert self.varsForBitwidth[revArgList["A"].idf] == 16
                funcName = "q15_v_tanh"
                args = {
                    revArgList["A"] : "vec",
                    revArgList["I"] : "len",
                    revArgList["B"] : "ret",
                    revArgList.get("scale_in", IR.Int(0)) : "scale_in",
                    revArgList.get("scale_out", IR.Int(0)) : "scale_out",
                    IR.Bool(use_tables) : "use_tables"
                }
                return funcName, args
            else:
                assert False, "Not Implemented for M3"
        elif name[:3] == "Exp": #Exp ExpNew16
            assert False, "Not Implemented for M3"
        elif name[:11] == "AdjustScale": #AdjustScaleShl AdjustScaleShr AdjustScaleShlSaturate
            if name[-8:] == "Saturate":
                assert False, "Not implemented for M3"
            assert bitwidths[0] == 16, "Not implemented for M3"
            shapeA = self.decls[revArgList["A"].idf].shape
            if name[-3:] == "Shl":
                assert len(shapeA) == 2, "Not implemented for M3"
            if shapeA[1] == 1:
                funcName = "q15_v_scale_%s" % ("up" if name[-3:] == "Shl" else "down")
                ret = IR.Var(revArgList["A"].idf)
                ret.inputVar = revArgList["A"].inputVar
                ret.internalVar = revArgList["A"].internalVar
                args = {
                    revArgList["A"] : "vec",
                    revArgList["I"] : "len",
                    ret : "ret",
                    revArgList["scale"] : "scvec"
                }
                return funcName, args
        elif name == "Transpose": #Transpose
            assert False, "Not implemented for M3"
        elif name == "Reverse2": #Reverse
            assert bitwidths[0] == 16, "Not implemented for M3"
            funcName = "q15_m_reverse"
            args = {
                revArgList["A"] : "mat",
                revArgList["I"] : "nrows",
                revArgList["J"] : "ncols",
                revArgList["axis"] : "axis",
                revArgList["B"] : "ret"
            }
            return funcName, args
        elif name == "ScalarMul": #ScalarMul
            shapeB = self.decls[revArgList["B"].idf].shape
            if shapeB[1] == 1:  
                assert bitwidths[0] == bitwidths[1] == bitwidths[3] == 16
                funcName = "q15_v_scalar_mul"
                scvec = revArgList["shrB"].n * revArgList["demote"].n
                args = {
                    IR.Int(self.cnsts[revArgList["A"].idf]) : "scalar", #revArgList["A"] : "scalar",
                    revArgList["B"] : "vec",
                    revArgList["I"] : "len",
                    revArgList["C"] : "ret",
                    revArgList["shrA"]: "scscalar",
                    IR.Int(scvec) : "scvec"
                }
                return funcName, args
            else:
                assert False, "Not Implemented for M3"
        elif name[:6] == "MatMul": #MatMulNN MatMulNC MatMulCC MatMulCN
            shapeB = self.decls[revArgList["B"].idf].shape
            if shapeB[1] == 1:  
                bwA = bitwidths[0] 
                bwB = bitwidths[1] 
                bwC = bitwidths[3]
                scvec = IR.Int(revArgList["shrA"].n * revArgList["demote"].n)
                shrB = IR.Int(revArgList["shrB"].n * revArgList["H1"].n)
                args = {
                    revArgList["A"] : "mat",
                    revArgList["B"] : "vec",
                    revArgList["I"] : "nrows",
                    revArgList["J"] : "ncols",
                    revArgList["C"] : "ret",
                    shrB : "scmat", #revArgList["shrB"]: "scmat",
                    scvec : "scvec",
                    revArgList["H1"] : "H1",
                    revArgList["H2"] : "H2",
                }
                if bwA == bwB == bwC == 16: #Note the order of inputs is reversed
                    bwString = "q15"
                elif bwA == bwC == 16 and bwB == 8:
                    bwString = "q15xq7_q15"
                else:
                    assert False, "Not implemented for M3"
                funcName = "%s_m_mulvec" % bwString
                return funcName, args
            else:
                assert False, "Not Implemented for M3"
        elif name[:12] == "SparseMatMul": #SparseMatMul
            shapeB = self.decls[revArgList["B"].idf].shape
            assert revArgList["B"].idf != "X", "Sparse MatMul for X not supported on M3"
            if shapeB[1] == 1:  
                bwA = bitwidths[0] 
                bwB = bitwidths[2] 
                bwC = bitwidths[4]
                shrC = IR.Int(revArgList["shrC"].n * revArgList["demote"].n)
                args = {
                    revArgList["Aidx"] : "row_indices",
                    revArgList["Aval"] : "mat_values",
                    revArgList["B"] : "vec",
                    revArgList["K"] : "nelem",
                    revArgList["C"] : "ret",
                    revArgList["shrA"] : "scmat", 
                    revArgList["shrB"] : "scvec",
                    shrC : "scret",
                }
                if bwA == bwB == bwC == 16: #Note the order of inputs is reversed
                    bwString = "q15"
                elif bwA == bwC == 16 and bwB == 8:
                    bwString = "q15xq7_q15"
                else:
                    assert False, "Not implemented for M3"
                funcName = "%s_m_sparse_mulvec" % bwString
                return funcName, args
            else:
                assert False, "Not Implemented for M3"
        elif name == "ArgMax": #ArgMax 
            shapeA = self.decls[revArgList["A"].idf].shape
            if shapeA[1] == 1: 
                assert bitwidths[0] == 16, "Not implemented for M3"
                funcName = "q15_v_argmax"
                args = {
                    revArgList["A"] : "vec",
                    revArgList["I"] : "len",
                    revArgList["index"] : "ret"
                }
                return funcName, args
            else:
                assert False, "Not implemented for M3"
        elif name[:4] == "Relu": #Relu2D Relu4D Relu6
            if name[-2:] == "4D" or name[-2:] == "2D":
                assert False, "Not implemented for M3"
            elif name[-1] == "6":
                assert bitwidths[0] == 8, "Not implemented for M3"
                funcName = "q7_t_relu"
                args = {
                    revArgList["A"] : "vec",
                    revArgList["N"] : "nbatches",
                    revArgList["H"] : "nrows",
                    revArgList["W"] : "ncols",
                    revArgList["C"] : "nchannels",
                    revArgList["B"] : "ret",
                    revArgList["six"] : "limit",
                    revArgList["div"] : "div"
                }
                return funcName, args
            else:
                assert False, "Not implemented for M3"
        elif name == "NormaliseL2": #NormaliseL2
            assert bitwidths[0] == 16, "Not implemented for M3"
            funcName = "q15_t_l2_norm"
            args = {
                revArgList["A"] : "ten",
                revArgList["N"] : "nbatches",
                revArgList["H"] : "nrows",
                revArgList["W"] : "ncols",
                revArgList["C"] : "nchannels",
                revArgList["B"] : "ret",
                revArgList["scaleA"] : "scale_in",
                revArgList["shrA"] : "scale_out"
            }
            return funcName, args
        elif name == "Maxpool": #Maxpool
            assert False, "Not implemented for M3"
        elif name == "Convolution": #Convolution
            bwA = bitwidths[0]
            bwB = bitwidths[1]
            bwC = bitwidths[3]
            shrA = IR.Int(revArgList["shrA"].n * 2 ** revArgList["H1"].n)
            args = {
                revArgList["A"] : "input",
                revArgList["B"] : "filter",
                revArgList["C"] : "output",
                # revArgList["tmp"] : "treesumBuffer",
                revArgList["N"] : "N",
                revArgList["H"] : "H",
                revArgList["W"] : "W",
                revArgList["CIN"] : "CIn",
                revArgList["HF"] : "HF",
                revArgList["WF"] : "WF",
                revArgList["CINF"] : "CF",
                revArgList["COUTF"] : "COut",
                revArgList["HOUT"] : "HOut",
                revArgList["WOUT"] : "WOut",
                revArgList["G"] : "G",
                revArgList["HPADL"] : "HPadU",
                revArgList["HPADR"] : "HPadD",
                revArgList["WPADL"] : "WPadL",
                revArgList["WPADR"] : "WPadR",
                revArgList["HSTR"] : "HStride",
                revArgList["WSTR"] : "WStride",
                revArgList["HDL"] : "HDilation",
                revArgList["WDL"] : "WDilation",
                # revArgList["H1"] : "H1",
                # revArgList["H2"] : "H2",
                shrA : "scinput", # revArgList["shrA"] : "scinput",
                revArgList["shrB"] : "scoutput",
                revArgList["demote"] : "demote"
            }
            if bwA == bwB == bwC == 16:
                bwString = "q15"
            elif bwA == 8 and bwB == 16:
                bwString = "q7xq15_q%d" % (bwC - 1)
            else:
                assert False, "Not implemented for M3"
            return "%s_convolution" % bwString, args
        elif name == "MBConv":
            bwA = bitwidths[0]
            bwF1 = bitwidths[1]
            bwW1 = bitwidths[2]
            bwB1 = bitwidths[3]
            bwF2 = bitwidths[4]
            bwW2 = bitwidths[5]
            bwB2 = bitwidths[6]
            bwF3 = bitwidths[7]
            bwW3 = bitwidths[8]
            bwB3 = bitwidths[9]
            assert bwF1 == bwW1 == bwB1 == bwF2 == bwW2 == bwB2 == bwF3 == bwW3 == bwB3, "Not implemented for M3"
            bwB = bwF1
            bwC = bitwidths[10]
            shr1 = IR.Int(revArgList["shr1"].n * 2 ** revArgList["D1"].n)
            shr4 = IR.Int(revArgList["shr4"].n * 2 ** revArgList["D2"].n)
            shr7 = IR.Int(revArgList["shr7"].n * 2 ** revArgList["D3"].n)
            args = {
                revArgList["A"] : "input",
                revArgList["F1"] : "filter1",
                revArgList["BN1W"] : "BN1W",
                revArgList["BN1B"] : "BN1B",
                revArgList["F2"] : "filter2",
                revArgList["BN2W"] : "BN2W",
                revArgList["BN2B"] : "BN2B",
                revArgList["F3"] : "filter3",
                revArgList["BN3W"] : "BN3W",
                revArgList["BN3B"] : "BN3B",
                revArgList["C"] : "output",
                revArgList["X"] : "convBuffer1",
                revArgList["T"] : "convBuffer2",
                # revArgList["U"] : "treesumBuffer",
                revArgList["N"] : "N",
                revArgList["H"] : "H",
                revArgList["W"] : "W",
                revArgList["Cin"] : "CIn",
                revArgList["Ct"] : "CTemp",
                revArgList["HF"] : "HF",
                revArgList["WF"] : "WF",
                revArgList["Cout"] : "COut",
                revArgList["Hout"] : "HOut",
                revArgList["Wout"] : "WOut",
                revArgList["HPADL"] : "HPadU",
                revArgList["HPADR"] : "HPadD",
                revArgList["WPADL"] : "WPadL",
                revArgList["WPADR"] : "WPadR",
                revArgList["HSTR"] : "HStride",
                revArgList["WSTR"] : "WStride",
                # revArgList["D1"] : "depth1",
                # revArgList["D2"] : "depth2",
                # revArgList["D3"] : "depth3",
                revArgList["SIX_1"] : "limit1",
                revArgList["SIX_2"] : "limit2",
                shr1 : "shrU1", #revArgList["shr1"] : "shrU1",
                # revArgList["shr2"] : "shrB1",
                revArgList["shr3"] : "shrX1",
                shr4 : "shrU2", #revArgList["shr4"] : "shrU2",
                # revArgList["shr5"] : "shrB2",
                revArgList["shr6"] : "shrX2",
                shr7 : "shrU3", #revArgList["shr7"] : "shrU3",
                # revArgList["shr8"] : "shrB3",
                revArgList["shr9"] : "shrX3",
                revArgList["shl1"] : "shlU1",
                # revArgList["shl2"] : "shlB1",
                revArgList["shl3"] : "shlX1",
                revArgList["shl4"] : "shlU2",
                # revArgList["shl5"] : "shlB2",
                revArgList["shl6"] : "shlX2",
                revArgList["shl7"] : "shlU3",
                # revArgList["shl8"] : "shlB3",
                revArgList["shl9"] : "shlX3",
            }
            if bwA == bwB == bwC:
                bwString = "q%d" % (bwA - 1)
            elif bwA == 8 and bwB == bwC == 16:
                bwString = "q7xq15_q15"
            elif bwA == 16 and bwB == 8:
                bwString = "q15xq7_q%d" % (bwC - 1)
            else:
                assert False, "Not implemented for M3"
            return "%s_mbconv_block" % bwString, args
        else:
            assert False, "Not implemented for M3"