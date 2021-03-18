# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from typing import Tuple
import numpy as np
import os

from seedot.compiler.codegen.codegenBase import CodegenBase

import seedot.compiler.ir.ir as IR
import seedot.compiler.ir.irUtil as IRUtil

import seedot.compiler.type as Type
from seedot.util import *
from seedot.writer import Writer

from bokeh.plotting import figure, output_file, show

import time

class EzPC(CodegenBase):

    def __init__(self, outputDir, decls, localDecls, scales, intvs, cnsts, expTables, globalVars, internalVars, floatConstants, substitutions, demotedVarsOffsets, varsForBitwidth, varLiveIntervals, notScratch, coLocatedVariables):
        self.outputDir = outputDir
        ezpcfile = os.path.join(
            self.outputDir, "predict.ezpc")
        
        self.out = Writer(ezpcfile)

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

    def printPrefix(self):

        self.printEzPCMain()

        self.printInputs()

        # self.computeScratchLocationsFirstFitPriority()
        self.printModelParamsWithBitwidth()

        self.printVarDecls()

        self.printConstDecls()

        self.out.printf('\n')
    
    def printModelParamsWithBitwidth(self):
        if config.vbwEnabled:
            for var in self.globalVars:
                if var + "idx" in self.globalVars and var + "val" in self.globalVars:
                    continue
                bw = self.varsForBitwidth[var]
                typ_str = "int64_al"
                size = self.decls[var].shape
                sizestr = ''.join(["*%dL" % (i) for i in size])
                if sizestr != "":
                    sizestr = '[' + sizestr[1:] + ']'

                self.out.printf(typ_str + " " + sizestr + " " + var + ";\n", indent = True)
                
                for i in range(len(size)):
                    self.out.printf("for i%d=[0L:%dL] {\n" % (i,size[i]), indent = True)
                    self.out.increaseIndent()
                
                indexstr = '['
                for i in range(len(size)):
                    indvar = "i"  + str(i)
                    dim_mul = ""
                    for j in range(i+1,len(size)):
                        dim_mul  = dim_mul + str(size[j])
                        if j != len(size) - 1:
                            dim_mul = dim_mul + "*"
                    if i != 0:
                        indexstr = indexstr + '+'
                    indexstr = indexstr + "(" + indvar
                    if (i!= len(size) - 1) and (dim_mul != ""):
                        indexstr = indexstr + "*" + dim_mul
                    indexstr = indexstr + ")"
                indexstr = indexstr + ']'

                divide = int(round(np.ldexp(1, config.wordLength - self.varsForBitwidth[var] + (self.demotedVarsOffsets.get(var, 0) if self.varsForBitwidth[var] != config.wordLength else 0) ))) if var[-3:] != "idx" and var != "X" else 1
                self.out.printf(var + indexstr + " = (" + var + "temp" + indexstr + "/" + str(divide) + "L);\n", indent = True)

                for i in range(len(size)):
                    self.out.decreaseIndent()
                    self.out.printf("};\n", indent = True)


    def printVarDecls(self):
        for decl in self.decls:
            if decl in self.globalVars:
                continue

            # typ_str = IR.DataType.getIntStr()
            # if config.vbwEnabled:
            #     if hasattr(self, 'varsForBitwidth'):
            #         typ_str = ("int%d_t" % (self.varsForBitwidth[decl])) if decl in self.varsForBitwidth else typ_str
            #     else:
            #         assert False, "VBW enabled but bitwidth info missing"
            typ_str = ""
            idf_str = decl
            shape_str = ""
            type = self.decls[decl]
            if Type.isInt(type):
                typ_str = "int64_pl"
                shape_str = ''
            elif Type.isTensor(type):
                typ_str = "int64_al"
                shape_str = ''.join(['*' + str(n) + 'L' for n in type.shape])
                if shape_str != "":
                    shape_str = '[' + shape_str[1:] + ']'
            if idf_str in self.varsForBitwidth and idf_str[:3] == "tmp":
                bw = self.varsForBitwidth.get(decl, config.wordLength)
                idf_str = idf_str + "bw" + str(bw)
            self.out.printf('%s %s %s;\n', typ_str, shape_str, idf_str,
                            indent=True)
            if Type.isTensor(type):
                if shape_str == "":
                    self.out.printf("%s = 0L;\n", idf_str, indent=True)
                else:
                    self.out.printf("for init%s=[0L:(%s)]{\n", idf_str, shape_str[1:-1], indent=True)
                    self.out.increaseIndent()
                    self.out.printf("%s[init%s] = 0L;\n", idf_str, idf_str, indent=True)
                    self.out.decreaseIndent()
                    self.out.printf("};\n", indent=True)
        self.out.printf('\n')

    def printVar(self, ir):
        if config.vbwEnabled:
            if hasattr(self, "varsForBitwidth"):
                if ir.idf in self.varsForBitwidth and ir.idf[:3] == "tmp" and ir.idf in self.decls:
                    self.out.printf("%sbw%d", ir.idf, self.varsForBitwidth[ir.idf])
                else:
                    self.out.printf("%s", ir.idf)
            else:
                assert False, "Illegal state, codegenBase must have variable bitwidth info for VBW mode"
        else:
            self.out.printf("%s", ir.idf)
        if len(ir.idx) != 0:
            try:
                var_shape = self.decls[ir.idf].shape
            except:
                var_shape = []

            self.out.printf('[')
            for i in range(len(ir.idx)):
                dim_mul = ""
                for j in range(i+1,len(var_shape)):
                    dim_mul  = dim_mul + str(var_shape[j])
                    if j != len(var_shape) - 1:
                        dim_mul = dim_mul + "*"
                if i != 0:
                    self.out.printf('+')
                self.out.printf("(")
                self.print(ir.idx[i])
                if (i!= len(ir.idx) - 1) and (dim_mul != ""):
                    self.out.printf("*" + dim_mul)
                self.out.printf(")")
            self.out.printf(']')


    def printEzPCMain(self):
        self.out.printf('def void main()\n', indent=True)
        self.out.printf('{\n', indent=True)
        self.out.increaseIndent()
        self.out.printf('initialize();\n', indent=True)
        

    def printInputs(self):
        for var in self.globalVars:
            
            bw = self.varsForBitwidth[var]
            
            typ_str = "int64_al"
            
            size = self.decls[var].shape
            sizestr = ''.join(["*%dL" % (i) for i in size])
            sizestr = '['  + sizestr[1:] + ']'
            inputPrefix = "input("

            if var[0:1] == 'X':
                party = "CLIENT"
            else:
                party = "SERVER"
            
            opt_temp = ""
            if config.vbwEnabled:
                opt_temp = "temp"

            self.out.printf(inputPrefix + party + ", " + var  + opt_temp + ", " + typ_str + sizestr + ");\n", indent = True)
    
    def printConstDecls(self):
        for cnst in self.cnsts:
            var, num = cnst, self.cnsts[cnst]

            if config.vbwEnabled and var in self.varsForBitwidth.keys():
                self.out.printf('%sbw%d = %dL;\n', var, self.varsForBitwidth[var], num, indent=True)
            else:
                self.out.printf('%s = %dL;\n', var, num, indent=True)
                
    def printCExpr(self, ir):
        self.out.printf('((')
        self.print(ir.cond)
        self.out.printf(' )?_bl (')
        self.print(ir.et)
        self.out.printf(') :( ')
        self.print(ir.ef)
        self.out.printf('))')
    
    def printTypeCast(self, ir):
        assert False, "Type case not allowed in EzPC"
    
    def printAssn(self, ir):
        self.out.printf('', indent=True)
        self.print(ir.var)
        self.out.printf(' = (')
        self.print(ir.e)
        self.out.printf(');\n')
    
    def printIf(self, ir):
        self.out.printf('if (', indent=True)
        self.print(ir.cond)
        self.out.printf(') \n')
        self.out.printf('{\n', indent=True)

        self.out.increaseIndent()
        for cmd in ir.trueCmds:
            self.print(cmd)
        self.out.decreaseIndent()

        if len(ir.falseCmds) == 0:
            self.out.printf('};\n', indent=True)
            return

        self.out.printf('} else {\n', indent=True)

        self.out.increaseIndent()
        for cmd in ir.falseCmds:
            self.print(cmd)
        self.out.decreaseIndent()

        self.out.printf('};\n', indent=True)

    def printFor(self, ir):
        self.printForHeader(ir)
        self.out.increaseIndent()
        self.printLocalVarDecls(ir)
        for cmd in ir.cmd_l:
            self.print(cmd)
        self.out.decreaseIndent()
        self.out.printf('};\n', indent=True)

    def printForHeader(self, ir):
        assert isinstance(ir.cond, IR.BoolCop), "Can't declare EzPC for loop"
        self.out.printf('for ', indent=True) #Loop counter must be int16 else indices can overflow
        self.out.printf('%s', ir.var.idf)
        self.out.printf('=[%dL:', ir.st)
        self.out.printf('%dL] {\n',ir.cond.e2.n)
        
    def printWhile(self, ir):
        self.out.printf('while (', indent=True)
        self.print(ir.expr)
        self.out.printf(') \n{\n')
        self.out.increaseIndent()
        for cmd in ir.cmds:
            self.print(cmd)
        self.out.decreaseIndent()
        self.out.printf('};\n', indent=True)

    def printLocalVarDecls(self, ir):
        for var in ir.varDecls.keys():
            #TODO: Add support for multiple datatypes
            typ_str = "int64_al"
            type = ir.varDecls[var]
            if Type.isInt(type):
                shape_str = ''
            elif Type.isTensor(type):
                shape_str = ''.join(['*' + str(n) + 'L' for n in type.shape])
                shape_str = '[' + shape_str[1:] + ']'
            if var.find('_') != -1:
                var.replace('_', 'bw') 
            self.out.printf(typ_str + shape_str + " " + var + ";\n", indent=True)
            
    def printMemset(self, ir):
        self.out.printf('for ', indent=True) #Loop counter must be int16 else indices can overflow
        self.out.printf('msetVar')
        self.out.printf('%s', ir.e.idf)
        self.out.printf('=[0L:')
        self.out.printf('%dL] {\n',ir.len)
        self.out.increaseIndent()
        vbwstr = ""
        if config.vbwEnabled and ir.e.idf in self.varsForBitwidth.keys():
            vbwstr = "bw" + str(self.varsForBitwidth[ir.e.idf])
        self.out.printf('%s%s', ir.e.idf, vbwstr, indent=True)
        self.out.printf('[msetVar')
        self.out.printf('%s', ir.e.idf)
        self.out.printf('] = 0L;\n')
        self.out.decreaseIndent()
        self.out.printf('};\n', indent=True)
        self.out.printf('\n')

    def printSuffix(self, expr: IR.Expr):
        self.out.printf('\n')

        # type = self.decls[expr.idf]

        # if Type.isInt(type) or (Type.isTensor(type) and type.dim == 0):
        #     self.out.printf('res[0] = ', indent = True)
        #     self.print(expr)
        #     self.out.printf(';\n')
        # elif Type.isTensor(type):
        #     idfr = expr.idf
        #     iters = []
        #     resIndex = ''
        #     remSize = np.prod(type.shape)
        #     for i in range(type.dim):
        #         s = chr(ord('i') + i)
        #         remSize = remSize // type.shape[i]
        #         resIndex += str(s) + '*' + str(remSize) + '+'
        #         tempVar = IR.Var(s)
        #         iters.append(tempVar)
        #     resIndex = resIndex[:-1]
        #     expr_1 = IRUtil.addIndex(expr, iters)
        #     cmds = IRUtil.loop(type.shape, iters, [
        #         IR.Assn(IRUtil.addIndex(IR.Var('res'), [IR.Var(resIndex)]), IRUtil.addIndex(expr, iters))
        #     ])
        #     self.print(IR.Prog(cmds))
        # else:
        #     assert False, "Illegal type of program output"
        type = self.decls[expr.idf]

        if Type.isInt(type) or (Type.isTensor(type)):
            self.out.printf('output(CLIENT, ', indent = True)
            self.print(expr)
            self.out.printf(');\n')
        self.out.printf('finalize();\n', indent=True)
        self.out.decreaseIndent()
        self.out.printf('}\n', indent=True)

        self.out.close()

    # def printArgMax(self, argLists):
    #     argList, extendList, nameargs = argLists
    #     name = "ArgMax"
    #     replicaidfname = (nameargs[-1]).idf
    #     self.out.printf("int64_al [1] %svarargmax;\n", replicaidfname, indent=True)
    #     self.out.print("%s(" %name, indent=True)
    #     keys = argList
    #     if keys != None:
    #         for i in range(len(keys)):
    #             arg = keys[i]
    #             self.print(arg)
    #             if i != len(keys) - 1:
    #                 self.out.printf(", ")
        
    #     if (len(extendList)) != 0:
    #         self.out.printf(", ")
    #     keys = extendList    
    #     for i in range(len(extendList)):
    #         arg = keys[i]
    #         self.out.printf(arg)
    #         if i != len(keys) - 1:
    #             self.out.printf(", ")
    #     if (len(nameargs)) != 0:
    #         self.out.printf(", ")
    #     keys = nameargs
    #     for i in range(len(nameargs)-1):
    #         arg = keys[i]
    #         self.print(arg)
    #         if i != len(keys) - 1:
    #             self.out.printf(", ")
    #     self.out.printf("%s", replicaidfname)
    #     self.out.printf(");\n")     
    
        

    def printFuncCall(self, ir):
        self.printLocalVarDecls(ir)
        # TODO: Not using the typeList at the moment
        name, revisedArgLists = self.translateToEzPC(ir)
        # if name == "ArgMax":
        #     self.printArgMax(revisedArgLists)
        #     return
        self.out.printf("%s(" % name, indent=True)
        argList, extendList, nameargs = revisedArgLists
        keys = argList
        if keys != None:
            for i in range(len(keys)):
                arg = keys[i]
                self.print(arg)
                if i != len(keys) - 1:
                    self.out.printf(", ")
        
        if (len(extendList)) != 0:
            self.out.printf(", ")
        keys = extendList    
        for i in range(len(extendList)):
            arg = keys[i]
            self.out.printf(arg)
            if i != len(keys) - 1:
                self.out.printf(", ")
        
        if (len(nameargs)) != 0:
            self.out.printf(", ")
        keys = nameargs
        for i in range(len(nameargs)):
            arg = keys[i]
            self.print(arg)
            if i != len(keys) - 1:
                self.out.printf(", ")
        
        self.out.printf(");\n")     
    
    def getEzPCName(self, ir_name):
        if ir_name[:-2] == 'MatAdd':
            return 'MatAdd'
        elif ir_name[:-2] == 'MatSub':
            return 'MatSub'
        elif ir_name[:-2] == 'MatMul':
            return 'MatMul'
        else:
            return ir_name

    def translateToEzPC(self, ir):
        ir_name = ir.name
        typeTemplateStart = ir_name.find('<')
        argListOrig = list(ir.argList)
        argListNew = []
        if typeTemplateStart != -1:
            substrName = ir_name[:typeTemplateStart]
            typeTemplateEnd = ir_name.find('>')
            funcname = self.getEzPCName(substrName)
            substrType = ir_name[typeTemplateStart+1:typeTemplateEnd]
            typeList = substrType.replace(' ', '').split(',')
            if len(typeList) != 0:
                for i in range(len(typeList)):
                    elem = typeList[i]
                    endidx= elem.find('_')
                    if elem[0] == 'u':
                        bwarg = elem[4:endidx]
                    else:
                        bwarg = elem[3:endidx]
                    argListNew.append(bwarg)
            revisedArgList = self.modifyArgList(funcname, argListOrig, argListNew, 0)        
            return funcname, revisedArgList
        else:
            funcname = self.getEzPCName(ir_name)
            revisedArgList = self.modifyArgList(funcname, argListOrig, [], 1)
            return funcname, revisedArgList

        
    def modifyArgList(self, name, initList, extendList, mode):
        if name == "MatMul" or name == "Convolution":
            if mode == 1:
                extendList = ["1L", "16L", "16L", "16L", "16L"]

            nameargs = initList[0:4]
            initList = initList[4:]
            return (initList, extendList, nameargs)

        elif (name == "MatSub") or (name == "MatAdd") or \
            (name == "MulCir") or (name == "ScalarMul") or\
            (name == "MatAddBroadCastB") or (name == "MatAddBroadCastA") or \
            (name == "MatSubBroadCastA") or (name == "MatSubBroadCastB") or \
            name == "AddOrSubCir4D" or name == "MatAdd4":
            if mode == 1:
                extendList = ["1L", "16L", "16L", "16L", "16L"]

            nameargs = initList[0:3]
            initList = initList[3:]
            return (initList, extendList, nameargs)

        elif (name == "Sigmoid"):
            if mode == 1:
                extendList = ["16L", "16L"]
            else:
                extendList.extend(extendList)
            nameargs = initList[0:1]
            nameargs.append(initList[8])
            initListPre = []
            initListPre.extend(initList[1:3])
            intiListPost = initList[6:8]
            initListPre.extend(intiListPost)
            initList = initListPre
            return (initList, extendList, nameargs)
        elif  (name == "TanH"):
            if mode == 1:
                extendList = ["16L", "16L"]
            else:
                extendList.extend(extendList)
            nameargs = initList[0:1]
            nameargs.append(initList[5])
            initListPre = []
            initListPre.extend(initList[1:5])
            initList = initListPre
            return (initList, extendList, nameargs)
        elif (name == "AdjustScaleShl"):
            nameargs = [initList[0]]
            initList = initList[1:]
            return (initList, [], nameargs)
        elif (name == "ArgMax"):
            nameargs = initList[0:1]
            nameargs.append(initList[3])
            initList = initList[1:3]
            extendList.extend(extendList)
            return (initList, extendList, nameargs)
        elif name == "Relu6":
            nameargs = initList[0:2]
            initList = initList[2:]
            return (initList, extendList, nameargs)
        elif name == "Reverse2":
            nameargs = initList[0:1]
            nameargs.append(initList[-1])
            initList = initList[1:-1]
            return (initList, extendList, nameargs)
        elif name == "MBConv":
            nameargs = initList[0:14]
            initList = initList[14:]
            return (initList, extendList, nameargs)
        elif name == "NormaliseL2":
            nameargs = initList[0:2]
            initList = initList[2:]
            return (initList, extendList, nameargs)
        else:
            return (initList, extendList, [])

    def printComment(self, ir):
        self.out.printf('\n')
        self.out.printf('(* ' + ir.msg + ' *)\n', indent=True)

    def printInt(self, ir):
        self.out.printf('%dL', ir.n)
    
    def printMemcpy(self, ir):
        def printIdx(idf, index):
            try:
                var_shape = self.decls[idf].shape
            except:
                var_shape = []
            for i in range(len(index)):
                dim_mul = ""
                for j in range(i+1,len(var_shape)):
                    dim_mul  = dim_mul + str(var_shape[j])
                    if j != len(var_shape) - 1:
                        dim_mul = dim_mul + "*"
                self.out.printf("(")
                self.print(index[i])
                if (i!= len(index) - 1) and (dim_mul != ""):
                    self.out.printf("*" + dim_mul)
                self.out.printf(")")
                self.out.printf(' + ')
        # typ_str = "MYINT"
        # if config.vbwEnabled:
        #     if hasattr(self, 'varsForBitwidth'):
        #         # Note ir.to and ir.start are constrained to have the same bitwidth
        #         typ_str = ("int%d_t" % (self.varsForBitwidth[ir.to.idf])) if ir.to.idf in self.varsForBitwidth else typ_str
        #     else:
        #         assert False, "Illegal state, VBW mode but no variable information present"
        # typ_str = "float" if forFloat() else typ_str
        # self.out.printf('memcpy(', indent=True)
        # toIndexed = IRUtil.addIndex(IR.Var(ir.to.idf), ir.to.idx + ir.toIndex)
        # startIndexed = IRUtil.addIndex(IR.Var(ir.start.idf), ir.start.idx + ir.startIndex)
        # self.out.printf("&")
        # self.print(toIndexed)
        # self.out.printf(", &")
        # self.print(startIndexed)
        # self.out.printf(', sizeof(%s) * %d);\n' % (typ_str, ir.length))
        
        vbwstrto = ""
        if config.vbwEnabled and ir.to.idf in self.varsForBitwidth.keys():
            vbwstrto = "bw" + str(self.varsForBitwidth[ir.to.idf])
        vbwstrstart = ""
        if config.vbwEnabled and ir.start.idf in self.varsForBitwidth.keys():
            vbwstrstart = "bw" + str(self.varsForBitwidth[ir.start.idf])

        self.out.printf('for ', indent=True) #Loop counter must be int16 else indices can overflow
        self.out.printf('mcpyVar')
        self.out.printf('%s%s%s%s', ir.start.idf, vbwstrstart, ir.to.idf, vbwstrto)
        self.out.printf('=[0L:')
        self.out.printf('%dL] {\n',ir.length)
        self.out.increaseIndent()
        self.out.printf('%s%s', ir.to.idf, vbwstrto, indent=True)
        self.out.printf('[')
        printIdx(ir.to.idf, ir.to.idx + ir.toIndex)

        self.out.printf('mcpyVar%s%s%s%s', ir.start.idf, vbwstrstart, ir.to.idf, vbwstrto)
        self.out.printf('] = ')
        self.out.printf('%s%s', ir.start.idf, vbwstrstart)
        self.out.printf('[')
        printIdx(ir.start.idf, ir.start.idx + ir.startIndex)
        self.out.printf('mcpyVar%s%s%s%s', ir.start.idf, vbwstrstart, ir.to.idf, vbwstrto)
        self.out.printf('];\n')
        
        self.out.decreaseIndent()
        self.out.printf('};\n', indent=True)
        self.out.printf('\n')
