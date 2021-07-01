# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import operator

from seedot.compiler.antlr.seedotParser import seedotParser as SeeDotParser

import seedot.compiler.ast.ast as AST
from seedot.compiler.ast.astVisitor import ASTVisitor

import seedot.compiler.ir.ir as IR
import seedot.compiler.ir.irUtil as IRUtil

import seedot.config as config
import seedot.compiler.type as Type
from seedot.util import *

'''
IRBuilder class converts the input AST into IR, which is a sequence of function calls.
Each node in the input grammar is handled with its own function in this class.
'''


class IRBuilder(ASTVisitor):

    def __init__(self, outputLog, ddsScaleInfo = {}, substitutions = {}, scaleForX = None, variableToBitwidthMap={}, sparseMatrixSizes={}, demotedVarsList=[], demotedVarsOffsets={}):
        self.log = outputLog
        self.intermediateVarScales = ddsScaleInfo
            # When data driven scaling is enabled (ddsEnabled is True), during fixed point compilation mode ddsScaleInfo
            # is populated with the data driven scales for each profiled variable.
        self.substitutions = substitutions
            # During the IR building process, variables are renamed. This variable maps from original names -> new names.
            # During fixed point code run, the input variable 'substitutions' contains all such mappings beforehand
            # (computed at first during floating point code run) as this info is needed for bitwidth and scale computation.
            # The substitutions carried out are the identical for both fixed point and floating point code.
        self.varsForBitwidth = variableToBitwidthMap
            # The input 'variableToBitwidthMap' is a map of all variables to default bitwidths.
            # The map 'varsForBitwidth' is modified during the IR generation and values of
            # demoted variables are set to config.wordlength // 2 bits.
        self.demotedVarsList = demotedVarsList
            # The input 'demotedVarsList' is a list of all demoted variables (using config.wordlength // 2 bits).
        self.demotedVarsOffsets = demotedVarsOffsets
            # The input 'demotedVarsOffset' contains scale offset while conversion from config.wordLength to config.wordlength // 2.
            # The scale of a demoted variable is original scale + config.wordLength // 2 + offset.
        self.independentBitwidthVars = [i for i in self.varsForBitwidth]
            # These are variables which are profiled and their scales are computed from the profile of the floating point code.
        self.ddsEnabled = config.ddsEnabled
        self.vbwEnabled = config.vbwEnabled and forFixed()
            # self.vbwEnabled = True means templated library functions are used.
            # Floating point library methods do not have a template implementation as the floating point code uses only FP32,
            # hence for floating point mode self.vbwEnabled is set to False.
        self.functionReducedProfiling = config.functionReducedProfiling
        self.sparseMatrixSizes = sparseMatrixSizes
        self.scaleForX = scaleForX

        for i in self.demotedVarsList:
            self.varsForBitwidth[i] = config.wordLength // 2

        if forFloat():
            self.independentVars = []
                # In floating point mode, this variable stores the variables whose bit-widths can be tweaked,
                # which can then be fed into the fixed point code generation.

        # Old SeeDot (PLDI'19) uses MAX_SCALE.
        # MAX_SCALE is used at each operation to compute scale parameters.
        # It is not used for floating-poing code generation.
        if self.ddsEnabled:
            self.MAX_SCALE = 1000
                # Large value which makes MAX_SCALE ineffective.
        elif getMaxScale() == None:
            if forFloat():
                getLogger().info(
                    "Setting MAX_SCALE = 0. This value will not affect the generated code.")
                self.MAX_SCALE = 0
            else:
                assert False, "MAX_SCALE not set for fixed-point code generation."
        else:
            self.MAX_SCALE = getMaxScale()

        # Variables used for exp() computation. Used in old SeeDot mode only.
        self.expTables = {}
        self.expProfileLoaded = False

        # Counter used while creating temp variables.
        self.counter_var = 0
        self.counter_iter = 0

        self.counter_inst = 0

        # List of variables declared in the SeeDot program whose definitions will be present in the model.h file.
        self.globalVars = []

        # Global variables
        # varDeclarations: Map of local variables (declared at the beginning) to their type used for declaring the variables in the generated C++ code.
        # varDeclarationsLocal: Same as varDeclarations, for variables which are declared locally within a For loop.
        # notScratch: Variables which would not be assigned in the scratch space.
        # varLiveIntervals: Map of variables to instructions across which the variable is used.
        # varScales: Map of variables to their scaling factors.
        # varIntervals: Map of variables to the range of values stored in the variable, which is obtained from range analysis.
        # internalVars: List of intermediate variables in the program whose type is always int irrespective of floating-point or fixed-point code generation.
        # intConstants: Map of integer constant variables to their value.
        # floatConstants: Map of float constant variables to their value.
        # curDepth: Depth of scope at current Instruction. Increases as we enter a loop, decreases as we exit a loop.
        # allDepths: Map from instruction ID to depth of scope.
        # biasShifts: For bias addition variables, the scaling shift operators are stored to help reduce number of scaling shift operators.
        # coLocatedVariables: Multiple variables used in one operation which need not be assigned separate memory locations.
        #                     eg. C = A + B -> C and A can occupy the same memory location.
        self.varDeclarations = {}
        self.varDeclarationsLocal = {}
        self.notScratch = ['X'] if not forM3() else []
        self.varLiveIntervals = {}
        self.varScales = {}
        self.varIntervals = {}
        self.internalVars = []
        self.intConstants = {}
        self.floatConstants = {}

        self.curDepth = 0
        self.allDepths = {}

        self.biasShifts = {}
        self.coLocatedVariables = {}

        # Used in old SeeDot.
        # Mutable variables declared in the 'loop' operator is stored in mutableVars.
        # The range of run-time values see by mutable variables is stored in mutableVarsProfile. This information is obtained from collecting run-time profile on the floating-point code.
        self.mutableVars = []
        self.mutableVarsProfile = []

    # Integers in the input code.
    def visitInt(self, node: AST.Int):
        val = node.value

        prog = IR.Prog([])
        expr = IR.Int(val)

        return (prog, expr)

    # Floating-point numbers in the input code.
    def visitFloat(self, node: AST.Float):
        val = node.value
        scale = self.getScale(abs(val))
        intv = self.getInterval(scale, val, val)
        val_int = IR.DataType.getInt(int(np.ldexp(val, -scale)))

        prog = IR.Prog([])
        expr = self.getTempVar()

        # Updating metadata.
        self.varDeclarations[expr.idf] = node.type
        self.varScales[expr.idf] = scale
        self.varIntervals[expr.idf] = intv
        self.intConstants[expr.idf] = val_int
        self.floatConstants[expr.idf] = val

        return (prog, expr)

    # Variables in the input code.
    def visitId(self, node: AST.ID):
        idf = node.name

        prog = IR.Prog([])

        expr = IR.Var(idf, inputVar=True if idf in self.globalVars else False)

        return (prog, expr)

    # Declaration for model parameters in the input code.
    def visitDecl(self, node: AST.Decl):
        minVal, maxVal = node.range

        assert minVal <= maxVal, "Range of a variable with values (%.6f, %.6f) is not valid" % (
            minVal, maxVal)

        # The range for model parameters is specified in the input code, which enables us to directly compute their scale.
        scale = self.getScale(max(abs(minVal), abs(maxVal)))
        intv = self.getInterval(scale, minVal, maxVal)

        prog = IR.Prog([])
        expr = self.getTempVar()
        expr.inputVar = True

        # Updating metadata.
        self.varScales[expr.idf] = scale
        self.varIntervals[expr.idf] = intv

        return (prog, expr)

    # Init is used for initializing mutable loop variables whose values are updated repeatedly.
    def visitInit(self, node: AST.Init):
        if node.value == 0:
            # getScale() fails for 0. Hence, replacing it with a very low value.
            minVal, maxVal = -0.000001, 0.000001
        else:
            minVal, maxVal = node.value, node.value

        expr = self.getTempVar()

        # Computing the scale of the variable, either using runtime profile data (new SeeDot OOPSLA '20) or using initial value (old SeeDot PLDI '19).
        if forFixed() and self.ddsEnabled:
            _, scale = self.getBitwidthAndScale(expr.idf)
        else:
            scale = self.getScale(max(abs(minVal), abs(maxVal)))
        intv = self.getInterval(scale, minVal, maxVal)

        comment = IR.Comment('init([%s], %.6f)' % (
            ', '.join(map(str, node.shape)), node.value), self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        # If the initial value is zero, memset is more efficient to set all values to zero.
        if node.value == 0:
            memset = IR.Memset(expr, node.type.size())
            prog_init = IR.Prog([comment, memset])
        # Using loops to initialize non-zero values instead of memset.
        else:
            iters_in = self.getTempIterators(len(node.shape))

            loopShape = []  # Contains the shape of the tensor being initialized.
            loopIters = []  # Iterators which will be used to iterate to each tensor element.

            for order in range(len(node.shape)):
                loopShape.append(node.shape[order])
                loopIters.append(iters_in[order])
                loop = IRUtil.loop(loopShape, loopIters, [
                IR.Assn(IRUtil.addIndex(expr, iters_in), 
                IR.Float(node.value) if forFloat() else self.getNumInFixedPoint(node.value, scale))
            ])

            prog_init = IR.Prog([comment] + loop)

        self.counter_inst += 1
        self.updateLiveRange(expr)

        prog_out = prog_init
        expr_out = expr

        # Updating metadata.
        self.varDeclarations[expr_out.idf] = node.type
        self.varScales[expr_out.idf] = scale
        self.varIntervals[expr_out.idf] = intv

        # Logging debug messages.
        self.log.print(comment.msg)
        self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

        return (prog_out, expr)

    # out = in ^ T
    def visitTransp(self, node: AST.Transp):
        (prog_in, expr_in) = self.visit(node.expr)

        expr_out = self.getTempVar()

        type_out = node.type
        [I, J] = type_out.shape

        # The input and output scale are same as the values of the input and output tensor are the same.
        bw_out, scale_out = self.getBitwidthAndScale(expr_in.idf)
        intv_out = self.varIntervals[expr_in.idf]

        expr_in.inputVar = False
        expr_out.inputVar = False

        comment = IR.Comment(expr_in.idf + "^T", self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        # If the input variable is demoted to lower bit-width, demote the output as well as no extra information can be stored in the extra bits.
        if forFixed():
            self.varsForBitwidth[expr_out.idf] = bw_out
            if bw_out != config.wordLength:
                self.demotedVarsList.append(expr_out.idf)
                self.demotedVarsOffsets[expr_out.idf] = self.getOffsetForDemotedVariable(expr_in.idf)

        funcCall = IR.FuncCall("Transpose", {
            expr_in: "A",
            expr_out: "B",
            IR.Int(I): "I",
            IR.Int(J): "J"
        }) if not self.vbwEnabled else IR.FuncCall("Transpose<int%d_t>" % (bw_out), {
            expr_in: "A",
            expr_out: "B",
            IR.Int(I): "I",
            IR.Int(J): "J"
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        prog_transp = IR.Prog([comment, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_transp)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = intv_out

        return (prog_out, expr_out)

    # out = in[a:+num][b:+num]...
    def visitSplice(self, node: AST.Splice):
        (prog_in, expr_in) = self.visit(node.expr)

        vars_in = []
        progs_in = []

        # Each indexing variable can be a complex expression so iterate through them all.
        for var in node.vars:
            part_prog_in, part_expr_in = self.visit(var)
            progs_in.append(part_prog_in)
            vars_in.append(part_expr_in)

        type_in = node.expr.type
        type_out = node.type

        # Keeping input and output scales same because the output tensor is a subtensor of the input, and generally the range of values remain the same in both.
        bw_out, scale_out = self.getBitwidthAndScale(expr_in.idf)

        expr_out = self.getTempVar()

        # If the input variable is demoted to lower bit-width, demote the output as well as no extra information can be stored in the extra bits.
        if forFixed():
            self.varsForBitwidth[expr_out.idf] = bw_out
            if self.varsForBitwidth[expr_out.idf] != config.wordLength:
                self.demotedVarsList.append(expr_out.idf)
                self.demotedVarsOffsets[expr_out.idf] = self.getOffsetForDemotedVariable(expr_in.idf)

        # Computing loop iterators for LHS and RHS.
        iters_in = self.getTempIterators(type_in.dim)
        iters_out = self.getTempVars(type_out.dim)

        loopShape = [] # Shape of the output tensor which will dictate the range of the iterators.
        loopIters = [] # Iterator which will iterate across different dimensions of the tensor.
        loopAssns = [] # Assignment carried out within one loop body.
        for order in range(type_in.dim):
            loopShape.append(node.sizes[order])
            loopIters.append(iters_in[order])
            loopAssns.append(IR.Assn(iters_out[order], IRUtil.add(iters_in[order], vars_in[order])))

        expr_out_idx = IRUtil.addIndex(expr_out, iters_in)
        expr_in_idx = IRUtil.addIndex(expr_in, iters_out)
        loop = IRUtil.loop(loopShape, loopIters, loopAssns + [
                IR.Assn(expr_out_idx, expr_in_idx)
            ])

        # Comment in the output code to show the input command for the corresponding output code.
        out_indices = ']['.join([i.idf for i in iters_in])
        in_indices = ']['.join([i.idf for i in iters_out])
        comment = IR.Comment("%s[%s] = %s[%s]"%(expr_out_idx.idf, out_indices, expr_in_idx.idf, in_indices), self.counter_inst+1)

        self.allDepths[self.counter_inst+1] = self.curDepth
        prog_splice = IR.Prog([comment] + loop)

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        # In case the target variable is contiguous, we can optimize (use memcpy instead of a loop).
        canOptimize = True
        loopShapeMustBeOne = False
        for i in range(len(loopShape) - 1, -1, -1):
            if loopShapeMustBeOne:
                if loopShape[i] != 1:
                    canOptimize = False
            else:
                if loopShape[i] == type_in.shape[i]:
                    continue
                elif loopShape[i] < type_in.shape[i]:
                    loopShapeMustBeOne = True
                    continue
                else:
                    assert False, "Illegal State, subtensor dimensions must be less than original tensor dimensions"
        canOptimize = canOptimize and (expr_in.idf not in self.globalVars)

        if canOptimize:
            prog_splice = IR.Prog([comment, IR.Memcpy(expr_out, expr_in, np.prod(loopShape), [IR.Int(0) for i in range(len(vars_in))], vars_in)])
        else:
            assert True

        # Concatenating the code for main expression and the indexing expressions.
        prog_out = IR.Prog([])
        prog_out = IRUtil.concatPrograms(prog_out, prog_in)
        for prog in progs_in:
            prog_out = IRUtil.concatPrograms(prog_out, prog)
        prog_out = IRUtil.concatPrograms(prog_out, prog_splice)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = (0,0)

        # Update declarations.
        for var in iters_out:
            self.varDeclarations[var.idf] = Type.Int()
            self.internalVars.append(var.idf)

        return (prog_out, expr_out)

    # out = reshape(in, shape, order)
    def visitReshape(self, node: AST.Reshape):
        (prog_in, expr_in) = self.visit(node.expr)

        '''
        reshape(A, (T1, T2), (N, H, W))

        cmd1:  t1 = t2 = 0;
        loop: for n in 0:N:
                 for h in 0:H:
                   for w in 0:W:
        cmd3:        B[t1][t2] = A[n][h][w]
        cmd5:        t2++;
                     if (t2 == T2)
                       t2 = 0;
        cmd5_:         t1++;
        '''

        type_in = node.expr.type
        type_out = node.type

        # Compute scaling factors.
        bw_out, scale_out = self.getBitwidthAndScale(expr_in.idf)
        intv_out = self.varIntervals[expr_in.idf]

        # Declare variables.
        expr_out = self.getTempVar()

        # If the input variable is demoted to lower bit-width, demote the output as well as no extra information can be stored in the extra bits.
        if forFixed():
            self.varsForBitwidth[expr_out.idf] = bw_out
            if self.varsForBitwidth[expr_out.idf] != config.wordLength:
                self.demotedVarsList.append(expr_out.idf)
                self.demotedVarsOffsets[expr_out.idf] = self.getOffsetForDemotedVariable(expr_in.idf)

        iters_in = self.getTempIterators(type_in.dim)
        iters_out = self.getTempVars(type_out.dim)

        # Initialize to 0.
        cmd1 = [IR.Assn(var, IRUtil.zero) for var in iters_out]

        # Incrementing the first index.
        first_iter = iters_out[0]
        cmd5_ = IRUtil.incCmd(first_iter)

        # Incrementing other indices using a loop.
        cmd5 = [cmd5_]
        for i in range(1, type_out.dim):
            curr_iter = iters_out[i]
            curr_size = IR.Int(type_out.shape[i])
            cmd5 = [IRUtil.incCmd(curr_iter), IR.If(IRUtil.eq(curr_iter, curr_size), [
                IRUtil.initVarToZero(curr_iter)] + cmd5)]

        # Outer loop.
        # The iterators are selected based on the selection order specified by the user.
        loopShape = []
        loopIters = []
        
        if node.order == None:
            node.order = [i+1 for i in range(type_in.dim)]

        for order in node.order:
            order = order - 1
            loopShape.append(type_in.shape[order])
            loopIters.append(iters_in[order])

        loop = IRUtil.loop(loopShape, loopIters, [IR.Assn(IRUtil.addIndex(
            expr_out, iters_out), IRUtil.addIndex(expr_in, iters_in))] + cmd5)

        # Finalize.
        comment = IR.Comment("reshape(" + expr_in.idf + ", (" + ', '.join(str(e)
            for e in type_out.shape) + "), (" + ', '.join(str(e) for e in node.order) + ")", self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        # In case the reshaped array's memory layout is identical to original array, we can optimize (use memcpy instead of loop).
        canOptimize = True
        for i in range(len(node.order)):
            if node.order[i] != i+1:
                canOptimize = False
        # The input variable 'X' is handled differently in M3 codegen.
        if not (forM3() and expr_in.idf == 'X'):
            canOptimize = canOptimize and expr_in.idf not in self.globalVars

        if canOptimize:
            prog_memcpy = IR.Memcpy(expr_out, expr_in, type_out.size(), [IR.Int(0) for i in range(type_out.dim)], [IR.Int(0) for i in range(type_in.dim)])
            prog_reshape = IR.Prog([comment] + [prog_memcpy])
        else:
            prog_reshape = IR.Prog([comment] + cmd1 + loop)

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        prog_out = IRUtil.concatPrograms(prog_in, prog_reshape)

        # Update context.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = intv_out

        # Update declarations.
        for var in iters_out:
            self.varDeclarations[var.idf] = Type.Int()
            self.internalVars.append(var.idf)

        return (prog_out, expr_out)

    # out = maxpool(in, stride)
    def visitMaxpool(self, node: AST.Maxpool):
        (prog_in, expr_in) = self.visit(node.expr)

        type_out = node.type
        kernelSize = node.kernelSize
        stride = node.stride
        padding = node.padding 

        # Declare variables.
        expr_out = self.getTempVar()

        # We use same scale for input and output as the output tensor has values sub-sampled from the input tensor, and the ranges of both input and output tensor generally remain the same.
        bw_in, scale_in = self.getBitwidthAndScale(expr_in.idf)
        bw_out, scale_out = bw_in, scale_in
        demote = 2 ** (scale_out - scale_in)

        assert demote == 1, "VBW not coded, Maxpool not profiled so this shouldnt happen (maxpool input and output scales must be same)"

        intv_out = self.varIntervals[expr_in.idf]

        [N, H, W, C] = node.expr.type.shape

        expr_in.inputVar = False
        expr_out.inputVar = False

        # If the input variable is demoted to lower bit-width, demote the output as well as no extra information can be stored in the extra bits.
        if bw_in != config.wordLength:
            self.demotedVarsList.append(expr_out.idf)
            self.demotedVarsOffsets[expr_out.idf] = 0
            self.varsForBitwidth[expr_out.idf] = config.wordLength // 2

        comment = IR.Comment(
            "maxpool(" + expr_in.idf + ", " + str(kernelSize) + ',' + str(padding) + ',' + str(stride) + ")", self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        funcCall = IR.FuncCall("Maxpool", {
            expr_in: "A",
            expr_out: "B",
            IR.Int(N): "N",
            IR.Int(H): "H",
            IR.Int(W): "W",
            IR.Int(C): "C",
            IR.Int(kernelSize[0]): "FH",
            IR.Int(kernelSize[1]): "FW",
            IR.Int(stride[0]): "stideH",
            IR.Int(stride[1]): "strideW",
            IR.Int(padding[0]): "HPADL",
            IR.Int(padding[1]): "HPADR",
            IR.Int(padding[2]): "WPADL",
            IR.Int(padding[3]): "WPADR"
        }) if not self.vbwEnabled else IR.FuncCall("Maxpool<int%d_t, int%d_t>"%(bw_in, bw_out), {
            expr_in: "A",
            expr_out: "B",
            IR.Int(N): "N",
            IR.Int(H): "H",
            IR.Int(W): "W",
            IR.Int(C): "C",
            IR.Int(kernelSize[0]): "FH",
            IR.Int(kernelSize[1]): "FW",
            IR.Int(stride[0]): "stideH",
            IR.Int(stride[1]): "strideW",
            IR.Int(padding[0]): "HPADL",
            IR.Int(padding[1]): "HPADR",
            IR.Int(padding[2]): "WPADL",
            IR.Int(padding[3]): "WPADR",
            IR.Int(demote): "demote"
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        prog_maxpool = IR.Prog([comment, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_maxpool)

        # Update declarations.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = intv_out

        return (prog_out, expr_out)

    # out = in[index]
    def visitIndex(self, node: AST.Index):

        (prog_in, expr_in) = self.visit(node.expr)
        (prog_idx, expr_idx) = self.visit(node.index)

        prog_out = IRUtil.concatPrograms(prog_in, prog_idx)
        expr_out = IRUtil.addIndex(expr_in, [expr_idx])

        return (prog_out, expr_out)

    # func(in_A, in_B, in_C, ... , in_n, out)
    # out.type = in_A.type = ... = in_n.type
    # out.scale = in_A.scale
    def visitFuncCall(self, node: AST.FuncCall):
        # The type of each argument is same and is equal to the type of the output.
        # The compiler assumes that the output of the uninterpreted function call is the last argument to the function.
        # Also assumes that the scale of the output is equal to the scale of the first argument.
        progs = []
        exprs = []
        for expr in node.exprList:
            (prog_in, expr_in) = self.visit(expr)
            progs.append(prog_in)
            exprs.append(expr_in)

        prog_out = IR.Prog([])
        for prog_funcCall in progs:
            prog_out = IRUtil.concatPrograms(prog_out, prog_funcCall)

        expr_out = self.getTempVar()

        # Scale of the output is the scale of the first argument.
        scale_out = self.varScales[exprs[0].idf]
        intv_out = self.varIntervals[exprs[0].idf]

        args = dict()
        ch = 'A'
        for expr in exprs:
            args[expr] = ch
            ch = chr(ord(ch) + 1) # Inputs would be labelled A, B, C, ... etc.
        args[expr_out] = expr_out.idf

        ch = 'I'
        for i in node.type.shape:
            args[IR.Int(i)] = ch
            ch = chr(ord(ch) + 1) # Indices would be named I, J, K, ... etc.

        comment = IR.Comment(
            node.name + '(' + ', '.join(expr.idf for expr in exprs) + ')', self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        funcCall = IR.FuncCall(node.name, args)

        prog_funcCall = IR.Prog([comment, funcCall])

        self.counter_inst += 1
        self.updateLiveRange([expr_in for expr_in in node.exprList] + [expr_out])

        prog_out = IRUtil.concatPrograms(prog_out, prog_funcCall)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = node.type
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = intv_out

        return (prog_out, expr_out)

    # B = reverse(A, axis=...)
    def visitReverse(self, node: AST.Reverse):
        (prog_in, expr_in) = self.visit(node.expr)

        prog_out = IR.Prog([])
        prog_out = IRUtil.concatPrograms(prog_out, prog_in)

        expr_out = self.getTempVar()

        # Scale of the output is the scale of the first argument as the values of output and first argument remain the same.
        intv_out = self.varIntervals[expr_in.idf]
        bitwidth_in, scale_in = self.getBitwidthAndScale(expr_in.idf)
        bw_out, scale_out = self.getBitwidthAndScale(expr_in.idf)

        args = dict()
        args[expr_in] = 'A'
        args[IR.Int(node.axis)] = 'axis'

        ch = 'I'
        for i in node.type.shape:
            args[IR.Int(i)] = ch
            ch = chr(ord(ch) + 1) # Indices will be labelled I, J, K, ... etc.

        args[expr_out] = 'B'

        comment = IR.Comment(
            "reverse" + '(' + expr_in.idf + ',' + str(node.axis) + ')', self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        funcCall = IR.FuncCall('Reverse' + str(len(node.type.shape)), args) if not self.vbwEnabled else IR.FuncCall('Reverse' + str(len(node.type.shape)) + '<int' + str(bitwidth_in) + '_t>', args)

        prog_funcCall = IR.Prog([comment, funcCall])

        # If the input variable is demoted to lower bit-width, demote the output as well as no extra information can be stored in the extra bits.
        if forFixed():
            self.varsForBitwidth[expr_out.idf] = bw_out
            if self.varsForBitwidth[expr_out.idf] != config.wordLength:
                self.demotedVarsList.append(expr_out.idf)
                self.demotedVarsOffsets[expr_out.idf] = self.getOffsetForDemotedVariable(expr_in.idf)

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        prog_out = IRUtil.concatPrograms(prog_out, prog_funcCall)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = node.type
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = intv_out

        return (prog_out, expr_out)

    # out = +- in
    def visitUop(self, node: AST.Uop):
        (prog_in, expr_in) = self.visit(node.expr)

        if node.op == SeeDotParser.ADD:
            return (prog_in, expr_in)

        assert node.op == SeeDotParser.SUB

        type_out = node.type
        
        self.allDepths[self.counter_inst+1] = self.curDepth

        # e : Int
        if Type.isInt(type_out):
            prog_out = prog_in
            expr_out = IRUtil.negate(expr_in)

            self.notScratch.append(expr_out.idf)

            # Just to be safe, check that the scaling factor of the integer variable is never tracked
            assert expr_in.idf not in self.varScales and expr_in.idf not in self.varIntervals
        # e: Tensor(), or Tensor(..)
        else:
            expr_out = self.getTempVar()
            iters = self.getTempIterators(type_out.dim)

            if type_out.isShapeOne():
                self.notScratch.append(expr_out.idf)

            bitwidth_out, scale_out = self.getBitwidthAndScale(expr_in.idf)

            # If the input variable is demoted to lower bit-width, demote the output as well as no extra information can be stored in the extra bits.
            if forFixed():
                self.varsForBitwidth[expr_out.idf] = bitwidth_out
                if bitwidth_out != config.wordLength:
                    self.demotedVarsList.append(expr_out.idf)
                    self.demotedVarsOffsets[expr_out.idf] = self.demotedVarsOffsets[expr_in.idf]

            (m, M) = self.varIntervals[expr_in.idf]
            intv_out = (-M, -m)

            lhs = IRUtil.addIndex(expr_out, iters)
            rhs = IRUtil.negate(IRUtil.addIndex(expr_in, iters))
            loop = IRUtil.loop(type_out.shape, iters, [IR.Assn(lhs, rhs)])
            prog_uop = IR.Prog(loop)

            prog_out = IRUtil.concatPrograms(prog_in, prog_uop)

            # Update metadata.
            self.varDeclarations[expr_out.idf] = type_out
            self.varScales[expr_out.idf] = scale_out
            self.varIntervals[expr_out.idf] = intv_out

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        return (prog_out, expr_out)

    # out = in_A op in_B
    def visitBop1(self, node: AST.Bop1):
        if node.op == SeeDotParser.MUL:
            return self.visitBopMul(node)
        elif node.op == SeeDotParser.SPARSEMUL:
            return self.visitBopSparseMul(node)
        elif node.op == SeeDotParser.MULCIR:
            return self.visitBopMulCir(node)
        elif node.op == SeeDotParser.ADDCIR:
            return self.visitBopAddOrSubCir(node)
        elif node.op == SeeDotParser.SUBCIR:
            return self.visitBopAddOrSubCir(node)
        else:
            assert False

    # out = in_A * in_B
    def visitBopMul(self, node: AST.Bop1):
        type_in_A = node.expr1.type
        type_in_B = node.expr2.type
        type_out = node.type

        if Type.isInt(type_out):
            return self.visitBopMulInt(node)
        elif type_in_A.dim == 0:
            return self.visitBopMul1DTensor(node)
        elif type_in_B.dim == 0:
            return self.visitBopMul1DTensor(node)
        else:
            return self.visitBopMul2DTensor(node)

    # out = in_A * in_B
    def visitBopMulInt(self, node: AST.Bop1):
        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B)
        expr_out = IRUtil.mul(expr_in_A, expr_in_B)

        # Just to be safe, check that the scaling factor of the integer variables is never tracked.
        if isinstance(expr_in_A, IR.Var):
            assert expr_in_A.idf not in self.varScales and expr_in_A.idf not in self.varIntervals
        if isinstance(expr_in_B, IR.Var):
            assert expr_in_B.idf not in self.varScales and expr_in_B.idf not in self.varIntervals

        return (prog_out, expr_out)

    # out = in_A * in_B
    def visitBopMul1DTensor(self, node: AST.Bop1):
        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        type_in_A, type_in_B = node.expr1.type, node.expr2.type
        type_out = node.type

        expr_out = self.getTempVar()

        # Read input scales and bit-widths.
        bitwidth_in_A, scale_in_A = self.getBitwidthAndScale(expr_in_A.idf)
        bitwidth_in_B, scale_in_B = self.getBitwidthAndScale(expr_in_B.idf)
        # Read output scales and bit-widths. In data-driven scaling, the output scale is directly profiled from floating-point runtime.
        # In static scaling used by old SeeDot (PLDI '19), output scale and bit-width is set to None is statically computed later.
        if self.ddsEnabled:
            bitwidth_out, scale_out = self.getBitwidthAndScale(expr_out.idf)
            bitwidth_temp, scale_temp = self.getBitwidthAndScale(expr_out.idf, native=True)
        else:
            bitwidth_out = config.wordLength // 2 if expr_out.idf in self.demotedVarsList else config.wordLength
            scale_out, scale_temp = None, None
            bitwidth_temp = bitwidth_out
        intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]

        # Compute scaling hyperparameters given input and output scales. If static scaling of old SeeDot is used, also compute the output scale and bit-width.
        shr_A, shr_B, H1, H2, demote, scale_out = self.getShrTreeSumAndDemoteParamsForMul(bitwidth_in_A, scale_in_A, bitwidth_in_B, scale_in_B, bitwidth_temp, scale_temp, bitwidth_out, scale_out, 1)

        intv_out = self.getIntvervalForMul(intv_in_A, shr_A, intv_in_B, shr_B)

        # Ensuring that in the generated code, the scalar is the first argument.
        if type_in_A.dim == 0:
            a, b = expr_in_A, expr_in_B
            bitwidth_in_A, bitwidth_in_B = bitwidth_in_A, bitwidth_in_B
            scale_in_A, scale_in_B = scale_in_A, scale_in_B
            [I, J] = type_in_B.shape
            shr_a, shr_b = shr_A, shr_B
        else:
            a, b = expr_in_B, expr_in_A
            bitwidth_in_A, bitwidth_in_B = bitwidth_in_B, bitwidth_in_A
            scale_in_A, scale_in_B = scale_in_B, scale_in_A
            [I, J] = type_in_A.shape
            shr_a, shr_b = shr_B, shr_A

        shr_a = self.formatShr(shr_a)
        shr_b = self.formatShr(shr_b)

        a.inputVar = False
        b.inputVar = False
        expr_out.inputVar = False

        comment = IR.Comment(expr_in_A.idf + ' * ' + expr_in_B.idf, self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth
        # Compute bit-width of intermediate variable.
        bitwidth_mul = self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "mul")
        funcCall = IR.FuncCall("ScalarMul", {
            a: "A",
            b: "B",
            expr_out: "C",
            IR.Int(I): "I",
            IR.Int(J): "J",
            shr_a: "shrA",
            shr_b: "shrB"
        }) if not self.vbwEnabled else IR.FuncCall("ScalarMul<int%d_t, int%d_t, int%d_t, int%d_t>"%(bitwidth_in_A, bitwidth_in_B, bitwidth_mul, bitwidth_out), {
            a: "A",
            b: "B",
            expr_out: "C",
            IR.Int(I): "I",
            IR.Int(J): "J",
            shr_a: "shrA",
            shr_b: "shrB",
            IR.Int(demote): "demote"
        })

        self.counter_inst += 1
        self.updateLiveRange([a, b, expr_out])

        profile = IR.FuncCall("Profile2", {
            expr_out: "Var",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.String(expr_out): "VarName"
        })
        if forFloat():
            self.independentVars.append(expr_out.idf)

        prog_mul = IR.Prog([comment, funcCall, profile] if forFloat() and self.ddsEnabled else [comment, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_mul)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = intv_out

        # Printing logs.
        self.log.print(comment.msg)
        self.log.print("\tInput1: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
        self.log.print("\tInput2: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
        self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

        return (prog_out, expr_out)

    # out = in_A * in_B
    def visitBopMul2DTensor(self, node: AST.Bop1):
        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        expr_treeSum = self.getTempVar()
        expr_out = self.getTempVar()

        # Read input scales and bit-widths.
        bitwidth_in_A, scale_in_A = self.getBitwidthAndScale(expr_in_A.idf)
        bitwidth_in_B, scale_in_B = self.getBitwidthAndScale(expr_in_B.idf)
        # Read output scales and bitwidths. In data-driven scaling, the output scale is directly profiled from floating-point runtime.
        # In static scaling used by old SeeDot (PLDI '19), output scale and bit-width is set to None is statically computed later.
        if self.ddsEnabled:
            bitwidth_out, scale_out = self.getBitwidthAndScale(expr_out.idf)
            bitwidth_temp, scale_temp = self.getBitwidthAndScale(expr_out.idf, native=True)
        else:
            bitwidth_out = config.wordLength // 2 if expr_out.idf in self.demotedVarsList else config.wordLength
            scale_out, scale_temp = None, None
            bitwidth_temp = bitwidth_out

        type_in_A, type_in_B = node.expr1.type, node.expr2.type
        type_out = node.type

        [I, J] = type_in_A.shape
        [J, K] = type_in_B.shape
        type_treeSum = Type.Tensor([J])

        intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]

        # Compute scaling hyperparameters given input and output scales. If static scaling of old SeeDot is used, also compute the output scale and bitwidth.
        shr_A, shr_B, H1, H2, demote, scale_out = self.getShrTreeSumAndDemoteParamsForMul(bitwidth_in_A, scale_in_A, bitwidth_in_B, scale_in_B, bitwidth_temp, scale_temp, bitwidth_out, scale_out, J)

        intv_out = (0,0)

        shr_A = self.formatShr(shr_A)
        shr_B = self.formatShr(shr_B)

        # If either of the input parameters are model parameters, change the function name which would read the model parameter differently on the target device (no difference in x86 mode).
        c = ''
        if expr_in_A.idf in self.globalVars:
            c += 'C'
        else:
            c += 'N'
        if expr_in_B.idf in self.globalVars:
            c += 'C'
        else:
            c += 'N'

        expr_in_A.inputVar = False
        expr_in_B.inputVar = False
        expr_out.inputVar = False
        expr_treeSum.inputVar = False

        if forFixed():
            self.varsForBitwidth[expr_treeSum.idf] = bitwidth_temp

        comment = IR.Comment(expr_in_A.idf + ' * ' + expr_in_B.idf, self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        # Bit-width for temporary variables.
        bitwidth_mul = self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "mul")
        if self.vbwEnabled:
            self.varsForBitwidth[expr_treeSum.idf] = bitwidth_mul

        # If one variable is already used as a sparse matrix, prevent further use as a dense matrix.
        assert expr_in_A.idf + "idx" not in self.sparseMatrixSizes.keys(), "Cannot use same matrix %s for both sparse and dense multiplication" % expr_in_A.idf

        funcCall = IR.FuncCall("MatMul" + c, {
            expr_in_A: "A",
            expr_in_B: "B",
            expr_out: "C",
            expr_treeSum: "T",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(K): "K",
            shr_A: "shrA",
            shr_B: "shrB",
            IR.Int(H1): "H1",
            IR.Int(H2): "H2"
        }) if not self.vbwEnabled else IR.FuncCall("MatMul" + c + ("<int%d_t, int%d_t, int%d_t, int%d_t>"%(bitwidth_in_A, bitwidth_in_B, bitwidth_mul, bitwidth_out)), {
            expr_in_A: "A",
            expr_in_B: "B",
            expr_out: "C",
            expr_treeSum: "T",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(K): "K",
            shr_A: "shrA",
            shr_B: "shrB",
            IR.Int(H1): "H1",
            IR.Int(H2): "H2",
            IR.Int(demote): "demote"
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in_A, expr_in_B, expr_out, expr_treeSum])

        # For floating point mode, profile the output for fixed-point scale computation.
        profile = IR.FuncCall("Profile2", {
            expr_out: "Var",
            IR.Int(I): "I",
            IR.Int(K): "J",
            IR.String(expr_out): "VarName"
        })
        if forFloat():
            self.independentVars.append(expr_out.idf)

        prog_mul = IR.Prog([comment, funcCall, profile] if forFloat() and self.ddsEnabled else [comment, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_mul)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = intv_out

        self.varDeclarations[expr_treeSum.idf] = type_treeSum
        self.varScales[expr_treeSum.idf] = scale_temp
        self.varIntervals[expr_treeSum.idf] = (0, 0)

        # Print logs.
        self.log.print(comment.msg)
        self.log.print("\tInput1: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
        self.log.print("\tInput2: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
        self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

        return (prog_out, expr_out)

    # out = in_A |*| in_B
    def visitBopSparseMul(self, node: AST.Bop1):
        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        [P, Q] = node.expr1.type.shape
        [Q, R] = node.expr2.type.shape

        assert R == 1, "Sparse matrix multiplication currently only support multiplication with a vector"

        expr_out = self.getTempVar()
        type_out = node.type

        # Reading input scales.
        bitwidth_in_A, scale_in_A = self.getBitwidthAndScale(expr_in_A.idf)
        bitwidth_in_Aid = (config.wordLength // 2) if (expr_in_A.idf + 'idx') in self.demotedVarsList else config.wordLength
        bitwidth_in_B, scale_in_B = self.getBitwidthAndScale(expr_in_B.idf)
        # Read output scales and bit-widths. In data-driven scaling, the output scale is directly profiled from floating-point runtime.
        # In static scaling used by old SeeDot (PLDI '19), output scale and bitwidth is set to None is statically computed later.
        if self.ddsEnabled:
            bitwidth_out, scale_out = self.getBitwidthAndScale(expr_out.idf)
            bitwidth_temp, scale_temp = self.getBitwidthAndScale(expr_out.idf, native=True)
        else:
            bitwidth_out = config.wordLength // 2 if expr_out.idf in self.demotedVarsList else config.wordLength
            scale_out, scale_temp = None, None
            bitwidth_temp = bitwidth_out

        intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]

        # Compute scaling hyperparameters given input and output scales. If static scaling of old SeeDot is used, also compute the output scale and bit-width.
        shr_A, shr_B, H1, H2, demote, scale_out = self.getShrTreeSumAndDemoteParamsForMul(bitwidth_in_A, scale_in_A, bitwidth_in_B, scale_in_B, bitwidth_temp, scale_temp, bitwidth_out, scale_out, Q)
        intv_out = (0, 0)

        in_A_idx = IR.Var(expr_in_A.idf +
                          'idx', expr_in_A.idx, inputVar=True)
        in_A_val = IR.Var(expr_in_A.idf +
                          'val', expr_in_A.idx, inputVar=True)

        shr_A = self.formatShr(shr_A)
        shr_B = self.formatShr(shr_B)
        height_shr = self.formatShr(H1)

        in_A_idx.inputVar = False
        in_A_val.inputVar = False
        expr_in_B.inputVar = False
        expr_out.inputVar = False

        comment = IR.Comment(expr_in_A.idf + ' |*| ' + expr_in_B.idf, self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        # The output variable needs to be set to zero as the matrix multiplication implementation assumes this.
        cmd1 = IR.Memset(expr_out, type_out.size())
        bitwidth_mul = self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "mul")

        # For input variable 'X', the data is streamed on the target device, which necessitates a different function implementation.
        if expr_in_B.idf == 'X':
            funcName = "SparseMatMulX"
        else:
            funcName = "SparseMatMul"

        funcCall = IR.FuncCall(funcName, {
            in_A_idx: "Aidx",
            in_A_val: "Aval",
            expr_in_B: "B",
            expr_out: "C",
            IR.Int(Q): "K",
            shr_A: "shrA",
            shr_B: "shrB",
            height_shr: "shrC"
        }) if not self.vbwEnabled else IR.FuncCall("%s<int%d_t, int%d_t, int%d_t, int%d_t, int%d_t>"%(funcName, bitwidth_in_A, bitwidth_in_Aid, bitwidth_in_B, bitwidth_mul, bitwidth_out), {
            in_A_idx: "Aidx",
            in_A_val: "Aval",
            expr_in_B: "B",
            expr_out: "C",
            IR.Int(Q): "K",
            shr_A: "shrA",
            shr_B: "shrB",
            height_shr: "shrC",
            IR.Int(demote): "demote"
        })

        self.counter_inst += 1
        self.updateLiveRange([in_A_idx, in_A_val, expr_in_B, expr_out])

        # Profiling the floating-point output for scale computation for the fixed-point code (only used if ddsEnabled is True).
        profile = IR.FuncCall("Profile2", {
                                expr_out: "Var",
                                IR.Int(P): "I",
                                IR.Int(R): "J",
                                IR.String(expr_out): "VarName"
                                })
        if forFloat():
            self.independentVars.append(expr_out.idf)

        prog_mul = IR.Prog([comment, cmd1, funcCall, profile] if forFloat() and self.ddsEnabled else [comment, cmd1, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_mul)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = intv_out

        # Update metadata for sparse matrix.
        self.varDeclarations.update({in_A_idx.idf: Type.Tensor([self.sparseMatrixSizes[expr_in_A.idf + 'idx']]),
                                     in_A_val.idf: Type.Tensor([self.sparseMatrixSizes[expr_in_A.idf + 'val']]),
                                     })

        # Include sparse matrices in global variables.
        if in_A_idx.idf not in self.globalVars:
            self.globalVars.append(in_A_idx.idf)
        if in_A_val.idf not in self.globalVars:
            self.globalVars.append(in_A_val.idf)

        # Print log.
        self.log.print(comment.msg)
        self.log.print("\tInput1: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
        self.log.print("\tInput2: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
        self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

        return (prog_out, expr_out)

    # out = in_A <*> in_B
    def visitBopMulCir(self, node: AST.Bop1):
        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        type_out = node.type

        expr_out = self.getTempVar()

        assert type_out.dim == 2

        [I, J] = type_out.shape

        # Read input scales.
        bitwidth_in_A, scale_in_A = self.getBitwidthAndScale(expr_in_A.idf)
        bitwidth_in_B, scale_in_B = self.getBitwidthAndScale(expr_in_B.idf)
        # Read output scales and bit-widths. In data-driven scaling, the output scale is directly profiled from floating-point runtime.
        # In static scaling used by old SeeDot (PLDI '19), output scale and bit-width is set to None is statically computed later.
        if self.ddsEnabled:
            bitwidth_out, scale_out = self.getBitwidthAndScale(expr_out.idf)
            bitwidth_temp, scale_temp = self.getBitwidthAndScale(expr_out.idf, native=True)
        else:
            bitwidth_out = config.wordLength // 2 if expr_out.idf in self.demotedVarsList else config.wordLength
            scale_out, scale_temp = None, None
            bitwidth_temp = bitwidth_out

        intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]

        # Compute scaling hyperparameters given input and output scales. If static scaling of old SeeDot is used, also compute the output scale and bit-width.
        shr_A, shr_B, H1, H2, demote, scale_raw = self.getShrTreeSumAndDemoteParamsForMul(bitwidth_in_A, scale_in_A, bitwidth_in_B, scale_in_B, bitwidth_temp, scale_temp, bitwidth_out, scale_out, 1)

        # The theoretical output scale in scale_raw might be different than profiled scale scale_out.
        # We perform a scale adjustment in this case for correctness.
        # TODO: Introduce a post-processing pass to merge consecutive scale adjustments hence generated.
        adjust = []
        if self.ddsEnabled:
            if scale_raw != scale_out:
                diff = 2 ** abs(scale_raw - scale_out)
                if scale_raw > scale_out:
                    adjust = [IR.FuncCall("AdjustScaleShl" + (("<int%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
                                expr_out: "A",
                                IR.Int(I): "I",
                                IR.Int(J): "J",
                                IR.Int(diff): "scale"
                            })]
                else:
                    adjust = [IR.FuncCall("AdjustScaleShr" + (("<int%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
                                expr_out: "A",
                                IR.Int(I): "I",
                                IR.Int(J): "J",
                                IR.Int(diff): "scale"
                            })]
        else:
            scale_out = scale_raw

        intv_out = self.getIntvervalForMul(intv_in_A, shr_A, intv_in_B, shr_B)

        shr_A = self.formatShr(shr_A)
        shr_B = self.formatShr(shr_B)

        expr_in_A.inputVar = False
        expr_in_B.inputVar = False
        expr_out.inputVar = False

        comment = IR.Comment(expr_in_A.idf + ' <*> ' + expr_in_B.idf, self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        bitwidth_mul = self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "mul")
        funcCall = IR.FuncCall("MulCir", {
            expr_in_A: "A",
            expr_in_B: "B",
            expr_out: "C",
            IR.Int(I): "I",
            IR.Int(J): "J",
            shr_A: "shrA",
            shr_B: "shrB"
        }) if not self.vbwEnabled else IR.FuncCall("MulCir<int%d_t, int%d_t, int%d_t, int%d_t>"%(bitwidth_in_A, bitwidth_in_B, bitwidth_mul, bitwidth_out), {
            expr_in_A: "A",
            expr_in_B: "B",
            expr_out: "C",
            IR.Int(I): "I",
            IR.Int(J): "J",
            shr_A: "shrA",
            shr_B: "shrB",
            IR.Int(demote): "demote"
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in_A, expr_in_B, expr_out])

        profile = IR.FuncCall("Profile2", {
            expr_out: "Var",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.String(expr_out): "VarName"
        })
        if forFloat():
            self.independentVars.append(expr_out.idf)

        prog_mul = IR.Prog([comment, funcCall, profile] if forFloat() and self.ddsEnabled else [comment, funcCall] + adjust)

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_mul)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = intv_out

        # Print logs.
        self.log.print(comment.msg)
        self.log.print("\tInput1: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
        self.log.print("\tInput2: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
        self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

        return (prog_out, expr_out)

    # out = mbconv(A, filters, weights, biases, <params>)
    # This is a specialised implementation of mobilenet conv layers which prevent excessive memory bloat during intermediate computations.
    def visitMbconv(self, node: AST.MBConv):
        if not (config.ddsEnabled and config.vbwEnabled):
            assert False, "MBConv is currently only supported if VBW and DDS modes are switched on"

        assert forX86() or forM3(), "MBConv not implemented for Arduino devices"

        # Process all inputs for MBConv.
        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_F1, expr_in_F1) = self.visit(node.exprF1)
        (prog_in_W1, expr_in_W1) = self.visit(node.exprW1)
        (prog_in_B1, expr_in_B1) = self.visit(node.exprB1)

        (prog_in_F2, expr_in_F2) = self.visit(node.exprF2)
        (prog_in_W2, expr_in_W2) = self.visit(node.exprW2)
        (prog_in_B2, expr_in_B2) = self.visit(node.exprB2)

        (prog_in_F3, expr_in_F3) = self.visit(node.exprF3)
        (prog_in_W3, expr_in_W3) = self.visit(node.exprW3)
        (prog_in_B3, expr_in_B3) = self.visit(node.exprB3)

        [expr_treeSum, expr_out] = self.getTempVars(2)
        [expr_bufX, expr_bufT] = self.getTempVars(2)

        [N, H, W, Cin] = node.expr1.type.shape
        [_, _, _, _, Ct] = node.exprF1.type.shape
        [_, Hf, Wf, _, _] = node.exprF2.type.shape
        [_, _, _, _, Cout] = node.exprF3.type.shape

        type_treeSum = Type.Tensor([np.max((Hf * Wf, Ct, Cin))])
        type_out = node.type
        type_bufX = Type.Tensor([Hf, W, Ct])
        type_bufT = Type.Tensor([Ct])

        # Process bit-width and scales for all inputs.
        bitwidth_in_A, scale_in_A = self.getBitwidthAndScale(expr_in_A.idf)

        bitwidth_in_F1, scale_in_F1 = self.getBitwidthAndScale(expr_in_F1.idf)
        bitwidth_in_W1, scale_in_W1 = self.getBitwidthAndScale(expr_in_W1.idf)
        bitwidth_in_B1, scale_in_B1 = self.getBitwidthAndScale(expr_in_B1.idf)

        bitwidth_in_F2, scale_in_F2 = self.getBitwidthAndScale(expr_in_F2.idf)
        bitwidth_in_W2, scale_in_W2 = self.getBitwidthAndScale(expr_in_W2.idf)
        bitwidth_in_B2, scale_in_B2 = self.getBitwidthAndScale(expr_in_B2.idf)

        bitwidth_in_F3, scale_in_F3 = self.getBitwidthAndScale(expr_in_F3.idf)
        bitwidth_in_W3, scale_in_W3 = self.getBitwidthAndScale(expr_in_W3.idf)
        bitwidth_in_B3, scale_in_B3 = self.getBitwidthAndScale(expr_in_B3.idf)

        bitwidth_out, scale_out = self.getBitwidthAndScale(expr_out.idf)

        shr = [0 for i in range(9)]
        shl = [0 for i in range(9)]

        # Compute intermediate scales and scaling factors for all operations which are included in MBConv.
        if not forFloat():
            # Stage 1 Step 1: Multiplication
            bitwidth_u1 = bitwidth_in_A + bitwidth_in_F1 - 1
            bitwidth_u1_code = self.getTempBitwidth(bitwidth_in_A, bitwidth_in_F1, "mul")
            scale_u1 = scale_in_A + scale_in_F1

            # Stage 1 Step 2: Tree Sum
            d1 = int(np.ceil(np.log2(Cin)))
            scale_u1 = scale_u1 + d1

            # Stage 1 Step 3: Batch Normalisation and ReLU6
            bitwidth_add1 = np.max((bitwidth_in_A, bitwidth_in_F1))
            bitwidth_reduction = config.wordLength - bitwidth_add1
            _, scale_add1 = self.getBitwidthAndScale(expr_out.idf + "t1") + bitwidth_reduction 
            shr[0] = (scale_add1 - scale_u1)
            shr[1] = (scale_add1 - scale_in_B1)
            bitwidth_mul1 = bitwidth_add1 + bitwidth_in_W1 - 1
            bitwidth_mul1_code = self.getTempBitwidth(bitwidth_add1, bitwidth_in_W1, "mul")
            scale_mul1 = scale_add1 + scale_in_W1
            six1 = 6 * (2 ** -scale_mul1)
            bitwidth_x = np.max((bitwidth_add1, bitwidth_in_W1))
            scale_x = -bitwidth_x + 1 + int(np.floor(np.log2(6)) + 1)
            scale_shift = scale_x - scale_mul1
            shr[2] = scale_shift

            # Stage 2 Step 4: Multiplication
            bitwidth_u2 = bitwidth_x + bitwidth_in_F2 - 1
            bitwidth_u2_code = self.getTempBitwidth(bitwidth_x, bitwidth_in_F2, "mul")
            scale_u2 = scale_x + scale_in_F2

            # Stage 2 Step 5: Tree Sum
            d2 = int(np.ceil(np.log2(Hf * Wf)))
            scale_u2 = scale_u2 + d2

            # Stage 2 Step 6: Batch Normalisation and ReLU6
            bitwidth_add2 = np.max((bitwidth_x, bitwidth_in_F2))
            bitwidth_reduction = config.wordLength - bitwidth_add2
            _, scale_add2 = self.getBitwidthAndScale(expr_out.idf + "t3") + bitwidth_reduction 
            shr[3] = (scale_add2 - scale_u2)
            shr[4] = (scale_add2 - scale_in_B2)
            bitwidth_mul2 = bitwidth_add2 + bitwidth_in_W2 - 1
            bitwidth_mul2_code = self.getTempBitwidth(bitwidth_add2, bitwidth_in_W2, "mul")
            scale_mul2 = scale_add2 + scale_in_W2
            six2 = 6 * (2 ** -scale_mul2)
            bitwidth_t = np.max((bitwidth_add2, bitwidth_in_W2))
            scale_t = -bitwidth_t + 1 + int(np.floor(np.log2(6)) + 1)
            scale_shift = scale_t - scale_mul2
            shr[5] = scale_shift

            # Stage 3 Step 7: Multiplication
            bitwidth_u3 = bitwidth_t + bitwidth_in_F3 - 1
            bitwidth_u3_code = self.getTempBitwidth(bitwidth_t, bitwidth_in_F3, "mul")
            scale_u3 = scale_t + scale_in_F3

            # Stage 3 Step 8: Tree Sum
            d3 = int(np.ceil(np.log2(Ct)))
            scale_u3 = scale_u3 + d3

            # Stage 3 Step 9: Batch Normalisation
            bitwidth_add3 = np.max((bitwidth_t, bitwidth_in_F3))
            bitwidth_reduction = config.wordLength - bitwidth_add3
            _, scale_add3 = self.getBitwidthAndScale(expr_out.idf + "t5") + bitwidth_reduction 
            shr[6] = (scale_add3 - scale_u3)
            shr[7] = (scale_add3 - scale_in_B3)
            bitwidth_mul3 = bitwidth_add3 + bitwidth_in_W3 - 1
            bitwidth_mul3_code = self.getTempBitwidth(bitwidth_add3, bitwidth_in_W3, "mul")
            scale_mul3 = scale_add3 + scale_in_W3
            scale_reduction = scale_out - scale_mul3
            shr[8] = scale_reduction

            for i in range(9):
                shl[i] = -shr[i]
        else:
            d1 = int(np.ceil(np.log2(Cin)))
            d2 = int(np.ceil(np.log2(Hf * Wf)))
            d3 = int(np.ceil(np.log2(Ct)))
            # In floating-point mode, none of the following values matter. Setting them to dummy values.
            for i in range(9):
                shr[i] = 1
                shl[i] = 1
            bitwidth_u = bitwidth_t = bitwidth_x = config.wordLength
            bitwidth_u1_code = bitwidth_u2_code = bitwidth_u3_code = config.wordLength
            six1 = six2 = 6.0
            scale_x = scale_t = 0

        for i in range(9):
            if shr[i] >= 0:
                shr[i] = self.formatShr(shr[i], saturate=False)
                shl[i] = self.formatShr(0)
            else:
                shr[i] = self.formatShr(0)
                shl[i] = self.formatShr(shl[i], saturate=False)

        expr_in_A.inputVar = False
        expr_in_F1.inputVar = False
        expr_in_W1.inputVar = False
        expr_in_B1.inputVar = False
        expr_in_F2.inputVar = False
        expr_in_W2.inputVar = False
        expr_in_B2.inputVar = False
        expr_in_F3.inputVar = False
        expr_in_W3.inputVar = False
        expr_in_B3.inputVar = False
        expr_out.inputVar = False
        expr_treeSum.inputVar = False
        expr_bufT.inputVar = False
        expr_bufX.inputVar = False

        bitwidth_u = np.max((bitwidth_u1_code, bitwidth_u2_code, bitwidth_u3_code))

        # Setting metadata.
        if forFixed():
            self.varsForBitwidth[expr_treeSum.idf] = bitwidth_u
            self.varsForBitwidth[expr_bufT.idf] = bitwidth_t
            self.varsForBitwidth[expr_bufX.idf] = bitwidth_x

        comment = IR.Comment('MBconv(%s)' %(expr_in_A.idf), self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        argMap = {
            expr_in_A: "A",
            expr_in_F1: "F1",
            expr_in_W1: "BN1W",
            expr_in_B1: "BN1B",
            expr_in_F2: "F2",
            expr_in_W2: "BN2W",
            expr_in_B2: "BN2B",
            expr_in_F3: "F3",
            expr_in_W3: "BN3W",
            expr_in_B3: "BN3B",
            expr_out: "C",
            expr_bufX: "X",
            expr_bufT: "T",
            expr_treeSum: "U",
            IR.Int(N): "N",
            IR.Int(H): "H",
            IR.Int(W): "W",
            IR.Int(Cin): "Cin",
            IR.Int(Ct): "Ct",
            IR.Int(Hf): "HF",
            IR.Int(Wf): "WF",
            IR.Int(Cout): "Cout",
            IR.Int(type_out.shape[1]): "Hout",
            IR.Int(type_out.shape[2]): "Wout",
            IR.Int(node.padding[0]): "HPADL",
            IR.Int(node.padding[1]): "HPADR",
            IR.Int(node.padding[2]): "WPADL",
            IR.Int(node.padding[3]): "WPADR",
            IR.Int(node.stride[0]): "HSTR",
            IR.Int(node.stride[1]): "WSTR",
            IR.Int(d1): "D1",
            IR.Int(d2): "D2",
            IR.Int(d3): "D3",
            IR.Int(six1): "SIX_1",
            IR.Int(six2): "SIX_2",
        }

        for i in range(9):
            argMap[shr[i]] = "shr%d" % (i+1)

        for i in range(9):
            argMap[shl[i]] = "shl%d" % (i+1)

        # These are used to optimise the m3 codegen to club multiple scale modification operators into one for faster code.
        self.biasShifts[expr_in_B1.idf] = int(np.log2(shr[1].n)) - int(np.log2(shl[1].n))
        self.biasShifts[expr_in_B2.idf] = int(np.log2(shr[4].n)) - int(np.log2(shl[4].n))
        self.biasShifts[expr_in_B3.idf] = int(np.log2(shr[7].n)) - int(np.log2(shl[7].n))

        if forFloat():
            argMap[IR.String(expr_out)] = "name"

        # Generating the argument map which is used in the codegen.
        localVarMap = {expr_treeSum.idf: type_treeSum, expr_bufX.idf: type_bufX, expr_bufT.idf: type_bufT}
        if forFloat():
            funcCall = IR.FuncCall("MBConv", argMap) 
        else:
            templateArgs = ("<int%s_t" + (", int%s_t" * 16) + ">") % (bitwidth_in_A, bitwidth_in_F1, bitwidth_in_W1, bitwidth_in_B1, bitwidth_in_F2, bitwidth_in_W2, bitwidth_in_B2, bitwidth_in_F3, bitwidth_in_W3, bitwidth_in_B3, bitwidth_out, bitwidth_x, bitwidth_t, bitwidth_u, bitwidth_mul1_code, bitwidth_mul2_code, bitwidth_mul3_code)
            funcCall = IR.FuncCall("MBConv" + templateArgs, argMap)

        self.counter_inst += 1
        self.updateLiveRange([expr_in_A, expr_in_F1, expr_in_F2, expr_in_F3, expr_in_W1, expr_in_W2, expr_in_W3, expr_in_B1, expr_in_B2, expr_in_B3, expr_out, expr_treeSum, expr_bufX, expr_bufT])

        # Profiling the output variable in floating-point mode for computing the scale of the fixed-point code.
        profile = IR.FuncCall("Profile4", {
            expr_out: "Var",
            IR.Int(N): "I",
            IR.Int(type_out.shape[1]): "J",
            IR.Int(type_out.shape[2]): "K",
            IR.Int(Cout): "L",
            IR.String(expr_out): "VarName"
        })
        if forFloat():
            self.independentVars.append(expr_out.idf)

        prog_mbconv = IR.Prog([comment, funcCall, profile] if forFloat() and self.ddsEnabled else [comment, funcCall])
        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_F1, prog_in_W1, prog_in_B1, prog_in_F2, prog_in_W2, prog_in_B2, prog_in_F3, prog_in_W3, prog_in_B3, prog_mbconv)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_out
        self.varDeclarations[expr_treeSum.idf] = type_treeSum
        self.varDeclarations[expr_bufX.idf] = type_bufX
        self.varDeclarations[expr_bufT.idf] = type_bufT

        self.varScales[expr_out.idf] = scale_out
        self.varScales[expr_treeSum.idf] = 0 # It changes across three stages above and an exact value not required outside of this method.
        self.varScales[expr_bufX.idf] = scale_x
        self.varScales[expr_bufT.idf] = scale_t

        # Intervals not needed necesarily for the compiler to run, updating this variable for being compatible with old SeeDot (PLDI '19).
        self.varIntervals[expr_out.idf] = (0, 0)
        self.varIntervals[expr_treeSum.idf] = (0, 0)
        self.varIntervals[expr_bufX.idf] = (0, 0)
        self.varIntervals[expr_bufT.idf] = (0, 0)

        # Printing log.
        self.log.print(comment.msg)
        self.log.print("\tInput1: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
        self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

        return (prog_out, expr_out)

    # out = conv(A, B, <params>)
    def visitConvolution(self, node: AST.Convolution):
        (prog_in_A, expr_in_A) = self.visit(node.expr1)
        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        [expr_treeSum, expr_out] = self.getTempVars(2)

        [N, H, W, Cin] = node.expr1.type.shape
        [G, Hf, Wf, CinF, CoutF] = node.expr2.type.shape

        type_treeSum = Type.Tensor([Hf * Wf * CinF])
        type_out = node.type

        # Read input scales.
        bitwidth_in_A, scale_in_A = self.getBitwidthAndScale(expr_in_A.idf)
        bitwidth_in_B, scale_in_B = self.getBitwidthAndScale(expr_in_B.idf)
        # Read output scales.
        if self.ddsEnabled:
            bitwidth_out, scale_out = self.getBitwidthAndScale(expr_out.idf)
            bitwidth_temp, scale_temp = self.getBitwidthAndScale(expr_out.idf, native=True)
        else:
            bitwidth_out = config.wordLength // 2 if expr_out.idf in self.demotedVarsList else config.wordLength
            scale_out, scale_temp = None, None
            bitwidth_temp = bitwidth_out

        intv_in_A, intv_in_B = (0, 0), (0, 0)
        intv_out = (0, 0) 

        # Compute scaling hyperparameters given input and output scales. If static scaling of old SeeDot is used, also compute the output scale and bit-width.
        shr_A, shr_B, H1, H2, demote, scale_out = self.getShrTreeSumAndDemoteParamsForMul(bitwidth_in_A, scale_in_A, bitwidth_in_B, scale_in_B, bitwidth_temp, scale_temp, bitwidth_out, scale_out, Hf * Wf * CinF)

        shr_A = self.formatShr(shr_A)
        shr_B = self.formatShr(shr_B)

        expr_in_A.inputVar = False
        expr_in_B.inputVar = False
        expr_out.inputVar = False
        expr_treeSum.inputVar = False

        if forFixed():
            self.varsForBitwidth[expr_treeSum.idf] = bitwidth_temp

        comment = IR.Comment('conv(%s, %s)' %(expr_in_A.idf,expr_in_B.idf), self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        # Compute bit-width of intermediate variables.
        bitwidth_mul = self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "mul")
        if self.vbwEnabled:
            self.varsForBitwidth[expr_treeSum.idf] = bitwidth_mul

        argMap = {
            expr_in_A: "A",
            expr_in_B: "B",
            expr_out: "C",
            expr_treeSum: "tmp",
            IR.Int(N): "N",
            IR.Int(H): "H",
            IR.Int(W): "W",
            IR.Int(Cin): "CIN",
            IR.Int(Hf): "HF",
            IR.Int(Wf): "WF",
            IR.Int(CinF): "CINF",
            IR.Int(CoutF): "COUTF",
            IR.Int(type_out.shape[1]): "HOUT",
            IR.Int(type_out.shape[2]): "WOUT",
            IR.Int(node.padding[0]): "HPADL",
            IR.Int(node.padding[1]): "HPADR",
            IR.Int(node.padding[2]): "WPADL",
            IR.Int(node.padding[3]): "WPADR",
            IR.Int(node.stride[0]): "HSTR",
            IR.Int(node.stride[1]): "WSTR",
            IR.Int(node.dilation[0]): "HDL",
            IR.Int(node.dilation[1]): "WDL",
            IR.Int(G): "G",
            shr_A: "shrA",
            shr_B: "shrB",
            IR.Int(H1): "H1",
            IR.Int(H2): "H2"
        }
        if self.vbwEnabled:
            argMap[IR.Int(demote)] = "demote"

        if not self.vbwEnabled:
            funcCall = IR.FuncCall("Convolution", argMap)
        else:
            funcCall = IR.FuncCall("Convolution" + ("<int%d_t, int%d_t, int%d_t, int%d_t>"%(bitwidth_in_A, bitwidth_in_B, bitwidth_mul, bitwidth_out)), argMap) #, {expr_treeSum.idf: type_treeSum})

        self.counter_inst += 1
        self.updateLiveRange([expr_in_A, expr_in_B, expr_out, expr_treeSum])

        if Hf == Wf == CinF == CoutF == 1 and bitwidth_in_A == bitwidth_out:
            self.setMemorySharableVariables(expr_in_A, expr_out)

        # Profile the output in floating-point mode to compute data-driven scale for fixed-point mode.
        profile = IR.FuncCall("Profile4", {
            expr_out: "Var",
            IR.Int(N): "I",
            IR.Int(type_out.shape[1]): "J",
            IR.Int(type_out.shape[2]): "K",
            IR.Int(CoutF * G): "L",
            IR.String(expr_out): "VarName"
        })
        if forFloat():
            self.independentVars.append(expr_out.idf)

        prog_conv = IR.Prog([comment, funcCall, profile] if forFloat() and self.ddsEnabled else [comment, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_conv)

        # Update context for output variable.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = intv_out

        self.varDeclarations[expr_treeSum.idf] = type_treeSum
        self.varScales[expr_treeSum.idf] = scale_temp
        self.varIntervals[expr_treeSum.idf] = (0, 0)

        # Print log.
        self.log.print(comment.msg)
        self.log.print("\tInput1: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
        self.log.print("\tInput2: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
        self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

        return (prog_out, expr_out)

    # out = in_A <+-> in_B
    def visitBopAddOrSubCir(self, node: AST.Bop1):
        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        type_out = node.type

        if node.op == SeeDotParser.ADDCIR:
            (op_ir, op_fn) = (IR.Op.Op['+'], operator.add)
            add = True
        elif node.op == SeeDotParser.SUBCIR:
            (op_ir, op_fn) = (IR.Op.Op['-'], operator.sub)
            add = False

        assert op_fn == operator.add, "Compiler currently does not support convolution-like subtraction."

        expr_out = self.getTempVar()

        # Read input scales and bit-widths.
        bitwidth_in_A, scale_in_A = self.getBitwidthAndScale(expr_in_A.idf)
        bitwidth_in_B, scale_in_B = self.getBitwidthAndScale(expr_in_B.idf)
        # Read output scales and bit-widths if in data-driven scaling mode.
        if self.ddsEnabled:
            bitwidth_out, scale_out = self.getBitwidthAndScale(expr_out.idf)
        else:
            bitwidth_out = config.wordLength // 2 if expr_out.idf in self.demotedVarsList else config.wordLength
            scale_out = None

        intv_in_A, intv_in_B = (0, 0), (0, 0)

        # Compute scaling hyperparameters given input and output scales. If static scaling of old SeeDot is used, also compute the output scale and bit-width.
        (scale_out_unadjusted, intv_out, [shr_A, shr_B, shr_out]) = self.getScaleForAddAndSub(scale_in_A, scale_in_B, scale_out, op_fn)
        if scale_out is None:
            scale_out = scale_out_unadjusted

        demoteLog = shr_out - 8 if shr_out >= 8 else 0
        shr_out = min(shr_out, 8)
        irdemote = self.formatShr(demoteLog)

        shr_A = self.formatShr(shr_A)
        shr_B = self.formatShr(shr_B)
        shr_out = self.formatShr(shr_out)

        expr_in_A.inputVar = False
        expr_in_B.inputVar = False
        expr_out.inputVar = False

        comment = IR.Comment(expr_in_A.idf + " <" +
                             op_ir.name + "> " + expr_in_B.idf, self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        # Generate function call for the output code depending on whether the input is 2-D or 4-D.
        if type_out.dim == 4:
            [N, H, W, C] = type_out.shape
            funcCall = IR.FuncCall("AddOrSubCir4D", {
                expr_in_A: "A",
                expr_in_B: "B",
                expr_out: "X",
                IR.Int(N): "N",
                IR.Int(H): "H",
                IR.Int(W): "W",
                IR.Int(C): "C",
                shr_A: "shrA",
                shr_B: "shrB",
                shr_out: "shrC",
                IR.Bool(add): "add"
            }) if not self.vbwEnabled else IR.FuncCall("AddOrSubCir4D" + ("<int%d_t, int%d_t, int%d_t, int%d_t>" % (bitwidth_in_A, bitwidth_in_B, self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "add", bitwidth_out), bitwidth_out)), {
                expr_in_A: "A",
                expr_in_B: "B",
                expr_out: "X",
                IR.Int(N): "N",
                IR.Int(H): "H",
                IR.Int(W): "W",
                IR.Int(C): "C",
                shr_A: "shrA",
                shr_B: "shrB",
                shr_out: "shrC",
                IR.Bool(add): "add",
                irdemote: "demote"
            })
            profile = IR.FuncCall("Profile4", {
                expr_out: "Var",
                IR.Int(N): "I",
                IR.Int(H): "J",
                IR.Int(W): "K",
                IR.Int(C): "L",
                IR.String(expr_out): "VarName"
            })
        elif type_out.dim == 2:
            [H, W] = type_out.shape
            funcCall = IR.FuncCall("AddOrSubCir2D", {
                expr_in_A: "A",
                expr_in_B: "B",
                expr_out: "X",
                IR.Int(H): "H",
                IR.Int(W): "W",
                shr_A: "shrA",
                shr_B: "shrB",
                shr_out: "shrC",
                IR.Bool(add): "add"
            }) if not self.vbwEnabled else IR.FuncCall("AddOrSubCir2D" + ("<int%d_t, int%d_t, int%d_t, int%d_t>" % (bitwidth_in_A, bitwidth_in_B, self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "add", bitwidth_out), bitwidth_out)), {
                expr_in_A: "A",
                expr_in_B: "B",
                expr_out: "X",
                IR.Int(H): "H",
                IR.Int(W): "W",
                shr_A: "shrA",
                shr_B: "shrB",
                shr_out: "shrC",
                IR.Bool(add): "add",
                irdemote: "demote"
            })
            profile = IR.FuncCall("Profile2", {
                expr_out: "Var",
                IR.Int(H): "I",
                IR.Int(W): "J",
                IR.String(expr_out): "VarName"
            })
        else:
            assert False, "AddCir only supports 2D and 4D tensors."

        if forFloat():
            self.independentVars.append(expr_out.idf)

        self.counter_inst += 1
        self.updateLiveRange([expr_in_A, expr_in_B, expr_out])

        if bitwidth_in_A == bitwidth_out:
            self.setMemorySharableVariables(expr_in_A, expr_out)

        # The theoretical output scale in scale_raw might be different than profiled scale scale_out.
        # We perform a scale adjustment in this case for correctness.
        # TODO: Introduce a post-processing pass to merge consecutive scale adjustments hence generated.
        adjust = []
        if forFixed():
            if scale_out_unadjusted != scale_out:
                if scale_out_unadjusted > scale_out:
                    diff_scale = 2 ** (scale_out_unadjusted - scale_out)
                    adjust = [IR.FuncCall("AdjustScaleShl" + (("<int%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
                                            expr_out: "A",
                                            IR.Int(N): "I",
                                            IR.Int(H): "J",
                                            IR.Int(W): "K",
                                            IR.Int(C): "L",
                                            IR.Int(diff_scale): "scale"
                                        } if type_out.dim == 4 else {
                                            expr_out: "A",
                                            IR.Int(H): "I",
                                            IR.Int(W): "J",
                                            IR.Int(diff_scale): "scale"
                                        })]
                elif scale_out_unadjusted < scale_out:
                    diff_scale = 2 ** (scale_out - scale_out_unadjusted)
                    adjust = [IR.FuncCall("AdjustScaleShr" + (("<int%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
                                         expr_out: "A",
                                            IR.Int(N): "I",
                                            IR.Int(H): "J",
                                            IR.Int(W): "K",
                                            IR.Int(C): "L",
                                            IR.Int(diff_scale): "scale"
                                        } if type_out.dim == 4 else {
                                            expr_out: "A",
                                            IR.Int(H): "I",
                                            IR.Int(W): "J",
                                            IR.Int(diff_scale): "scale"
                                        })]

        prog_cir = IR.Prog([comment, funcCall, profile] if forFloat() and self.ddsEnabled else [comment, funcCall] + adjust)

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_cir)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = intv_out

        # Print log.
        self.log.print(comment.msg)
        self.log.print("\tInput1: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
        self.log.print("\tInput2: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
        self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

        return (prog_out, expr_out)

    # out = in_A 'op' in_B
    def visitBop2(self, node: AST.Bop2):
        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        type_out = node.type

        if node.op == SeeDotParser.ADD:
            (op_ir, op_fn) = (IR.Op.Op['+'], operator.add)
            funcName = "MatAdd"
        elif node.op == SeeDotParser.SUB:
            (op_ir, op_fn) = (IR.Op.Op['-'], operator.sub)
            funcName = "MatSub"

        # e : Int
        if Type.isInt(type_out):
            prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B)
            expr_out = IR.IntBop(expr_in_A, op_ir, expr_in_B)

            # Just to be safe that the scaling factor of the integer variable is never tracked.
            if isinstance(expr_in_A, IR.Var):
                assert expr_in_A.idf not in self.varScales and expr_in_A.idf not in self.varIntervals
            if isinstance(expr_in_B, IR.Var):
                assert expr_in_B.idf not in self.varScales and expr_in_B.idf not in self.varIntervals
        # e : Tensor(), or Tensor(..)
        else:
            assert type_out.dim == 2 or (type_out.dim == 4 and config.vbwEnabled), "Addition/subtraction of tensors is currently only supported for 2D tensors. Addition for 4D tensors is supported when VBW is enabled"

            type_A = node.expr1.type
            type_B = node.expr2.type

            assert (not type_out.dim == 4) or (type_A.dim == type_B.dim and expr_in_A.idf not in self.globalVars and expr_in_B.idf not in self.globalVars and node.op == SeeDotParser.ADD), "For 4D operation, no broadcasting supported, inputs should not be model parameters, and operation cannot be subtraction"

            # Depending on whether one of the inputs is a model parameter, change the function name so that the model parameter is read differently in the arduino codegen. No difference in case of x86 code.
            c = ''
            if op_fn == operator.add:
                if expr_in_A.idf in self.globalVars:
                    c += 'C'
                else:
                    c += 'N'
                if expr_in_B.idf in self.globalVars:
                    c += 'C'
                else:
                    c += 'N'

            # If one of the inputs is a scalar, the operator will broadcast that input.
            if type_A.dim == 0:
                funcName += 'BroadCastA'
                c = ''
            elif type_B.dim == 0:
                funcName += 'BroadCastB'
                c = ''

            expr_out = self.getTempVar()

            # Read input scale.
            bitwidth_in_A, scale_in_A = self.getBitwidthAndScale(expr_in_A.idf)
            bitwidth_in_B, scale_in_B = self.getBitwidthAndScale(expr_in_B.idf)
            # Read output scale.
            if self.ddsEnabled:
                bitwidth_out, scale_out = self.getBitwidthAndScale(expr_out.idf)
            else:
                bitwidth_out = config.wordLength // 2 if expr_out.idf in self.demotedVarsList else config.wordLength
                scale_out = None

            # Compute scaling hyperparameters given input and output scales. If static scaling of old SeeDot is used, also compute the output scale and bit-width.
            (scale_out_unadjusted, intv_out, [shr_A, shr_B, shr_out]) = self.getScaleForAddAndSub(scale_in_A, scale_in_B, scale_out, op_fn)
            if scale_out is None:
                    scale_out = scale_out_unadjusted

            intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]

            demoteLog = shr_out - 8 if shr_out >= 8 else 0
            shr_out = min(shr_out, 8)
            irdemote = self.formatShr(demoteLog)

            if type_out.dim == 2:
                [I, J] = type_out.shape
            elif type_out.dim == 4:
                [N, H, W, C] = type_out.shape
            else:
                assert False, "Unsupported dimension for addition"

            shr_A = self.formatShr(shr_A)
            shr_B = self.formatShr(shr_B)
            shr_out = self.formatShr(shr_out)

            expr_in_A.inputVar = False
            expr_in_B.inputVar = False
            expr_out.inputVar = False

            comment = IR.Comment(expr_in_A.idf + ' ' +
                                 op_ir.name + ' ' + expr_in_B.idf, self.counter_inst+1)
            self.allDepths[self.counter_inst+1] = self.curDepth

            # Generate output function call depending on dimensionality of the input / output.
            if type_out.dim == 2:
                funcCall = IR.FuncCall(funcName + c, {
                    expr_in_A: "A",
                    expr_in_B: "B",
                    expr_out: "C",
                    IR.Int(I): "I",
                    IR.Int(J): "J",
                    shr_A: "shrA",
                    shr_B: "shrB",
                    shr_out: "shrC"
                }) if not self.vbwEnabled else IR.FuncCall(funcName + c + ("<int%d_t, int%d_t, int%d_t, int%d_t>" % (bitwidth_in_A, bitwidth_in_B, self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "add", bitwidth_out), bitwidth_out)), {
                    expr_in_A: "A",
                    expr_in_B: "B",
                    expr_out: "C",
                    IR.Int(I): "I",
                    IR.Int(J): "J",
                    shr_A: "shrA",
                    shr_B: "shrB",
                    shr_out: "shrC",
                    irdemote: "demote"
                })
            elif type_out.dim == 4:
                funcCall = IR.FuncCall(funcName + "4", {
                    expr_in_A: "A",
                    expr_in_B: "B",
                    expr_out: "X",
                    IR.Int(N): "N",
                    IR.Int(H): "H",
                    IR.Int(W): "W",
                    IR.Int(C): "C",
                    shr_A: "shrA",
                    shr_B: "shrB",
                    shr_out: "shrC",
                }) if not self.vbwEnabled else IR.FuncCall(funcName + "4" + ("<int%d_t, int%d_t, int%d_t, int%d_t>" % (bitwidth_in_A, bitwidth_in_B, self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "add", bitwidth_out), bitwidth_out)), {
                    expr_in_A: "A",
                    expr_in_B: "B",
                    expr_out: "X",
                    IR.Int(N): "N",
                    IR.Int(H): "H",
                    IR.Int(W): "W",
                    IR.Int(C): "C",
                    shr_A: "shrA",
                    shr_B: "shrB",
                    shr_out: "shrC",
                    irdemote: "demote"
                })

            self.counter_inst += 1
            self.updateLiveRange([expr_in_A, expr_in_B, expr_out])

            if type_out.dim == 4:
                if expr_in_A.idf not in self.globalVars and bitwidth_in_A == bitwidth_out:
                    self.setMemorySharableVariables(expr_in_A, expr_out)
                elif expr_in_B.idf not in self.globalVars and bitwidth_in_B == bitwidth_out:
                    self.setMemorySharableVariables(expr_in_B, expr_out)

            # Profile the output variable in the floating point version to obtain fixed-point scale.
            if type_out.dim == 2:
                profile = IR.FuncCall("Profile2", {
                                expr_out: "Var",
                                IR.Int(I): "I",
                                IR.Int(J): "J",
                                IR.String(expr_out): "VarName"
                                })
            elif type_out.dim == 4:
                profile = IR.FuncCall("Profile4", {
                                expr_out: "Var",
                                IR.Int(N): "N",
                                IR.Int(H): "H",
                                IR.Int(W): "W",
                                IR.Int(C): "C",
                                IR.String(expr_out): "VarName"
                                })
            else:
                assert False, "Illegal number of dimensions"

            if forFloat():
                self.independentVars.append(expr_out.idf)

            # The theoretical output scale in scale_raw might be different than profiled scale scale_out.
            # We perform a scale adjustment in this case for correctness.
            # TODO: Introduce a post-processing pass to merge consecutive scale adjustments hence generated.
            if type_out.dim == 2:
                adjust = []
                if forFixed():
                    if scale_out_unadjusted != scale_out:
                        if scale_out_unadjusted > scale_out:
                            diff_scale = 2 ** (scale_out_unadjusted - scale_out)
                            adjust = [IR.FuncCall("AdjustScaleShl" + (("<int%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
                                                expr_out: "A",
                                                IR.Int(I): "I",
                                                IR.Int(J): "J",
                                                IR.Int(diff_scale): "scale"
                                                })]
                        elif scale_out_unadjusted < scale_out:
                            diff_scale = 2 ** (scale_out - scale_out_unadjusted)
                            adjust = [IR.FuncCall("AdjustScaleShr" + (("<int%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
                                                expr_out: "A",
                                                IR.Int(I): "I",
                                                IR.Int(J): "J",
                                                IR.Int(diff_scale): "scale"
                                                })]
            elif type_out.dim == 4:
                adjust = []
                if forFixed():
                    if scale_out_unadjusted != scale_out:
                        if scale_out_unadjusted > scale_out:
                            diff_scale = 2 ** (scale_out_unadjusted - scale_out)
                            adjust = [IR.FuncCall("AdjustScaleShl" + (("<int%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
                                                expr_out: "A",
                                                IR.Int(N): "N",
                                                IR.Int(H): "H",
                                                IR.Int(W): "W",
                                                IR.Int(C): "C",
                                                IR.Int(diff_scale): "scale"
                                                })]
                        elif scale_out_unadjusted < scale_out:
                            diff_scale = 2 ** (scale_out - scale_out_unadjusted)
                            adjust = [IR.FuncCall("AdjustScaleShr" + (("<int%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
                                                expr_out: "A",
                                                IR.Int(N): "N",
                                                IR.Int(H): "H",
                                                IR.Int(W): "W",
                                                IR.Int(C): "C",
                                                IR.Int(diff_scale): "scale"
                                                })]
            else:
                assert False, "Illegal number of dimensions"

            prog_bop = IR.Prog([comment, funcCall, profile] if forFloat() and self.ddsEnabled else [comment, funcCall] + adjust)

            prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_bop)

            # Updating metadata.
            self.varDeclarations[expr_out.idf] = type_out
            self.varScales[expr_out.idf] = scale_out
            self.varIntervals[expr_out.idf] = intv_out

            # Print log.
            self.log.print(comment.msg)
            self.log.print("\tInput1: scale = %d, interval = [%d, %d]" % (
                (self.varScales[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
            self.log.print("\tInput2: scale = %d, interval = [%d, %d]" % (
                (self.varScales[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
            self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % (
                (self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

        return (prog_out, expr_out)

    # out = func(in)
    def visitFunc(self, node: AST.Func):
        if node.op == SeeDotParser.RELU:
            return self.visitRelu(node)
        if node.op == SeeDotParser.RELU6:
            return self.visitRelu6(node)
        elif node.op == SeeDotParser.EXP:
            return self.visitExp(node)
        elif node.op == SeeDotParser.ARGMAX:
            return self.visitArgMax(node)
        elif node.op == SeeDotParser.SGN:
            return self.visitSgn(node)
        elif node.op == SeeDotParser.TANH:
            if useNewTableExp() and forFixed():
                return self.visitNewTableTanH(node)
            else:
                return self.visitTanh(node)
        elif node.op == SeeDotParser.SIGMOID:
            if useNewTableExp() and forFixed():
                return self.visitNewTableSigmoid(node)
            else:
                return self.visitSigmoid(node)
        elif node.op == SeeDotParser.NORMALISEL2:
            return self.visitNormaliseL2(node)
        else:
            assert False

    def visitNormaliseL2(self, node:AST.Func):
        (prog_in, expr_in) = self.visit(node.expr)
        intv_out = (0, 0)

        expr_out = self.getTempVar()
        bw_in, scale_in = self.getBitwidthAndScale(expr_in.idf)

        bw_out = bw_in

        # Calculating scale.
        # We downscale by (bw_in//2) while taking calculating square.
        sum_square_scale = 2*(scale_in + bw_in//2) 

        # y in binary search converges to 1/sqrt(sum_square).
        # We choose y to have bitwidth of (bw_in/2) and scale of (bw_in/2 -1).

        # For the binary search this is the scale of y * y * sum_square.
        # cmp_val_scale = sum_square_scale + bw_in - 2

        # scale_out = scale_in + bw_in/2 + y_scale
        # since we assume y_scale = (bw_in/2 - 1)
        scale_out = scale_in + 1 

        # We propagate the demotion of bit-width.
        if forFixed() and bw_out != config.wordLength:
            self.demotedVarsList.append(expr_out.idf)
            self.demotedVarsOffsets[expr_out.idf] = 0
        self.varsForBitwidth[expr_out.idf] = bw_out

        expr_in.inputVar = False

        comment = IR.Comment("normaliseL2(" + expr_in.idf + ")", self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        # Since NormaliseL2 does not get profiled now. We do not demote the output.
        if node.type.dim == 4:
            [N, H, W, C] = node.type.shape
            funcCall = IR.FuncCall("NormaliseL2", {
                expr_in: "A",
                expr_out: "B",
                IR.Int(N): "N",
                IR.Int(H): "H",
                IR.Int(W): "W",
                IR.Int(C): "C",
                IR.Int(scale_in): "scaleA",
                IR.Int(bw_in/2): "shrA"
            }) if not self.vbwEnabled else IR.FuncCall("NormaliseL2<int%d_t>"%(bw_in), {
                expr_in: "A",
                expr_out: "B",
                IR.Int(N): "N",
                IR.Int(H): "H",
                IR.Int(W): "W",
                IR.Int(C): "C",
                IR.Int(scale_in): "scaleA",
                IR.Int(bw_in/2): "shrA"
            })
        else:
            assert False, "inverseL2Norm only supports 4D tensors."

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        self.setMemorySharableVariables(expr_in, expr_out)

        prog_func = IR.Prog([comment, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_func)

        self.varDeclarations[expr_out.idf] = node.type
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = (0, 0)

        return (prog_out, expr_out)

    # out = relu(in)
    def visitRelu(self, node: AST.Func):
        (prog_in, expr_in) = self.visit(node.expr)
        intv_out = (0, 0)

        scale_in = self.varScales[expr_in.idf]

        expr_in.inputVar = False

        comment = IR.Comment("relu(" + expr_in.idf + ")", self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        if node.type.dim == 4:
            [N, H, W, C] = node.type.shape
            funcCall = IR.FuncCall("Relu4D", {
                expr_in: "A",
                IR.Int(N): "N",
                IR.Int(H): "H",
                IR.Int(W): "W",
                IR.Int(C): "C"
            })
        elif node.type.dim == 2:
            [H, W] = node.type.shape
            funcCall = IR.FuncCall("Relu2D", {
                expr_in: "A",
                IR.Int(H): "H",
                IR.Int(W): "W"
            })
        else:
            assert False, "Relu operator currently only supports 2D and 4D tensors."

        self.counter_inst += 1
        self.updateLiveRange([expr_in])

        prog_relu = IR.Prog([comment, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_relu)

        self.varIntervals[expr_in.idf] = intv_out

        return (prog_out, expr_in)

    # out = relu(in)
    def visitRelu6(self, node: AST.Func):
        (prog_in, expr_in) = self.visit(node.expr)
        intv_out = (0, 0)

        type_in = node.expr.type

        expr_out = self.getTempVar()

        # Read input scale and bit-width. Output scale and bit-width are the same as input.
        bitwidth_in, scale_in = self.getBitwidthAndScale(expr_in.idf)
        bitwidth_out, scale_out = bitwidth_in, scale_in
        
        # If input variable is demoted to 8 bits, demote the output variable to 8 bits too.
        tmp_var = expr_in.idf
        while tmp_var in self.substitutions.keys():
            tmp_var = self.substitutions[tmp_var]
        if tmp_var in self.demotedVarsList:
            self.demotedVarsList.append(expr_out.idf)
            self.demotedVarsOffsets[expr_out.idf] = 0
            self.varsForBitwidth[expr_out.idf] = config.wordLength // 2

        # Scaling hyperparameters in the fixed-point code.
        divide = 2 ** (scale_out - scale_in)
        cap = 6 * (2 ** -scale_in)

        expr_in.inputVar = False

        comment = IR.Comment("relu6(" + expr_in.idf + ")", self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        assert node.type.dim == 4, "Relu6 only implemented for 4 dimensional tensors"
        [N, H, W, C] = node.type.shape
        funcCall = IR.FuncCall("Relu6", {
            expr_in: "A",
            expr_out: "B",
            IR.Int(N): "N",
            IR.Int(H): "H",
            IR.Int(W): "W",
            IR.Int(C): "C",
            IR.Int(cap): "six",
            IR.Int(divide): "div"
        }) if not self.vbwEnabled else IR.FuncCall("Relu6<int%d_t, int%d_t>" % (bitwidth_in, bitwidth_out), {
            expr_in: "A",
            expr_out: "B",
            IR.Int(N): "N",
            IR.Int(H): "H",
            IR.Int(W): "W",
            IR.Int(C): "C",
            IR.Int(cap): "six",
            IR.Int(divide): "div"
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        prog_relu = IR.Prog([comment, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_relu)

        # Update metadata.
        self.varIntervals[expr_out.idf] = intv_out
        self.varDeclarations[expr_out.idf] = type_in
        self.varScales[expr_out.idf] = scale_out

        return (prog_out, expr_out)

    # out = exp(in)
    def visitExp(self, node: AST.Func):
        if forFloat() or useMathExp():
            return self.visitMathExp(node)
        elif useTableExp():
            self.readExpProfileFile()
            return self.visitTableExp(node)
        elif useNewTableExp():
            return self.visitNewTableExp(node)
        else:
            assert False

    def visitNewTableExp(self, node: AST.Func):
        # For a theoretical explation, please refer to OOPSLA '20 paper Section 5.4.
        assert self.vbwEnabled, "VBW must be enabled for new table"

        (prog_in, expr_in) = self.visit(node.expr)

        type_in = node.expr.type

        # Raw input scale, which can be any value.
        bitwidth_in_raw, scale_in_raw = self.getBitwidthAndScale(expr_in.idf)

        MIN = 0.1
        maxExp = np.exp(MIN)

        expr_out = self.getTempVar()

        # Output bit-width.
        if self.ddsEnabled:
            bitwidth_out, _ = self.getBitwidthAndScale(expr_out.idf)
        else:
            bitwidth_out = config.wordLength // 2 if expr_out.idf in self.demotedVarsList else config.wordLength
        
        # Output scale, refer to the paper for a detailed explanation.
        # The output scale is fixed depending on only the output bit-width.
        scale_out = self.getScale(maxExp) + config.wordLength // 2 if expr_out.idf in self.demotedVarsList else self.getScale(maxExp)
        # Adjusting the input scale given the bit-width, refer to the OOPSLA '20 paper for explanation.
        # The input scale if fixed depending on only the input bit-width.
        scale_in_adjusted = self.getScale(maxExp) + config.wordLength // 2 if expr_in.idf in self.demotedVarsList else self.getScale(maxExp)

        [I, J] = type_in.shape

        scale_in = -4 if expr_in.idf in self.demotedVarsList else -11

        # Scaling hyperparameters.
        scaling = 1
        if expr_in.idf in self.demotedVarsList and expr_out.idf not in self.demotedVarsList:
            scaling = 256
        elif expr_in.idf not in self.demotedVarsList and expr_out.idf in self.demotedVarsList:
            scaling = 256

        # Refer to OOPSLA '20 paper Section 5.4 which explains the computation of input and output scales for TanH and Sigmoid and Exp.
        # The input scales are adjusted to the theoretically known value for different bit-widths.
        adjust = []
        if scale_in_raw != scale_in:
            diff_scale = abs(scale_in_raw - scale_in)
            if scale_in_raw > scale_in:
                saturate = (2 ** (-scale_in_adjusted - diff_scale))
                adjust = [IR.FuncCall("AdjustScaleShlSaturate<int%d_t>" %(bitwidth_in_raw), {
                                            expr_in : "A",
                                            IR.Int(I): "I",
                                            IR.Int(J): "J",
                                            IR.Int(2 ** diff_scale): "scale",
                                            IR.Int(saturate): "saturate"
                })]
            else:
                adjust = [IR.FuncCall("AdjustScaleShr<int%d_t>" %(bitwidth_in_raw), {
                                            expr_in : "A",
                                            IR.Int(I): "I",
                                            IR.Int(J): "J",
                                            IR.Int(2 ** diff_scale): "scale"
                })]

        comm = IR.Comment('exp(' + expr_in.idf + ')', self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        # TODO: Check if saturation/overflows are handled.
        funcCall = IR.FuncCall("ExpNew%d<int%d_t>" %(bitwidth_in_raw, bitwidth_out), {
            expr_in: "A",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(scaling): "adjust",
            expr_out: "B"
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        prog_exp = IR.Prog([comm] + adjust + [funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_exp)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_in
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = (0, 0)

        return (prog_out, expr_out)

    # Used in floating point mode always for profiling. May or may not be used in the fixed-point mode.
    # Note: We assume e<=0 for exp(e).
    def visitMathExp(self, node: AST.Func):
        # Used in the old SeeDot (PLDI '19) version.
        # Tunable parameter.
        MIN = 0.1

        (prog_in, expr_in) = self.visit(node.expr)

        type_in = node.expr.type

        # Reading input scale and bit-width.
        bitwidth_in, scale_in = self.getBitwidthAndScale(expr_in.idf)

        '''
        1.  y = ((int) (exp(((float)e) / shr1) * shr2))
        '''

        maxExp = np.exp(MIN)

        expr_out = self.getTempVar()

        # Reading / Computing output bit-width.
        if self.ddsEnabled:
            bitwidth_out, _ = self.getBitwidthAndScale(expr_out.idf)
        else:
            bitwidth_out = config.wordLength // 2 if expr_out.idf in self.demotedVarsList else config.wordLength
        # Computing output scale.
        scale_out = self.getScale(maxExp) + config.wordLength // 2 if expr_out.idf in self.demotedVarsList else self.getScale(maxExp)

        intv_out = self.getInterval(scale_out, maxExp, maxExp)

        [I, J] = type_in.shape

        # Scaling hyperparameters.
        shr1 = 2 ** -scale_in
        shr2 = 2 ** -scale_out

        shr1 = self.formatShr(shr1, "shr")
        shr2 = self.formatShr(shr2, "shr")

        cmd0 = IR.Comment('exp(' + expr_in.idf + ')', self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        funcCall = IR.FuncCall("Exp", {
            expr_in: "A",
            IR.Int(I): "I",
            IR.Int(J): "J",
            shr1: "shrA",
            shr2: "shrB",
            expr_out: "B"
        }) if not self.vbwEnabled else IR.FuncCall("Exp<int%d_t, int%d_t>"%(bitwidth_in, bitwidth_out), {
            expr_in: "A",
            IR.Int(I): "I",
            IR.Int(J): "J",
            shr1: "shrA",
            shr2: "shrB",
            expr_out: "B",
            IR.Int(1): "demote"
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        # This method is used in the profiling floating point stage to check whether the input values are beyond a threshold.
        # Input values beyond a threshold are always mapped to zero in fixed-point code, hence these datapoints hold little use in the fixed-point mode.
        rangeCheck = IR.FuncCall("checkRange2", {
            expr_in: "A",
            IR.Int(I): "I",
            IR.Int(J): "J"
        })  if self.functionReducedProfiling and forFloat() else IR.Comment("Recommend switching on Function Reduced Profiling for sound output")

        profile = IR.FuncCall("Profile2", {
            expr_out: "Var",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.String(expr_out): "VarName"
        })
        if forFloat():
            self.independentVars.append(expr_out.idf)

        prog_exp = IR.Prog([cmd0, rangeCheck, funcCall, profile] if forFloat() and self.ddsEnabled else [cmd0, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_exp)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_in
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = intv_out

        return (prog_out, expr_out)

    # Used by old SeeDot (PLDI '19).
    # Note: We assume e<=0 for exp(e).
    def visitTableExp(self, node: AST.Func):
        assert False, "Must redesign IMHO"

        (prog_in, expr_in) = self.visit(node.expr)

        # TODO: Use MAX_VAL_EXP.
        type_in = node.expr.type

        scale_in = self.varScales[expr_in.idf]

        [m, M] = self.expRange
        [m_scale, M_scale] = [
            int(np.ldexp(m, -scale_in)), int(np.ldexp(M, -scale_in))]

        max = int(np.ldexp(M - m, -scale_in))
        shl = self.getShl(max)

        input = self.getTempVar()
        [i, j] = self.getTempVars(2)
        expr_out = self.getTempVar()

        '''
        1.  if ((-x) < min) {
        2.      i = 0;
        3.      j = 0;
        4.  }
        5.  else {
        6.      y = ((-x) - min) << shl
        7.      i = (y >> shrI) & (2^b-1)
        8.      j = (y >> shrJ) & (2^b-1)
        9.  }
        10. ans = T[i] * U[j]
        '''

        mask = IR.Int(2 ** self.expB - 1)
        shrI = config.wordLength - self.expB
        shrJ = config.wordLength - self.expB * 2
        table = self.getExpTable(scale_in)

        scale1 = self.getScale(1)
        scale2 = self.getScale(abs(np.exp(-m)))

        [shr1, shr2] = self.getShrForMul(scale1, scale2)

        expr_1_elt = IRUtil.addIndex(expr_in, [IRUtil.zero] * type_in.dim)
        expr_2_elt = IRUtil.addIndex(expr_out, [IRUtil.zero] * type_in.dim)

        cond = IRUtil.lt(IRUtil.negate(expr_1_elt), IR.Int(m_scale))

        cmd2 = IR.Assn(i, IR.Int(0))
        cmd3 = IR.Assn(j, IR.Int(0))

        cmd6 = IR.Assn(input, IRUtil.shl(IRUtil.sub(
            IRUtil.negate(expr_1_elt), IR.Int(m_scale)), shl))
        cmd7 = IR.Assn(i, IRUtil.bitAnd(IRUtil.shrUint(input, shrI), mask))
        cmd8 = IR.Assn(j, IRUtil.bitAnd(IRUtil.shrUint(input, shrJ), mask))

        cmd1 = IR.If(cond, [cmd2, cmd3], [cmd6, cmd7, cmd8])
        cmd10 = IR.Assn(expr_2_elt, IRUtil.mul(IRUtil.shrUint(IRUtil.addIndex(
            table[0], [i]), shr1), IRUtil.shrUint(IRUtil.addIndex(table[1], [j]), shr2)))

        scale_out = self.getScaleForExp(scale1, shr1, scale2, shr2)
        intv_out = self.getIntervalForExp(scale_out, [-m_scale, -M_scale])

        cmd0 = IR.Comment('exp(' + expr_in.idf + ')')

        prog_exp = IR.Prog([cmd0, cmd1, cmd10])

        prog_out = IRUtil.concatPrograms(prog_in, prog_exp)

        self.varDeclarations[expr_out.idf] = type_in
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = intv_out

        self.varDeclarations.update(
            dict((var.idf, Type.Int()) for var in [input, i, j]))

        return (prog_out, expr_out)

    def getShl(self, n: int):
        assert n != 0

        shl = 0
        while(n != 0):
            n = n >> 1
            shl += 1
        return min(config.wordLength - shl, config.wordLength - self.expB * 2)

    # Used by old SeeDot.
    def getExpTable(self, p):
        table = self.expTables.get(p)
        if table == None:
            table = self.populateExpTable(p)
            self.expTables[p] = table

        return table[1]

    # Used by old SeeDot.
    def populateExpTable(self, p):
        [table_m, table_n] = self.expTableShape
        b = np.log2(table_n)

        # Currently looking at only 2D arrays.
        assert table_m == 2

        [m, M] = self.expRange
        max = int(np.ldexp(M - m, -p))
        shl = self.getShl(max)

        alpha_count = table_n
        beta_count = table_n

        table = [[0 for _ in range(alpha_count)], [
            0 for _ in range(beta_count)]]

        alpha = config.wordLength - shl - b
        pRes = self.getScale(1)
        for i in range(alpha_count):
            num = i * 2 ** (alpha + p)
            exp = np.exp(-num)
            table[0][i] = int(np.ldexp(exp, -pRes))

        beta = alpha - b
        pRes = self.getScale(abs(np.exp(-m)))
        for i in range(beta_count):
            num = m + i * 2 ** (beta + p)
            exp = np.exp(-num)
            table[1][i] = int(np.ldexp(exp, -pRes))

        tableVar = [IR.Var('EXP' + str(abs(p)) + 'A', inputVar=True),
                    IR.Var('EXP' + str(abs(p)) + 'B', inputVar=True)]

        return [table, tableVar]

    def getAlphaCount(self, max, shl):
        mask = 2 ** self.expB - 1
        shr = config.wordLength - shl - self.expB
        return ((max >> shr) & mask) + 1

    # Used by old SeeDot.
    def readExpProfileFile(self):
        '''
        This function reads the profile generated by the floating-point program.
        The profile will consist of the range of run-time values seen for the exp() function.
        This range is stored and will be used while generating look-up tables for exp() function.
        '''
        if self.expProfileLoaded == True:
            return
        self.expProfileLoaded = True

        inputFile = getProfileLogFile()

        data = []
        with open(inputFile, 'r') as f:
            for line in f:
                entries = line.strip().split(", ")
                row = list(map(float, entries))
                data.append(row)

        [min_exp, max_exp] = data[1]

        assert max_exp >= min_exp >= 0, "The range of values for exp() is not as expected."

        # Data for computing exp.
        self.expRange = [min_exp, max_exp]
        self.expB = getExpBitLength()
        self.expTableShape = [2, 2 ** self.expB]

        self.MAX_VAL_EXP = max_exp

    # out = argmax(in)
    def visitArgMax(self, node: AST.Func):
        (prog_in, expr_in) = self.visit(node.expr)

        type_out = node.expr.type

        assert type_out.dim == 2, "'argmax' operator currently only supports 2D tensors."

        # Read input scale.
        bitwidth_in, scale_in = self.getBitwidthAndScale(expr_in.idf)

        [I, J] = type_out.shape

        expr_out = self.getTempVar()

        expr_in.inputVar = False

        comment = IR.Comment('argmax(' + expr_in.idf + ')', self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        funcCall = IR.FuncCall("ArgMax", {
            expr_in: "A",
            IR.Int(I): "I",
            IR.Int(J): "J",
            expr_out: "index"
        }) if not self.vbwEnabled else IR.FuncCall("ArgMax<int%d_t>"%(bitwidth_in), {
            expr_in: "A",
            IR.Int(I): "I",
            IR.Int(J): "J",
            expr_out: "index"
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        prog_argmax = IR.Prog([comment, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_argmax)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = Type.Int()
        self.internalVars.append(expr_out.idf)

        return (prog_out, expr_out)

    # out = sgn(in)
    # if in > 0:
    #    out = 1
    # else
    #    out = 0
    def visitSgn(self, node: AST.Func):
        (prog_in, expr_in) = self.visit(node.expr)

        expr_out = self.getTempVar()
        type_in = node.expr.type

        expr_in_idx = IRUtil.addIndex(expr_in, [IRUtil.zero] * type_in.dim)

        bitwidth_in, scale_in = self.getBitwidthAndScale(expr_in.idf)

        comment = IR.Comment('sgn(' + expr_in.idf + ')', self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        cmd1 = IR.Assn(expr_out, IRUtil.cond_zero(
            expr_in_idx, IRUtil.one, IRUtil.zero))

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        prog_sgn = IR.Prog([comment, cmd1])

        prog_out = IRUtil.concatPrograms(prog_in, prog_sgn)

        self.varDeclarations[expr_out.idf] = Type.Int()
        self.internalVars.append(expr_out.idf)

        return (prog_out, expr_out)

    # out = tanh(in)
    def visitTanh(self, node: AST.Func):
        # Old implementation of TanH, where a linear approximation is used.
        # The floating-point version of this method uses math.h implementation of exp(x).
        (prog_in, expr_in) = self.visit(node.expr)

        type_in = node.expr.type
        [I, J] = type_in.shape

        expr_out = self.getTempVar()

        # Read input scale.
        bitwidth_in, scale_in = self.getBitwidthAndScale(expr_in.idf)
        intv_in = self.varIntervals[expr_in.idf]

        # If input is demoted to lower bit-width, demote the output to lower bit-width as well.
        tmp_var = expr_in.idf
        while tmp_var in self.substitutions.keys():
            tmp_var = self.substitutions[tmp_var]
        if tmp_var in self.demotedVarsList:
            self.demotedVarsList.append(expr_out.idf)
            self.demotedVarsOffsets[expr_out.idf] = 0
            self.varsForBitwidth[expr_out.idf] = config.wordLength // 2

        if forFloat():
            tanh_limit = IR.Float(config.tanhLimit)
        else:
            # Scale TanH limit.
            tanh_limit = self.getNumInFixedPoint(config.tanhLimit, scale_in)

        tanh_intv = self.getInterval(
            scale_in, config.tanhLimit, config.tanhLimit)
        intv_out = self.updateTanhIntv(intv_in, tanh_intv)

        scale_new = self.getScale(config.tanhLimit)
        getLogger().debug("Scale changes in TanH operation: old = %d, new = %d, diff = %d" % (
            scale_in, scale_new, abs(scale_in - scale_new)))

        expr_in.inputVar = False

        comment = IR.Comment("tanh(" + expr_in.idf + ")", self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        scale_out = scale_in # self.getScale(1.5)
        tanh_limit_out = 2 ** -scale_out

        funcCall = IR.FuncCall("TanH", {
            expr_in: "A",
            IR.Int(I): "I",
            IR.Int(J): "J",
            tanh_limit: "scale_in",
            IR.Int(tanh_limit_out): "scale_out",
            expr_out: "B"
        }) if not self.vbwEnabled else IR.FuncCall("TanH<int%d_t>"%(bitwidth_in), {
            expr_in: "A",
            IR.Int(I): "I",
            IR.Int(J): "J",
            tanh_limit: "scale_in",
            IR.Int(tanh_limit_out): "scale_out",
            expr_out: "B"
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        prog_tanh = IR.Prog([comment, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_tanh)

        # Updating metadata.
        self.varDeclarations[expr_out.idf] = type_in
        self.varIntervals[expr_out.idf] = intv_out
        self.varScales[expr_out.idf] = scale_out

        return (prog_out, expr_out)

    def visitNewTableTanH(self, node: AST.Func):
        # Refer to OOPSLA '20 paper Section 5.4.1 for a theoretical explanation.
        assert self.vbwEnabled, "VBW must be enabled for new table"

        (prog_in, expr_in) = self.visit(node.expr)

        type_in = node.expr.type

        bitwidth_in_raw, scale_in_raw = self.getBitwidthAndScale(expr_in.idf)

        MIN = 0.1
        maxExp = np.exp(MIN)

        expr_out = self.getTempVar()

        # If the input is demoted to 8 bits, demote the output to the lower bit-width as well.
        if bitwidth_in_raw != config.wordLength:
            self.demotedVarsList.append(expr_out.idf)
            self.demotedVarsOffsets[expr_out.idf] = 0
            self.varsForBitwidth[expr_out.idf] = config.wordLength // 2

        # Reading / Computing output bit-width.
        bitwidth_out = bitwidth_in_raw
        scale_out = self.getScale(maxExp) + config.wordLength // 2 if expr_out.idf in self.demotedVarsList else self.getScale(maxExp)
        scale_in_adjusted = scale_out

        [I, J] = type_in.shape

        scale_in = -4 if bitwidth_in_raw != config.wordLength else -11

        # Refer to OOPSLA '20 paper Section 5.4 which explains the computation of input and output scales for TanH and Sigmoid.
        # The input scales are adjusted to the theoretically known value for different bit-widths.
        adjust = []
        if scale_in_raw != scale_in:
            diff_scale = abs(scale_in_raw - scale_in)
            if scale_in_raw > scale_in:
                saturate = (2 ** (-scale_in_adjusted - diff_scale))
                adjust = [IR.FuncCall("AdjustScaleShlSaturate<int%d_t>" %(bitwidth_in_raw), {
                                            expr_in : "A",
                                            IR.Int(I): "I",
                                            IR.Int(J): "J",
                                            IR.Int(2 ** diff_scale): "scale",
                                            IR.Int(saturate): "saturate"
                })]
            else:
                adjust = [IR.FuncCall("AdjustScaleShr<int%d_t>" %(bitwidth_in_raw), {
                                            expr_in : "A",
                                            IR.Int(I): "I",
                                            IR.Int(J): "J",
                                            IR.Int(2 ** diff_scale): "scale"
                })]

        comm = IR.Comment('tanh(' + expr_in.idf + ')', self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        funcCall = IR.FuncCall("TanHNew%d<0>" %(bitwidth_in_raw), {
            expr_in: "A",
            IR.Int(I): "I",
            IR.Int(J): "J",
            expr_out: "B"
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        prog_tanh = IR.Prog([comm] + adjust + [funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_tanh)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_in
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = (0, 0)

        return (prog_out, expr_out)

    # out = sigmoid(in)
    def visitSigmoid(self, node: AST.Func):
        # y = max(min( x/4 + 2/4 , 1), 0), 1).
        # Old implementation, fixed point code uses linear approximation.
        # Floating point implementation uses math.h.
        denominator = 2
        addition = 0.5
        sigmoid_limit = 1

        (prog_in, expr_in) = self.visit(node.expr)

        type_in = node.expr.type
        [I, J] = type_in.shape

        expr_out = self.getTempVar()

        # Read input scales and bit-width.
        bitwidth_in, scale_in = self.getBitwidthAndScale(expr_in.idf)
        intv_in = self.varIntervals[expr_in.idf]

        # If input is demoted to lower bit-width, demote the output variable to lower bit-width as well.
        tmp_var = expr_in.idf
        while tmp_var in self.substitutions.keys():
            tmp_var = self.substitutions[tmp_var]
        if tmp_var in self.demotedVarsList:
            self.demotedVarsList.append(expr_out.idf)
            self.demotedVarsOffsets[expr_out.idf] = 0
            self.varsForBitwidth[expr_out.idf] = config.wordLength // 2

        # Scale sigmoid limit and other constants.
        addition_int = self.getNumInFixedPoint(addition, scale_in)
        sigmoid_limit_int = self.getNumInFixedPoint(sigmoid_limit, scale_in)

        # Compute new interval.
        [m, M] = intv_in
        m_new = max(min((m / denominator) + addition_int.n,
                        sigmoid_limit_int.n), 0)
        M_new = max(min((M / denominator) + addition_int.n,
                        sigmoid_limit_int.n), 0)
        assert m_new <= M_new, "The range of sigmoid has changed. Re-check the assertion."

        intv_out = (m_new, M_new)

        scale_out = self.getScale(1.5) + ((config.wordLength // 2 + self.demotedVarsOffsets[expr_in.idf]) if expr_in.idf in self.demotedVarsList else 0)

        # Computing hyperparameters for linear approximation of Sigmoid.
        max_val = max(abs(m_new), abs(M_new))
        max_val_f = np.ldexp(max_val, scale_in)

        if forFloat():
            addition_ir = IR.Float(addition)
            sigmoid_limit_ir = IR.Float(sigmoid_limit)
        else:
            addition_ir = addition_int
            sigmoid_limit_ir = sigmoid_limit_int

        scale_in_num = 2 ** -scale_in
        scale_out_num = 2 ** -scale_out

        expr_in.inputVar = False

        comment = IR.Comment("Sigmoid(" + expr_in.idf + ")", self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        funcCall = IR.FuncCall("Sigmoid", {
            expr_in: "A",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(denominator): "div",
            addition_ir: "add",
            sigmoid_limit_ir: "sigmoid_limit",
            IR.Int(scale_in_num): "scale_in",
            IR.Int(scale_out_num): "scale_out",
            expr_out: "B"
        }) if not self.vbwEnabled else IR.FuncCall("Sigmoid<int%d_t>"%(bitwidth_in), {
            expr_in: "A",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(denominator): "div",
            addition_ir: "add",
            sigmoid_limit_ir: "sigmoid_limit",
            IR.Int(scale_in_num): "scale_in",
            IR.Int(scale_out_num): "scale_out",
            expr_out: "B"
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        prog_sigmoid = IR.Prog([comment, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_sigmoid)

        # Updating metadata.
        self.varDeclarations[expr_out.idf] = type_in
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = intv_out

        # Print log.
        self.log.print(comment.msg)
        self.log.print("\tInput:  scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in.idf],) + self.varIntervals[expr_in.idf]))
        self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

        return (prog_out, expr_out)

    def visitNewTableSigmoid(self, node: AST.Func):
        # Please refer to OOPSLA '20 paper Section 5.4.1 for explanation.
        assert self.vbwEnabled, "VBW must be enabled for new table"

        (prog_in, expr_in) = self.visit(node.expr)

        type_in = node.expr.type

        bitwidth_in_raw, scale_in_raw = self.getBitwidthAndScale(expr_in.idf)

        MIN = 0.1
        maxExp = np.exp(MIN)

        expr_out = self.getTempVar()
        
        # If input variable is demoted to lower bit-width, demote the output variable to lower bit-width as well.
        if bitwidth_in_raw != config.wordLength:
            self.demotedVarsList.append(expr_out.idf)
            self.demotedVarsOffsets[expr_out.idf] = 0
            self.varsForBitwidth[expr_out.idf] = config.wordLength // 2

        # Compute / Read output scale and bit-width.
        bitwidth_out = bitwidth_in_raw
        scale_out = self.getScale(maxExp) + config.wordLength // 2 if expr_out.idf in self.demotedVarsList else self.getScale(maxExp)
        scale_in_adjusted = scale_out

        [I, J] = type_in.shape

        scale_in = -4 if bitwidth_in_raw != config.wordLength else -11

        # Refer to OOPSLA '20 paper Section 5.4 which explains the computation of input and output scales for TanH and Sigmoid.
        # The input scales are adjusted to the theoretically known value for different bit-widths.
        adjust = []
        if scale_in_raw != scale_in:
            diff_scale = abs(scale_in_raw - scale_in)
            if scale_in_raw > scale_in:
                saturate = (2 ** (-scale_in_adjusted - diff_scale))
                adjust = [IR.FuncCall("AdjustScaleShlSaturate<int%d_t>" %(bitwidth_in_raw), {
                                            expr_in : "A",
                                            IR.Int(I): "I",
                                            IR.Int(J): "J",
                                            IR.Int(2 ** diff_scale): "scale",
                                            IR.Int(saturate): "saturate"
                })]
            else:
                adjust = [IR.FuncCall("AdjustScaleShr<int%d_t>" %(bitwidth_in_raw), {
                                            expr_in : "A",
                                            IR.Int(I): "I",
                                            IR.Int(J): "J",
                                            IR.Int(2 ** diff_scale): "scale"
                })]

        comm = IR.Comment('sigmoid(' + expr_in.idf + ')', self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        funcCall = IR.FuncCall("SigmoidNew%d<0>" %(bitwidth_in_raw), {
            expr_in: "A",
            IR.Int(I): "I",
            IR.Int(J): "J",
            expr_out: "B"
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        prog_sigmoid = IR.Prog([comm] + adjust + [funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_sigmoid)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_in
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = (0, 0)

        return (prog_out, expr_out)

    # let a[i1:+n1][i2:+n2]... = ...
    def visitLeftSplice(self, node: AST.LeftSplice, expr_in, nodeVarType):
        # Used to assign a splice of a tensor to some value.
        vars_in = []
        progs_in = []
        for var in node.vars:
            part_prog_in, part_expr_in = self.visit(var)
            progs_in.append(part_prog_in)
            vars_in.append(part_expr_in)

        expr_out = IR.Var(node.expr) 

        # Read input and output scales.
        bw_in, scale_in = self.getBitwidthAndScale(expr_in.idf)
        bw_out, scale_out = self.getBitwidthAndScale(expr_out.idf)

        loop_dim = len(node.sizes)

        # Computing indices on LHS and RHS in for loop.
        iters_in = self.getTempIterators(loop_dim) 
        iters_out = self.getTempVars(loop_dim)

        loopShape = [] # Shape of tensor to be assigned, limits of the iterators.
        loopIters = [] # Iterators to iterate across different dimensions.
        loopAssns = [] # Assignment statements within the loop body.
        for order in range(loop_dim):
            loopShape.append(node.sizes[order])
            loopIters.append(iters_in[order])
            loopAssns.append(IR.Assn(iters_out[order], IRUtil.add(iters_in[order], vars_in[order])))

        expr_in_idx = IRUtil.addIndex(expr_in, iters_in)
        expr_out_idx = IRUtil.addIndex(expr_out, iters_out)

        # Adjusting scale in the input and output code.
        if scale_in > scale_out:
            cmd2 = IR.Assn(expr_out_idx, IRUtil.shl(expr_in_idx, scale_in - scale_out))
        elif scale_in < scale_out:
            cmd2 = IR.Assn(expr_out_idx, IRUtil.shr(expr_in_idx, scale_out - scale_in))
        else:
            cmd2 = IR.Assn(expr_out_idx, expr_in_idx)

        # Compared to right splice iters_out is the index for LHS.
        loop = IRUtil.loop(loopShape, loopIters, loopAssns + [
                cmd2
            ])

        # Comments for showing the input used to generate the given output.
        out_indices = ']['.join([i.idf for i in iters_out])
        in_indices = ']['.join([i.idf for i in iters_in])
        comment = IR.Comment("%s[%s] = %s[%s]"%(expr_out_idx.idf, out_indices, expr_in_idx.idf, in_indices), self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth
        prog_splice = IR.Prog([comment] + loop)

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        # In case the target variable is contiguous, we can optimize (use memcpy instead of a loop).
        canOptimize = True
        loopShapeMustBeOne = False
        for i in range(len(loopShape) - 1, -1, -1):
            if loopShapeMustBeOne:
                if loopShape[i] != 1:
                    canOptimize = False
            else:
                if loopShape[i] == nodeVarType.shape[i]:
                    continue
                elif loopShape[i] < nodeVarType.shape[i]:
                    loopShapeMustBeOne = True
                    continue
                else:
                    assert False, "Illegal State, subtensor dimensions must be less than original tensor dimensions"
        canOptimize = canOptimize and (expr_in.idf not in self.globalVars) and bw_in == bw_out and scale_in == scale_out

        if canOptimize:
            prog_splice = IR.Prog([comment, IR.Memcpy(expr_out, expr_in, np.prod(loopShape), vars_in, [IR.Int(0) for i in range(len(vars_in))])])
        else:
            assert True

        prog_out = IR.Prog([])
        for prog in progs_in:
            prog_out = IRUtil.concatPrograms(prog_out, prog)
        prog_out = IRUtil.concatPrograms(prog_out, prog_splice)

        # Update declarations.
        for var in iters_out:
            self.varDeclarations[var.idf] = Type.Int()
            self.internalVars.append(var.idf)

        return (prog_out, expr_out)

    # out = $x[start:end] in
    def visitSum(self, node: AST.Sum):
        '''
        expr_out
        i = 0
        for (j = 0; j < n; j++)
          expr_in = prog_in
          expr_out = expr_out + expr_in
          i++

        1.  for i in [0, C]:
        2.    expr_out[i] = expr_out[i] + shr(expr_in[i])
        '''

        var_idf = node.name
        self.varDeclarations[var_idf] = Type.Int()
        self.internalVars.append(var_idf)

        start, end = node.start, node.end
        comment = IR.Comment("sum(i = [%d, %d])" % (start, end), self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth
        self.counter_inst += 1
        self.curDepth += 1

        (prog_in, expr_in) = self.visit(node.expr)

        self.curDepth -= 1

        expr_out = self.getTempVar()
        type_out = node.type

        var = IR.Var(var_idf)
        var_iter = self.getTempIterator()
        iters = self.getTempIterators(type_out.dim)

        # Read input scale and bitwidth.
        bitwidth_in, scale_in = self.getBitwidthAndScale(expr_in.idf)
        # Read output scale and bitwidth if known from profiling.
        if self.ddsEnabled:
            bitwidth_out, scale_out = self.getBitwidthAndScale(expr_out.idf)
        else:
            bitwidth_out, scale_out = None, None
        intv_in = self.varIntervals[expr_in.idf]

        # Read / Compute the output scale and associated hyperparameters.
        if self.ddsEnabled:
            scale_raw, height_shr = self.getDDVBRScaleForTreeSum(scale_in, scale_out, end-start)
            height_noshr = 0 # For interval calculation, not used.
        else:
            (scale_out, height_shr, height_noshr) = self.getScaleForTreeSum(
            scale_in, end - start)
        intv_out = (0,0) 

        expr_in_idx = IRUtil.addIndex(expr_in, iters)
        expr_out_idx = IRUtil.addIndex(expr_out, iters)

        # Adjusting scale of input and output in the fixed-point code.
        cmd1 = IR.Memset(expr_out, type_out.size())
        if self.ddsEnabled:
            if scale_raw > scale_out:
                cmd2 = IR.Assn(expr_out_idx, IRUtil.add(expr_out_idx, IRUtil.shr(IRUtil.shl(expr_in_idx, scale_raw - scale_out), height_shr)))
            elif scale_raw < scale_out:
                cmd2 = IR.Assn(expr_out_idx, IRUtil.add(expr_out_idx, IRUtil.shr(IRUtil.shr(expr_in_idx, scale_out - scale_raw), height_shr)))
            else:
                cmd2 = IR.Assn(expr_out_idx, IRUtil.add(expr_out_idx, IRUtil.shr(expr_in_idx, height_shr)))
        else:
            cmd2 = IR.Assn(expr_out_idx, IRUtil.add(expr_out_idx, IRUtil.shr(expr_in_idx, height_shr)))
        treeSum = IRUtil.loop(type_out.shape, iters, [cmd2])

        assert type_out.dim == 2, "Only 2 dim Summation supported for now due to laziness of programmer"
        if forFloat():
            profile = [IR.FuncCall("Profile2", {
                expr_out: "Var",
                IR.Int(type_out.shape[0]): "I",
                IR.Int(type_out.shape[1]): "J",
                IR.String(expr_out): "VarName"
            })]
        if forFloat():
            self.independentVars.append(expr_out.idf)

        # Final program to sum output of each iteration.
        prog_sum = [cmd1,
                    IR.Assn(var, IR.Int(start)),
                    IR.For(var_iter, 0, IRUtil.lt(var_iter, IR.Int(end - start)),
                           prog_in.cmd_l + treeSum + (profile if forFloat() and self.ddsEnabled else []) + [IR.Assn(var, IRUtil.inc(var))])
                    ]

        self.updateLiveRange([expr_in, expr_out])

        prog_out = IR.Prog([comment] + prog_sum)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = intv_out

        return (prog_out, expr_out)

    # out = loop(x[start:end]) (expr) in
    def visitLoop(self, node: AST.Loop):
        '''
        for (i = 0; i < n; i++)
          prog_in
        '''

        idf = node.mutableVar.name
        self.mutableVars.append(idf)

        # Update the scale and interval of the mutable variable only during fixed-point code generation.
        if forFixed():
            scale, intv = self.readProfileForMutableVars(idf)
            bitwidth, _ = self.getBitwidthAndScale(idf) # (init 0 default scale currently stored in varScales which has to be overwritten).
            if bitwidth != config.wordLength:
                idfs = idf
                while idfs in self.substitutions.keys():
                    idfs = self.substitutions[idfs]
                scale += config.wordLength // 2 + self.demotedVarsOffsets[idfs]
            self.varScales[idf] = scale
            self.varIntervals[idf] = intv

        prevVarDecls = dict(self.varDeclarations)

        start, end = node.start, node.end

        comment = IR.Comment("loop(%s = [%d, %d], %s)" % (
            node.name, start, end, idf), self.counter_inst+1) # The comment is before visiting the loop statements so the self.counter_inst's earlier value is used.
        self.allDepths[self.counter_inst+1] = self.curDepth

        self.counter_inst += 1
        self.curDepth += 1

        (prog_in, expr_in) = self.visit(node.expr)

        self.curDepth -=1

        # This variable contains variables that need to be declared within the for loop.
        # No longer needed as the current memory management (config.x86MemoryOptimize) takes care of variables declared locally within loop.
        forDecls = {}

        assert start == 0, "'loop' operator currently supports only iterations starting from 0."

        var = IR.Var(node.name)

        loop = IR.For(var, 0, IRUtil.lt(
            var, IR.Int(end - start)), prog_in.cmd_l, 0, forDecls)

        self.updateLiveRange([expr_in])

        # Generate code for profiling.
        if forFloat() and getTarget() == config.Target.x86:
            mVar = IR.Var(node.mutableVar.name)
            mVar_type = node.mutableVar.type
            profile_iters = self.getTempIterators(mVar_type.dim)
            mVar_idx = IRUtil.addIndex(mVar, profile_iters)
            funcCall = IR.FuncCall("updateRange", {
                mVar_idx: "A"
            })
            profile = IRUtil.loop(mVar_type.shape, profile_iters, [funcCall])
        else:
            profile = []

        prog_out = IR.Prog([comment, loop] + profile)

        return (prog_out, expr_in)

    # out = in_cond > 0? in_A: in_B
    def visitCond(self, node: AST.Cond):
        (prog_in_cond, expr_in_cond) = self.visit(node.expr)

        (prog_in_A, expr_in_A) = self.visit(node.trueBlock)

        (prog_in_B, expr_in_B) = self.visit(node.falseBlock)

        type_in_cond = node.expr.type
        type_in_A = node.trueBlock.type

        if Type.isInt(type_in_cond):
            expr_in_cond_idx = expr_in_cond
        else:
            expr_in_cond_idx = IRUtil.addIndex(
                expr_in_cond, [IRUtil.zero] * type_in_cond.dim)

        # e2, e3 : Int
        if Type.isInt(type_in_A):
            # TODO: Update the scale and intv of expr_out based on in_A and in_B.
            prog_out = IRUtil.concatPrograms(
                prog_in_cond, prog_in_A, prog_in_B)
            expr_out = IRUtil.cond_zero(expr_in_cond_idx, expr_in_A, expr_in_B)

            if isinstance(expr_in_A, IR.Var):
                assert expr_in_A.idf not in self.varScales and expr_in_A.idf not in self.varIntervals
            if isinstance(expr_in_B, IR.Var):
                assert expr_in_B.idf not in self.varScales and expr_in_B.idf not in self.varIntervals
        # e2, e3 : Tensor(), or Tensor(..)
        else:
            expr_out = self.getTempVar()
            iters = self.getTempIterators(type_in_A.dim)

            # Read input scales and bit-widths.
            bitwidth_in_A, scale_in_A = self.getBitwidthAndScale(expr_in_A.idf)
            bitwidth_in_B, scale_in_B = self.getBitwidthAndScale(expr_in_B.idf)
            intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]

            m_A, M_A = intv_in_A
            m_B, M_B = intv_in_B

            if scale_in_A >= scale_in_B:
                shr_A, shr_B = 0, scale_in_A - scale_in_B
            else:
                shr_A, shr_B = scale_in_B - scale_in_A, 0

            scale_out = max(scale_in_A, scale_in_B)
            intv_out = (min(m_A >> shr_A, m_B >> shr_B),
                        max(M_A >> shr_A, M_B >> shr_B))

            # prog_assn
            expr_in_A_idx = IRUtil.addIndex(expr_in_A, iters)
            expr_in_B_idx = IRUtil.addIndex(expr_in_B, iters)
            expr_out_idx = IRUtil.addIndex(expr_out, iters)

            rhs = IRUtil.cond_zero(expr_in_cond_idx,
                                   IRUtil.shr(expr_in_A_idx, shr_A),
                                   IRUtil.shr(expr_in_B_idx, shr_B))
            cmdl_assn = IRUtil.loop(type_in_A.shape, iters, [
                                    IR.Assn(expr_out_idx, rhs)])
            prog_cond = IR.Prog(cmdl_assn)

            prog_out = IRUtil.concatPrograms(
                prog_in_cond, prog_in_A, prog_in_B, prog_cond)

            # Update metadata.
            self.varDeclarations[expr_out.idf] = type_in_A
            self.varScales[expr_out.idf] = scale_out
            self.varIntervals[expr_out.idf] = intv_out

        self.allDepths[self.counter_inst+1] = self.curDepth
        self.counter_inst += 1
        self.updateLiveRange([expr_in_A, expr_in_B, expr_in_cond, expr_out])

        return (prog_out, expr_out)

    # let LHS = decl 'in' in
    def visitLet(self, node: AST.Let):
        # Visit RHS of the let statement.
        (prog_decl, expr_decl) = self.visit(node.decl)

        type_decl = node.decl.type
        idf = node.name

        # e1 : Int
        if Type.isInt(type_decl):
            # LHS is a new integer variable and needs to be assigned to the list of variables.
            self.varDeclarations[idf] = Type.Int()
            self.internalVars.append(idf)

            # Visit remainder of the program.
            (prog_in, expr_in) = self.visit(node.expr)

            cmd = IR.Assn(IR.Var(idf), expr_decl)
            prog_let = IR.Prog([cmd])

            prog_out = IRUtil.concatPrograms(prog_decl, prog_let, prog_in)

            return (prog_out, expr_in)

        # Left Splice case.
        elif node.leftSplice is not None:
            # We have to assign the value of decl (RHS) into a splice of the LHS variable.
            parentVar = node.name
            while parentVar in self.substitutions:
                parentVar = self.substitutions[parentVar] # Done as all metadata is stored at the end of the substitution chain.
            # Assign the RHS to a splice of LHS.
            (prog_splice, expr_splice) = self.visitLeftSplice(node.leftSplice, expr_decl, self.varDeclarations[parentVar])
            (prog_in, expr_in) = self.visit(node.expr)

            # Profile the LHS as the value would have been updated, hence the scale required for LHS in the floating-point code may be different.
            profile = IR.Prog([])
            if forFloat():
                profile = IR.Prog([IR.FuncCall("Profile2", {
                    expr_decl: "Var",
                    IR.Int(node.decl.type.shape[0]): "I",
                    IR.Int(node.decl.type.shape[1]): "J",
                    IR.String(expr_splice): "VarName"
                })])
            if forFloat():
                self.independentVars.append(expr_splice.idf)

            prog_out = IRUtil.concatPrograms(prog_decl, prog_splice, profile, prog_in)

            return (prog_out, expr_in)
        # e1 : Tensor{(),(..)}
        else:
            # Compute the scale of the LHS variable. RHS/decl may have a different bit-width, hence the scale of LHS has to be adjusted accordingly.
            self.varScales[idf] = self.varScales[expr_decl.idf] + (config.wordLength//2 + self.demotedVarsOffsets.get(idf, 0) if idf in self.demotedVarsList else 0)
            self.varIntervals[idf] = self.varIntervals[expr_decl.idf]

            # If LHS is demoted to lower bit-width, the RHS should also be in a lower bit-width, so scale of RHS is also adjusted.
            if idf in self.demotedVarsList:
                self.varScales[expr_decl.idf] += self.demotedVarsOffsets[idf] + config.wordLength // 2
                self.demotedVarsList.append(expr_decl.idf)
                self.demotedVarsOffsets[expr_decl.idf] = self.demotedVarsOffsets[idf]
                self.varsForBitwidth[expr_decl.idf] = config.wordLength // 2
            else:
                if expr_decl.idf not in self.varsForBitwidth:
                    self.varsForBitwidth[expr_decl.idf] = config.wordLength

            # For input X, scale is computed as follows.
            if idf == "X" and self.scaleForX is not None:
                self.varScales[idf] = self.scaleForX + (config.wordLength // 2 + self.demotedVarsOffsets.get("X", 0) if 'X' in self.demotedVarsList else 0)

            # If the let statement is a model parameter declaration, then the following is invoked.
            if isinstance(node.decl, AST.Decl):
                self.globalVars.append(idf)
                # TODO: Do I need to update varDeclarations or is it handled already?
                self.varDeclarations[idf] = node.decl.type
                expr_decl.idf = idf
                expr_decl.inputVar = True

            # For mutable variables of a loop, such variables are substituted  later and the details are captured here.
            if idf in self.mutableVars:
                if forFloat():
                    if expr_decl.idf != idf:
                        self.substitutions[expr_decl.idf] = idf
                expr_decl.idf = idf

            # In fixed-point mode, for mutable variables the scales need to be adjusted which is done here.
            if forFixed() and idf in self.mutableVars:
                # Add a loop to adjust the scale back to the original one.
                curr_scale = self.varScales[idf]
                idfs = idf
                while idfs in self.substitutions.keys():
                    idfs = self.substitutions[idfs]
                # Read profiled scale of the LHS (profile assumes 16-bit variables) and compute final scale depending on actual bitwidth of LHS.
                if self.ddsEnabled:
                    _, raw_new_scale = self.getBitwidthAndScale(idfs)
                    new_scale = raw_new_scale + (config.wordLength // 2 + self.demotedVarsOffsets[idfs] if idfs in self.demotedVarsList else 0)
                    new_intv = (0, 0)
                else:
                    [minVal, maxVal] = self.mutableVarsProfile[0] # TODO: This function may not work for multiple loops in a code.
                    new_scale = self.getScale(max(abs(minVal), abs(maxVal))) + (config.wordLength // 2 + self.demotedVarsOffsets[idfs] if idfs in self.demotedVarsList else 0)
                    new_intv = self.getInterval(new_scale, minVal, maxVal)

                diff_scale = 2 ** (curr_scale - new_scale) if curr_scale > new_scale else 2 ** (new_scale - curr_scale)

                [I, J] = type_decl.shape
                bitwidth_decl, scale_decl = self.getBitwidthAndScale(expr_decl.idf)

                # The mutable loop variable needs to have it's scale adjusted so that it remains the same across iterations for correctness.
                adjust = []
                if curr_scale != new_scale:
                    if curr_scale > new_scale:
                        adjust = [IR.FuncCall("AdjustScaleShl", {
                                            IR.Var(idf): "A",
                                            IR.Int(I): "I",
                                            IR.Int(J): "J",
                                            IR.Int(diff_scale): "scale"
                                 })] if not self.vbwEnabled else [IR.FuncCall("AdjustScaleShl<int%d_t>"%(bitwidth_decl), {
                                            IR.Var(idf): "A",
                                            IR.Int(I): "I",
                                            IR.Int(J): "J",
                                            IR.Int(diff_scale): "scale"
                                 })]
                    elif curr_scale < new_scale:
                        adjust = [IR.FuncCall("AdjustScaleShr", {
                                            IR.Var(idf): "A",
                                            IR.Int(I): "I",
                                            IR.Int(J): "J",
                                            IR.Int(diff_scale): "scale"
                                 })] if not self.vbwEnabled else [IR.FuncCall("AdjustScaleShr<int%d_t>"%(bitwidth_decl), {
                                            IR.Var(idf): "A",
                                            IR.Int(I): "I",
                                            IR.Int(J): "J",
                                            IR.Int(diff_scale): "scale"
                                 })]

                prog_for_mutable = IR.Prog(adjust)

                # Reset the self.scale value to the profile generated one.
                self.varScales[idf] = new_scale
                self.varIntervals[idf] = new_intv
            else:
                prog_for_mutable = IR.Prog([])

            # In floating point mode, the details of substitutions are stored for use in the fixed-point version.
            # Independent variables (which are profiled) are also stored in a way to avoid duplication (profiling two names of the same entity).
            if forFloat():
                if expr_decl.idf != idf:
                    if idf in self.substitutions.keys():
                        assert False, "What kind of subtitutions are going on?"
                    self.substitutions[idf] = expr_decl.idf

                # To ensure loop variable is correctly fed for data driven scaling.
                for i in range(len(self.independentVars)):
                    while self.independentVars[i] in self.substitutions.keys():
                        self.independentVars[i] = self.substitutions[self.independentVars[i]]

            (prog_in, expr_in) = self.visit(node.expr)

            # TODO: When is this triggered and why is this required?
            if forFixed() and idf in self.mutableVars:
                getLogger().warning("TODO: Fix this if condition")
                idfs = idf
                while idfs in self.substitutions.keys():
                    idfs = self.substitutions[idfs]
                if self.ddsEnabled:
                    _, raw_new_scale = self.getBitwidthAndScale(idfs)
                    new_scale = raw_new_scale + (config.wordLength // 2 + self.demotedVarsOffsets[idfs] if idfs in self.demotedVarsList else 0)
                    new_intv = (0, 0)
                else:
                    [minVal, maxVal] = self.mutableVarsProfile[0]
                    new_scale = self.getScale(max(abs(minVal), abs(maxVal)))
                    new_intv = self.getInterval(new_scale, minVal, maxVal)
                self.varScales[expr_decl.idf] = new_scale
                self.varIntervals[expr_decl.idf] = new_intv

            prog_decl = IRUtil.concatPrograms(
                prog_decl, IR.Prog([prog_for_mutable]))

            # Perform substitutions to consolidate generated names and user-provided names.
            prog_in = prog_in.subst(idf, expr_decl)
            expr_in = expr_in.subst(idf, expr_decl)

            # Consolidate the information about live ranges for lhs and rhs, given the substitutions performed above.
            if idf != expr_decl.idf and idf in self.varLiveIntervals and expr_decl.idf in self.varLiveIntervals:
                self.varLiveIntervals[idf] = [min(self.varLiveIntervals[idf][0], self.varLiveIntervals[expr_decl.idf][0]), max(self.varLiveIntervals[idf][1], self.varLiveIntervals[expr_decl.idf][1])]
                self.varLiveIntervals[expr_decl.idf] = list(self.varLiveIntervals[idf])

            prog_out = IRUtil.concatPrograms(prog_decl, prog_in)

            return (prog_out, expr_in)

    # Used by old SeeDot for reading profile for exponentiation.
    # NOTE: Works only when there is one variable for exponentiation.
    # New SeeDot uses same data driven scaling platform for all variables.
    def readProfileForMutableVars(self, idf):
        # Data-driven parameters.
        inputFile = getProfileLogFile()

        with open(inputFile, 'r') as f:
            for line in f:
                entries = line.strip().split(", ")
                row = list(map(float, entries))
                self.mutableVarsProfile.append(row)

        [minVal, maxVal] = self.mutableVarsProfile[0]

        scale = self.getScale(max(abs(minVal), abs(maxVal)))
        intv = self.getInterval(scale, minVal, maxVal)

        return scale, intv

    # Used in old SeeDot.
    # Computing exponent and intervals.
    def getScale(self, val_max: float):  # -> int
        return computeScalingFactor(val_max)

    # Used in old SeeDot.
    # Takes range [val_min, val_max] and returns the interval in fixed-point.
    def getInterval(self, scale: int, val_min: float, val_max: float):
        return (int(np.ldexp(val_min, -scale)), int(np.ldexp(val_max, -scale)))

    # Used in old SeeDot.
    # A * B
    def getScaleForMul(self, scale_A: int, shr_A: int, scale_B: int, shr_B: int) -> int:
        return (scale_A + shr_A) + (scale_B + shr_B)

    # Used in old SeeDot.
    # int^2 * int^2 -> int^2
    def getIntvervalForMul(self, intv_A, shr_A: int, intv_B, shr_B: int):
        return (0, 0)
        (minVal_A, maxVal_A) = intv_A
        (minVal_A, maxVal_A) = (minVal_A >> shr_A, maxVal_A >> shr_A)

        (minVal_B, maxVal_B) = intv_B
        (minVal_B, maxVal_B) = (minVal_B >> shr_B, maxVal_B >> shr_B)

        values = [minVal_A * minVal_B, minVal_A * maxVal_B,
                  maxVal_A * minVal_B, maxVal_A * maxVal_B]

        minVal_out, maxVal_out = min(values), max(values)

        return (minVal_out, maxVal_out)

    # Used in old SeeDot.
    def getScaleForTreeSum(self, scale: int, length: int):
        height = int(np.ceil(np.log2(length)))

        if scale >= self.MAX_SCALE:
            scale_out = scale
        else:
            scale_out = min(scale + height, self.MAX_SCALE)

        height_shr = scale_out - scale
        assert height_shr >= 0

        height_noshr = height - height_shr
        assert height_noshr >= 0

        return (scale_out, height_shr, height_noshr)

    # Compute scale for treeSum operation given input and output scales.
    def getDDVBRScaleForTreeSum(self, scaleIn:int, scaleOut:int, length:int):
        height = int(np.ceil(np.log2(length)))

        scaleAfterAdds = scaleIn + height

        scaleAfterAdds = max(scaleIn, min(scaleAfterAdds, scaleOut))

        height_shr = scaleAfterAdds - scaleIn
        assert height_shr >= 0
        assert height - height_shr >= 0

        return scaleAfterAdds, height_shr

    # Used by old SeeDot.
    def getIntervalForTreeSum(self, intv, count, height_shr, height_noshr):
        (minVal, maxVal) = intv

        arr_min = [minVal for i in range(count)]
        arr_max = [maxVal for i in range(count)]

        minVal_out = self.treeSum(arr_min, count, height_shr, height_noshr)
        maxVal_out = self.treeSum(arr_max, count, height_shr, height_noshr)

        return (minVal_out, maxVal_out)

    # Used by old SeeDot.
    def getScaleAndIntervalForAddAndSub(self, scale_A: int, scale_B: int, intv_A, intv_B, op_fn):
        if op_fn == operator.add:
            return self.getScaleAndIntervalForAdd(scale_A, scale_B, intv_A, intv_B)
        elif op_fn == operator.sub:
            return self.getScaleAndIntervalForSub(scale_A, scale_B, intv_A, intv_B)
        else:
            assert False, "Operator other than add and sub not supported"

    # Used by old SeeDot.
    def getScaleAndIntervalForAdd(self, scale_A: int, scale_B: int, intv_A, intv_B):
        (minVal_A, maxVal_A) = intv_A
        (minVal_B, maxVal_B) = intv_B

        if scale_A >= scale_B:
            shr_all = [0, scale_A - scale_B, 0]
            scale_common = scale_A
        else:
            shr_all = [scale_B - scale_A, 0, 0]
            scale_common = scale_B

        minVal_out = (minVal_A >> shr_all[0]) + (minVal_B >> shr_all[1])
        maxVal_out = (maxVal_A >> shr_all[0]) + (maxVal_B >> shr_all[1])

        if scale_common < self.MAX_SCALE:
            shr_all[2] = 1
            scale_common += 1
        max_abs = (1 << config.wordLength - 2) - 1

        minVal_out = max(minVal_out >> shr_all[2], -max_abs)
        maxVal_out = min(maxVal_out >> shr_all[2],  max_abs)

        return (scale_common, (minVal_out, maxVal_out), shr_all)

    # Used by old SeeDot.
    def getScaleAndIntervalForSub(self, scale_A: int, scale_B: int, intv_A, intv_B):
        (minVal_A, maxVal_A) = intv_A
        (minVal_B, maxVal_B) = intv_B

        if scale_A >= scale_B:
            shr_all = [0, scale_A - scale_B, 0]
            scale_common = scale_A
        else:
            shr_all = [scale_B - scale_A, 0, 0]
            scale_common = scale_B

        minVal_out = (minVal_A >> shr_all[0]) - (minVal_B >> shr_all[1])
        maxVal_out = (maxVal_A >> shr_all[0]) - (maxVal_B >> shr_all[1])

        # if max(abs(minVal_out), abs(maxVal_out)) >= (1 << (config.wordLength - 2)) and scale_common < self.MAX_SCALE:
        if scale_common < self.MAX_SCALE:
            shr_all[2] = 1
            scale_common += 1
        max_abs = (1 << config.wordLength - 2) - 1

        minVal_out = max(minVal_out >> shr_all[2], -max_abs)
        maxVal_out = min(maxVal_out >> shr_all[2],  max_abs)

        return (scale_common, (minVal_out, maxVal_out), shr_all)

    # Used for old SeeDot.
    def getScaleForExp(self, scale_A: int, shr_A: int, scale_B: int, shr_B: int):
        return (scale_A + shr_A) + (scale_B + shr_B)

    # Used for old SeeDot.
    def getIntervalForExp(self, scale: int, intv):  # int^2 -> int^2
        (m, M) = intv
        assert m < np.ldexp(self.MAX_VAL_EXP, -scale)
        M = min(M, np.ldexp(self.MAX_VAL_EXP, -scale))
        return self.getInterval(scale, np.exp(np.ldexp(m, scale)), np.exp(np.ldexp(M, scale)))

    # Used for old SeeDot.
    def getShrForMul(self, scale_A, scale_B):
        shr1, shr2 = config.wordLength // 2, (config.wordLength // 2) - 1
        pRes = (scale_A + shr1) + (scale_B + shr2)

        if pRes <= self.MAX_SCALE:
            if scale_A <= scale_B:
                shrA, shrB = shr1, shr2
            else:
                shrA, shrB = shr2, shr1
            return [shrA, shrB]
        else:
            save = abs(abs(pRes) - abs(self.MAX_SCALE))
            if save % 2 == 1:
                shr1 -= 1
                save -= 1
            save = save // 2
            if scale_A <= scale_B:
                shrA = max(shr1 - save, 0)
                shrB = max(shr2 - save, 0)
            else:
                shrA = max(shr2 - save, 0)
                shrB = max(shr1 - save, 0)

            return [shrA, shrB]

    def getNumInFixedPoint(self, num_float, scale):
        # num_float as python int
        num_py = int(np.ldexp(num_float, -scale))
        assert np.iinfo(IR.DataType.getIntClass()).min <= num_py <= np.iinfo(IR.DataType.getIntClass(
        )).max, "%.6f in fixed-point representation using numpy will overflow." % (num_float)
        # num as numpy int
        num_np = IR.DataType.getInt(num_py)
        # num as SeeDot int
        num_ir = IR.Int(num_np)
        return num_ir

    # Used for old SeeDot.
    def updateTanhIntv(self, intv_A, intv_tanh):
        minVal_A, maxVal_A = intv_A
        minVal_tanh, maxVal_tanh = intv_tanh
        return min(minVal_A, minVal_tanh), min(maxVal_A, maxVal_tanh)

    # Variable and iterators creation.
    def getTempVars(self, num: int):
        return [self.getTempVar() for i in range(num)]

    def getTempVar(self):
        var = IR.Var('tmp' + str(self.counter_var))
        self.counter_var += 1
        return var

    def getTempIterators(self, num: int):
        return [self.getTempIterator() for i in range(num)]

    def getTempIterator(self):
        var = IR.Var('i' + str(self.counter_iter))
        self.counter_iter += 1
        return var

    # Return scaling parameters as required for codegen.
    def formatShr(self, num, shrt=getShrType(), saturate=True):
        assert num >= 0

        shrType = shrt 
        if shrType == "shr" or shrType == "shr+":
            return IR.Int(num)
        elif shrType == "div":
            if num >= config.wordLength - 1 and saturate:
                return IR.Int(IR.Int.max())
            else:
                intVar = IR.Int(2 ** num)
                if intVar.n == 0:
                    assert False
                return intVar
        else:
            assert False

    # Get bit-width of temporary variables used in multiplication.
    def getTempBitwidth(self, bitwidthA, bitwidthB, op, bitwidthC=None):
        if op == "mul":
            assert bitwidthC is None, "Illegal call to getTempBitwidth()"
            biggerBitWidth = max(bitwidthA, bitwidthB)
            return biggerBitWidth * 2
        elif op == "add":
            assert bitwidthC is not None, "Illegal call to getTempBitwidth()"
            biggerBitWidth = max(bitwidthA, bitwidthB, bitwidthC)
            return biggerBitWidth * 2 if isSaturate() else biggerBitWidth
        else:
            assert False, "Illegal operation specified for temp bitwidth"

    # Compute scale for addition and subtraction given input and output scales.
    def getScaleForAddAndSub(self, scale_A:int, scale_B:int,scale_out:int, op_fn):
        if op_fn == operator.add or op_fn == operator.sub:
            if scale_out is None:
                if scale_A >= scale_B:
                    scale_out = scale_A
                else:
                    scale_out = scale_B
                if scale_out < self.MAX_SCALE:
                    scale_out += 1

            if scale_A >= scale_B:
                shr_all = [0, scale_A - scale_B, 0]
                scale_common = scale_A
            else:
                shr_all = [scale_B - scale_A, 0, 0]
                scale_common = scale_B

            diff = scale_out - scale_common
            if diff >= 0:
                shr_all[2] += diff
                scale_common = scale_out
                return (scale_common, (0,0), shr_all)
            else:
                return (scale_common, (0,0), shr_all)
        else:
            assert False, "Op_fn can be add or sub only"

    # Compute multiplication of hyperparameters given input and output scales and bit-widths.
    def getShrTreeSumAndDemoteParamsForMul(self,
            bitwidth_in_A, scale_in_A,
            bitwidth_in_B, scale_in_B,
            bitwidth_temp, scale_temp,
            bitwidth_out, scale_out,
            hiddenDim
        ):
        bitwidth_mul = max(bitwidth_in_A, bitwidth_in_B) 
        bitwidth_mul = bitwidth_mul * 2 if config.vbwEnabled else bitwidth_mul
        height = int(np.ceil(np.log2(hiddenDim)))
        if forFloat():
            return 1, 1, 0, height, 1, scale_out if scale_out is not None else 0
        if scale_temp is None and scale_out is None:
            scale_temp = min(scale_in_A + scale_in_B + config.wordLength - 1 + height, self.MAX_SCALE)
            scale_out = scale_temp
        elif scale_temp is None or scale_out is None:
            assert False, "Illegal state, check function arguments"
        totalShr = 0
        # Raw multiplication.
        scaleAfterMulOp = scale_in_A + scale_in_B
        bitsAfterMulOp = bitwidth_in_A + bitwidth_in_B - 1
        # Saving raw multiplication result into a variable.
        bitsAfterMulStore = bitwidth_mul
        scaleAfterMulStore = min(scaleAfterMulOp + max(bitsAfterMulOp - bitsAfterMulStore, 0), self.MAX_SCALE - max(bitsAfterMulStore - config.wordLength, 0))
        totalShr += (scaleAfterMulStore - scaleAfterMulOp)
        if scale_in_A <= scale_in_B:
            shr_B, shr_A = totalShr // 2, totalShr - totalShr // 2
        else:
            shr_A, shr_B = totalShr // 2, totalShr - totalShr // 2
        assert totalShr <= config.wordLength - 1, "Values wont fit in Stage 2 of MatMul"
        # After addition.
        bitsAfterAddOp = bitwidth_temp
        scaleAfterAddOp = max(scale_temp, scaleAfterMulStore)
        totalShr += (scaleAfterAddOp - scaleAfterMulStore)
        H1 = (scaleAfterAddOp - scaleAfterMulStore)
        assert H1 >= 0, "Invalid state"
        # After adjusting according to bitwidth of output.
        bitsAfterAddStore = bitwidth_out
        scaleAfterAddStore = scale_out
        totalShr += (scaleAfterAddStore - scaleAfterAddOp)
        # Last stage, adjusting scale to avoid invalid values.
        demote = totalShr - shr_A - shr_B - H1
        if height < H1:
            getLogger().info("Rolling back H1 in matrix multiplication. Current H1: %d height: %d" % (H1, height))
        if height < H1:
            diff = H1 - height
            H1 = height
            if shr_A >= shr_B:
                shr_A += diff // 2
                shr_B += diff - diff // 2
            else:
                shr_B += diff // 2
                shr_A += diff - diff // 2
        if demote < 0:
            getLogger().info("Rolling back shr in matrix multiplication. Current demote: %d" % (demote))
        if demote < 0:
            if demote + H1 >= 0:
                H1 += demote
                demote = totalShr - shr_A - shr_B - H1
            else:
                H1 = 0
                demote = totalShr - shr_A - shr_B - H1
                if demote + shr_A + shr_B >= 0:
                    toAdd = demote
                    shr_A += toAdd // 2
                    shr_B += toAdd - toAdd // 2
                    demote = totalShr - shr_A - shr_B - H1
                else:
                    shr_A = 0
                    shr_B = 0
                    demote = 0
                    scale_out = scale_in_A + scale_in_B
        demote = 2 ** demote
        assert shr_A >= 0 and shr_B >= 0, "Invalid state"
        return shr_A, shr_B, H1, height - H1, demote, scale_out

    # If a variable is demoted, get its scale offset.
    def getOffsetForDemotedVariable(self, varName):
        if forFloat():
            return 0
        if self.ddsEnabled:
            while varName not in self.independentBitwidthVars:
                if varName in self.substitutions:
                    varName = self.substitutions[varName]
                else:
                    break
        return self.demotedVarsOffsets.get(varName, 0)

    # For any variable, get its bitwidth and scale given the bitwidth assignment.
    def getBitwidthAndScale(self, varName, native=False):
        if forFloat():
            return 0,0

        if self.ddsEnabled or self.vbwEnabled: # If not enabled, all scales statically computed.
            while varName in self.substitutions:
                varName = self.substitutions[varName]

        if varName in self.varScales.keys(): # Function has been called on this variable or scale has been manually computed.
            if varName in self.demotedVarsList:
                return config.wordLength // 2, self.varScales[varName]
            else:
                return config.wordLength, self.varScales[varName]
        elif varName in self.intermediateVarScales.keys(): # This will be populated for DDS mode.
            if varName in self.demotedVarsList and native == False:
                return config.wordLength // 2, self.intermediateVarScales[varName] + config.wordLength // 2 + self.demotedVarsOffsets[varName]
            else:
                return config.wordLength, self.intermediateVarScales[varName]
        else:
            assert False, "No root found"

    # Update the last instruction where a variable was active / used.
    def updateLiveRange(self, exprs, counter_inst= -1):
        if counter_inst == -1:
            counter_inst = self.counter_inst
        if not isinstance(exprs, list):
            exprs = [exprs]
        for expr in exprs:
            if hasattr(expr, 'idf'):
                varName = expr.idf
                if forFixed():
                    while varName in self.substitutions:
                        varName = self.substitutions[varName]
                if varName in self.varLiveIntervals.keys():
                    self.varLiveIntervals[varName][1] = counter_inst
                else:
                    self.varLiveIntervals[varName] = [counter_inst, counter_inst]

    # Set that two variables will use the same memory location in the generated code, if x86MemoryOptimize is True.
    def setMemorySharableVariables(self, expr_in, expr_out):
        assert hasattr(expr_in, 'idf') and hasattr(expr_out, 'idf'), "Illegal State"
        expr_in_idf = expr_in.idf
        expr_out_idf = expr_out.idf
        while expr_in_idf in self.substitutions:
            expr_in_idf = self.substitutions[expr_in_idf]
        while expr_out_idf in self.substitutions:
            expr_out_idf = self.substitutions[expr_out_idf]
        self.coLocatedVariables[expr_in_idf] = expr_out_idf
