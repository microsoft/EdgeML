# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import argparse
import datetime
from distutils.dir_util import copy_tree
import os
import shutil
import sys
import operator
import tempfile
import traceback
from tqdm import tqdm
import numpy as np

from seedot.compiler.converter.converter import Converter

import seedot.config as config
from seedot.compiler.compiler import Compiler
from seedot.predictor import Predictor
import seedot.util as Util

'''
Overall compiler logic is maintained in this file. Please refer to architecture.md for a 
detailed explanation of how the various modules interact with each other.
'''


class Main:

    def __init__(self, algo, encoding, target, trainingFile, testingFile, modelDir, sf, metric, dataset, numOutputs, source):
        self.algo, self.encoding, self.target = algo, encoding, target
        self.trainingFile, self.testingFile, self.modelDir = trainingFile, testingFile, modelDir
        self.sf = sf
            # MaxScale factor. Used in the original version of SeeDot.
            # Refer to PLDI'19 paper: maxscale parameter P.
        self.dataset = dataset
            # Dataset which is being evaluated.
        self.accuracy = {}
            # SeeDot examines accuracy of multiple codes.
            # This variable contains a map from code ID -> corresponding accuracy.
        self.metric = metric
            # This can be accuracy, disagreements (see OOPSLA'20 paper: disagreement ratio) or
            # reduced disagreement (disagreement ratio for only those parameters where float model prediction is correct).
        self.numOutputs = numOutputs
            # Number of outputs, it is 1 for a single-class prediction. For n simultaneous predictions, it is n.
        self.source = source
            # SeeDot or ONNX or TensorFlow.
        self.variableSubstitutions = {}
            # Evaluated during profiling code run (runForFloat). During compilation, variable names get substituted
            # into other names, which is stored in this variable. It is required in the case that an attribute
            # computed for some variable might have to be propagated to other variables, like scales and bitwidths.
        self.scalesForX = {}
            # Populated for multiple code generation (performSearch: if vbwEnabled is True). SeeDot carries out an
            # exploration (OOPSLA'20 paper, Section 6.2) where multiple codes are compiled and evaluated. This variable
            # stored a map from code ID -> corresponding scale of input variable 'X'.
        self.scalesForY = {}
            # Populated for multiple code generation (performSearch: if vbwEnabled is True). SeeDot carries out an
            # exploration (OOPSLA'20 paper, Section 6.2) where multiple codes are compiled and evaluated. This variable
            # stored a map from code ID -> corresponding scale of output variable 'Y' (if the problem is regression,
            # for classification problems 'Y' is already an integer hence does not need to be scaled).
        self.problemType = config.ProblemType.default
            # Edited when converter module determines whether problem is regression or classification.
        self.variableToBitwidthMap = {}
            # This variable holds the bit-width assignment for all variables in the code. The keys (variable names)
            # are identified during profiling run (runForFloat) and the values (bitwidths) are evaluated during
            # multiple code generation (performSearch: if vbwEnabled is True). By default, the bit-widths are set to 16.
        self.sparseMatrixSizes = {}
            # Populated during profiling code run (runForFloat). For sparse matrix multiplication, the matrices
            # in the generated code have different sizes than in the input code due to CSR represenation, and
            # the generated matrix sizes are stored here.
        self.varDemoteDetails = []
            # Populated during variable demotion in VBW mode. This stores the resultant code performance
            # when a subset of variables is demoted.
        self.flAccuracy = -1
            # Populated during profiling code run. Accuracy of the floating point code.
        self.allScales = {}
            # This stores the final scale assignment of every variable in the code (considering their
            # bit-width assignments). This variable is updated after every code generated, so eventually it
            # is populated with the bit-width assignment of the final generated code.
        self.demotedVarsList = []
            # Populated in VBW mode after exploration is completed. After the compiler determines the variables
            # to be demoted, this variable is populated with a list of those variables.
        self.demotedVarsOffsets = {}
            # Populated in VBW mode after exploration is completed. After the compiler determines the variables
            # to be demoted, this variable is populated with a map of variables to the scale offset which gives the best accuracy.
        self.biasShifts = {}
            # For simplifying bias addition, populated after every code run, used for M3 codegen.
            # In operations like WX + B, B is mostly used once in the code. So all the fixed point computations are clubbed into one.
        self.varSizes = {}
            # Map from a variable to number of elements it holds. Populated in floating point mode.

    # This function is invoked right at the beginning for moving around files into the working directory.
    def setup(self):
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        copy_tree(os.path.join(curr_dir, "Predictor"), os.path.join(config.tempdir, "Predictor"))

        if self.target == config.Target.arduino:
            for fileName in ["arduino.ino", "config.h", "predict.h"]:
                srcFile = os.path.join(curr_dir, "arduino", fileName)
                destFile = os.path.join(config.outdir, fileName)
                shutil.copyfile(srcFile, destFile)
        elif self.target == config.Target.m3:
            for fileName in ["datatypes.h", "mbconv.h", "utils.h"]:
                srcFile = os.path.join(curr_dir, "..", "..", "..",  "c_reference", "include", "quantized_%s"%fileName)
                destFile = os.path.join(config.outdir, "quantized_%s"%fileName)
                shutil.copyfile(srcFile, destFile)
            for fileName in ["mbconv.c", "utils.c"]:
                srcFile = os.path.join(curr_dir, "..", "..", "..",  "c_reference", "src", "quantized_%s"%fileName)
                destFile = os.path.join(config.outdir, "quantized_%s"%fileName)
                shutil.copyfile(srcFile, destFile)
            for fileName in ["main.c", "predict.h", "Makefile"]:
                srcFile = os.path.join(curr_dir, "m3", fileName)
                destFile = os.path.join(config.outdir, fileName)
                shutil.copyfile(srcFile, destFile)

    def get_input_file(self):
        if self.source == config.Source.seedot:
            return os.path.join(self.modelDir, "input.sd")
        elif self.source == config.Source.onnx:
            return os.path.join(self.modelDir, "input.onnx")
        else:
            return os.path.join(self.modelDir, "input.pb")

    # Generates one particular fixed-point or floating-point code.
    # Arguments:
    #   encoding:                float or fixed
    #   target:                 target device (x86, arduino or m3)
    #   sf:                     maxScale factor (check description above)
    #   The next three parameters are used to control how many candidate codes are built at once. Generating
    #   multiple codes at once and building them simultaneously helps avoid multiple build overheads as well
    #   as saves data processing overhead at runtime.
    #   generateAllFiles:       if True, it generates multiple files like datasets, configuration etc. if False,
    #                           only generates the inference code.
    #   id:                     Multiple inference codes are designated by a code ID 'N' (function names are
    #                           seedot_fixed_'N').
    #   printSwitch:            whether or not to print a switch between multiple inference codes (only needs to
    #                           be True at the last code being generated as the switch is print right after that).
    #   scaleForX:              scale of input X for the particular fixed-point code.
    #   variableToBitwidthMap:  bitwidth assignments for the particular fixed-point code.
    #   demotedVarsList:        set of variables using 8-bits in the particular fixed-point code.
    #   demotedVarsOffsets:     map from variables to scale offsets for particular fixed-point code.
    #   paramInNativeBitwidth:  if False, it means model parameters are stored as 8-bit/16-bit integers mixed.
    #                           If True, it means model parameters are stored as 16-bit integers only (16 is native bit-width).
    def compile(self, encoding, target, sf, generateAllFiles=True, id=None, printSwitch=-1, scaleForX=None, variableToBitwidthMap=None, demotedVarsList=[], demotedVarsOffsets={}, paramInNativeBitwidth=True):
        Util.getLogger().debug("Generating code...")

        if variableToBitwidthMap is None:
            variableToBitwidthMap = dict(self.variableToBitwidthMap)

        # Set input and output files.
        inputFile = self.get_input_file()
        profileLogFile = os.path.join(
            config.tempdir, "Predictor", "output", "float", "profile.txt")

        logDir = os.path.join(config.outdir, "output")
        os.makedirs(logDir, exist_ok=True)
        if encoding == config.Encoding.floatt:
            outputLogFile = os.path.join(logDir, "log-float.txt")
        else:
            if config.ddsEnabled:
                outputLogFile = os.path.join(logDir, "log-fixed-" + str(abs(scaleForX)) + ".txt")
            else:
                outputLogFile = os.path.join(logDir, "log-fixed-" + str(abs(sf)) + ".txt")

        if target == config.Target.arduino:
            outdir = os.path.join(config.outdir, str(config.wordLength), self.algo, self.dataset)
            os.makedirs(outdir, exist_ok=True)
            outputDir = os.path.join(outdir)
        elif target == config.Target.m3:
            outdir = os.path.join(config.outdir)
            os.makedirs(outdir, exist_ok=True)
            outputDir = os.path.join(outdir)
        elif target == config.Target.x86:
            outputDir = os.path.join(config.tempdir, "Predictor")

        obj = Compiler(self.algo, encoding, target, inputFile, outputDir,
                        profileLogFile, sf, self.source, outputLogFile,
                        generateAllFiles, id, printSwitch, self.variableSubstitutions,
                        scaleForX,
                        variableToBitwidthMap, self.sparseMatrixSizes, demotedVarsList, demotedVarsOffsets,
                        paramInNativeBitwidth)
        obj.run()
        self.biasShifts = obj.biasShifts
        self.allScales = dict(obj.varScales)
        if encoding == config.Encoding.floatt:
            self.variableSubstitutions = obj.substitutions
            self.variableToBitwidthMap = dict.fromkeys(obj.independentVars, config.wordLength)
            self.varSizes = obj.varSizes

        self.problemType = obj.problemType
        if id is None:
            self.scaleForX = obj.scaleForX
            self.scaleForY = obj.scaleForY
        else:
            self.scalesForX[id] = obj.scaleForX
            self.scalesForY[id] = obj.scaleForY

        Util.getLogger().debug("Completed")
        return True

    # Runs the converter project to generate the input files using reading the training model.
    # Arguments:
    #   version:                float or fixed.
    #   datasetType:            train or test.
    #   target:                 target device (x86, arduino or m3).
    #   varsForBitwidth:        bitwidth assignments used to generate model files. If none,
    #                           default bitwidth 16 used for all variables.
    #   demotedVarsOffsets:     Keys are list of variables which use 8 bits.
    def convert(self, encoding, datasetType, target, varsForBitwidth={}, demotedVarsOffsets={}):
        Util.getLogger().debug("Generating input files for %s %s dataset..." %
              (encoding, datasetType))

        # Create output dirs.
        if target == config.Target.arduino:
            outputDir = os.path.join(config.outdir, "input")
            datasetOutputDir = outputDir
        elif target == config.Target.m3:
            outputDir = os.path.join(config.outdir, "input")
            datasetOutputDir = outputDir
        elif target == config.Target.x86:
            outputDir = os.path.join(config.tempdir, "Predictor")
            datasetOutputDir = os.path.join(config.tempdir, "Predictor", "input")
        else:
            assert False

        os.makedirs(datasetOutputDir, exist_ok=True)
        os.makedirs(outputDir, exist_ok=True)

        inputFile = self.get_input_file()

        try:
            varsForBitwidth = dict(varsForBitwidth)
            for var in demotedVarsOffsets:
                varsForBitwidth[var] = config.wordLength // 2
            obj = Converter(self.algo, encoding, datasetType, target, self.source,
                            datasetOutputDir, outputDir, varsForBitwidth, self.allScales, self.numOutputs, self.biasShifts, self.scaleForY if hasattr(self, "scaleForY") else None)
            obj.setInput(inputFile, self.modelDir,
                         self.trainingFile, self.testingFile)
            obj.run()
            if encoding == config.Encoding.floatt:
                self.sparseMatrixSizes = obj.sparseMatrixSizes
        except Exception as e:
            traceback.print_exc()
            return False

        Util.getLogger().debug("Done")
        return True

    # Build and run the Predictor project.
    def predict(self, encoding, datasetType):
        outputDir = os.path.join("output", encoding)

        curDir = os.getcwd()
        os.chdir(os.path.join(config.tempdir, "Predictor"))

        obj = Predictor(self.algo, encoding, datasetType,
                        outputDir, self.scaleForX, self.scalesForX, self.scaleForY, self.scalesForY, self.problemType, self.numOutputs)
        execMap = obj.run()

        os.chdir(curDir)
        return execMap

    # Compile and run the generated code once for a given scaling factor.
    # The arguments are explain in the description of self.compile().
    # The function is named partial compile as in one C++ output file multiple inference codes are generated.
    # One invocation of partialCompile generates only one of the multiple inference codes.
    def partialCompile(self, encoding, target, scale, generateAllFiles, id, printSwitch, variableToBitwidthMap=None, demotedVarsList=[], demotedVarsOffsets={}, paramInNativeBitwidth=True):
        if config.ddsEnabled:
            res = self.compile(encoding, target, None, generateAllFiles, id, printSwitch, scale, variableToBitwidthMap, demotedVarsList, demotedVarsOffsets, paramInNativeBitwidth)
        else:
            res = self.compile(encoding, target, scale, generateAllFiles, id, printSwitch, None, variableToBitwidthMap, demotedVarsList, demotedVarsOffsets, paramInNativeBitwidth)
        if res == False:
            return False
        else:
            return True

    # Runs the C++ file which contains multiple inference codes. Reads the output of all inference codes,
    # arranges them and returns a map of inference code descriptor to performance.
    def runAll(self, encoding, datasetType, codeIdToScaleFactorMap, demotedVarsToOffsetToCodeId=None, doNotSort=False, printAlso=False):
        execMap = self.predict(encoding, datasetType)
        if execMap == None:
            return False, True

        # Used by test module.
        if self.algo == config.Algo.test:
            for codeId, sf in codeIdToScaleFactorMap.items():
                self.accuracy[sf] = execMap[str(codeId)]
                Util.getLogger().debug("The 95th percentile error for sf" + str(sf) + "with respect to dataset is " + str(execMap[str(codeId)][0]) + "%.")
                Util.getLogger().debug("The 95th percentile error for sf" + str(sf) + "with respect to float execution is " + str(execMap[str(codeId)][1]) + "%.\n")
            return True, False

        # During the third exploration phase, when multiple codes are generated at once, codeIdToScaleFactorMap
        # is populated with the codeID to the code description (bitwidth assignments of different variables).
        # After executing the code, print out the accuracy of the code against the code ID.
        if codeIdToScaleFactorMap is not None:
            for codeId, sf in codeIdToScaleFactorMap.items():
                self.accuracy[sf] = execMap[str(codeId)]
                if printAlso:
                    print("Accuracy at scale factor %d is %.3f%%, Disagreement Count is %d, Reduced Disagreement Count is %d" % (sf, execMap[str(codeId)][0], execMap[str(codeId)][1], execMap[str(codeId)][2]))
                else:
                    Util.getLogger().info("Accuracy at scale factor %d is %.3f%%, Disagreement Count is %d, Reduced Disagreement Count is %d\n" % (sf, execMap[str(codeId)][0], execMap[str(codeId)][1], execMap[str(codeId)][2]))
                if datasetType == config.DatasetType.testing and self.target == config.Target.arduino:
                    outdir = os.path.join(config.outdir, str(config.wordLength), self.algo, self.dataset)
                    os.makedirs(outdir, exist_ok=True)
                    file = open(os.path.join(outdir, "res"), "w")
                    file.write("Demoted Vars:\n")
                    file.write(str(self.demotedVarsOffsets) if hasattr(self, 'demotedVarsOffsets') else "")
                    file.write("\nAll scales:\n")
                    file.write(str(self.allScales))
                    file.write("\nAccuracy at scale factor %d is %.3f%%, Disagreement Count is %d, Reduced Disagreement Count is %d\n" % (sf, execMap[str(codeId)][0], execMap[str(codeId)][1], execMap[str(codeId)][2]))
                    file.close()
        else:
            # During fourth exploration phase, when the accuracy drops of every variable is known, the variables are cumulatively demoted
            # in order of better accuracy/disagreement count which is handled in this block.
            def getMetricValue(a):
                if self.metric == config.Metric.accuracy:
                    return (a[1][0], -a[1][1], -a[1][2])
                elif self.metric == config.Metric.disagreements:
                    return (-a[1][1], -a[1][2], a[1][0])
                elif self.metric == config.Metric.reducedDisagreements:
                    return (-a[1][2], -a[1][1], a[1][0])
            allVars = []
            for demotedVars in demotedVarsToOffsetToCodeId:
                offsetToCodeId = demotedVarsToOffsetToCodeId[demotedVars]
                Util.getLogger().debug("Demoted vars: %s\n" % str(demotedVars))

                x = [(i, execMap[str(offsetToCodeId[i])]) for i in offsetToCodeId]
                x.sort(key=getMetricValue, reverse=True)
                allVars.append(((demotedVars, x[0][0]), x[0][1]))

                for offset in offsetToCodeId:
                    codeId = offsetToCodeId[offset]
                    Util.getLogger().debug("Offset %d (Code ID %d): Accuracy %.3f%%, Disagreement Count %d, Reduced Disagreement Count %d\n" %(offset, codeId, execMap[str(codeId)][0], execMap[str(codeId)][1], execMap[str(codeId)][2]))
            self.varDemoteDetails += allVars
            # For the sec
            if not doNotSort:
                self.varDemoteDetails.sort(key=getMetricValue, reverse=True)
        return True, False

    # This function performs an exploration and determines the scales and bit-widths of all variables. Described in OOPSLA'20 Paper Section 6.2.
    # The exploration is across 4 stages:
    # STAGE I: For input 'X', we perform an exploration trying out scales from 0 to -15. (First stage exploration in this function).
    # STAGE II: Determining Scale all other variables in 16 bits.
    #   For all variables except input 'X', the scale is dynamically computed using the dataset (OOPSLA'20 paper, Section 6.1).
    #   This data is captured during the floating-point code run (collectProfileData), not performSearch.
    #   During collectProfileData() call, self.allScales is populated which contains data driven scaling info.
    # STAGE III: Determining accuracy when each variable is demoted to 8 bits one at a time.
    #   Multiple fixed point codes are generated, each with one variable demoted, are generated. Data driven scales do not work for 8-bits,
    #   so we try out multiple scales (config.offsetsPerDemotedVariable different scales).
    # STAGE IV: Cumulatively demoting variables one after the other to reduce latency and model size, as long as accuracy remains good.
    def performSearch(self):
        start, end = config.maxScaleRange
        lastStageAcc = -1

        fixedPointCounter = 0
        while True:
            # STAGE I exploration.
            print("Stage I Exploration: Determining scale for input \'X\'...")
            fixedPointCounter += 1
            if config.fixedPointVbwIteration:
                Util.getLogger().debug("Will compile until conversion to fixed point. Iteration %d"%fixedPointCounter)
            highestValidScale = start
            firstCompileSuccess = False
            # Bar longer than actually required
            stage_1_bar = tqdm(total=(2 * abs(start - end) + 2), mininterval=0, miniters=1, leave=True)
            while firstCompileSuccess == False:
                if highestValidScale == end:
                    Util.getLogger().error("Compilation not possible for any scale factor of variable \'X\'. Aborting code!")
                    return False

                # Refactor and remove this try/catch block in the future.
                try:
                    firstCompileSuccess = self.partialCompile(config.Encoding.fixed, config.Target.x86, highestValidScale, True, None, 0, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))
                except:
                    firstCompileSuccess = False

                if firstCompileSuccess:
                    stage_1_bar.update(highestValidScale - end + 1)
                    break
                highestValidScale -= 1
                stage_1_bar.update(1)
            
            lowestValidScale = end + 1
            firstCompileSuccess = False
            while firstCompileSuccess == False:
                try:
                    firstCompileSuccess = self.partialCompile(config.Encoding.fixed, config.Target.x86, lowestValidScale, True, None, 0, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))
                except:
                    firstCompileSuccess = False
                if firstCompileSuccess:
                    stage_1_bar.update(start - lowestValidScale + 2)
                    break
                lowestValidScale += 1
                stage_1_bar.update(1)
            stage_1_bar.close()

            # Ignored.
            self.partialCompile(config.Encoding.fixed, config.Target.x86, lowestValidScale, True, None, -1, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))

            print("Stage II Exploration: Determining scale for all non-\'X\' variables...")
            # The iterator logic is as follows:
            # Search begins when the first valid scaling factor is found (runOnce returns True).
            # Search ends when the execution fails on a particular scaling factor (runOnce returns False).
            # This is the window where valid scaling factors exist and we
            # select the one with the best accuracy.
            numCodes = highestValidScale - lowestValidScale + 1
            codeId = 0
            codeIdToScaleFactorMap = {}
            for i in tqdm(range(highestValidScale, lowestValidScale - 1, -1)):
                if config.ddsEnabled:
                    Util.getLogger().debug("Testing with DDS and scale of X as " + str(i) + "\n")
                else:
                    Util.getLogger().debug("Testing with max scale factor of " + str(i) + "\n")

                codeId += 1
                try:
                    compiled = self.partialCompile(
                        config.Encoding.fixed, config.Target.x86, i, False, codeId, -1 if codeId != numCodes else codeId, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))
                except: # If some code in the middle fails to compile.
                    codeId -=1
                    continue
                if compiled == False:
                    return False
                codeIdToScaleFactorMap[codeId] = i

            print("Stage II Code Run Started...")
            res, exit = self.runAll(config.Encoding.fixed, config.DatasetType.training, codeIdToScaleFactorMap)
            print("Stage II Code Run Completed!\n")
            if exit == True or res == False:
                return False

            Util.getLogger().info("\nSearch completed\n")
            Util.getLogger().info("----------------------------------------------\n\n")
            Util.getLogger().info("Best performing scaling factors with accuracy, disagreement, reduced disagreement:")

            self.sf = self.getBestScale()
            if self.accuracy[self.sf][0] != lastStageAcc:
                lastStageAcc = self.accuracy[self.sf][0]
            elif config.fixedPointVbwIteration:
                Util.getLogger().info("No difference in iteration %d Stage 2 and iteration %d Stage 1. Stopping search\n"%(fixedPointCounter-1, fixedPointCounter))
                break

            if config.vbwEnabled:
                # Stage III exploration.
                print("Stage III Exploration: Demoting variables one at a time...")

                assert config.ddsEnabled, "Currently VBW on maxscale not supported"
                if config.wordLength != 16:
                    assert False, "VBW mode only supported if native bitwidth is 16"
                Util.getLogger().debug("Scales computed in native bitwidth. Starting exploration over other bitwidths.")

                # We attempt to demote all possible variables in the code. We try out multiple different scales
                # (controlled by config.offsetsPerDemotedVariable) for each demoted variable. When a variable is
                # demoted, it is assigned a scale given by :
                # demoted Scale = self.allScales[var] + 8 - offset

                attemptToDemote = [var for var in self.variableToBitwidthMap if (var[-3:] != "val" and var not in self.demotedVarsList)]
                tmpAttemptToDemote = []
                for var in attemptToDemote:
                    tmp_var = var
                    while tmp_var in self.variableSubstitutions.keys():
                        tmp_var = self.variableSubstitutions[tmp_var]
                    if tmp_var not in tmpAttemptToDemote:
                        tmpAttemptToDemote.append(tmp_var)
                    else:
                        print("Repeated Var: " + tmp_var)
                attemptToDemote = tmpAttemptToDemote
                numCodes = config.offsetsPerDemotedVariable * len(attemptToDemote) + ((9 - config.offsetsPerDemotedVariable) if 'X' in attemptToDemote else 0)
                # 9 offsets tried for X while 'offsetsPerDemotedVariable' tried for other variables.

                # We approximately club batchSize number of codes in one generated C++ code, so that one generated code does
                # not become too large.
                batchSize = int(np.ceil(50 / np.ceil(len(attemptToDemote) / 50)))
                redBatchSize = np.max((batchSize, 16)) / config.offsetsPerDemotedVariable

                totalSize = len(attemptToDemote)
                numBatches = int(np.ceil(totalSize / redBatchSize))

                self.varDemoteDetails = []
                for i in tqdm(range(numBatches)):
                    Util.getLogger().info("=====\nBatch %i out of %d\n=====\n" %(i + 1, numBatches))

                    firstVarIndex = (totalSize * i) // numBatches
                    lastVarIndex = (totalSize * (i + 1)) // numBatches
                    demoteBatch = [attemptToDemote[i] for i in range(firstVarIndex, lastVarIndex)]
                    numCodes = config.offsetsPerDemotedVariable * len(demoteBatch) + ((9 - config.offsetsPerDemotedVariable) if 'X' in demoteBatch else 0)
                    # 9 offsets tried for X while 'config.offsetsPerDemotedVariable' tried for other variables.

                    self.partialCompile(config.Encoding.fixed, config.Target.x86, self.sf, True, None, -1 if len(demoteBatch) > 0 else 0, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))
                    codeId = 0
                    contentToCodeIdMap = {}

                    for demoteVar in demoteBatch:
                        # For each variable being demoted, we populate some variables containing information regarding demoted variable.
                        newbitwidths = dict(self.variableToBitwidthMap)
                        newbitwidths[demoteVar] = config.wordLength // 2
                        if demoteVar + "val" in newbitwidths:
                            newbitwidths[demoteVar + "val"] = config.wordLength // 2
                        for alreadyDemotedVars in self.demotedVarsList: # In subsequent iterations during fixed point compilation, this variable will have the variables demoted during the previous runs.
                            newbitwidths[alreadyDemotedVars] = config.wordLength // 2
                        demotedVarsList = [i for i in newbitwidths.keys() if newbitwidths[i] != config.wordLength]
                        demotedVarsOffsets = {}
                        for key in self.demotedVarsList:
                            demotedVarsOffsets[key] = self.demotedVarsOffsets[key]

                        contentToCodeIdMap[tuple(demotedVarsList)] = {}
                        # We try out multiple offsets for each variable to find best scale assignment for each variable.
                        for demOffset in (range(0, -config.offsetsPerDemotedVariable, -1) if demoteVar != 'X' else range(0, -9, -1)):
                            codeId += 1
                            for k in demotedVarsList:
                                if k not in self.demotedVarsList:
                                    demotedVarsOffsets[k] = demOffset
                            contentToCodeIdMap[tuple(demotedVarsList)][demOffset] = codeId
                            compiled = self.partialCompile(config.Encoding.fixed, config.Target.x86, self.sf, False, codeId, -1 if codeId != numCodes else codeId, dict(newbitwidths), list(demotedVarsList), dict(demotedVarsOffsets))
                            if compiled == False:
                                Util.getLogger().error("Variable bitwidth exploration resulted in a compilation error\n")
                                return False

                    res, exit = self.runAll(config.Encoding.fixed, config.DatasetType.training, None, contentToCodeIdMap)
                
                print("Stage IV Exploration: Cumulatively demoting variables...")
                # Stage IV exploration.
                # Again, we compute only a limited number of inference codes per generated C++ so as to not bloat up the memory usage of the compiler.
                redBatchSize *= config.offsetsPerDemotedVariable
                totalSize = len(self.varDemoteDetails)
                numBatches = int(np.ceil(totalSize / redBatchSize))

                sortedVars1 = []
                sortedVars2 = []
                for ((demoteVars, offset), _) in self.varDemoteDetails:
                    variableInMap = False
                    for demoteVar in demoteVars:
                        if demoteVar in self.varSizes:
                            variableInMap = True
                            if self.varSizes[demoteVar] >= Util.Config.largeVariableLimit:
                                sortedVars1.append((demoteVars, offset))
                                break
                            else:
                                sortedVars2.append((demoteVars, offset))
                                break
                    if not variableInMap:
                        sortedVars2.append((demoteVars, offset))

                sortedVars = sortedVars1 + sortedVars2

                self.varDemoteDetails = []
                demotedVarsOffsets = dict(self.demotedVarsOffsets)
                demotedVarsList = list(self.demotedVarsList)
                demotedVarsListToOffsets = {}

                # Knowing the accuracy when each single variable is demoted to 8-bits one at a time, we proceed to cumulatively
                # demoting all of them one after the other ensuring accuracy of target code does not fall below a threshold. The
                # following for loop controls generation of inference codes.
                for i in tqdm(range(numBatches)):
                    Util.getLogger().info("=====\nBatch %i out of %d\n=====\n" %(i + 1, numBatches))

                    firstVarIndex = (totalSize * i) // numBatches
                    lastVarIndex = (totalSize * (i+1)) // numBatches
                    demoteBatch = [sortedVars[i] for i in range(firstVarIndex, lastVarIndex)]

                    self.partialCompile(config.Encoding.fixed, config.Target.x86, self.sf, True, None, -1 if len(attemptToDemote) > 0 else 0, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))
                    contentToCodeIdMap = {}
                    codeId = 0
                    numCodes = len(demoteBatch)
                    for (demoteVars, offset) in demoteBatch:
                        newbitwidths = dict(self.variableToBitwidthMap)
                        for var in demoteVars:
                            if var not in self.demotedVarsList:
                                newbitwidths[var] = config.wordLength // 2
                                demotedVarsOffsets[var] = offset
                            if var not in demotedVarsList:
                                demotedVarsList.append(var)
                        codeId += 1
                        contentToCodeIdMap[tuple(demotedVarsList)] = {}
                        contentToCodeIdMap[tuple(demotedVarsList)][offset] = codeId
                        demotedVarsListToOffsets[tuple(demotedVarsList)] = dict(demotedVarsOffsets)
                        compiled = self.partialCompile(config.Encoding.fixed, config.Target.x86, self.sf, False, codeId, -1 if codeId != numCodes else codeId, dict(newbitwidths), list(demotedVarsList), dict(demotedVarsOffsets))
                        if compiled == False:
                            Util.getLogger().error("Variable bitwidth exploration resulted in another compilation error\n")
                            return False

                    res, exit = self.runAll(config.Encoding.fixed, config.DatasetType.training, None, contentToCodeIdMap, True)

                if exit == True or res == False:
                    return False

                # The following for loop controls how many variables are actually demoted in the final output code, which has
                # as many variables as possible in 8-bits, while ensuring accuracy drop compared to floating point is reasonable:
                okToDemote = ()
                acceptedAcc = lastStageAcc
                for ((demotedVars, _), metrics) in self.varDemoteDetails:
                    acc = metrics[0]
                    if self.problemType == config.ProblemType.classification and (self.flAccuracy - acc) > config.permittedClassificationAccuracyLoss:
                        break
                    elif self.problemType == config.ProblemType.regression and acc > config.permittedRegressionNumericalLossMargin:
                        break
                    else:
                        okToDemote = demotedVars
                        acceptedAcc = acc

                self.demotedVarsList = [i for i in okToDemote] + [i for i in self.demotedVarsList]
                self.demotedVarsOffsets.update(demotedVarsListToOffsets.get(okToDemote, {}))

                if acceptedAcc != lastStageAcc:
                    lastStageAcc = acceptedAcc
                else:
                    Util.getLogger().warning("No difference in iteration %d's stages 1 & 2. Stopping search."%fixedPointCounter)
                    break

            if not config.vbwEnabled or not config.fixedPointVbwIteration:
                break

        return True

    # Reverse sort the accuracies, print the top 5 accuracies and return the
    # best scaling factor.
    def getBestScale(self):
        def getMaximisingMetricValue(a):
            if self.metric == config.Metric.accuracy:
                return (a[1][0], -a[1][1], -a[1][2]) if not config.higherOffsetBias else (a[1][0], -a[0])
            elif self.metric == config.Metric.disagreements:
                return (-a[1][1], -a[1][2], a[1][0]) if not config.higherOffsetBias else (-max(5, a[1][1]), -a[0])
            elif self.metric == config.Metric.reducedDisagreements:
                return (-a[1][2], -a[1][1], a[1][0]) if not config.higherOffsetBias else (-max(5, a[1][2]), -a[0])
            elif self.algo == config.Algo.test:
                # Minimize regression error.
                return (-a[1][0])

        x = [(i, self.accuracy[i]) for i in self.accuracy]
        x.sort(key=getMaximisingMetricValue, reverse=True)
        sorted_accuracy = x[:5]
        Util.getLogger().info(sorted_accuracy)
        return sorted_accuracy[0][0]

    # Find the scaling factor which works best on the training dataset and
    # predict on the testing dataset.
    def findBestScalingFactor(self):
        print("-------------------------")
        print("Performing Exploration...")
        print("-------------------------\n")

        # Generate input files for training dataset.
        res = self.convert(config.Encoding.fixed,
                           config.DatasetType.training, config.Target.x86)
        if res == False:
            return False

        # Search for the best scaling factor.
        res = self.performSearch()
        if res == False:
            return False

        Util.getLogger().info("Best scaling factor = %d" % (self.sf))
        return True

    # After exploration is completed, this function is invoked to show the performance of the final quantised code on a testing dataset,
    # which is ideally different than the training dataset on which the bitwidth and scale tuning was done.
    def runOnTestingDataset(self):
        print("\n-------------------------------")
        print("Prediction on testing dataset")
        print("-------------------------------\n")

        Util.getLogger().debug("Setting max scaling factor to %d\n" % (self.sf))

        if config.vbwEnabled:
            Util.getLogger().debug("Demoted Vars with Offsets: %s\n" % (str(self.demotedVarsOffsets)))

        # Generate files for the testing dataset.
        res = self.convert(config.Encoding.fixed,
                           config.DatasetType.testing, config.Target.x86)
        if res == False:
            return False

        # Compile and run code using the best scaling factor.
        if config.vbwEnabled:
            compiled = self.partialCompile(config.Encoding.fixed, config.Target.x86, self.sf, True, None, 0, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))
        else:
            compiled = self.partialCompile(config.Encoding.fixed, config.Target.x86, self.sf, True, None, 0)
        if compiled == False:
            return False

        res, exit = self.runAll(config.Encoding.fixed, config.DatasetType.testing, {"default" : self.sf}, printAlso=True)
        if res == False:
            return False

        return True

    # This function is invoked before the exploration to obtain floating point accuracy, as well as profiling each variable
    # in the floating point code to compute their ranges and consequently their fixed-point ranges.
    def collectProfileData(self):
        Util.getLogger().info("-----------------------\n")
        Util.getLogger().info("Collecting profile data\n")
        Util.getLogger().info("-----------------------\n")

        res = self.convert(config.Encoding.floatt,
                           config.DatasetType.training, config.Target.x86)
        if res == False:
            return False

        res = self.compile(config.Encoding.floatt, config.Target.x86, self.sf)
        if res == False:
            return False

        execMap = self.predict(config.Encoding.floatt, config.DatasetType.training)
        if execMap == None:
            return False

        self.flAccuracy = execMap["default"][0]
        Util.getLogger().info("Accuracy is %.3f%%\n" % (execMap["default"][0]))
        Util.getLogger().info("Disagreement is %.3f%%\n" % (execMap["default"][1]))
        Util.getLogger().info("Reduced Disagreement is %.3f%%\n" % (execMap["default"][2]))

    # Generate code for Arduino.
    def compileFixedForTarget(self):
        print("------------------------------")
        print("Generating code for %s..." % (self.target))
        print("------------------------------\n")

        demotedVarsOffsets = dict(self.demotedVarsOffsets) if hasattr(self, 'demotedVarsOffsets') else {}
        variableToBitwidthMap = dict(self.variableToBitwidthMap) if hasattr(self, 'variableToBitwidthMap') else {}
        res = self.convert(config.Encoding.fixed,
                           config.DatasetType.testing, self.target, variableToBitwidthMap, demotedVarsOffsets)
        if res == False:
            return False

        # Copy files.
        if self.target == config.Target.arduino:
            srcFile = os.path.join(config.outdir, "input", "model_fixed.h")
            destFile = os.path.join(config.outdir, str(config.wordLength), self.algo, self.dataset, "model.h")
            os.makedirs(os.path.join(config.outdir, str(config.wordLength), self.algo, self.dataset), exist_ok=True)
        elif self.target == config.Target.m3:
            srcFile = os.path.join(config.outdir, "input", "model_fixed.h")
            destFile = os.path.join(config.outdir, "model.h")
            os.makedirs(os.path.join(config.outdir), exist_ok=True)
            shutil.copyfile(srcFile, destFile)
            srcFile = os.path.join(config.outdir, "input", "scales.h")
            destFile = os.path.join(config.outdir, "scales.h")
        shutil.copyfile(srcFile, destFile)

        # Copy library.h file.
        curr_dir = os.path.dirname(os.path.realpath(__file__))

        if self.target == config.Target.arduino:
            srcFile = os.path.join(curr_dir, self.target, "library", "library_fixed.h")
            destFile = os.path.join(config.outdir, "library.h")
            shutil.copyfile(srcFile, destFile)

        modifiedBitwidths = dict.fromkeys(self.variableToBitwidthMap.keys(), config.wordLength) if hasattr(self, 'variableToBitwidthMap') else {}
        if hasattr(self, 'demotedVarsList'):
            for i in self.demotedVarsList:
                modifiedBitwidths[i] = config.wordLength // 2
        res = self.partialCompile(config.Encoding.fixed, self.target, self.sf, True, None, 0, dict(modifiedBitwidths), list(self.demotedVarsList) if hasattr(self, 'demotedVarsList') else [], dict(demotedVarsOffsets))
        if res == False:
            return False

        return True

    def runForFixed(self):
        # Collect runtime profile.
        res = self.collectProfileData()
        if res == False:
            return False

        # Obtain best scaling factor.
        if self.sf == None:
            res = self.findBestScalingFactor()
            if res == False:
                return False

        res = self.runOnTestingDataset()
        if res == False:
            return False
        else:
            self.testingAccuracy = self.accuracy[self.sf]

        # Generate code for target.
        if self.target != config.Target.x86:
            self.compileFixedForTarget()

            print("%s sketch dumped in the folder %s\n" % (self.target, config.outdir))

        return True

    # Generate Arduino floating point code.
    def compileFloatForTarget(self):
        assert self.target == config.Target.arduino, "Floating point code supported for Arduino only"

        print("------------------------------")
        print("Generating code for %s..." % (self.target))
        print("------------------------------\n")

        res = self.convert(config.Encoding.floatt,
                           config.DatasetType.testing, self.target)
        if res == False:
            return False

        res = self.compile(config.Encoding.floatt, self.target, self.sf)
        if res == False:
            return False

        # Copy model.h.
        srcFile = os.path.join(config.outdir, "Streamer", "input", "model_float.h")
        destFile = os.path.join(config.outdir, self.target, "model.h")
        shutil.copyfile(srcFile, destFile)

        # Copy library.h file.
        srcFile = os.path.join(config.outdir, self.target, "library", "library_float.h")
        destFile = os.path.join(config.outdir, self.target, "library.h")
        shutil.copyfile(srcFile, destFile)

        return True

    # Floating point x86 code.
    def runForFloat(self):
        Util.getLogger().info("---------------------------")
        Util.getLogger().info("Executing for X86 target...")
        Util.getLogger().info("---------------------------\n")

        res = self.convert(config.Encoding.floatt,
                           config.DatasetType.testing, config.Target.x86)
        if res == False:
            return False

        res = self.compile(config.Encoding.floatt, config.Target.x86, self.sf)
        if res == False:
            return False

        execMap = self.predict(config.Encoding.floatt, config.DatasetType.testing)
        if execMap == None:
            return False
        else:
            self.testingAccuracy = execMap["default"][0]

        print("Accuracy is %.3f%%\n" % (self.testingAccuracy))

        if self.target == config.Target.arduino:
            self.compileFloatForTarget()
            print("\nArduino sketch dumped in the folder %s\n" % (config.outdir))

        return True

    def run(self):
        sys.setrecursionlimit(10000)
        self.setup()

        if self.encoding == config.Encoding.fixed:
            return self.runForFixed()
        else:
            return self.runForFloat()
