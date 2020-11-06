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

from seedot.compiler.converter.converter import Converter

import seedot.config as config
from seedot.compiler.compiler import Compiler
from seedot.predictor import Predictor
import seedot.util as Util


class Main:

    def __init__(self, algo, version, target, trainingFile, testingFile, modelDir, sf, maximisingMetric, dataset, numOutputs, source):
        self.algo, self.version, self.target = algo, version, target
        self.trainingFile, self.testingFile, self.modelDir = trainingFile, testingFile, modelDir
        self.sf = sf
        self.dataset = dataset
        self.accuracy = {}
        self.maximisingMetric = maximisingMetric
        self.numOutputs = numOutputs
        self.source = source
        self.variableSubstitutions = {} #evaluated during profiling code run
        self.scalesForX = {} #populated for multiple code generation
        self.scalesForY = {} #populated for multiple code generation
        self.problemType = config.ProblemType.default
        self.variableToBitwidthMap = {} #Populated during profiling code run
        self.sparseMatrixSizes = {} #Populated during profiling code run
        self.varDemoteDetails = [] #Populated during variable demotion in VBW mode
        self.flAccuracy = -1 #Populated during profiling code run
        self.allScales = {} #Eventually populated with scale assignments in final code
        self.demotedVarsList = [] #Populated in VBW mode after exploration completed
        self.demotedVarsOffsets = {} #Populated in VBW mode after exploration completed
        self.biasShifts = {} #For simplifying bias addition, populated after every code run, used for M3 codegen

    def setup(self):
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        
        copy_tree(os.path.join(curr_dir, "Predictor"), os.path.join(config.tempdir, "Predictor"))

        if self.target == config.Target.arduino:
            for fileName in ["arduino.ino", "config.h", "predict.h"]:
                srcFile = os.path.join(curr_dir, "arduino", fileName)
                destFile = os.path.join(config.outdir, fileName)
                shutil.copyfile(srcFile, destFile)
        elif self.target == config.Target.m3:
            for fileName in ["datatypes.h", "mbconv.h", "mbconv.c", "utils.h", "utils.c"]:
                srcFile = os.path.join(curr_dir, "m3", "library", "quantized_%s"%fileName)
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

    # Generate the fixed-point code using the input generated from the
    # Converter project
    def compile(self, version, target, sf, generateAllFiles=True, id=None, printSwitch=-1, scaleForX=None, variableToBitwidthMap=None, demotedVarsList=[], demotedVarsOffsets={}, paramInNativeBitwidth=True):
        print("Generating code...", end='')

        if variableToBitwidthMap is None:
            variableToBitwidthMap = dict(self.variableToBitwidthMap)

        # Set input and output files
        inputFile = self.get_input_file()
        profileLogFile = os.path.join(
            config.tempdir, "Predictor", "output", "float", "profile.txt")

        logDir = os.path.join(config.outdir, "output")
        os.makedirs(logDir, exist_ok=True)
        if version == config.Version.floatt:
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

        obj = Compiler(self.algo, version, target, inputFile, outputDir,
                        profileLogFile, sf, self.source, outputLogFile, 
                        generateAllFiles, id, printSwitch, self.variableSubstitutions, 
                        scaleForX,
                        variableToBitwidthMap, self.sparseMatrixSizes, demotedVarsList, demotedVarsOffsets,
                        paramInNativeBitwidth)
        obj.run()
        self.biasShifts = obj.biasShifts
        self.allScales = dict(obj.varScales)
        if version == config.Version.floatt:
            self.variableSubstitutions = obj.substitutions
            self.variableToBitwidthMap = dict.fromkeys(obj.independentVars, config.wordLength)

        self.problemType = obj.problemType
        if id is None:
            self.scaleForX = obj.scaleForX
            self.scaleForY = obj.scaleForY
        else:
            self.scalesForX[id] = obj.scaleForX
            self.scalesForY[id] = obj.scaleForY

        print("completed")
        return True

    # Run the converter project to generate the input files using reading the
    # training model
    def convert(self, version, datasetType, target, varsForBitwidth={}, demotedVarsOffsets={}):
        print("Generating input files for %s %s dataset..." %
              (version, datasetType), end='')

        # Create output dirs
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
            obj = Converter(self.algo, version, datasetType, target, self.source,
                            datasetOutputDir, outputDir, varsForBitwidth, self.allScales, self.numOutputs, self.biasShifts, self.scaleForY if hasattr(self, "scaleForY") else None)
            obj.setInput(inputFile, self.modelDir,
                         self.trainingFile, self.testingFile)
            obj.run()
            if version == config.Version.floatt:
                self.sparseMatrixSizes = obj.sparseMatrixSizes
        except Exception as e:
            traceback.print_exc()
            return False

        print("done\n")
        return True

    # Build and run the Predictor project
    def predict(self, version, datasetType):
        outputDir = os.path.join("output", version)

        curDir = os.getcwd()
        os.chdir(os.path.join(config.tempdir, "Predictor"))

        obj = Predictor(self.algo, version, datasetType,
                        outputDir, self.scaleForX, self.scalesForX, self.scaleForY, self.scalesForY, self.problemType, self.numOutputs)
        execMap = obj.run()

        os.chdir(curDir)

        return execMap

    # Compile and run the generated code once for a given scaling factor
    def partialCompile(self, version, target, scale, generateAllFiles, id, printSwitch, variableToBitwidthMap=None, demotedVarsList=[], demotedVarsOffsets={}, paramInNativeBitwidth=True):
        if config.ddsEnabled:
            res = self.compile(version, target, None, generateAllFiles, id, printSwitch, scale, variableToBitwidthMap, demotedVarsList, demotedVarsOffsets, paramInNativeBitwidth)
        else:
            res = self.compile(version, target, scale, generateAllFiles, id, printSwitch, None, variableToBitwidthMap, demotedVarsList, demotedVarsOffsets, paramInNativeBitwidth)
        if res == False:
            return False
        else:
            return True

    def runAll(self, version, datasetType, codeIdToScaleFactorMap, demotedVarsToOffsetToCodeId=None, doNotSort=False):
        execMap = self.predict(version, datasetType)
        if execMap == None:
            return False, True

        if self.algo == config.Algo.test:
            for codeId, sf in codeIdToScaleFactorMap.items():
                self.accuracy[sf] = execMap[str(codeId)]
                print("The 95th percentile error for sf" + str(sf) + "with respect to dataset is " + str(execMap[str(codeId)][0]) + "%.")
                print("The 95th percentile error for sf" + str(sf) + "with respect to float execution is " + str(execMap[str(codeId)][1]) + "%.")     
                print("\n")
            return True,False    
                
        if codeIdToScaleFactorMap is not None:
            for codeId, sf in codeIdToScaleFactorMap.items():
                self.accuracy[sf] = execMap[str(codeId)]
                print("Accuracy at scale factor %d is %.3f%%, Disagreement Count is %d, Reduced Disagreement Count is %d\n" % (sf, execMap[str(codeId)][0], execMap[str(codeId)][1], execMap[str(codeId)][2]))
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
            def getMaximisingMetricValue(a):
                if self.maximisingMetric == config.MaximisingMetric.accuracy:
                    return (a[1][0], -a[1][1], -a[1][2])
                elif self.maximisingMetric == config.MaximisingMetric.disagreements:
                    return (-a[1][1], -a[1][2], a[1][0])
                elif self.maximisingMetric == config.MaximisingMetric.reducedDisagreements:
                    return (-a[1][2], -a[1][1], a[1][0])
            allVars = []
            for demotedVars in demotedVarsToOffsetToCodeId:
                offsetToCodeId = demotedVarsToOffsetToCodeId[demotedVars]
                print("Demoted vars: %s\n" % str(demotedVars))
                
                x = [(i, execMap[str(offsetToCodeId[i])]) for i in offsetToCodeId]
                x.sort(key=getMaximisingMetricValue, reverse=True)
                allVars.append(((demotedVars, x[0][0]), x[0][1]))

                for offset in offsetToCodeId:
                    codeId = offsetToCodeId[offset]
                    print("Offset %d (Code ID %d): Accuracy %.3f%%, Disagreement Count %d, Reduced Disagreement Count %d\n" %(offset, codeId, execMap[str(codeId)][0], execMap[str(codeId)][1], execMap[str(codeId)][2]))
            if not doNotSort:
                allVars.sort(key=getMaximisingMetricValue, reverse=True)
            self.varDemoteDetails = allVars
        return True, False

    # Iterate over multiple scaling factors and store their accuracies
    def performSearch(self):

        # # #for face-2

        # keys1 = {'X':0, 'tmp22':0, 'tmp23':-1, 'tmp25':0} #before rnnpool
        # keys2 = {'tmp27':0} #before mbonv
        # keys3 = []
        # for i in [9, 10, 11, 12, 13]:
        #     for j in range(1, 4):
        #         keys3.append('L%dF%d'%(i, j))
        #         keys3.append('L%dW%d'%(i, j))
        #         keys3.append('L%dB%d'%(i, j))
        # keys4 = {'tmp433':0, 'tmp446':0, 'tmp449':0, 'tmp460':0, 'tmp463':0, 'tmp469':0, 'tmp470':0} #after 3rd detection mbconvs

        # #for face 3                             
        # # keys1 = {'X':0, 'tmp22':-1, 'tmp23':-1, 'tmp25':0} #before rnnpool
        # # keys2 = {}#{'tmp27':0} #before mbonv
        # # keys3 = []
        # # for i in []: #[2, 3]:
        # #     for j in range(1, 4):
        # #         keys3.append('L%dF%d'%(i, j))
        # #         keys3.append('L%dW%d'%(i, j))
        # #         keys3.append('L%dB%d'%(i, j)) # 1. not needed for face-3, if done it tanks accuracy
        # # keys4 = {}#{'tmp433':0, 'tmp446':0, 'tmp449':0, 'tmp460':0, 'tmp463':0, 'tmp469':0, 'tmp470':0} #after 3rd detection mbconvs

        # #for face 4
        # # keys1 = {'X':0, 'tmp22':0, 'tmp23':0, 'tmp25':0}
        # # keys2 = {'tmp27':0}
        # # keys3 = []
        # # keys4 = {}

        # a = {}
        # b = {}
        # c = []
        # for key in self.variableToBitwidthMap.keys():
        #     a[key] = 16
        # for key in keys1.keys():
        #     a[key] = 8
        #     b[key] = keys1[key]
        #     c.append(key)
        # for key in keys2.keys():
        #     a[key] = 8
        #     b[key] = keys2[key]
        #     c.append(key)
        # for key in keys3:
        #     a[key] = 8
        #     b[key] = 0
        #     c.append(key)
        # for key in keys4.keys():
        #     a[key] = 8
        #     b[key] = keys4[key]
        #     c.append(key)
        # self.sf = -7
        # self.partialCompile(config.Version.fixed, config.Target.x86, self.sf, True, None, 0, a, c, b, paramInNativeBitwidth=False)
        # self.convert(config.Version.fixed,
        #             config.DatasetType.testing, config.Target.x86, a, b)
        # self.runAll(config.Version.fixed, config.DatasetType.testing, None, {}, True)

        # self.demotedVarsOffsets = b
        # self.variableToBitwidthMap = a
        # self.demotedVarsList = c

        # return


        # #######################################################

        start, end = config.maxScaleRange

        lastStageAcc = -1

        fixedPointCounter = 0
        while True:
            fixedPointCounter += 1
            if config.fixedPointVbwIteration:
                print("Will compile until conversion to fixed point. Iteration %d"%fixedPointCounter)
            highestValidScale = start
            firstCompileSuccess = False
            while firstCompileSuccess == False:
                if highestValidScale == end:
                    print("Compilation not possible for any Scale Factor. Abort")
                    return False
                
                # Refactor and remove this try/catch block in the futur 
                try:
                    firstCompileSuccess = self.partialCompile(config.Version.fixed, config.Target.x86, highestValidScale, True, None, 0, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))
                except:	
                    firstCompileSuccess = False

                if firstCompileSuccess:
                    break
                highestValidScale -= 1
                
            lowestValidScale = end + 1
            firstCompileSuccess = False
            while firstCompileSuccess == False:
                try:
                    firstCompileSuccess = self.partialCompile(config.Version.fixed, config.Target.x86, lowestValidScale, True, None, 0, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))
                except:
                    firstCompileSuccess = False
                if firstCompileSuccess:
                    break
                lowestValidScale += 1
                
            #Ignored
            self.partialCompile(config.Version.fixed, config.Target.x86, lowestValidScale, True, None, -1, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))

            # The iterator logic is as follows:
            # Search begins when the first valid scaling factor is found (runOnce returns True)
            # Search ends when the execution fails on a particular scaling factor (runOnce returns False)
            # This is the window where valid scaling factors exist and we
            # select the one with the best accuracy
            numCodes = highestValidScale - lowestValidScale + 1
            codeId = 0
            codeIdToScaleFactorMap = {}
            for i in range(highestValidScale, lowestValidScale - 1, -1):
                if config.ddsEnabled:
                    print("Testing with DDS and scale of X as " + str(i))
                else:
                    print("Testing with max scale factor of " + str(i))

                codeId += 1
                try:
                    compiled = self.partialCompile(
                        config.Version.fixed, config.Target.x86, i, False, codeId, -1 if codeId != numCodes else codeId, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))
                except: #If some code in the middle fails to compile
                    codeId -=1
                    continue
                if compiled == False:
                    return False
                codeIdToScaleFactorMap[codeId] = i

            res, exit = self.runAll(config.Version.fixed, config.DatasetType.training, codeIdToScaleFactorMap)

            if exit == True or res == False:
                return False

            print("\nSearch completed\n")
            print("----------------------------------------------")
            print("Best performing scaling factors with accuracy, disagreement, reduced disagreement:")

            self.sf = self.getBestScale()
            if self.accuracy[self.sf][0] != lastStageAcc:
                lastStageAcc = self.accuracy[self.sf][0]
            elif config.fixedPointVbwIteration:
                print("No difference in iteration %d Stage 2 and iteration %d Stage 1. Stopping search"%(fixedPointCounter-1, fixedPointCounter))
                break 

            #break
            if config.vbwEnabled:
                assert config.ddsEnabled, "Currently VBW on maxscale not supported"
                if config.wordLength != 16:
                    assert False, "VBW mode only supported if native bitwidth is 16"
                print("Scales computed in native bitwidth. Starting exploration over other bitwidths.")

                attemptToDemote = [var for var in self.variableToBitwidthMap if (var[-3:] != "val" and var not in self.demotedVarsList)]
                numCodes = 3 * len(attemptToDemote) + (6 if 'X' in attemptToDemote else 0) # 9 offsets tried for X while 3 tried for other variables
                
                self.partialCompile(config.Version.fixed, config.Target.x86, self.sf, True, None, -1 if len(attemptToDemote) > 0 else 0, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))
                codeId = 0
                contentToCodeIdMap = {}
                for demoteVar in attemptToDemote:
                    newbitwidths = dict(self.variableToBitwidthMap)
                    newbitwidths[demoteVar] = config.wordLength // 2
                    if demoteVar + "val" in newbitwidths:
                        newbitwidths[demoteVar + "val"] = config.wordLength // 2
                    for alreadyDemotedVars in self.demotedVarsList: # In subsequent iterations during fixed point compilation, this variable will have the variables demoted during the previous runs
                        newbitwidths[alreadyDemotedVars] = config.wordLength // 2
                    demotedVarsList = [i for i in newbitwidths.keys() if newbitwidths[i] != config.wordLength]
                    demotedVarsOffsets = {}
                    for key in self.demotedVarsList:
                        demotedVarsOffsets[key] = self.demotedVarsOffsets[key]

                    contentToCodeIdMap[tuple(demotedVarsList)] = {}
                    for demOffset in ([0, -1, -2] if demoteVar != 'X' else [0, -1, -2, -3, -4, -5, -6, -7, -8]):
                        codeId += 1
                        for k in demotedVarsList:
                            if k not in self.demotedVarsList:
                                demotedVarsOffsets[k] = demOffset
                        contentToCodeIdMap[tuple(demotedVarsList)][demOffset] = codeId
                        compiled = self.partialCompile(config.Version.fixed, config.Target.x86, self.sf, False, codeId, -1 if codeId != numCodes else codeId, dict(newbitwidths), list(demotedVarsList), dict(demotedVarsOffsets))
                        if compiled == False:
                            print("Variable Bitwidth exploration resulted in a compilation error")
                            return False
                
                res, exit = self.runAll(config.Version.fixed, config.DatasetType.training, None, contentToCodeIdMap)

                self.partialCompile(config.Version.fixed, config.Target.x86, self.sf, True, None, -1 if len(attemptToDemote) > 0 else 0, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))
                contentToCodeIdMap = {}
                demotedVarsOffsets = dict(self.demotedVarsOffsets)
                demotedVarsList = list(self.demotedVarsList)
                codeId = 0
                numCodes = len(attemptToDemote)
                demotedVarsListToOffsets = {}
                for ((demoteVars, offset), metrics) in self.varDemoteDetails:
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
                    compiled = self.partialCompile(config.Version.fixed, config.Target.x86, self.sf, False, codeId, -1 if codeId != numCodes else codeId, dict(newbitwidths), list(demotedVarsList), dict(demotedVarsOffsets))
                    if compiled == False:
                        print("Variable Bitwidth exploration resulted in another compilation error")
                        return False

                res, exit = self.runAll(config.Version.fixed, config.DatasetType.training, None, contentToCodeIdMap, True)

                if exit == True or res == False:
                    return False

                okToDemote = ()
                acceptedAcc = lastStageAcc
                for ((demotedVars, _), metrics) in self.varDemoteDetails:
                    acc = metrics[0]
                    if self.problemType == config.ProblemType.classification and (self.flAccuracy - acc) > 2.0:
                        break
                    elif self.problemType == config.ProblemType.regression and acc > 90.0:
                        break
                    else:
                        okToDemote = demotedVars
                        acceptedAcc = acc
                
                self.demotedVarsList = [i for i in okToDemote] + [i for i in self.demotedVarsList]
                self.demotedVarsOffsets.update(demotedVarsListToOffsets.get(okToDemote, {}))

                if acceptedAcc != lastStageAcc:
                    lastStageAcc = acceptedAcc
                else:
                    print("No difference in iteration %d's stages 1 & 2. Stopping search."%fixedPointCounter)
                    break


            if not config.vbwEnabled or not config.fixedPointVbwIteration:
                break

        return True

    # Reverse sort the accuracies, print the top 5 accuracies and return the
    # best scaling factor
    def getBestScale(self):
        def getMaximisingMetricValue(a):
            if self.maximisingMetric == config.MaximisingMetric.accuracy:
                return (a[1][0], -a[1][1], -a[1][2]) if not config.higherOffsetBias else (a[1][0], -a[0])
            elif self.maximisingMetric == config.MaximisingMetric.disagreements:
                return (-a[1][1], -a[1][2], a[1][0]) if not config.higherOffsetBias else (-max(5, a[1][1]), -a[0])
            elif self.maximisingMetric == config.MaximisingMetric.reducedDisagreements:
                return (-a[1][2], -a[1][1], a[1][0]) if not config.higherOffsetBias else (-max(5, a[1][2]), -a[0])
            elif self.algo == config.Algo.test:
                # minimize regression error
                return (-a[1][0])    

        x = [(i, self.accuracy[i]) for i in self.accuracy]
        x.sort(key=getMaximisingMetricValue, reverse=True)
        sorted_accuracy = x[:5]
        print(sorted_accuracy)
        return sorted_accuracy[0][0]

    # Find the scaling factor which works best on the training dataset and
    # predict on the testing dataset
    def findBestScalingFactor(self):
        print("-------------------------------------------------")
        print("Performing search to find the best scaling factor")
        print("-------------------------------------------------\n")

        # Generate input files for training dataset
        res = self.convert(config.Version.fixed,
                           config.DatasetType.training, config.Target.x86)
        if res == False:
            return False

        # Search for the best scaling factor
        res = self.performSearch()
        if res == False:
            return False

        print("Best scaling factor = %d" % (self.sf))

        return True

    def runOnTestingDataset(self):
        print("\n-------------------------------")
        print("Prediction on testing dataset")
        print("-------------------------------\n")

        print("Setting max scaling factor to %d\n" % (self.sf))

        if config.vbwEnabled:
            print("Demoted Vars with Offsets: %s\n" % (str(self.demotedVarsOffsets)))

        # Generate files for the testing dataset
        res = self.convert(config.Version.fixed,
                           config.DatasetType.testing, config.Target.x86)
        if res == False:
            return False

        # Compile and run code using the best scaling factor
        if config.vbwEnabled:
            compiled = self.partialCompile(config.Version.fixed, config.Target.x86, self.sf, True, None, 0, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))
        else:
            compiled = self.partialCompile(config.Version.fixed, config.Target.x86, self.sf, True, None, 0)
        if compiled == False:
            return False
            
        res, exit = self.runAll(config.Version.fixed, config.DatasetType.testing, {"default" : self.sf})
        if res == False:
            return False

        return True

    # Generate files for training dataset and perform a profiled execution
    def collectProfileData(self):
        print("-----------------------")
        print("Collecting profile data")
        print("-----------------------")

        res = self.convert(config.Version.floatt,
                           config.DatasetType.training, config.Target.x86)
        if res == False:
            return False

        res = self.compile(config.Version.floatt, config.Target.x86, self.sf)
        if res == False:
            return False

        execMap = self.predict(config.Version.floatt, config.DatasetType.training)
        if execMap == None:
            return False

        self.flAccuracy = execMap["default"][0]
        print("Accuracy is %.3f%%\n" % (execMap["default"][0]))
        print("Disagreement is %.3f%%\n" % (execMap["default"][1]))
        print("Reduced Disagreement is %.3f%%\n" % (execMap["default"][2]))

    # Generate code for Arduino
    def compileFixedForTarget(self):
        print("------------------------------")
        print("Generating code for %s..." % (self.target))
        print("------------------------------\n")

        demotedVarsOffsets = dict(self.demotedVarsOffsets) if hasattr(self, 'demotedVarsOffsets') else {}
        variableToBitwidthMap = dict(self.variableToBitwidthMap) if hasattr(self, 'variableToBitwidthMap') else {}
        res = self.convert(config.Version.fixed,
                           config.DatasetType.testing, self.target, variableToBitwidthMap, demotedVarsOffsets)
        if res == False:
            return False

        # Copy file
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

        # Copy library.h file
        curr_dir = os.path.dirname(os.path.realpath(__file__))

        if self.target == config.Target.arduino:
            srcFile = os.path.join(curr_dir, self.target, "library", "library_fixed.h")
            destFile = os.path.join(config.outdir, "library.h")
            shutil.copyfile(srcFile, destFile)


        modifiedBitwidths = dict.fromkeys(self.variableToBitwidthMap.keys(), config.wordLength) if hasattr(self, 'variableToBitwidthMap') else {}
        if hasattr(self, 'demotedVarsList'):
            for i in self.demotedVarsList:
                modifiedBitwidths[i] = config.wordLength // 2
        res = self.partialCompile(config.Version.fixed, self.target, self.sf, True, None, 0, dict(modifiedBitwidths), list(self.demotedVarsList) if hasattr(self, 'demotedVarsList') else [], dict(demotedVarsOffsets))
        if res == False:
            return False

        return True

    def runForFixed(self):
        # Collect runtime profile
        res = self.collectProfileData()
        if res == False:
            return False

        # Obtain best scaling factor
        if self.sf == None:
            res = self.findBestScalingFactor()
            if res == False:
                return False

        res = self.runOnTestingDataset()
        if res == False:
            return False
        else:
            self.testingAccuracy = self.accuracy[self.sf]

        # Generate code for target
        if self.target != config.Target.x86:
            self.compileFixedForTarget()

            print("\%s sketch dumped in the folder %s\n" % (self.target, config.outdir))

        return True

    def compileFloatForTarget(self):
        assert self.target == config.Target.arduino, "Floating point code supported for Arduino only"

        print("------------------------------")
        print("Generating code for %s..." % (self.target))
        print("------------------------------\n")

        res = self.convert(config.Version.floatt,
                           config.DatasetType.testing, self.target)
        if res == False:
            return False

        res = self.compile(config.Version.floatt, self.target, self.sf)
        if res == False:
            return False

        # Copy model.h
        srcFile = os.path.join(config.outdir, "Streamer", "input", "model_float.h")
        destFile = os.path.join(config.outdir, self.target, "model.h")
        shutil.copyfile(srcFile, destFile)

        # Copy library.h file
        srcFile = os.path.join(config.outdir, self.target, "library", "library_float.h")
        destFile = os.path.join(config.outdir, self.target, "library.h")
        shutil.copyfile(srcFile, destFile)

        return True

    def runForFloat(self):
        print("---------------------------")
        print("Executing for X86 target...")
        print("---------------------------\n")

        res = self.convert(config.Version.floatt,
                           config.DatasetType.testing, config.Target.x86)
        if res == False:
            return False

        res = self.compile(config.Version.floatt, config.Target.x86, self.sf)
        if res == False:
            return False

        execMap = self.predict(config.Version.floatt, config.DatasetType.testing)
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

        if self.version == config.Version.fixed:
            return self.runForFixed()
        else:
            return self.runForFloat()
