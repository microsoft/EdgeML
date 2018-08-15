# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import helpermethods
import os
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def min_max(A, name):
    print(name + " has max: " + str(np.max(A)) + " min: " + str(np.min(A)))
    return np.max([np.abs(np.max(A)), np.abs(np.min(A))])


def quantizeFastModels(modelDir, maxValue=127, scalarScaleFactor=1000):
    ls = os.listdir(modelDir)
    paramNameList = []
    paramWeightList = []
    paramLimitList = []

    classifierNameList = []
    classifierWeightList = []
    classifierLimitList = []

    scalarNameList = []
    scalarWeightList = []

    for file in ls:
        if file.endswith("npy"):
            if file.startswith("W"):
                paramNameList.append(file)
                temp = np.load(modelDir + "/" + file)
                paramWeightList.append(temp)
                paramLimitList.append(min_max(temp, file))
            elif file.startswith("U"):
                paramNameList.append(file)
                temp = np.load(modelDir + "/" + file)
                paramWeightList.append(temp)
                paramLimitList.append(min_max(temp, file))
            elif file.startswith("B"):
                paramNameList.append(file)
                temp = np.load(modelDir + "/" + file)
                paramWeightList.append(temp)
                paramLimitList.append(min_max(temp, file))
            elif file.startswith("FC"):
                classifierNameList.append(file)
                temp = np.load(modelDir + "/" + file)
                classifierWeightList.append(temp)
                classifierLimitList.append(min_max(temp, file))
            else:
                scalarNameList.append(file)
                scalarWeightList.append(np.load(modelDir + "/" + file))

    paramLimit = np.max(paramLimitList)
    classifierLimit = np.max(classifierLimitList)

    paramScaleFactor = np.round(255.0 / (2.0 * paramLimit))
    classifierScaleFactor = 255.0 / (2.0 * classifierLimit)

    quantParamWeights = []
    for param in paramWeightList:
        temp = np.round(paramScaleFactor * param)
        temp[temp[:] > maxValue] = maxValue

        if maxValue <= 127:
            temp = temp.astype('int8')
        elif maxValue <= 32767:
            temp = temp.astype('int16')
        else:
            temp = temp.astype('int32')

        quantParamWeights.append(temp)

    quantClassifierWeights = []
    for param in classifierWeightList:
        temp = np.round(classifierScaleFactor * param)
        temp[temp[:] > maxValue] = maxValue

        if maxValue <= 127:
            temp = temp.astype('int8')
        elif maxValue <= 32767:
            temp = temp.astype('int16')
        else:
            temp = temp.astype('int32')

        quantClassifierWeights.append(temp)

    quantScalarWeights = []
    for scalar in scalarWeightList:
        quantScalarWeights.append(
            np.round(scalarScaleFactor * sigmoid(scalar)).astype('int32'))

    if os.path.isdir(modelDir + '/QuantizedFastModel') is False:
        try:
            os.mkdir(modelDir + '/QuantizedFastModel')
            quantModelDir = modelDir + '/QuantizedFastModel'
        except OSError:
            print("Creation of the directory %s failed" %
                  modelDir + '/QuantizedFastModel')

    np.save(quantModelDir + "/paramScaleFactor.npy",
            paramScaleFactor.astype('int32'))
    np.save(quantModelDir + "/classifierScaleFactor.npy",
            classifierScaleFactor)
    np.save(quantModelDir + "/scalarScaleFactor", scalarScaleFactor)

    for i in range(0, len(scalarNameList)):
        np.save(quantModelDir + "/q" +
                scalarNameList[i], quantScalarWeights[i])

    for i in range(len(classifierNameList)):
        np.save(quantModelDir + "/q" +
                classifierNameList[i], quantClassifierWeights[i])

    for i in range(len(paramNameList)):
        np.save(quantModelDir + "/q" + paramNameList[i], quantParamWeights[i])

    print("\n\nQuantized Model Dir: " + quantModelDir)


def main():
    args = helpermethods.getQuantArgs()
    quantizeFastModels(args.model_dir, int(
        args.max_val), int(args.scalar_scale))


if __name__ == '__main__':
    main()
