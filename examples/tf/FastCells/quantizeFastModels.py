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
                temp = np.load(os.path.join(modelDir, file))
                paramWeightList.append(temp)
                paramLimitList.append(min_max(temp, file))
            elif file.startswith("U"):
                paramNameList.append(file)
                temp = np.load(os.path.join(modelDir, file))
                paramWeightList.append(temp)
                paramLimitList.append(min_max(temp, file))
            elif file.startswith("B"):
                paramNameList.append(file)
                temp = np.load(os.path.join(modelDir, file))
                paramWeightList.append(temp)
                paramLimitList.append(min_max(temp, file))
            elif file.startswith("FC"):
                classifierNameList.append(file)
                temp = np.load(os.path.join(modelDir, file))
                classifierWeightList.append(temp)
                classifierLimitList.append(min_max(temp, file))
            elif file.startswith("mean") or file.startswith("std"):
                continue
            else:
                scalarNameList.append(file)
                scalarWeightList.append(np.load(os.path.join(modelDir, file)))

    paramLimit = np.max(paramLimitList)
    classifierLimit = np.max(classifierLimitList)

    paramScaleFactor = np.round((2.0 * maxValue + 1.0) / (2.0 * paramLimit))
    classifierScaleFactor = (2.0 * maxValue + 1.0) / (2.0 * classifierLimit)

    quantParamWeights = []
    for param in paramWeightList:
        temp = np.round(paramScaleFactor * param)
        temp[temp[:] > maxValue] = maxValue
        temp[temp[:] < -maxValue] = -1 * (maxValue + 1)

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
        temp[temp[:] < -maxValue] = -1 * (maxValue + 1)

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

    quantModelDir = os.path.join(modelDir, 'QuantizedModel')
    if not os.path.isdir(quantModelDir):
        try:
            os.makedirs(quantModelDir, exist_ok=True)
        except OSError:
            print("Creation of the directory %s failed" % quantModelDir)

    np.save(os.path.join(quantModelDir, "paramScaleFactor.npy"),
            paramScaleFactor.astype('int32'))
    np.save(os.path.join(quantModelDir, "classifierScaleFactor.npy"),
            classifierScaleFactor)
    np.save(os.path.join(quantModelDir, "scalarScaleFactor"), scalarScaleFactor)

    for i in range(0, len(scalarNameList)):
        np.save(os.path.join(quantModelDir, "q" +
                scalarNameList[i]), quantScalarWeights[i])

    for i in range(len(classifierNameList)):
        np.save(os.path.join(quantModelDir, "q" +
                classifierNameList[i]), quantClassifierWeights[i])

    for i in range(len(paramNameList)):
        np.save(os.path.join(quantModelDir, "q" + paramNameList[i]),
                quantParamWeights[i])

    print("\n\nQuantized Model Dir: " + quantModelDir)


def main():
    args = helpermethods.getQuantArgs()
    quantizeFastModels(args.model_dir, int(
        args.max_val), int(args.scalar_scale))


if __name__ == '__main__':
    main()
