# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import math
import numpy as np
import os
from sklearn.datasets import load_svmlight_file

import seedot.common as Common

# Utility functions commonly used by both Bonsai and Protonn

# Configurations class which can be modified based on the requirement


class Config:
    # If False, datasets are not generated again which reduces the processing
    # time
    dumpDataset = True
    # To use sparse matrix representation whenever required
    sparseMat = True


# Bonsai or Protonn
def getAlgo():
    return Config.algo


def setAlgo(algo: str):
    Config.algo = algo


# Fixed-point or float-point
def getVersion():
    return Config.version


def setVersion(version: str):
    Config.version = version


# training or testing dataset
def getDatasetType():
    return Config.datasetType


def setDatasetType(datasetType: str):
    Config.datasetType = datasetType


def usingTrainingDataset():
    return getDatasetType() == Common.DatasetType.Training


# Arduino code or desktop code (aka plain C++ code)
def getTarget():
    return Config.target


def setTarget(target: str):
    Config.target = target


def forArduino():
    return getTarget() == Common.Target.Arduino


def getDatasetOutputDir():
    return Config.datasetOuputDir


def setDatasetOutputDir(datasetOuputDir):
    Config.datasetOuputDir = datasetOuputDir


def getOutputDir():
    return Config.outputDir


def setOutputDir(outputDir: str):
    Config.outputDir = outputDir


def getModelDir():
    return Config.modelDir


def setModelDir(modelDir):
    Config.modelDir = modelDir


def setDatasetInput(trainingFile, testingFile):
    Config.trainingFile = trainingFile
    Config.testingFile = testingFile


def usingLibSVM():
    return Common.inputFileType == "libsvm"


def usingTSV():
    return Common.inputFileType == "tsv"


def usingCSV():
    return Common.inputFileType == "csv"


def usingNPY():
    return Common.inputFileType == "npy"


def dumpDataset():
    return Config.dumpDataset


def useSparseMat():
    return Config.sparseMat


def getNormType():
    return Config.norm


def setNormType(normType):
    Config.norm = normType


def noNorm():
    return getNormType() == 0


def minMaxNorm():
    return getNormType() == 1


def l2Norm():
    return getNormType() == 2


def meanVarNorm():
    return getNormType() == 3


def getMaxInt():
    return (2 ** (Common.wordLength - 1)) - 1


# Format specifiers for various datatypes
def getDataType(num):
    if isinstance(num, int):
        return 'MYINT', '%d'
    elif isinstance(num, float):
        return 'float', '%.6f'
    else:
        raise Exception(
            "Format specifier not found for the type: " + str(type(num)))


def matMin(mat):
    return min([min(x) for x in mat])


def matMax(mat):
    return max([max(x) for x in mat])


def matRange(mat):
    return matMin(mat), matMax(mat)


def matShape(mat):
    return len(mat), len(mat[0])


def listRange(list):
    return min(list), max(list)


def readXandY(useTrainingSet=False):
    train_ext = os.path.splitext(Config.trainingFile)[1]
    test_ext = os.path.splitext(Config.testingFile)[1]

    if train_ext == test_ext == ".npy":
        return readXandYasNPY(useTrainingSet)
    elif train_ext == test_ext == ".tsv":
        return readXandYasTSV(useTrainingSet)
    elif train_ext == test_ext == ".csv":
        return readXandYasCSV(useTrainingSet)
    elif train_ext == test_ext == ".txt":
        return readXandYasLibSVM(useTrainingSet)
    else:
        assert False


def zeroIndexLabels(Y):
    lab = np.array(Y)
    lab = lab.astype('uint8')
    lab = np.array(lab) - min(lab)
    return lab.tolist()


def readXandYasLibSVM(trainingDataset):
    if trainingDataset == True or usingTrainingDataset() == True:
        inputFile = Config.trainingFile
    else:
        inputFile = Config.testingFile

    data = load_svmlight_file(inputFile)

    X = data[0].todense().tolist()

    Y = data[1].tolist()
    Y = list(map(int, Y))
    Y = [[classID] for classID in Y]

    Y = zeroIndexLabels(Y)

    return X, Y


def readXandYasTSV(trainingDataset):
    '''
    In TSV format, the input is a file containing tab seperated values.
    In each row of the TSV file, the class ID will be the first entry followed by the feature vector of the data point
    The file is initially read as a matrix and later X and Y are extracted
    '''
    if trainingDataset == True or usingTrainingDataset() == True:
        mat = readFileAsMat(Config.trainingFile, "\t", float)
    else:
        mat = readFileAsMat(Config.testingFile, "\t", float)
    X, Y = extractXandYfromMat(mat)

    Y = zeroIndexLabels(Y)

    return X, Y


def extractXandYfromMat(mat):
    '''
    The first entry is cast to int (since it is the class ID) and used as Y
    The remaining entries are part of X
    '''
    X = []
    Y = []
    for i in range(len(mat)):
        classID = int(mat[i][0])

        temp = mat[i]
        temp.pop(0)

        X.append(temp)
        Y.append([classID])
    return X, Y


def readXandYasCSV(trainingDataset):
    '''
    In CSV format, the input is a folder containing two files "X.csv" and "Y.csv"
    Each file contains comma seperated values.
    X contains feature vector and Y contains the class ID of each data point
    '''
    if trainingDataset == True or usingTrainingDataset() == True:
        X = readFileAsMat(os.path.join(
            Config.trainingFile, "X.csv"), ", ", float)
        Y = readFileAsMat(os.path.join(
            Config.trainingFile, "Y.csv"), ", ", int)
    else:
        X = readFileAsMat(os.path.join(
            Config.testingFile, "X.csv"), ", ", float)
        Y = readFileAsMat(os.path.join(Config.testingFile, "Y.csv"), ", ", int)

    Y = zeroIndexLabels(Y)

    return X, Y


def readXandYasNPY(trainingDataset):
    '''
    In TSV format, the input is a file containing tab seperated values.
    In each row of the TSV file, the class ID will be the first entry followed by the feature vector of the data point
    The file is initially read as a matrix and later X and Y are extracted
    '''
    if trainingDataset == True or usingTrainingDataset() == True:
        mat = np.load(Config.trainingFile).tolist()
    else:
        mat = np.load(Config.testingFile).tolist()
    X, Y = extractXandYfromMat(mat)

    Y = zeroIndexLabels(Y)

    return X, Y


# Parse the file using the delimited and store it as a matrix
def readFileAsMat(fileName: str, delimiter: str, dataType):
    mat = []
    rowLength = -1

    with open(fileName, 'r') as f:
        for line in f:
            # If the delimiter is ' ', use split() without parameters to parse
            # the line even if there are consecutive spaces
            if delimiter == " ":
                entries = line.strip().split()
            else:
                entries = line.strip().split(delimiter)

            if rowLength == -1:
                rowLength = len(entries)

            assert rowLength == len(entries)

            # Cast each entry to the datatype specified
            row = list(map(dataType, entries))
            mat.append(row)
    return mat


# Write the matrix as a CSV file
def writeMatAsCSV(mat, fileName: str):
    m, n = matShape(mat)
    _, formatSpecifier = getDataType(mat[0][0])

    with open(fileName, 'w') as file:
        for i in range(m):
            for j in range(n):
                file.write(formatSpecifier % mat[i][j])
                if j != (n - 1):
                    file.write(", ")
            file.write("\n")


def writeMatsAsArray(mats: dict, fileName: str, shapeStr=None):
    for key in mats:
        writeMatAsArray(mats[key], key, fileName, shapeStr)


def writeMatAsArray(mat, name: str, fileName: str, shapeStr=None):
    m, n = matShape(mat)

    dataType, formatSpecifier = getDataType(mat[0][0])

    # Add the 'f' specifier for each float to supress compiler warnings
    if dataType == "float":
        formatSpecifier += 'f'

    # If custom matrix shape is not specified, use default
    if shapeStr == None:
        shapeStr = "[%d]" * 2 % (m, n)

    # Use Arduino pragma
    arduinoStr = ""
    if forArduino():
        arduinoStr = "PROGMEM "

    with open(fileName, 'a') as file:
        file.write('const %s%s %s%s = {\n' %
                   (arduinoStr, dataType, name, shapeStr))

        for row in mat:
            file.write('\t')
            for cell in row:
                file.write((formatSpecifier + ", ") % cell)
            file.write('\n')
        file.write('};\n\n')


def writeListsAsArray(lists: dict, fileName: str, shapeStr=None):
    for key in lists:
        writeListAsArray(lists[key], key, fileName, shapeStr)


def writeListAsArray(list, name: str, fileName: str, shapeStr=None):
    n = len(list)

    dataType, formatSpecifier = getDataType(list[0])

    # Add the 'f' specifier for each float to supress compiler warnings
    if dataType == "float":
        formatSpecifier += 'f'

    # If custom matrix shape is not specified, use default
    if shapeStr == None:
        shapeStr = "[%d]" % (n)

    # Use Arduino pragma
    arduinoStr = ""
    if forArduino():
        arduinoStr = "PROGMEM "

    with open(fileName, 'a') as file:
        file.write('const %s%s %s%s = {\n' %
                   (arduinoStr, dataType, name, shapeStr))

        file.write('\t')
        for cell in list:
            file.write((formatSpecifier + ", ") % cell)
        file.write('\n};\n\n')


def hex2(n):
    return hex(n & 0xffffffff)


def writeListsAsLUTs(lists: dict, dirName: str):
    os.makedirs(dirName, exist_ok=True)

    for key in lists:
        fileName = os.path.join(dirName, (key + '.lut'))
        writeListAsLUT(lists[key], key, fileName)


def writeListAsLUT(list, name: str, fileName: str):
    n = len(list)
    file = open(fileName, "w")
    for i in list:
        file.write(hex2(i)[2:])
        file.write('\n')
    file.close()


def writeVars(vars: dict, fileName: str):
    with open(fileName, 'a') as file:
        for key in vars:
            dataType, formatSpecifier = getDataType(vars[key])

            # Add the 'f' specifier for each float to supress compiler warnings
            if dataType == "float":
                formatSpecifier += 'f'
                file.write(("const %s %s = " + formatSpecifier + ";\n") %
                           (dataType, key, vars[key]))
            # todo: Why is this required when I am already taking care of
            # writing the datatype?
            else:
                file.write(("const %s %s = " + formatSpecifier + ";\n") %
                           ("int", key, vars[key]))
        file.write("\n")


def matMul(X, Y):
    X_m, X_n = matShape(X)
    Y_m, Y_n = matShape(Y)

    assert X_n == Y_m

    Z = [[0 for _ in range(Y_n)] for _ in range(X_m)]

    for i in range(X_m):
        for j in range(Y_n):
            sum = 0
            for k in range(X_n):
                sum += X[i][k] * Y[k][j]
            Z[i][j] = sum
    return Z


def matTranspose(mat):
    m, n = matShape(mat)
    transp = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(m):
        for j in range(n):
            transp[j][i] = mat[i][j]
    return transp


def convertToSparse(mat):
    '''
    Convert a sparse matrix into two arrays M_val and M_idx
    M_val contains all the non-zero elements in the matrix
    M_idx contains the row index of each non-zero element in the matrix which are delimited at each column using '0'
    '''
    m, n = matShape(mat)

    matVal = []
    matIdx = []

    for i in range(m):
        for j in range(n):
            if mat[i][j] != 0:
                matVal.append(mat[i][j])
                matIdx.append(int(j + 1))
        matIdx.append(int(0))

    return matVal, matIdx


# Custom function to compute the maximum scaling factor which can fit M
# into an integer of Common.wordLength length
def computeScale(m, M):
    maxAbs = max(abs(m), abs(M))
    return int(math.ceil(math.log2(maxAbs) - math.log2((1 << (Common.wordLength - 2)) - 1)))


# Scaling the matrix using the scaling factor computed
def scaleMat(mat, scale=None):
    if scale == None:
        scale = computeScale(*matRange(mat))

    scaledMat = [[int(math.ldexp(cell, -scale))
                  for cell in row] for row in mat]

    return scaledMat, scale


# Scaling an array using the scaling factor computed
def scaleList(list, scale=None):
    if scale == None:
        scale = computeScale(*listRange(list))

    scaledList = [int(math.ldexp(cell, -scale)) for cell in list]

    return scaledList, scale


# Remove some data points in X whose value is an outlier compared to the
# distribution of X
def trimMatrix(X, Y=None):
    # The matrix is trimmed only if the range of the matrix is more than this threshold
    # Used to skip trimming when the range is already low
    matThreshold = 2.1

    # The percentage of data points used to performa trimming
    ratio = 0.9

    # Skip trimming if within the threshold
    matMin, matmax = matRange(X)
    if abs(matmax - matMin) < matThreshold:
        return X, Y

    # Find the max of each data point
    rowMax = []
    for i in range(len(X)):
        m, M = listRange(X[i])
        maxAbs = max(abs(m), abs(M))
        rowMax.append(maxAbs)

    # Sort and find the trim threshold
    rowMaxSorted = list(rowMax)
    rowMaxSorted.sort()
    trimThreshold = rowMaxSorted[int(len(X) * ratio) - 1]

    # Only store data points which are beyond the threshold
    X_trim = []
    Y_trim = []
    for i in range(len(rowMax)):
        if rowMax[i] < trimThreshold:
            X_trim.append(X[i])
            if Y != None:
                Y_trim.append(Y[i])

    return X_trim, Y_trim
