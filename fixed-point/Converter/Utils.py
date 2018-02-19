import os
import math

# Utility functions commonly used by both Bonsai and Protonn


# Configurations class which can be modified based on the requirement
class Config:
    # If False, datasets are not generated again which reduces the time taken
    dumpDataset = True
    # To use sparse matrix representation whenever required
    sparseMat = True
    # Target word length. Currently set to match the word length of Arduino (2 bytes)
    bits = 16


def getCurDir():
    return os.path.dirname(__file__)


def getAlgo():
    '''
    Bonsai or Protonn
    '''
    return Config.algo


def setAlgo(algo: str):
    Config.algo = algo


def getVersion():
    '''
    Fixed-point or float-point
    '''
    return Config.version


def setVersion(version: str):
    Config.version = version


def getDatasetType():
    '''
    training or testing dataset
    '''
    return Config.datasetType


def setDatasetType(datasetType: str):
    Config.datasetType = datasetType


def usingTrainingDataset():
    return Config.datasetType == "training"


def getTarget():
    '''
    Arduino code or desktop code (aka plain C++ code)
    '''
    return Config.target


def setTarget(target: str):
    Config.target = target


def forArduino():
    return Config.target == "arduino"


def getDatasetOutputDir():
    return Config.datasetOuputDir


def setDatasetOutputDir(datasetOuputDir):
    Config.datasetOuputDir = datasetOuputDir


def getOutputDir():
    return Config.outputDir


def setOutputDir(outputDir: str):
    Config.outputDir = outputDir


def getFileType():
    '''
    Tab seperated file (tsv) or comma seperated file (csv)
    '''
    return Config.fileType


def setFileType(fileType):
    Config.fileType = fileType


def usingTSV():
    return getFileType() == "tsv"


def setTSVinputFiles(trainingFile, testingFile):
    setFileType("tsv")
    Config.trainingFile = trainingFile
    Config.testingFile = testingFile


def setCSVinputDirs(trainingDir, testingDir):
    setFileType("csv")
    Config.trainingDir = trainingDir
    Config.testingDir = testingDir


def dumpDataset():
    return Config.dumpDataset


def useSparseMat():
    return Config.sparseMat


def getMaxInt():
    return (2 ** (Config.bits - 1)) - 1


def getModelDir():
    return Config.modelDir


def setModelDir(modelDir):
    Config.modelDir = modelDir


def getDataType(num):
    '''
    Format specifiers for various datatypes
    '''
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


def readXandY(trainingDataset=False):
    if usingTSV():
        return readXandYasTSV(trainingDataset)
    else:
        return readXandYasCSV(trainingDataset)


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
    X, Y = extractXandY(mat)
    return X, Y


def readXandYasCSV(trainingDataset):
    '''
    In CSV format, the input is a folder containing two files "X.csv" and "Y.csv"
    Each file contains comma seperated values.
    X contains feature vector and Y contains the class ID of each data point
    '''
    if trainingDataset == True or usingTrainingDataset() == True:
        X = readFileAsMat(os.path.join(
            Config.trainingDir, "X.csv"), ", ", float)
        Y = readFileAsMat(os.path.join(Config.trainingDir, "Y.csv"), ", ", int)
    else:
        X = readFileAsMat(os.path.join(
            Config.testingDir, "X.csv"), ", ", float)
        Y = readFileAsMat(os.path.join(Config.testingDir, "Y.csv"), ", ", int)
    return X, Y


def extractXandY(mat):
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


def readFileAsMat(fileName: str, delimiter: str, dataType):
    '''
    Parse the file using the delimited and store it as a matrix
    '''
    mat = []
    rowLength = -1

    with open(fileName, 'r') as f:
        for line in f:
            # If the delimiter is ' ', use split() without parameters to parse the line even if there are consecutive spaces
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


def writeMatAsCSV(mat, fileName: str):
    '''
    Write the matrix as a CSV file
    '''
    m, n = matShape(mat)
    _, formatSpecifier = getDataType(mat[0][0])

    with open(fileName, 'w') as file:
        for i in range(m):
            for j in range(n):
                file.write(formatSpecifier % mat[i][j])
                if j != (n - 1):
                    file.write(", ")
            file.write("\n")


def writeMatsAsArray(mats: dict, fileName: str):
    for key in mats:
        writeMatAsArray(mats[key], key, fileName)


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


def writeListsAsArray(lists: dict, fileName: str):
    for key in lists:
        writeListAsArray(lists[key], key, fileName)


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


def writeVars(vars: dict, fileName: str):
    with open(fileName, 'a') as file:
        for key in vars:
            dataType, formatSpecifier = getDataType(vars[key])

            # Add the 'f' specifier for each float to supress compiler warnings
            if dataType == "float":
                formatSpecifier += 'f'

            file.write(("const %s %s = " + formatSpecifier + ";\n") %
                       (dataType, key, vars[key]))
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


def computeScaleSpecial(m, M):
    '''
    Custom function to compute the maximum scaling factor which can fit M into an integer of Config.bits length
    '''
    maxAbs = max(abs(m), abs(M))
    return int(math.ceil(math.log2(maxAbs) - math.log2((1 << (Config.bits - 2)) - 1)))


def scaleMatSpecial(mat, scale=None):
    '''
    Scaling the matrix using the scaling factor computed
    '''
    if scale == None:
        scale = computeScaleSpecial(*matRange(mat))

    scaledMat = [[int(math.ldexp(cell, -scale))
                  for cell in row] for row in mat]

    return scaledMat, scale


def scaleListSpecial(list, scale=None):
    '''
    Scaling an array using the scaling factor computed
    '''
    if scale == None:
        scale = computeScaleSpecial(*listRange(list))

    scaledList = [int(math.ldexp(cell, -scale)) for cell in list]

    return scaledList, scale


def trimMatrix(X, Y=None):
    '''
    Remove some data points in X whose value is an outlier compared to the distribution of X
    '''

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
