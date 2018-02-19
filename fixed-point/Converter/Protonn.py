import os

from Utils import *

# Class to read ProtoNN model dumps and generate input files (C header file and the compiler input)
# The two classes ProtonnFixed and ProtonnFloat are for generating fixed model and float model respectively
# The baseclass Protonn collects some of the common functions between them.


class Protonn:

    def readDataset(self):
        self.X, self.Y = readXandY()

    def writeDataset(self):
        writeMatAsCSV(self.X, os.path.join(getDatasetOutputDir(), "X.csv"))
        writeMatAsCSV(self.Y, os.path.join(getDatasetOutputDir(), "Y.csv"))

    def processDataset(self):
        self.readDataset()
        assert len(self.X) == len(self.Y)
        self.transformDataset()
        self.writeDataset()

    def readModel(self):
        self.W = readFileAsMat(os.path.join(getModelDir(), "W"), "\t", float)
        self.B = readFileAsMat(os.path.join(getModelDir(), "B"), "\t", float)
        self.Z = readFileAsMat(os.path.join(getModelDir(), "Z"), "\t", float)
        self.gamma = readFileAsMat(os.path.join(
            getModelDir(), "gamma"), "\t", float)
        self.MinMax = readFileAsMat(os.path.join(
            getModelDir(), "minMaxParams"), "\t", float)

    def validateModel(self):
        W_m, W_n = matShape(self.W)
        B_m, B_n = matShape(self.B)
        Z_m, Z_n = matShape(self.Z)
        gamma_m, gamma_n = matShape(self.gamma)
        MinMax_m, MinMax_n = matShape(self.MinMax)

        assert W_m == B_m
        assert B_n == Z_n
        assert gamma_m == gamma_n == 1
        assert MinMax_m == 2
        assert MinMax_n == W_n

    def computeVars(self):
        self.D = len(self.W[0])
        self.d = len(self.W)
        self.p = len(self.B[0])
        self.c = len(self.Z)

    # Precompute some values to speedup prediction
    def formatModel(self):
        # Precompute g2
        self.g2 = self.gamma[0][0] * self.gamma[0][0]

        # Extract Min and Max
        Min = []
        Max = []
        for i in range(len(self.MinMax[0])):
            Min.append([self.MinMax[0][i]])
            Max.append([self.MinMax[1][i]])

        # Precompute W * (X-m)/(M-m) by absorbing m, M into W
        for i in range(len(self.W)):
            for j in range(len(self.W[0])):
                self.W[i][j] = self.W[i][j] / (Max[j][0] - Min[j][0])

        self.Min = matMul(self.W, Min)

        assert len(self.Min[0]) == 1
        self.Min = [x[0] for x in self.Min]

        self.computeVars()

    # Write macros and namespace declarations
    def writeHeader(self):
        with open(self.headerFile, 'a') as file:
            file.write("#pragma once\n\n")
            if useSparseMat():
                file.write("#define P_SPARSE_W 1\n\n")
            else:
                file.write("#define P_SPARSE_W 0\n\n")

            s = 'protonn_' + getVersion()
            file.write("namespace %s {\n\n" % (s))

    def writeFooter(self):
        with open(self.headerFile, 'a') as file:
            file.write("}\n")

    def processModel(self):
        self.readModel()
        self.validateModel()
        self.formatModel()
        self.transformModel()
        self.writeModel()

    def run(self):
        self.headerFile = os.path.join(getOutputDir(), "model.h")
        self.inputFile = os.path.join(getOutputDir(), "input.txt")
        self.infoFile = os.path.join(getOutputDir(), "info.txt")

        open(self.headerFile, 'w').close()
        open(self.inputFile, 'w').close()

        if dumpDataset():
            self.processDataset()

        self.processModel()

        if dumpDataset():
            assert len(self.X[0]) == len(self.W[0])


class ProtonnFixed(Protonn):

    # The X matrix is quantized using a scale factor computed from the training dataset.
    # The range of X_train is used to compute the scale factor.
    # Since the range of X_train depends on its distribution, the scale computed may be imprecise.
    # To avoid this, any outliers in X_train is trimmed off using a threshold to get a more precise range and a more precise scale.
    def transformDataset(self):
        # If X itself is X_train, reuse it. Otherwise, read it from file
        if usingTrainingDataset():
            self.X_train = list(self.X)
        else:
            self.X_train, _ = readXandY(trainingDataset=True)

        # Trim some data points from X_train
        self.X_train, _ = trimMatrix(self.X_train)

        # Compute range and scale and quantize X
        testDatasetRange = matRange(self.X)
        self.trainDatasetRange = matRange(self.X_train)

        scale = computeScaleSpecial(*self.trainDatasetRange)
        self.X, _ = scaleMatSpecial(self.X, scale)

        with open(self.infoFile, 'w') as file:
            file.write("Range of test dataset: [%.6f, %.6f]\n" % (
                testDatasetRange))
            file.write("Range of training dataset: [%.6f, %.6f]\n" % (
                self.trainDatasetRange))
            file.write("Test dataset scaled by: %d\n\n" % (scale))

    # Write the ProtoNN algorithm in terms of the compiler DSL.
    # Compiler takes this as input and generates fixed-point code.
    def genInputForCompiler(self):
        with open(self.inputFile, 'w') as file:
            # Matrix declarations
            file.write("let XX   = X(%d, 1)   in [%.6f, %.6f] in\n" % (
                (len(self.X[0]),) + self.trainDatasetRange))
            file.write("let WW  = W(%d, %d)    in [%.6f, %.6f] in\n" % (
                (self.d, self.D) + matRange(self.W)))
            file.write("let BB  = B(%d, %d, 1) in [%.6f, %.6f] in\n" % (
                (self.p, self.d) + matRange(self.B)))
            file.write("let ZZ  = Z(%d, %d, 1) in [%.6f, %.6f] in\n" % (
                (self.p, self.c) + matRange(self.Z)))
            file.write("let Min = min(%d, 1)   in [%.6f, %.6f] in\n" % (
                (self.d,) + listRange(self.Min)))
            file.write("let g2 = %.6f in\n\n" % (self.g2))

            # Algorithm
            file.write("let WX = WW |*| XX - Min in\n")
            file.write("let res = $(i = [0:%d])\n" % (self.p))
            file.write("(\n")
            file.write("\tlet del = WX - BB[i] in\n")
            file.write("\tZZ[i] * exp(-g2 * (del^T * del))\n")
            file.write(") in\n")
            file.write("argmax(res)\n")

    # Quantize the matrices
    def transformModel(self):
        if dumpDataset():
            self.genInputForCompiler()

        self.W, _ = scaleMatSpecial(self.W)
        self.B, _ = scaleMatSpecial(self.B)
        self.Z, _ = scaleMatSpecial(self.Z)
        self.Min, _ = scaleListSpecial(self.Min)

    # Writing the model as a bunch of variables, arrays and matrices to a file
    def writeModel(self):
        self.writeHeader()

        writeListAsArray(self.Min, 'min', self.headerFile,
                         shapeStr="[%d]" * 2 % (self.d, 1))

        # Sparse matrices are converted in to two arrays containing values and indices to reduce space
        if useSparseMat():
            W_transp = matTranspose(self.W)
            Wval, Widx = convertToSparse(W_transp)
            writeListsAsArray({'Wval': Wval, 'Widx': Widx}, self.headerFile)
        else:
            writeMatAsArray(self.W, 'W', self.headerFile)

        # Transpose B and Z to satisfy the declarations in the generated DSL input
        B_transp = matTranspose(self.B)
        Z_transp = matTranspose(self.Z)

        writeMatAsArray(B_transp, 'B', self.headerFile,
                        shapeStr="[%d]" * 3 % (self.p, self.d, 1))
        writeMatAsArray(Z_transp, 'Z', self.headerFile,
                        shapeStr="[%d]" * 3 % (self.p, self.c, 1))

        self.writeFooter()


class ProtonnFloat(Protonn):

    # Float model is generated for for training dataset to profile the prediction
    # Hence, X is trimmed down to remove outliers. Prediction profiling is performed on the trimmed X to generate more precise profile data
    def transformDataset(self):
        if usingTrainingDataset():
            beforeLen = len(self.X)
            beforeRange = matRange(self.X)

            self.X, self.Y = trimMatrix(self.X, self.Y)

            afterLen = len(self.X)
            afterRange = matRange(self.X)

            with open(self.infoFile, 'w') as file:
                file.write("Old range of X: [%.6f, %.6f]\n" % (beforeRange))
                file.write("Trimmed the dataset from %d to %d data points; %.3f%%\n" % (
                    beforeLen, afterLen, float(beforeLen - afterLen) / beforeLen * 100))
                file.write("New range of X: [%.6f, %.6f]\n" % (afterRange))

    def transformModel(self):
        pass

    # Writing the model as a bunch of variables, arrays and matrices to a file
    def writeModel(self):
        mats = {}
        lists = {'min': self.Min}

        # Sparse matrices are converted in to two arrays containing values and indices to reduce space
        if useSparseMat():
            W_transp = matTranspose(self.W)
            Wval, Widx = convertToSparse(W_transp)
            lists.update({'Wval': Wval, 'Widx': Widx})
        else:
            mats['W'] = self.W

        mats.update({'B': self.B, 'Z': self.Z})

        self.writeHeader()
        writeVars({'D': self.D, 'd': self.d, 'p': self.p,
                   'c': self.c, 'g2': self.g2}, self.headerFile)
        writeListsAsArray(lists, self.headerFile)
        writeMatsAsArray(mats, self.headerFile)
        self.writeFooter()
