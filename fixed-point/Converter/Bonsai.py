import math
import os

from Utils import *

class Bonsai:

	def readDataset(self):
		self.X, self.Y = readXandY()

	def formatDataset(self):
		assert len(self.X) == len(self.Y)
		for i in range(len(self.X)):
			self.X[i].append(1)

	def writeDataset(self):
		writeMatAsCSV(self.X, os.path.join(getDatasetOutputDir(), "X.csv"))
		writeMatAsCSV(self.Y, os.path.join(getDatasetOutputDir(), "Y.csv"))

	def processDataset(self):
		self.readDataset()
		self.formatDataset()
		self.transformDataset()
		self.writeDataset()

	def readModel(self):
		self.Z = readFileAsMat(os.path.join(getModelDir(), "Z"), "\t", float)
		self.W = readFileAsMat(os.path.join(getModelDir(), "W"), "\t", float)
		self.V = readFileAsMat(os.path.join(getModelDir(), "V"), "\t", float)
		self.T = readFileAsMat(os.path.join(getModelDir(), "Theta"), "\t", float)
		self.Sigma = readFileAsMat(os.path.join(getModelDir(), "Sigma"), "\t", float)
		self.Mean = readFileAsMat(os.path.join(getModelDir(), "Mean"), "\t", float)
		self.Variance = readFileAsMat(os.path.join(getModelDir(), "Variance"), "\t", float)

	def validateModel(self):
		Z_m, Z_n = matShape(self.Z)
		W_m, W_n = matShape(self.W)
		V_m, V_n = matShape(self.V)
		# If T is empty
		if len(self.T) == 0:
			T_m, T_n = 0, Z_m
		else:
			T_m, T_n = matShape(self.T)
		Sigma_m, Sigma_n = matShape(self.Sigma)
		Mean_m, Mean_n = matShape(self.Mean)
		Variance_m, Variance_n = matShape(self.Variance)

		assert Z_n == Mean_m == Variance_m
		assert Z_m == W_n == V_n == T_n 
		assert W_m == V_m
		assert Sigma_m == Sigma_n == 1
		assert Mean_n == Variance_n == 1

	def rearrangeWandV(self, mat):
		matNew = [[] for _ in range(len(mat))]
		for i in range(self.numClasses):
			for j in range(self.totalNodes):
				matNew[i + j * self.numClasses] = mat[i * self.totalNodes + j]
		return matNew

	# Note: Following computations assume that the tree is always complete
	def computeVars(self):
		self.D = len(self.Z[0])
		self.d = len(self.Z)
		self.internalNodes = len(self.T)
		self.depth = int(math.log2(self.internalNodes + 1))
		self.totalNodes = 2 ** (self.depth + 1) - 1
		self.numClasses = len(self.W) // self.totalNodes

	def formatModel(self):
		self.Mean[len(self.Mean) - 1] = [0]
		self.Variance[len(self.Variance) - 1] = [1]

		sigma = self.Sigma[0][0]

		self.computeVars()

		self.V = [[x * sigma for x in y] for y in self.V]
		self.Z = [[x / self.d for x in y] for y in self.Z]
		self.Z = [[x / v[0] for x, v in zip(y, self.Variance)] for y in self.Z]

		self.mean = matMul(self.Z, self.Mean)

		assert len(self.mean[0]) == 1
		self.mean = [x[0] for x in self.mean]

		self.W = self.rearrangeWandV(self.W)
		self.V = self.rearrangeWandV(self.V)

	def verifyModel(self):
		if len(self.T) == 0:
			print("Warning: Empty matrix T\n")
			self.T = [[-0.000001, 0.000001]]

	def writeHeader(self):
		with open(self.headerFile, 'a') as file:
			file.write("#pragma once\n\n")
			if useSparseMat():
				file.write("#define B_SPARSE_Z 1\n\n")
			else:
				file.write("#define B_SPARSE_Z 0\n\n")

			s = 'bonsai_' + getVersion()
			file.write("namespace %s {\n\n" % (s))

	def writeFooter(self):
		with open(self.headerFile, 'a') as file:
			file.write("}\n")

	def writeModel(self):
		lists = {'mean': self.mean}
		mats = {}
	
		if useSparseMat():
			Z_transp = matTranspose(self.Z)
			Zval, Zidx = convertToSparse(Z_transp)
			lists.update({'Zval': Zval, 'Zidx': Zidx})
		else:
			mats['Z'] = self.Z

		mats.update({'W': self.W, 'V': self.V, 'T': self.T})

		self.writeHeader()
		writeVars({'D': self.D, 'd': self.d, 'c': self.numClasses, 'depth': self.depth,
					'totalNodes': self.totalNodes, 'internalNodes': self.internalNodes,
					'tanh_limit': self.tanh_limit}, self.headerFile)
		writeListsAsArray(lists, self.headerFile)
		writeMatsAsArray(mats, self.headerFile)
		self.writeFooter()

	def processModel(self):
		self.readModel()
		self.validateModel()
		self.formatModel()
		self.verifyModel()
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
			assert len(self.X[0]) == len(self.Z[0])

class BonsaiFixed(Bonsai):

	def transformDataset(self):
		if usingTrainingDataset():
			self.X_train = list(self.X)
		else:
			self.X_train, _ = readXandY(trainingDataset = True)

		self.X_train, _ = trimMatrix(self.X_train)

		testDatasetRange = matRange(self.X)
		self.trainDatasetRange = matRange(self.X_train)

		scale = computeScaleSpecial(*self.trainDatasetRange)
		self.X, _ = scaleMatSpecial(self.X, scale)

		with open(self.infoFile, 'w') as file:
			file.write("Range of test dataset: [%.6f, %.6f]\n" % (testDatasetRange))
			file.write("Range of training dataset: [%.6f, %.6f]\n" % (self.trainDatasetRange))
			file.write("Test dataset scaled by: %d\n\n" % (scale))

	def genInputForCompiler(self):
		m_mean, M_mean = listRange(self.mean)
		if abs(m_mean) < 0.0000005: m_mean = -0.000001
		if abs(M_mean) < 0.0000005: M_mean = 0.000001

		with open(self.inputFile, 'w') as file:
			file.write("let XX   = X(%d, 1)   in [%.6f, %.6f] in\n" % ((len(self.X[0]),) + self.trainDatasetRange))
			file.write("let ZZ   = Z(%d, %d)  in [%.6f, %.6f] in\n" % ((self.d, self.D) + matRange(self.Z)))
			file.write("let WW   = W(%d, %d, %d) in [%.6f, %.6f] in\n" % ((self.totalNodes, self.numClasses, self.d) + matRange(self.W)))
			file.write("let VV   = V(%d, %d, %d) in [%.6f, %.6f] in\n" % ((self.totalNodes, self.numClasses, self.d) + matRange(self.V)))
			file.write("let TT   = T(%d, 1, %d) in [%.6f, %.6f] in\n" % ((self.internalNodes, self.d) + matRange(self.T)))
			file.write("let Mean = mean(%d, 1) in [%.6f, %.6f] in\n\n" % (self.d, m_mean, M_mean))
			
			file.write("let ZX = ZZ |*| XX - Mean in\n\n")
			
			file.write("// depth 0\n")
			file.write("let node0   = 0    in\n")
			file.write("let W0      = WW[node0] * ZX in\n")
			file.write("let V0      = VV[node0] * ZX in\n")
			file.write("let V0_tanh = tanh(V0) in\n")
			file.write("let score0  = W0 <*> V0_tanh in\n\n")
			
			for i in range(1, self.depth + 1):
				file.write("// depth %d\n" % (i))
				file.write("let node%d   = (TT[node%d] * ZX) >= 0? 2 * node%d + 1 : 2 * node%d + 2 in\n" % (i, i - 1, i - 1, i - 1))
				file.write("let W%d      = WW[node%d] * ZX in\n" % (i, i))
				file.write("let V%d      = VV[node%d] * ZX in\n" % (i, i))
				file.write("let V%d_tanh = tanh(V%d) in\n" % (i, i))
				file.write("let score%d  = score%d + W%d <*> V%d_tanh in\n\n" % (i, i - 1, i, i))
			
			if self.numClasses <= 2:
				file.write("sgn(score%d)\n" % (self.depth))
			else:
				file.write("argmax(score%d)\n" % (self.depth))

	def transformModel(self):
		if dumpDataset():
			self.genInputForCompiler()

		self.Z, _ = scaleMatSpecial(self.Z)
		self.W, _ = scaleMatSpecial(self.W)
		self.V, _ = scaleMatSpecial(self.V)
		self.T, _ = scaleMatSpecial(self.T)
		self.mean, _ = scaleListSpecial(self.mean)

	def writeModel(self):
		self.writeHeader()

		writeListAsArray(self.mean, 'mean', self.headerFile, shapeStr = "[%d]" * 2 % (self.d, 1))

		if useSparseMat():
			Z_transp = matTranspose(self.Z)
			Zval, Zidx = convertToSparse(Z_transp)
			writeListsAsArray({'Zval': Zval, 'Zidx': Zidx}, self.headerFile)
		else:
			writeMatAsArray(self.Z, 'Z', self.headerFile)

		T_m = self.internalNodes
		if T_m == 0: T_m = 1

		writeMatAsArray(self.W, 'W', self.headerFile, shapeStr = "[%d]" * 3 % (self.totalNodes, self.numClasses, self.d))
		writeMatAsArray(self.V, 'V', self.headerFile, shapeStr = "[%d]" * 3 % (self.totalNodes, self.numClasses, self.d))
		writeMatAsArray(self.T, 'T', self.headerFile, shapeStr = "[%d]" * 3 % (T_m, 1, self.d))

		self.writeFooter()

class BonsaiFloat(Bonsai):

	def transformDataset(self):
		if usingTrainingDataset():
			beforeLen = len(self.X)
			beforeRange = matRange(self.X)

			self.X, self.Y = trimMatrix(self.X, self.Y)

			afterLen = len(self.X)
			afterRange = matRange(self.X)

			with open(self.infoFile, 'w') as file:
				file.write("Old range of X: [%.6f, %.6f]\n" % (beforeRange))
				file.write("Trimmed the dataset from %d to %d data points; %.3f%%\n" % (beforeLen, afterLen, float(beforeLen - afterLen) / beforeLen * 100))
				file.write("New range of X: [%.6f, %.6f]\n" % (afterRange))

	def transformModel(self):
		self.tanh_limit = 1.0
