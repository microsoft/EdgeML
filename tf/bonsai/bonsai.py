import utils
import tensorflow as tf
import numpy as np

## Bonsai Class
class Bonsai:
	## Constructor
	def __init__(self, C, F, P, D, S, lW, lT, lV, lZ, 
		sW, sT, sV, sZ, lr = None, W = None, T = None, 
		V = None, Z = None, feats = None):

		self.dataDimension = F + 1
		self.projectionDimension = P
		
		if (C > 2):
			self.numClasses = C
		elif (C == 2):
			self.numClasses = 1

		self.treeDepth = D
		self.sigma = S
		
		## Regularizer coefficients
		self.lW = lW
		self.lV = lV
		self.lT = lT
		self.lZ = lZ
		
		## Sparsity hyperparams
		self.sW = sW
		self.sV = sV
		self.sT = sT
		self.sZ = sZ

		self.internalNodes = 2**self.treeDepth - 1
		self.totalNodes = 2*self.internalNodes + 1

		## The Parameters of Bonsai
		self.W = self.initW(W)
		self.V = self.initV(V)
		self.T = self.initT(T)
		self.Z = self.initZ(Z)

		## Placeholders for Hard Thresholding and Sparse Training
		self.Wth = tf.placeholder(tf.float32, name='Wth')
		self.Vth = tf.placeholder(tf.float32, name='Vth')
		self.Zth = tf.placeholder(tf.float32, name='Zth')
		self.Tth = tf.placeholder(tf.float32, name='Tth')
		
		## Placeholders for Features and labels
		## feats are to be fed when joint training is being done with Bonsai as end classifier
		if feats is None:
			self.x = tf.placeholder("float", [None, self.dataDimension])
		else:
			self.x = feats

		self.y = tf.placeholder("float", [None, self.numClasses])

		## Placeholder for batch size, needed for Multiclass hinge loss
		self.batch_th = tf.placeholder(tf.int64, name='batch_th')

		self.sigmaI = 1.0
		if lr is not None:
			self.learningRate = lr
		else:
			self.learningRate = 0.01

		## Functions to setup required graphs
		self.hardThrsd()
		self.sparseTraining()
		self.lossGraph()
		self.trainGraph()
		self.accuracyGraph()

	## Functions to initilaise Params (Warm start possible with given numpy matrices)
	def initZ(self, Z):
		if Z is None:
			Z = tf.random_normal([self.projectionDimension, self.dataDimension])
		Z = tf.Variable(Z, name='Z', dtype=tf.float32)
		return Z

	def initW(self, W):
		if W is None:
			W = tf.random_normal([self.numClasses*self.totalNodes, self.projectionDimension])
		W = tf.Variable(W, name='W', dtype=tf.float32)
		return W

	def initV(self, V):
		if V is None:
			V = tf.random_normal([self.numClasses*self.totalNodes, self.projectionDimension])
		V = tf.Variable(V, name='V', dtype=tf.float32)
		return V

	def initT(self, T):
		if T is None:
			T = tf.random_normal([self.internalNodes, self.projectionDimension])
		T = tf.Variable(T, name='T', dtype=tf.float32)
		return T

	## Function to get aimed model size
	def getModelSize(self):	
		nnzZ = np.ceil(int(self.Z.shape[0]*self.Z.shape[1])*self.sZ)
		nnzW = np.ceil(int(self.W.shape[0]*self.W.shape[1])*self.sW)
		nnzV = np.ceil(int(self.V.shape[0]*self.V.shape[1])*self.sV)
		nnzT = np.ceil(int(self.T.shape[0]*self.T.shape[1])*self.sT)
		return int((nnzZ+nnzT+nnzV+nnzW)*8)

	## Function to build the Bonsai Tree graph
	def bonsaiGraph(self, X):
		X = tf.reshape(X, [-1,self.dataDimension])
		X_ = tf.divide(tf.matmul(self.Z, X, transpose_b=True), self.projectionDimension)
		
		W_ = self.W[0:(self.numClasses)]
		V_ = self.V[0:(self.numClasses)]

		self.nodeProb = []
		self.nodeProb.append(1)
		
		score_ = self.nodeProb[0]*tf.multiply(tf.matmul(W_, X_), tf.tanh(self.sigma*tf.matmul(V_, X_)))
		for i in range(1, self.totalNodes):
			W_ = self.W[i*self.numClasses:((i+1)*self.numClasses)]
			V_ = self.V[i*self.numClasses:((i+1)*self.numClasses)]

			prob = (1+((-1)**(i+1))*tf.tanh(tf.multiply(self.sigmaI, 
				tf.matmul(tf.reshape(self.T[int(np.ceil(i/2)-1)], [-1, self.projectionDimension]), X_))))

			prob = tf.divide(prob, 2)
			prob = self.nodeProb[int(np.ceil(i/2)-1)]*prob
			self.nodeProb.append(prob)
			score_ += self.nodeProb[i]*tf.multiply(tf.matmul(W_, X_), tf.tanh(self.sigma*tf.matmul(V_, X_)))
			
		return score_, X_, self.T, self.W, self.V, self.Z

	## Functions setting up graphs for IHT and Sparse Retraining
	def hardThrsd(self):
		self.Woph = self.W.assign(self.Wth)
		self.Voph = self.V.assign(self.Vth)
		self.Toph = self.T.assign(self.Tth)
		self.Zoph = self.Z.assign(self.Zth)
		self.hardThresholdGroup = tf.group(self.Woph, self.Voph, self.Toph, self.Zoph)

	def runHardThrsd(self, sess):
		currW = self.Weval.eval()
		currV = self.Veval.eval()
		currZ = self.Zeval.eval()
		currT = self.Teval.eval()

		self.thrsdW = utils.hardThreshold(currW, self.sW)
		self.thrsdV = utils.hardThreshold(currV, self.sV)
		self.thrsdZ = utils.hardThreshold(currZ, self.sZ)
		self.thrsdT = utils.hardThreshold(currT, self.sT)

		fd_thrsd = {self.Wth:self.thrsdW, self.Vth:self.thrsdV, self.Zth:self.thrsdZ, self.Tth:self.thrsdT}
		sess.run(self.hardThresholdGroup, feed_dict=fd_thrsd)

	def sparseTraining(self):
		self.Wops = self.W.assign(self.Wth)
		self.Vops = self.V.assign(self.Vth)
		self.Zops = self.Z.assign(self.Zth)
		self.Tops = self.T.assign(self.Tth)
		self.sparseRetrainGroup = tf.group(self.Wops, self.Vops, self.Tops, self.Zops)

	def runSparseTraining(self, sess):
		currW = self.Weval.eval()
		currV = self.Veval.eval()
		currZ = self.Zeval.eval()
		currT = self.Teval.eval()

		newW = utils.copySupport(self.thrsdW, currW)
		newV = utils.copySupport(self.thrsdV, currV)
		newZ = utils.copySupport(self.thrsdZ, currZ)
		newT = utils.copySupport(self.thrsdT, currT)

		fd_st = {self.Wth:newW, self.Vth:newV, self.Zth:newZ, self.Tth:newT}
		sess.run(self.sparseRetrainGroup, feed_dict=fd_st)

	## Function to build a Loss graph for Bonsai
	def lossGraph(self):
		self.score, self.Xeval, self.Teval, self.Weval, self.Veval, self.Zeval = self.bonsaiGraph(self.x)

		self.regLoss = 0.5*(self.lZ*tf.square(tf.norm(self.Z)) + self.lW*tf.square(tf.norm(self.W)) + 
			self.lV*tf.square(tf.norm(self.V)) + self.lT*tf.square(tf.norm(self.T)))

		if (self.numClasses > 2):
			## need to give users an option to choose the loss
			if True:
				self.marginLoss = utils.multiClassHingeLoss(tf.transpose(self.score), tf.argmax(self.y,1), self.batch_th)
			else:
				self.marginLoss = utils.crossEntropyLoss(tf.transpose(self.score), self.y)
			self.loss = self.marginLoss + self.regLoss
		else:
			self.marginLoss = tf.reduce_mean(tf.nn.relu(1.0 - (2*self.y-1)*tf.transpose(self.score)))
			self.loss = self.marginLoss + self.regLoss

	## Function to set up optimisation for Bonsai
	def trainGraph(self):
		self.trainStep = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)

	## Function to run training step on Bonsai
	def runTraining(self, sess, _feed_dict):
		sess.run([self.trainStep], feed_dict=_feed_dict)

	## Function to build a graph to compute accuracy of the current model
	def accuracyGraph(self):
		if (self.numClasses > 2):
			correctPrediction = tf.equal(tf.argmax(tf.transpose(self.score),1), tf.argmax(self.y,1))
			self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
		else:
			y_ = self.y*2-1
			correctPrediction = tf.multiply(tf.transpose(self.score), y_)
			correctPrediction = tf.nn.relu(correctPrediction)
			correctPrediction = tf.ceil(tf.tanh(correctPrediction))
			self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

