import utils
import tensorflow as tf
import numpy as np

class Bonsai:
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
		
		self.lW = lW
		self.lV = lV
		self.lT = lT
		self.lZ = lZ
		
		self.sW = sW
		self.sV = sV
		self.sT = sT
		self.sZ = sZ

		self.internalNodes = 2**self.treeDepth - 1
		self.totalNodes = 2*self.internalNodes + 1

		self.W = self.initW(W)
		self.V = self.initV(V)
		self.T = self.initT(T)
		self.Z = self.initZ(Z)

		self.W_th = tf.placeholder(tf.float32, name='W_th')
		self.V_th = tf.placeholder(tf.float32, name='V_th')
		self.Z_th = tf.placeholder(tf.float32, name='Z_th')
		self.T_th = tf.placeholder(tf.float32, name='T_th')

		self.W_st = tf.placeholder(tf.float32, name='W_st')
		self.V_st = tf.placeholder(tf.float32, name='V_st')
		self.Z_st = tf.placeholder(tf.float32, name='Z_st')
		self.T_st = tf.placeholder(tf.float32, name='T_st')
		
		if feats is None:
			self.x = tf.placeholder("float", [None, self.dataDimension])
		else:
			self.x = feats

		self.y = tf.placeholder("float", [None, self.numClasses])
		self.batch_th = tf.placeholder(tf.int64, name='batch_th')

		self.sigmaI = 1.0
		if lr is not None:
			self.learning_rate = lr
		else:
			self.learning_rate = 0.01

		self.hardThrsd()
		self.sparseTraining()
		self.lossGraph()
		self.trainGraph()
		self.accuracyGraph()

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

	def getModelSize(self):	
		nnzZ = np.ceil(int(self.Z.shape[0]*self.Z.shape[1])*self.sZ)
		nnzW = np.ceil(int(self.W.shape[0]*self.W.shape[1])*self.sW)
		nnzV = np.ceil(int(self.V.shape[0]*self.V.shape[1])*self.sV)
		nnzT = np.ceil(int(self.T.shape[0]*self.T.shape[1])*self.sT)
		return int((nnzZ+nnzT+nnzV+nnzW)*8)

	def bonsaiGraph(self, X):
		X = tf.reshape(X,[-1,self.dataDimension])
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
			score_ = score_ + self.nodeProb[i]*tf.multiply(tf.matmul(W_, X_), tf.tanh(self.sigma*tf.matmul(V_, X_)))
			
		return score_, X_, self.T, self.W, self.V, self.Z

	def hardThrsd(self):
		self.W_op1 = self.W.assign(self.W_th)
		self.V_op1 = self.V.assign(self.V_th)
		self.T_op1 = self.T.assign(self.T_th)
		self.Z_op1 = self.Z.assign(self.Z_th)
		self.hard_thrsd_grp = tf.group(self.W_op1, self.V_op1, self.T_op1, self.Z_op1)

	def runHardThrsd(self, sess):
		self.W_old = self.W_eval.eval()
		self.V_old = self.V_eval.eval()
		self.Z_old = self.Z_eval.eval()
		self.T_old = self.T_eval.eval()

		self.W_new = utils.hard_thrsd(self.W_old, self.sW)
		self.V_new = utils.hard_thrsd(self.V_old, self.sV)
		self.Z_new = utils.hard_thrsd(self.Z_old, self.sZ)
		self.T_new = utils.hard_thrsd(self.T_old, self.sT)

		fd_thrsd = {self.W_th:self.W_new, self.V_th:self.V_new, self.Z_th:self.Z_new, self.T_th:self.T_new}
		sess.run(self.hard_thrsd_grp, feed_dict=fd_thrsd)

	def sparseTraining(self):
		self.W_op2 = self.W.assign(self.W_st)
		self.V_op2 = self.V.assign(self.V_st)
		self.Z_op2 = self.Z.assign(self.Z_st)
		self.T_op2 = self.T.assign(self.T_st)
		self.sparse_retrain_grp = tf.group(self.W_op2, self.V_op2, self.T_op2, self.Z_op2)

	def runSparseTraining(self, sess):
		self.W_old = self.W_eval.eval()
		self.V_old = self.V_eval.eval()
		self.Z_old = self.Z_eval.eval()
		self.T_old = self.T_eval.eval()

		W_new1 = utils.copy_support(self.W_new, self.W_old)
		V_new1 = utils.copy_support(self.V_new, self.V_old)
		Z_new1 = utils.copy_support(self.Z_new, self.Z_old)
		T_new1 = utils.copy_support(self.T_new, self.T_old)

		fd_st = {self.W_st:W_new1, self.V_st:V_new1, self.Z_st:Z_new1, self.T_st:T_new1}
		sess.run(self.sparse_retrain_grp, feed_dict=fd_st)

	def lossGraph(self):
		self.score, self.X_eval, self.T_eval, self.W_eval, self.V_eval, self.Z_eval = self.bonsaiGraph(self.x)

		self.reg_loss = 0.5*(self.lZ*tf.square(tf.norm(self.Z)) + self.lW*tf.square(tf.norm(self.W)) + 
			self.lV*tf.square(tf.norm(self.V)) + self.lT*tf.square(tf.norm(self.T)))

		if (self.numClasses > 2):
			self.margin_loss = utils.multi_class_hinge_loss(tf.transpose(self.score), tf.argmax(self.y,1), self.batch_th)
			self.loss = self.margin_loss + self.reg_loss
		else:
			self.margin_loss = tf.reduce_mean(tf.nn.relu(1.0 - (2*self.y-1)*tf.transpose(self.score)))
			self.loss = self.margin_loss + self.reg_loss

	def trainGraph(self):
		self.train_stepW = (tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.W]))
		self.train_stepV = (tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.V]))
		self.train_stepT = (tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.T]))
		self.train_stepZ = (tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.Z]))

	def runTraining(self, sess, _feed_dict):
		## Done independently to allow the order to remain the same to match C++ version and to avoid race during concurrency
		sess.run([self.train_stepW], feed_dict=_feed_dict)
		sess.run([self.train_stepV], feed_dict=_feed_dict)
		sess.run([self.train_stepT], feed_dict=_feed_dict)
		sess.run([self.train_stepZ], feed_dict=_feed_dict)

	def accuracyGraph(self):
		if (self.numClasses > 2):
			correct_prediction = tf.equal(tf.argmax(tf.transpose(self.score),1), tf.argmax(self.y,1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		else:
			y_ = self.y*2-1
			correct_prediction = tf.multiply(tf.transpose(self.score), y_)
			correct_prediction = tf.nn.relu(correct_prediction)
			correct_prediction = tf.ceil(tf.tanh(correct_prediction))
			self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

