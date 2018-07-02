from __future__ import print_function
import tensorflow as tf
import pandas as pd
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import edgeml.utils as utils


class ProtoNNTrainer:
	def __init__(self, currDir,protoNNObj, regW, regB, regZ,
				 sparcityW, sparcityB, sparcityZ,
				 learningRate, X, Y , lossType='l2'):
		'''
		protoNNObj: An instance of ProtoNN class. This instance
			will be trained.
		regW, regB, regZ: Regularization constants for W, B, and
			Z matrices of protoNN.
		sparcityW, sparcityB, sparcityZ: Sparcity constraints
			for W, B and Z matrices.
		X, Y : Placeholders for data and labels.
			X [-1, featureDimension]
			Y [-1, num Labels]
		lossType: ['l2', 'xentropy']
		'''
		self.protoNNObj = protoNNObj
		self.__regW = regW
		self.__regB = regB
		self.__regZ = regZ
		self.__sW = sparcityW
		self.__sB = sparcityB
		self.__sZ = sparcityZ
		self.__lR = learningRate
		self.X = X
		self.Y = Y
		self.sparseTraining = True
		if (sparcityW == 1.0) and (sparcityB == 1.0) and (sparcityZ == 1.0):
			self.parseTraining = False
			print("Sparse training disabled.", file = sys.stderr)
		# Define placeholders for sparse training
		self.W_th = None
		self.B_th = None
		self.Z_th = None

		self.__lossType = lossType
		self.__validInit = False
		self.__validInit = self.__validateInit()
		self.__protoNNOut = protoNNObj(X, Y)
		self.loss = self.__lossGraph()
		self.trainStep = self.__trainGraph()
		self.__hthOp = self.__getHardThresholdOp()
		self.accuracy = protoNNObj.getAccuracyOp()

	def __validateInit(self):
		self.__validInit = False
		msg = "Sparcity value should be between"
		msg += " 0 and 1 (both inclusive)."
		assert self.__sW >= 0. and self.__sW <= 1., 'W:' + msg
		assert self.__sB >= 0. and self.__sB <= 1., 'B:' + msg
		assert self.__sZ >= 0. and self.__sZ <= 1., 'Z:' + msg
		d, dcap, m, L, _ = self.protoNNObj.getHyperParams()
		msg = 'Y should be of dimension [-1, num labels/classes]'
		msg += ' specified as part of ProtoNN object.'
		assert (len(self.Y.shape)) == 2, msg
		assert (self.Y.shape[1] == L), msg
		msg = 'X should be of dimension [-1, featureDimension]'
		msg += ' specified as part of ProtoNN object.'
		assert (len(self.X.shape) == 2), msg
		assert (self.X.shape[1] == d), msg
		self.__validInit = True
		msg = 'Values can be \'l2\', or \'xentropy\''
		if self.__lossType not in ['l2', 'xentropy']:
			raise ValueError(msg)
		return True

	def __lossGraph(self):
		pnnOut = self.__protoNNOut
		l1, l2, l3 = self.__regW, self.__regB, self.__regZ
		W, B, Z, _ = self.protoNNObj.getModelMatrices()
		if self.__lossType == 'l2':
			with tf.name_scope('protonn-l2-loss'):
				#Maybe a huber loss ?
				#loss_0 = tf.nn.l2_loss(self.Y - pnnOut)
				loss_0 = tf.losses.huber_loss(self.Y , pnnOut)
				reg = l1 * tf.nn.l2_loss(W) + l2 * tf.nn.l2_loss(B)
				reg += l3 * tf.nn.l2_loss(Z)
				loss = loss_0 + reg
		elif self.__lossType == 'xentropy':
			with tf.name_scope('protonn-xentropy-loss'):
				loss_0 = tf.nn.softmax_cross_entropy_with_logits(logits=pnnOut,
																 labels=self.Y)
				loss_0 = tf.reduce_mean(loss_0)
				reg = l1 * tf.nn.l2_loss(W) + l2 * tf.nn.l2_loss(B)
				reg += l3 * tf.nn.l2_loss(Z)
				loss = loss_0 + reg

		return loss

	def __trainGraph(self):
		with tf.name_scope('protonn-gradient-adam'):
			trainStep = tf.train.AdamOptimizer(self.__lR)
			trainStep = trainStep.minimize(self.loss)
		return trainStep

	def __getHardThresholdOp(self):
		W, B, Z, _ = self.protoNNObj.getModelMatrices()
		self.W_th = tf.placeholder(tf.float32, name='W_th')
		self.B_th = tf.placeholder(tf.float32, name='B_th')
		self.Z_th = tf.placeholder(tf.float32, name='Z_th')
		with tf.name_scope('hard-threshold-assignments'):
			hard_thrsd_W = W.assign(self.W_th)
			hard_thrsd_B = B.assign(self.B_th)
			hard_thrsd_Z = Z.assign(self.Z_th)
			hard_thrsd_op = tf.group(hard_thrsd_W, hard_thrsd_B, hard_thrsd_Z)
		return hard_thrsd_op

	def train(self, currDir,batchSize, totalEpochs, sess,
			  x_train, x_val, y_train, y_val,
			  noInit=False, redirFile=sys.stdout ,printStep=10):
		'''
		Dense + IHT training of ProtoNN
		noInit: if not to perform initialization (reuse previous init)
		printStep: Number of batches after which loss is to be printed
		TODO: Implement dense - IHT - sparse
		'''
		ndict = {}
		d, d_cap, m, L, gamma = self.protoNNObj.getHyperParams()
		assert batchSize >= 1, 'Batch size should be positive integer'
		assert totalEpochs >= 1, 'Total epochs should be psotive integer'
		assert x_train.ndim == 2, 'Expected training data to be of rank 2'
		assert x_train.shape[1] == d, 'Expected x_train to be [-1, %d]' % d
		assert x_val.ndim == 2, 'Expected validation data to be of rank 2'
		assert x_val.shape[1] == d, 'Expected x_val to be [-1, %d]' % d
		assert y_train.ndim == 2, 'Expected training labels to be of rank 2'
		assert y_train.shape[1] == L, 'Expected y_train to be [-1, %d]' % L
		assert y_val.ndim == 2, 'Expected valing labels to be of rank 2'
		assert y_val.shape[1] == L, 'Expected y_val to be [-1, %d]' % L

		# Numpy will throw asserts for arrays
		if sess is None:
			raise ValueError('sess must be valid tensorflow session.')

		print ("trainNumBatches : ",len(x_train))
		trainNumBatches = int(np.ceil(len(x_train) / batchSize))
		valNumBatches = int(np.ceil(len(x_val) / batchSize))
		print ("valNumBatches : ",len(x_val))
		x_train_batches = np.array_split(x_train, trainNumBatches)
		y_train_batches = np.array_split(y_train, trainNumBatches)
		x_val_batches = np.array_split(x_val, valNumBatches)
		y_val_batches = np.array_split(y_val, valNumBatches)
		if not noInit:
			#sess.run(tf.global_variables_initializer())
			sess.run(tf.group(tf.initialize_all_variables(),
							  tf.initialize_variables(tf.local_variables())))
		X, Y = self.X, self.Y
		W, B, Z, _ = self.protoNNObj.getModelMatrices()
		preds = []
		test = []
		train_preds = []
		train_test = []

		train_loss = []
		test_loss = []

		print ("Length of x_train_batches : ",len(x_train_batches[0]))
		for epoch in range(totalEpochs):
			'''
			Gamma value will be computed inside the function and updated.
			'''
			self.protoNNObj.updateGamma(sess = sess,x_train = x_train)

			ntrain_p = []
			ntrain_t = []
			trainloss = 0.0
			trainacc=0.0
			for i in range(len(x_train_batches)):
				batch_x = x_train_batches[i]
				batch_y = y_train_batches[i]
				feed_dict = {
					X: batch_x,
					Y: batch_y
				}
				sess.run(self.trainStep, feed_dict=feed_dict)

				loss, acc , train_predictions = sess.run([self.loss, self.accuracy,self.protoNNObj.predictions],
									 feed_dict=feed_dict)
				trainacc += (acc[0]+acc[1])/2.0
				trainloss += loss
				'''
				if i % printStep == 0:
					msg = "Epoch: %3d Batch: %3d" % (epoch, i)
					msg += " Loss: %3.5f Accuracy: %2.5f" % (loss, (acc[0]+acc[1])/2.0)
					print(msg, file=redirFile)
				'''
				ntrain_p.append(np.concatenate(list(train_predictions),axis=0))
				ntrain_t.append(np.concatenate(list(batch_y),axis=0))

			trainacc /= len(x_train_batches)
			trainloss /= len(x_train_batches)
			msg = "Epoch: %3d" % (epoch)
			msg += " Loss: %3.5f Accuracy: %2.5f" % (trainloss, (acc[0]+acc[1])/2.0)
			print(msg, file=redirFile)

			if(epoch == totalEpochs-1):
				train_preds = ntrain_p
				train_test = ntrain_t

			# Perform Hard thresholding
			print ("Epoch : ",epoch)
			if self.sparseTraining:
				W_, B_, Z_ = sess.run([W, B, Z])
				print ("W : ",np.isnan(W_))
				print ("B : ",np.isnan(B_))
				print ("Z : ",np.isnan(Z_))
				fd_thrsd = {
					self.W_th: utils.hardThreshold(W_, self.__sW),
					self.B_th: utils.hardThreshold(B_, self.__sB),
					self.Z_th: utils.hardThreshold(Z_, self.__sZ)
				}
				sess.run(self.__hthOp, feed_dict=fd_thrsd)
			train_loss.append(trainloss)

			del preds[:]
			del test [:]
			#if (epoch + 1) % 3 == 0:
			#if(epoch == totalEpochs-1):
			if (True):
			#Print the validation loss after every epcoh.
				acc = 0.0
				testloss = 0.0
				for j in range(len(x_val_batches)):
					npreds = []
					batch_x = x_val_batches[j]
					batch_y = y_val_batches[j]
					feed_dict = {
						X: batch_x,
						Y: batch_y
					}
					acc_, loss_,npreds = sess.run([self.accuracy, self.loss,self.protoNNObj.predictions],feed_dict=feed_dict)
					preds.append(np.concatenate(npreds,axis=0))
					test.append(np.concatenate(batch_y,axis=0))
					acc += (acc_[0]+acc_[1])/2.0
					testloss += loss_
				acc /= len(y_val_batches)
				testloss /= len(y_val_batches)
				test_loss.append(testloss)
				print("Test Loss: %2.5f Accuracy: %2.5f" % (testloss, acc))


		#---------Saving the loss across training and validation set-----------#
		print ("Saving both train and test losses.")
		#pickle.dump(train_loss,open("train_loss.pkl","wb"))
		#pickle.dump(test_loss,open("test_loss.pkl","wb"))
		np.save("train_loss.npy",np.array(train_loss))
		np.save("test_loss.npy",np.array(test_loss))
		print ("Saved the train and test losses.")
		'''
		print ("Plots of train and validation losses.")
		plt.figure(figsize=(10,10))
		plt.plot(train_loss)
		plt.plot(test_loss)
		plt.legend(["Train Loss","Test Loss"])
		plt.show()
		'''
		print ("Combining both train and test preds. ")
		print ("Lens : ",len(train_preds) , len(train_test))
		split = np.concatenate(train_preds,axis=0).shape[0]
		print (split)
		trainer = np.hstack((np.concatenate(train_preds,axis=0).reshape((-1,1)),(np.concatenate(train_test,axis=0).reshape((-1,1)))))
		tester = np.hstack((np.concatenate(preds,axis=0).reshape((-1,1)),(np.concatenate(test,axis=0).reshape((-1,1)))))
		print ("Shapes of train and test : ",trainer.shape , tester.shape)
		saver = np.vstack((trainer,tester))
		print ("Final Shape of Train and Test combined : ",saver.shape)

		print ("Saving the predictions of the model !!")
		pd.DataFrame(saver).to_csv(currDir+"/"+"Ppreds.csv",index=False)
		preds = np.concatenate(preds,axis=0)
		test = np.concatenate(test,axis=0)
		print ("---------------------------------------------")
		print ("Metrics")
		print ("R2 : ",r2_score(test,preds))
		ndict["r2"] = r2_score(test,preds)
		print ("MAE : ",mean_absolute_error(test,preds))
		ndict["mae"] = mean_absolute_error(test,preds)
		print ("RMSE : ",np.sqrt(mean_squared_error(test,preds)))
		ndict["rmse"] = np.sqrt(mean_squared_error(test,preds))
		print ("---------------------------------------------")


		fig, ax = plt.subplots( nrows=1, ncols=1)
		plt.grid(True)
		ax.plot(preds)
		ax.plot(test)
		ax.legend(["Preds","Test"])
		plt.show()


		return saver[:,0].reshape((-1,1)) , saver[:,1].reshape((-1,1))
