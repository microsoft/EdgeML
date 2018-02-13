import utils
import tensorflow as tf
import numpy as np
import sys
from bonsai import Bonsai

## Fixing seeds for reproducibility
tf.set_random_seed(42)
np.random.seed(42)

## Hyper Param pre-processing
args = utils.getArgs()

sigma = args.sigma
depth = args.depth

projectionDimension = args.projDim
regZ = args.rZ
regT = args.rT
regW = args.rW
regV = args.rV

totalEpochs = args.epochs

learningRate = args.learningRate

data_dir = args.data_dir

(dataDimension, numClasses, 
	Xtrain, Ytrain, Xtest, Ytest) = utils.preProcessData(data_dir)

sparZ = args.sZ

if numClasses == 2:
	sparW = 1
	sparV = 1
	sparT = 1
else:
	sparW = 0.2
	sparV = 0.2
	sparT = 0.2

if args.sW is not None:
	sparW = args.sW
if args.sV is not None:
	sparV = args.sV
if args.sT is not None:
	sparT = args.sT

if args.batchSize is None:
	batchSize = np.maximum(100, int(np.ceil(np.sqrt(Ytrain.shape[0]))))
else:
	batchSize = args.batchSize


## Creation of Bonsai Object
bonsaiObj = Bonsai(numClasses, dataDimension, projectionDimension, depth, sigma, 
	regW, regT, regV, regZ, sparW, sparT, sparV, sparZ, lr = learningRate)

sess = tf.InteractiveSession()
sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_variables(tf.local_variables())))
saver = tf.train.Saver()   ## Use it incase of saving the model

numIters = Xtrain.shape[0]/batchSize

totalBatches = numIters*totalEpochs

counter = 0
if bonsaiObj.numClasses > 2:
	trimlevel = 15
else:
	trimlevel = 5
ihtDone = 0


for i in range(totalEpochs):
	print("\nEpoch Number: "+str(i))

	trainAcc = 0.0
	for j in range(numIters):

		if counter == 0:
			print("\n******************** Dense Training Phase Started ********************\n")

		## Updating the indicator sigma
		if ((counter == 0) or (counter == int(totalBatches/3)) or (counter == int(2*totalBatches/3))):
			bonsaiObj.sigmaI = 1
			itersInPhase = 0

		elif (itersInPhase%100 == 0):
			indices = np.random.choice(Xtrain.shape[0],100)
			batchX = Xtrain[indices,:]
			batchY = Ytrain[indices,:]
			batchY = np.reshape(batchY, [-1, bonsaiObj.numClasses])

			_feed_dict = {bonsaiObj.x: batchX, bonsaiObj.y: batchY}
			Xcapeval = bonsaiObj.Xeval.eval(feed_dict=_feed_dict)
			Teval = bonsaiObj.Teval.eval()
			
			sum_tr = 0.0
			for k in range(0, bonsaiObj.internalNodes):
				sum_tr += (np.sum(np.abs(np.dot(Teval[k], Xcapeval))))

			if(bonsaiObj.internalNodes > 0):
				sum_tr /= (100*bonsaiObj.internalNodes)
				sum_tr = 0.1/sum_tr
			else:
				sum_tr = 0.1
			sum_tr = min(1000,sum_tr*(2**(float(itersInPhase)/(float(totalBatches)/30.0))))

			bonsaiObj.sigmaI = sum_tr
		
		itersInPhase += 1
		batchX = Xtrain[j*batchSize:(j+1)*batchSize]
		batchY = Ytrain[j*batchSize:(j+1)*batchSize]
		batchY = np.reshape(batchY, [-1, bonsaiObj.numClasses])

		if bonsaiObj.numClasses > 2:
			_feed_dict = {bonsaiObj.x: batchX, bonsaiObj.y: batchY, bonsaiObj.batch_th: batchY.shape[0]}
		else:
			_feed_dict = {bonsaiObj.x: batchX, bonsaiObj.y: batchY}

		## Mini-batch training
		batchLoss = bonsaiObj.runTraining(sess, _feed_dict)

		batchAcc = bonsaiObj.accuracy.eval(feed_dict=_feed_dict)
		trainAcc += batchAcc

		## Training routine involving IHT and sparse retraining
		if (counter >= int(totalBatches/3) and (counter < int(2*totalBatches/3)) and counter%trimlevel == 0):
			bonsaiObj.runHardThrsd(sess)
			if ihtDone == 0:
				print("\n******************** IHT Phase Started ********************\n")
			ihtDone = 1
		elif ((ihtDone == 1 and counter >= int(totalBatches/3) and (counter < int(2*totalBatches/3)) 
			and counter%trimlevel != 0) or (counter >= int(2*totalBatches/3))):
			bonsaiObj.runSparseTraining(sess)
			if counter == int(2*totalBatches/3):
				print("\n******************** Sprase Retraining Phase Started ********************\n")
		counter += 1

	print("Train accuracy "+str(trainAcc/numIters)) 

	if bonsaiObj.numClasses > 2:
		_feed_dict = {bonsaiObj.x: Xtest, bonsaiObj.y: Ytest, bonsaiObj.batch_th: Ytest.shape[0]}
	else:
		_feed_dict = {bonsaiObj.x: Xtest, bonsaiObj.y: Ytest}

	## this helps in direct testing instead of extracting the model out
	oldSigmaI = bonsaiObj.sigmaI
	bonsaiObj.sigmaI = 1e9

	testAcc = bonsaiObj.accuracy.eval(feed_dict=_feed_dict)
	if ihtDone == 0:
		maxTestAcc = -10000
		maxTestAccEpoch = i
	else:
		if maxTestAcc <= testAcc:
			maxTestAccEpoch = i
			maxTestAcc = testAcc

	print("Test accuracy %g"%testAcc)

	testLoss = bonsaiObj.loss.eval(feed_dict=_feed_dict)
	regTestLoss = bonsaiObj.regLoss.eval(feed_dict=_feed_dict)
	print("MarginLoss + RegLoss: " + str(testLoss - regTestLoss) + " + " + str(regTestLoss) + " = " + str(testLoss) + "\n")

	bonsaiObj.sigmaI = oldSigmaI
	sys.stdout.flush()

print("Maximum Test accuracy at compressed model size(including early stopping): " 
	+ str(maxTestAcc) + " at Epoch: " + str(maxTestAccEpoch) + "\nFinal Test Accuracy: " + str(testAcc))
print("\nNon-Zeros: " + str(bonsaiObj.getModelSize()) + " Model Size: " + 
	str(float(bonsaiObj.getModelSize())/1024.0) + " KB \n")

np.save("W.npy", bonsaiObj.Weval.eval())
np.save("V.npy", bonsaiObj.Veval.eval())
np.save("Z.npy", bonsaiObj.Zeval.eval())
np.save("T.npy", bonsaiObj.Teval.eval())