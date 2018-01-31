import utils
import tensorflow as tf
import numpy as np
from bonsai import Bonsai

tf.set_random_seed(42)
np.random.seed(42)

args = utils.getArgs()

sigma = args.sigma
depth = args.depth

embed_dims = args.proj_dim
reg_Z = args.rZ
reg_T = args.rT
reg_W = args.rW
reg_V = args.rV

total_epochs = args.epochs

learning_rate = args.learning_rate

data_dir = args.data_dir

(inp_dims, n_classes, 
	train_feats, train_labels, test_feats, test_labels) = utils.preProcessData(data_dir)

spar_Z = args.sZ

if n_classes == 2:
	spar_W = 1
	spar_V = 1
	spar_T = 1
else:
	spar_W = 0.2
	spar_V = 0.2
	spar_T = 0.2

if args.sW is not None:
	spar_W = args.sW
if args.sV is not None:
	spar_V = args.sV
if args.sT is not None:
	spar_T = args.sT

if args.batch_size is None:
	batch_size = np.maximum(100, int(np.ceil(np.sqrt(train_labels.shape[0]))))
else:
	batch_size = args.batch_size

bonsaiObj = Bonsai(n_classes, inp_dims, embed_dims, depth, sigma, 
	reg_W, reg_T, reg_V, reg_Z, spar_W, spar_T, spar_V, spar_Z, lr = learning_rate)

sess = tf.InteractiveSession()
sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_variables(tf.local_variables())))
saver = tf.train.Saver()   ##for saving the model

num_iters=train_feats.shape[0]/batch_size

total_batches = num_iters*total_epochs

counter = 0
if bonsaiObj.numClasses > 2:
	trimlevel = 15
else:
	trimlevel = 5
iht_done = 0


for i in range(total_epochs):
	print("\nEpoch Number: "+str(i))

	accu = 0.0
	for j in range(num_iters):

		if counter == 0:
			print("\n******************** Dense Training Phase Started ********************\n")

		if ((counter == 0) or (counter == int(total_batches/3)) or (counter == int(2*total_batches/3))):
			bonsaiObj.sigmaI = 1
			iters_phase = 0

		elif (iters_phase%100 == 0):
			indices = np.random.choice(train_feats.shape[0],100)
			batch_x = train_feats[indices,:]
			batch_y = train_labels[indices,:]
			batch_y = np.reshape(batch_y, [-1, bonsaiObj.numClasses])
			_feed_dict = {bonsaiObj.x: batch_x, bonsaiObj.y: batch_y}
			x_cap_eval = bonsaiObj.X_eval.eval(feed_dict=_feed_dict)
			T_eval = bonsaiObj.T_eval.eval()
			sum_tr = 0.0
			for k in range(0, bonsaiObj.internalNodes):
				sum_tr = sum_tr + (np.sum(np.abs(np.dot(T_eval[k], x_cap_eval))))

			if(bonsaiObj.internalNodes > 0):
				sum_tr = sum_tr/(100*bonsaiObj.internalNodes)
				sum_tr = 0.1/sum_tr
			else:
				sum_tr = 0.1
			sum_tr = min(1000,sum_tr*(2**(float(iters_phase)/(float(total_batches)/30.0))))

			bonsaiObj.sigmaI = sum_tr
		
		iters_phase = iters_phase + 1
		batch_x = train_feats[j*batch_size:(j+1)*batch_size]
		batch_y = train_labels[j*batch_size:(j+1)*batch_size]
		batch_y = np.reshape(batch_y, [-1, bonsaiObj.numClasses])

		if bonsaiObj.numClasses > 2:
			_feed_dict = {bonsaiObj.x: batch_x, bonsaiObj.y: batch_y, bonsaiObj.batch_th: batch_y.shape[0]}
		else:
			_feed_dict = {bonsaiObj.x: batch_x, bonsaiObj.y: batch_y}


		batchLoss = bonsaiObj.runTraining(sess, _feed_dict)

		temp = bonsaiObj.accuracy.eval(feed_dict=_feed_dict)
		accu = temp+accu

		if (counter >= int(total_batches/3) and (counter < int(2*total_batches/3)) and counter%trimlevel == 0):
			bonsaiObj.runHardThrsd(sess)
			if iht_done == 0:
				print("\n******************** IHT Phase Started ********************\n")
			iht_done = 1
		elif ((iht_done == 1 and counter >= int(total_batches/3) and (counter < int(2*total_batches/3)) 
			and counter%trimlevel != 0) or (counter >= int(2*total_batches/3))):
			bonsaiObj.runSparseTraining(sess)
			if counter == int(2*total_batches/3):
				print("\n******************** Sprase Retraining Phase Started ********************\n")
		counter = counter + 1

	print("Train accuracy "+str(accu/num_iters)) 

	if bonsaiObj.numClasses > 2:
		_feed_dict={bonsaiObj.x: test_feats, bonsaiObj.y: test_labels, bonsaiObj.batch_th: test_labels.shape[0]}
	else:
		_feed_dict={bonsaiObj.x: test_feats, bonsaiObj.y: test_labels}

	old = bonsaiObj.sigmaI
	bonsaiObj.sigmaI = 1e9

	test_acc = bonsaiObj.accuracy.eval(feed_dict=_feed_dict)
	if iht_done == 0:
		max_acc = -10000
	else:
		max_acc = max(max_acc, test_acc)

	print("Test accuracy %g"%test_acc)

	loss_new = bonsaiObj.loss.eval(feed_dict=_feed_dict)
	reg_loss_new = bonsaiObj.reg_loss.eval(feed_dict=_feed_dict)
	print("Margin_Loss + Reg_Loss: " + str(loss_new - reg_loss_new) + " + " + str(reg_loss_new) + " = " + str(loss_new) + "\n")

	bonsaiObj.sigmaI = old  

print("Maximum Test accuracy at compressed model size(including early stopping): " + str(max_acc) + " Final Test Accuracy: " + str(test_acc))
print("\nNon-Zeros: " + str(bonsaiObj.getModelSize()) + " Model Size: " + 
	str(float(bonsaiObj.getModelSize())/1024.0) + " KB \n")

np.save("W.npy", bonsaiObj.W_eval.eval())
np.save("V.npy", bonsaiObj.V_eval.eval())
np.save("Z.npy", bonsaiObj.Z_eval.eval())
np.save("T.npy", bonsaiObj.T_eval.eval())