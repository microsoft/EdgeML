import tensorflow as tf
import numpy as np
import argparse

def check_int_positive(value):
	ivalue = int(value)
	if ivalue <= 0:
		raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
	return ivalue

def check_int_nonneg(value):
	ivalue = int(value)
	if ivalue < 0:
		raise argparse.ArgumentTypeError("%s is an invalid non-neg int value" % value)
	return ivalue

def check_float_nonneg(value):
	fvalue = float(value)
	if fvalue < 0:
		raise argparse.ArgumentTypeError("%s is an invalid non-neg float value" % value)
	return fvalue

def check_float_positive(value):
	fvalue = float(value)
	if fvalue <= 0:
		raise argparse.ArgumentTypeError("%s is an invalid positive float value" % value)
	return fvalue

def getArgs():
	parser = argparse.ArgumentParser(description='HyperParams for Bonsai Algorithm')
	parser.add_argument('-dir', '--data_dir', required=True, help='Data directory containing train.npy and test.npy')
	
	parser.add_argument('-d', '--depth', type=check_int_nonneg, default=2, help='Depth of Bonsai Tree (default: 2 try: [0, 1, 3])')
	parser.add_argument('-p', '--proj_dim', type=check_int_positive, default=10, help='Projection Dimension (default: 20 try: [5, 10, 30])')
	parser.add_argument('-s', '--sigma', type=float, default=1.0, help='Parameter for sigmoid sharpness (default: 1.0 try: [3.0, 0.05, 0.1]')
	parser.add_argument('-e', '--epochs', type=check_int_positive, default=42, help='Total Epochs (default: 42 try:[100, 150, 60])')
	parser.add_argument('-b', '--batch_size', type=check_int_positive, help='Batch Size to be used (default: max(100, sqrt(train_samples)))')
	parser.add_argument('-lr', '--learning_rate', type=check_float_positive, default=0.01, help='Initial Learning rate for Adam Oprimizer (default: 0.01)')

	parser.add_argument('-rW', type=float, default=0.0001, help='Regularizer for predictor parameter W  (default: 0.0001 try: [0.01, 0.001, 0.00001])')
	parser.add_argument('-rV', type=float, default=0.0001, help='Regularizer for predictor parameter V  (default: 0.0001 try: [0.01, 0.001, 0.00001])')
	parser.add_argument('-rT', type=float, default=0.0001, help='Regularizer for branching parameter Theta  (default: 0.0001 try: [0.01, 0.001, 0.00001])')
	parser.add_argument('-rZ', type=float, default=0.00001, help='Regularizer for projection parameter Z  (default: 0.00001 try: [0.001, 0.0001, 0.000001])')

	parser.add_argument('-sW', type=check_float_positive, help='Sparsity for predictor parameter W  (default: For Binary classification 1.0 else 0.2 try: [0.1, 0.3, 0.5])')
	parser.add_argument('-sV', type=check_float_positive, help='Sparsity for predictor parameter V  (default: For Binary classification 1.0 else 0.2 try: [0.1, 0.3, 0.5])')
	parser.add_argument('-sT', type=check_float_positive, help='Sparsity for branching parameter Theta  (default: For Binary classification 1.0 else 0.2 try: [0.1, 0.3, 0.5])')
	parser.add_argument('-sZ', type=check_float_positive, default=0.2, help='Sparsity for projection parameter Z  (default: 0.2 try: [0.1, 0.3, 0.5])')	

	return parser.parse_args()


def multi_class_hinge_loss(logits, label, batch_th):
	flat_logits = tf.reshape(logits, [-1,])
	correct_id = tf.range(0, batch_th) * logits.shape[1] + label
	correct_logit = tf.gather(flat_logits, correct_id)

	max_label = tf.argmax(logits, 1)
	top2, _ = tf.nn.top_k(logits, k=2, sorted=True)

	wrong_max_logit = tf.where(tf.equal(max_label, label), top2[:,1], top2[:,0])

	return tf.reduce_mean(tf.nn.relu(1. + wrong_max_logit - correct_logit))

def hard_thrsd(A, s):
	A_ = np.copy(A)
	A_ = A_.ravel()
	if len(A_) > 0:
		th = np.percentile(np.abs(A_), (1 - s)*100.0, interpolation='higher')
		A_[np.abs(A_)<th] = 0.0
	A_ = A_.reshape(A.shape)
	return A_

def copy_support(src, dest):
	support = np.nonzero(src)
	dest_ = dest
	dest = np.zeros(dest_.shape)
	dest[support] = dest_[support]
	return dest

def preProcessData(data_dir):
	train = np.load(data_dir + '/train.npy')
	test = np.load(data_dir + '/test.npy')

	inp_dims = int(train.shape[1]) - 1

	train_feats = train[:,1:inp_dims+1]
	train_lbl = train[:,0]
	n_classes = max(train_lbl) - min(train_lbl) + 1

	test_feats = test[:,1:inp_dims+1]
	test_lbl = test[:,0]

	n_classes = int(max(n_classes, max(test_lbl) - min(test_lbl) + 1))

	# Mean Var Normalisation
	mean = np.mean(train_feats,0)
	std = np.std(train_feats,0)
	std[std[:]<0.000001]=1
	train_feats = (train_feats-mean)/std

	test_feats = (test_feats-mean)/std
	# End Mean Var normalisation

	lab = train_lbl.astype('uint8')
	lab = np.array(lab) - min(lab)

	lab1 = np.zeros((train_feats.shape[0], n_classes))
	lab1[np.arange(train_feats.shape[0]), lab] = 1
	train_labels = lab1
	if (n_classes == 2):
		train_labels = lab
		train_labels = np.reshape(train_labels, [-1, 1])

	lab = test_lbl.astype('uint8')
	lab = np.array(lab)-min(lab)
	lab1 = np.zeros((test_feats.shape[0], n_classes))
	lab1[np.arange(test_feats.shape[0]), lab] = 1
	test_labels = lab1
	train_bias = np.ones([train_feats.shape[0],1]);
	train_feats = np.append(train_feats,train_bias,axis=1)
	test_bias = np.ones([test_feats.shape[0],1]);
	test_feats = np.append(test_feats,test_bias,axis=1)
	if (n_classes == 2):
		test_labels = lab
		test_labels = np.reshape(test_labels, [-1, 1])

	return inp_dims, n_classes, train_feats, train_labels, test_feats, test_labels
