import bonsaiPreProcess
import tensorflow as tf
import numpy as np
from bonsaiTrainer import BonsaiTrainer
from bonsai import Bonsai


# Fixing seeds for reproducibility
tf.set_random_seed(42)
np.random.seed(42)

# Hyper Param pre-processing
args = bonsaiPreProcess.getArgs()

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
    Xtrain, Ytrain, Xtest, Ytest) = bonsaiPreProcess.preProcessData(data_dir)

sparZ = args.sZ

if numClasses > 2:
    sparW = 0.2
    sparV = 0.2
    sparT = 0.2
else:
    sparW = 1
    sparV = 1
    sparT = 1

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

useMCHLoss = True

X = tf.placeholder("float32", [None, dataDimension])
Y = tf.placeholder("float32", [None, numClasses])

bonsaiObj = Bonsai(numClasses, dataDimension,
                   projectionDimension, depth, sigma)

bonsaiTrainer = BonsaiTrainer(bonsaiObj,
                              regW, regT, regV, regZ,
                              sparW, sparT, sparV, sparZ,
                              learningRate, X, Y, useMCHLoss)

sess = tf.InteractiveSession()
sess.run(tf.group(tf.initialize_all_variables(),
                  tf.initialize_variables(tf.local_variables())))
saver = tf.train.Saver()

bonsaiTrainer.train(batchSize, totalEpochs, sess, Xtrain, Xtest, Ytrain, Ytest)

# TODO: Write a model saver for storing the params

# For the following command:
# python train.py -dir ../../../../../../deepBonsai/DeepBonsai/Bonsai_tf/data/curet/ -d 2 -p 22 -rW 0.00001 -rZ 0.0000001 -rV 0.00001 -rT 0.000001 -sZ 0.4 -sW 0.5 -sV 0.5 -sT 1 -e 300 -s 0.1 -b 20
# Final Output
# Maximum Test accuracy at compressed model size(including early stopping): 0.94583 at Epoch: 156
# Final Test Accuracy: 0.92516
# Non-Zeros: 118696 Model Size: 115.9140625 KB
