import utils
import tensorflow as tf
import numpy as np
from bonsaiTrainer import BonsaiTrainer
from bonsai import Bonsai

# Fixing seeds for reproducibility
tf.set_random_seed(42)
np.random.seed(42)

# Hyper Param pre-processing
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

lossFlag = True

X = tf.placeholder("float32", [None, dataDimension])
Y = tf.placeholder("float32", [None, numClasses])

bonsaiObj = Bonsai(numClasses, dataDimension,
                   projectionDimension, depth, sigma, X)

bonsaiTrainer = BonsaiTrainer(bonsaiObj,
                              regW, regT, regV, regZ,
                              sparW, sparT, sparV, sparZ,
                              learningRate, Y, lossFlag)

sess = tf.InteractiveSession()
sess.run(tf.group(tf.initialize_all_variables(),
                  tf.initialize_variables(tf.local_variables())))
saver = tf.train.Saver()

bonsaiTrainer.train(batchSize, totalEpochs, sess, Xtrain, Xtest, Ytrain, Ytest)

print(bonsaiTrainer.bonsaiObj.score.eval(feed_dict={X: Xtest}))
print(bonsaiObj.score.eval(feed_dict={X: Xtest}))

np.save("W.npy", bonsaiTrainer.bonsaiObj.W.eval())
np.save("V.npy", bonsaiTrainer.bonsaiObj.V.eval())
np.save("Z.npy", bonsaiTrainer.bonsaiObj.Z.eval())
np.save("T.npy", bonsaiTrainer.bonsaiObj.T.eval())
