from __future__ import print_function
import os
import pickle
from sklearn.grid_search import ParameterGrid

import bonsaipreprocess
import tensorflow as tf
import numpy as np
import multiprocessing
import os
import sys
import pickle
sys.path.insert(0, '../')

# Fixing seeds for reproducibility
tf.set_random_seed(42)
np.random.seed(42)

from edgeml.trainer.bonsaiTrainer import BonsaiTrainer
from edgeml.graph.bonsai import Bonsai

def model_create(vals,dir="../../plantA"):

    sigma = float(vals["s"])
    depth = int(vals["d"])

    projectionDimension = 5
    regZ = float(vals["rZ"])
    #regT = args.rT
    regT = float(vals["rW"])


    regW = float(vals["rW"])
    #regV = args.rV
    regV = float(vals["rW"])

    totalEpochs = 42

    learningRate = float(vals["lr"])
    data_dir = dir

    outFile = None

    (dataDimension, numClasses,
        Xtrain, Ytrain, Xtest, Ytest) = bonsaipreprocess.preProcessData(data_dir)

    sparZ = 1

    if numClasses > 2:
        sparW = 0.2
        sparV = 0.2
        sparT = 0.2
    else:
        sparW = 1
        sparV = 1
        sparT = 1

    sparW = 1
    sparV = 1
    sparT = 1

    batchSize = int(vals["b"])

    useMCHLoss = False

    if numClasses == 2:
        numClasses = 1

    X = tf.placeholder("float32", [None, dataDimension])
    Y = tf.placeholder("float32", [None, numClasses])

    #currDir = bonsaipreprocess.createDir(data_dir)
    currDir = data_dir

    # numClasses = 1 for binary case
    bonsaiObj = Bonsai(numClasses, dataDimension,
                       projectionDimension, depth, sigma)

    """
    Argument that determines, whether or not to test on entire data or only on test data.
    type = 1 -> Test only on test data.
    type = 2 -> Test on entire data.
    """
    type = 2

    split = int(0.8*(np.vstack((Xtrain,Xtest)).shape[0]))
    print ("Total Size : ",np.vstack((Xtrain,Xtest)).shape)
    print ("Split : ",split,"\n\n")

    bonsaiTrainer = BonsaiTrainer(bonsaiObj,
                                  regW, regT, regV, regZ,
                                  sparW, sparT, sparV, sparZ,
                                  learningRate, X, Y, split,useMCHLoss, outFile,type)

    sess = tf.InteractiveSession(config=tf.ConfigProto(
  intra_op_parallelism_threads = 4))
    sess.run(tf.group(tf.initialize_all_variables(),
                      tf.initialize_variables(tf.local_variables())))
    saver = tf.train.Saver()
    #Getting the size of the tensorflow model, by saving itself.
    #saver.save(sess,"tf-model.ckpt")

    print("DONE")
    dict = bonsaiTrainer.train(batchSize, totalEpochs, sess,
                        Xtrain, Xtest, Ytrain, Ytest, split,data_dir, currDir,type)

    #print dict
    return dict

param_grid = {
"e" : [500],
"d" : [1,2,4,6,7],
"p" : [10,15,20],
"b"  : [32,64,128],
"s" : [3.0],
"lr" : [0.01,0.05],
"rZ" : [0.001, 0.0001, 0.000001],
"rW" : [0.01, 0.001, 0.00001]
}

grid = ParameterGrid(param_grid)
list_dicts = []
bestr2 = 0.0
curr2 = {}
for params in grid:
    #os.system("python bonsai_example.py -dir ../../Inverter_1_2017-03-30"+" -d "+str(params["d"])+" -b "+str(params["b"])+" -s "+str(params["s"])+" -lr "+str(params["lr"])+" -rZ "+str(params["rZ"])+" -rW "+str(params["rW"]))
    vals = {"e" : str(params["e"]) ,"d" : str(params["d"]) , "b" : str(params["b"]) , "s" : str(params["s"]) , "lr" : str(params["lr"]) , "rZ" : str(params["rZ"]) , "rW" : str(params["rW"]), "p" : str(params["p"]) }
    curr2 = model_create(vals)
    curr2["e"] = params["e"]
    curr2["d"] = params["d"]
    curr2["p"] = params["p"]
    curr2["b"] = params["b"]
    curr2["lr"] = params["lr"]
    curr2["s"] = params["s"]
    curr2["rZ"] = params["rZ"]
    curr2["rW"] = params["rW"]
    print ("Parameters : ",curr2,file = sys.stdout)
    list_dicts.append(curr2)
    #For each run, append the results and then pickle it on top. Useful when we want to stop early.
    #Change the directory as per the dataset.
    print ("Dumping !!")
    pickle.dump(list_dicts,open("../../plantA/list_dicts-s4.pkl","wb"))

print ("---------------------------")
print ("End of Param Search : ")
best_params = {}
for i in range(len(list_dicts)):
    if(i==0):
        best_params = list_dicts[i].copy()
        continue
    #Focus is only on RMSE.
    if (list_dicts[i]["rmse"] < best_params["rmse"]):
        best_params = list_dicts[i].copy()

print ("Best so far : ",best_params)
sys.stdout.close()
