import argparse
import protoNNpreprocess
import datetime
import pickle
import sys
import os
import pandas as pd
sys.path.insert(0, '../')
import numpy as np
import tensorflow as tf
from edgeml.trainer.protoNNTrainer import ProtoNNTrainer
from edgeml.graph.protoNN import ProtoNN
import matplotlib.pyplot as plt
import edgeml.utils as utils
from sklearn import linear_model
from sklearn.metrics import r2_score
np.random.seed(42)

def getModelSize(matrixList, sparcityList, expected=True, bytesPerVar=4):
    '''
    expected: Expected size according to the parameters set. The number of
              zeros could actually be more than the applied sparsity constraint.
    '''
    nnzList, sizeList, isSparseList = [], [], []
    hasSparse = False
    for i in range(len(matrixList)):
        A, s = matrixList[i], sparcityList[i]
        assert A.ndim == 2
        # 's - sparsity factor' should be between 0 and 1.
        assert s >= 0
        assert s <= 1
        nnz, size, sparse = utils.countnnZ(A, s, bytesPerVar=bytesPerVar)
        nnzList.append(nnz)
        sizeList.append(size)
        hasSparse = (hasSparse or sparse)

    totalnnZ = np.sum(nnzList)
    totalSize = np.sum(sizeList)
    #By default, go into this part, fails only when the bytesPerVar need to be changed or the sparsity factor.
    if expected:
        return totalnnZ, totalSize, hasSparse

    numNonZero = 0
    totalSize = 0
    hasSparse = False
    for i in range(len(matrixList)):
        A, s = matrixList[i], sparcityList[i]
        numNonZero_ = np.count_nonzero(A)
        numNonZero += numNonZero_
        hasSparse = (hasSparse or (s < 0.5))
        if s <= 0.5:
            totalSize += numNonZero_ * 2 * bytesPerVar
        else:
            totalSize += A.size* bytesPerVar
    return numNonZero, totalSize, hasSparse


def loadData(dataDir):
    train = np.load(dataDir + '/train.npy')
    test = np.load(dataDir + '/test.npy')

    dataDimension = int(train.shape[1]) - 1
    x_train = train[:, 1:dataDimension + 1]
    y_train_ = train[:, 0]
    x_test = test[:, 1:dataDimension + 1]
    y_test_ = test[:, 0]

    #numClasses = max(y_train_) - min(y_train_) + 1
    #numClasses = max(numClasses, max(y_test_) - min(y_test_) + 1)
    #numClasses = int(numClasses)

    #To use as a regressor.
    numClasses = 1

    # mean-var normalization.
    mean = np.mean(x_train, 0)
    std = np.std(x_train, 0)
    std[std[:] < 0.000001] = 1
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    print ("Inside loadData.")
    print ("Mean : ",mean)
    print ("Std  : ",std)

    #print ("Xtrain : ",x_train[:5,:])
    #print ("Xtrain : ",x_test[:5,:])

    """
    # one hot y-train
    lab = y_train_.astype('uint8')
    lab = np.array(lab) - min(lab)
    lab_ = np.zeros((x_train.shape[0], numClasses))
    lab_[np.arange(x_train.shape[0]), lab] = 1
    y_train = lab_

    # one hot y-test
    lab = y_test_.astype('uint8')
    lab = np.array(lab) - min(lab)
    lab_ = np.zeros((x_test.shape[0], numClasses))
    lab_[np.arange(x_test.shape[0]), lab] = 1
    y_test = lab_
    """
    # Don's original piece of line.
    #return dataDimension, numClasses, x_train, y_train, x_test, y_test

    return dataDimension , numClasses, x_train, y_train_.reshape((-1,1)), x_test, y_test_.reshape((-1,1)),mean,std


def main(**kwargs):
    # -----------------
    # Configuration
    # -----------------
    #Get the directory path, as a command line argument.

    args = protoNNpreprocess.getArgs()
    DATA_DIR = args.data_dir

    PROJECTION_DIM = args.projDim
    NUM_PROTOTYPES = args.num_proto
    GAMMA = args.gamma

    REG_W = args.rW
    REG_B = args.rB
    REG_Z = args.rZ

    # 1.0 implies dense matrix.
    SPAR_W = 1.0
    SPAR_B = 1.0
    SPAR_Z = 1.0
    batchSize = args.batchSize

    LEARNING_RATE = args.learningRate
    NUM_EPOCHS = args.num_epochs
    # -----------------
    # End configuration
    # -----------------

    #if GAMMA is None:
    #else:
#        gamma = GAMMA
#        W, B = None, None

    out = loadData(DATA_DIR)
    dataDimension = out[0]
    numClasses = out[1]
    x_train, y_train = out[2], out[3]
    x_test, y_test = out[4], out[5]

    split = y_train.shape[0]

    print("Using median heuristc to estimate gamma")
    centers , gamma, W, B = utils.medianHeuristic(x_train, PROJECTION_DIM,
                                        NUM_PROTOTYPES,W_init=np.eye(PROJECTION_DIM))
    print ("Before run : ",np.linalg.norm(B,ord="fro"))

    X = tf.placeholder(tf.float32, [None, dataDimension], name='X')
    Y = tf.placeholder(tf.float32, [None, numClasses], name='Y')
    '''
    print ("------------------------------")
    print ("ProtoNN 1 : ")

    protoNN = ProtoNN(dataDimension, PROJECTION_DIM,
                      NUM_PROTOTYPES, numClasses,
                      gamma,W=W,B=B)

    trainer = ProtoNNTrainer(DATA_DIR,protoNN, REG_W, REG_B, REG_Z,
                             SPAR_W, SPAR_B, SPAR_Z,
                             LEARNING_RATE, X, Y,lossType='l2')
    sess = tf.Session()
    sess.run(tf.group(tf.initialize_all_variables(),
                      tf.initialize_variables(tf.local_variables())))

    #A numpy array.
    y_hat , y = trainer.train(DATA_DIR,batchSize, NUM_EPOCHS, sess, x_train, x_test, y_train, y_test,
                  DATA_DIR,printStep=200)
    acc = sess.run(protoNN.accuracy, feed_dict={X: x_test, Y:y_test})
    W, B, Z, _ = protoNN.getModelMatrices()
    #print ("W : ",W)
    matrixList = sess.run([W, B, Z])
    sparcityList = [SPAR_W, SPAR_B, SPAR_Z]
    print ("Metrics on Final Value : ",r2_score(y,y_hat))
    sess.close()
    #print ("Output of ProtoNN 1 :")
    #print ("y_hat : ",y_hat)
    #print ("y : ",y)
    y_hat_testpreds = y_hat[split:]
    y_hat = y_hat[:split]
    '''

    lin_model = linear_model.LinearRegression()
    lin_model.fit(x_train,y_train)

    #Save the base model predictions for later.
    y_hat_testpreds = lin_model.predict(x_test)

    #y_hat = lin_model.predict(np.vstack((x_train,x_test)))
    y_hat = lin_model.predict(x_train)

    print ("Train Set Metrics for base Linear Regressor : ",r2_score(y_train,lin_model.predict(x_train)))
    print ("Test Set Metrics for base Linear Regressor : ",r2_score(y_test,y_hat_testpreds))
    #y_hat = lin_model.predict(x_train)
    #y = np.vstack((y_train,y_test))

    y = y_train
    final_pred = y_hat

    #y_hat_test = lin_model.predict(x_test)
    #final_pred = y_hat

    theta = None
    #Now iteratively building the residual model.
    theta_arr = [1.0,1.0]
    theta_inter = [0.0,0.0]
    dicti = {}
    for i in range(0,2):

        print ("-------------------------")
        print ("Iter : ",i)
        y_residuals = (y - final_pred)
        protoNN = ProtoNN(dataDimension, PROJECTION_DIM,
                          NUM_PROTOTYPES, numClasses,
                          gamma,W=W,B=B)

        trainer = ProtoNNTrainer(DATA_DIR,protoNN, REG_W, REG_B, REG_Z,
                                 SPAR_W, SPAR_B, SPAR_Z,
                                 LEARNING_RATE, X, Y,lossType='l2')
        sess = tf.Session()
        sess.run(tf.group(tf.initialize_all_variables(),
                          tf.initialize_variables(tf.local_variables())))

        #A numpy array.
        #y_train_res =  y_residuals[:split]
        #y_test_res = y_residuals[split:]
        y_train_res = y_residuals
        y_test_res = y_test - y_hat_testpreds

        print ("Residual Y Shapes : ",y_train_res.shape , y_test_res.shape)

        y_hat_res , y_hat_res_truth = trainer.train(DATA_DIR,batchSize, NUM_EPOCHS, sess, x_train, x_test, y_train_res, y_test_res,
                      DATA_DIR,printStep=200)

        y_hat_res_test = y_hat_res[split:]
        dicti[i] = y_hat_res_test
        #y_hat_res_truth = y_hat_res_truth[split:]

        y_hat_res_train = y_hat_res[:split]
        y_hat_res_truth = y_hat_res_truth[:split]

        '''
        acc = sess.run(protoNN.accuracy, feed_dict={X: x_test, Y:y_test_res})
        acc = sess.run(protoNN.accuracy, feed_dict={X: x_train, Y:y_test_res})

        print ("Predictions :")
        print (y_hat_res_test)
        print ("Truth : ")
        print (y_hat_res_truth)
        '''

        cl_x = np.hstack((y_hat,y_hat_res_train))
        cl_y = y
        print ("Shape of x : ",cl_x.shape)
        #print ("Shape of Inverse : ",np.linalg.inv(np.matmul(np.transpose(x_train),x_train)).shape)
        #print ("Shape of term 1 : ",np.matmul(np.linalg.inv(np.matmul(x_train,np.transpose(x_train))),np.transpose(x_train)).shape)
        #print ("Shape of term 2 : ",y_train_res.shape)
        theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(cl_x),cl_x)),np.transpose(cl_x)),cl_y)
        print ("Theta : ",theta)
        #final_pred = (final_pred + np.transpose(np.matmul(theta,np.transpose(y_hat_res))))


        new_lin_model = linear_model.LinearRegression()
        #new_lin_model.fit(np.hstack((y_hat,y_hat_res)),y)
        print ("Shapes : ",y_hat.reshape((-1,1)).shape)
        print ("Shapes : ",y_hat_res_train.reshape((-1,1)).shape)
        new_lin_model.fit(np.hstack((y_hat.reshape((-1,1)),y_hat_res_train.reshape((-1,1)))),y)
        print ("Coefficients of Linear Model : ",new_lin_model.coef_)
        print ("Coefficients of Linear Model : ",new_lin_model.coef_[0][1])
        theta_arr[i] = (new_lin_model.coef_[0][1])
        theta_inter[i] = (new_lin_model.intercept_[0])
        #theta_arr[i] = theta[1][0]
        print ("Intercept of Lin Model : ",new_lin_model.intercept_)

        final_pred = final_pred + (theta_arr[i]*y_hat_res_train+theta_inter[i])
        print ("Metrics on the Train Set : ",r2_score(y_train,final_pred))
        #print ("Metrics on Final Value : ",r2_score(y,final_pred))
        #print ("Metrics on Final Value : ",r2_score(y[split:],final_pred[split:]))
        print ("Final Pred : ",final_pred.shape)
        #print ("Metrics on Final Value : ",r2_score(y,final_pred))
        #print ("Metrics on Final Value : ",r2_score(y_test,final_pred[split:]))
        #W, B, Z, _ = protoNN.getModelMatrices()
        #print ("W : ",W)
        #matrixList = sess.run([W, B, Z])
        #sparcityList = [SPAR_W, SPAR_B, SPAR_Z]
        sess.close()

    #print ("Yhat : ",y_hat_testpreds)
    #print ("dicti[0] : ",dicti[0])
    #print ("dicti[1] : ",dicti[1])
    final_prediction = y_hat_testpreds
    for i in dicti.keys():
        final_prediction += (dicti[i]*theta_arr[i]*0.1+theta_inter[i])
        #+ (dicti[i]*theta_arr[1]+theta_inter[1])
    print ("Metrics after the end of the run : ",r2_score(y_test,final_prediction))
    '''
    nnz, size, sparse = getModelSize(matrixList, sparcityList)
    print("Final test accuracy", acc)
    print("Model size constraint (Bytes): ", size)
    print("Number of non-zeros: ", nnz)
    nnz, size, sparse = getModelSize(matrixList, sparcityList, expected=False)
    print("Actual model size: ", size)
    print("Actual non-zeros: ", nnz)


    #Save the value of the 'Gamma' in the dictionary.
    ndict["g"] = gamma
    #B : num_prototypes.
    print ("Prototypes (B) : ",matrixList[1].shape)
    #Z : what it predicts.
    print ("Z : ",matrixList[2].shape)
    #print ("Z : ",matrixList[2])
    #(np.save("B.npy",matrixList[1]))
    #print ("B : ",matrixList[1])
    print ("After run : ",np.linalg.norm(matrixList[1],ord="fro"))
    pd.DataFrame(matrixList[1]).to_csv("B.csv")
    pd.DataFrame(matrixList[2]).to_csv("Z.csv")
    B = pd.DataFrame(matrixList[1])
    print ("W : ",matrixList[0])
    '''
if __name__ == '__main__':
    main()

'''
NOTES:
    1. Curet:
        Data dimension: 610
        num Classes: 61
        Reasonable parameters:
            Projection_Dim = 60
            PROJECTION_DIM = 60
            NUM_PROTOTYPES = 80
            GAMMA = 0.0015
            REG_W = 0.000
            REG_B = 0.000
            REG_Z = 0.000
            SPAR_W = 1.0
            SPAR_B = 1.0
            SPAR_Z = 1.0
            LEARNING_RATE = 0.1
            Expected test accuracy: 87-88%

            PROJECTION_DIM = 60
            NUM_PROTOTYPES = 60
            GAMMA = 0.0015
            REG_W = 0.000005
            REG_B = 0.000
            REG_Z = 0.00005
            SPAR_W = .8
            SPAR_B = 1.0
            SPAR_Z = 1.0
            LEARNING_RATE = 0.05
            Expected test accuracy: 89-90%
'''
