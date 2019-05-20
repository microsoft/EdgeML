# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import edgeml.utils as utils


class ProtoNNTrainer:
    def __init__(self, protoNNObj, regW, regB, regZ,
                 sparcityW, sparcityB, sparcityZ,
                 learningRate, X, Y, lossType='l2'):
        '''
        A wrapper for the various techniques used for training ProtoNN. This
        subsumes both the responsibility of loss graph construction and
        performing training. The original training routine that is part of the
        C++ implementation of EdgeML used iterative hard thresholding (IHT),
        gamma estimation through median heuristic and other tricks for
        training ProtoNN. This module implements the same in Tensorflow
        and python.

        protoNNObj: An instance of ProtoNN class defining the forward
            computation graph. The loss functions and training routines will be
            attached to this instance.
        regW, regB, regZ: Regularization constants for W, B, and
            Z matrices of protoNN.
        sparcityW, sparcityB, sparcityZ: Sparsity constraints
            for W, B and Z matrices. A value between 0 (exclusive) and 1
            (inclusive) is expected. A value of 1 indicates dense training.
        learningRate: Initial learning rate for ADAM optimizer.
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
            self.sparseTraining = False
            print("Sparse training disabled.", file=sys.stderr)
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
        msg = "Sparsity value should be between"
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
            with tf.compat.v1.name_scope('protonn-l2-loss'):
                loss_0 = tf.nn.l2_loss(self.Y - pnnOut)
                reg = l1 * tf.nn.l2_loss(W) + l2 * tf.nn.l2_loss(B)
                reg += l3 * tf.nn.l2_loss(Z)
                loss = loss_0 + reg
        elif self.__lossType == 'xentropy':
            with tf.compat.v1.name_scope('protonn-xentropy-loss'):
                loss_0 = tf.nn.softmax_cross_entropy_with_logits(logits=pnnOut,
                                                         labels=tf.stop_gradient(self.Y))
                loss_0 = tf.reduce_mean(input_tensor=loss_0)
                reg = l1 * tf.nn.l2_loss(W) + l2 * tf.nn.l2_loss(B)
                reg += l3 * tf.nn.l2_loss(Z)
                loss = loss_0 + reg
        return loss

    def __trainGraph(self):
        with tf.compat.v1.name_scope('protonn-gradient-adam'):
            trainStep = tf.compat.v1.train.AdamOptimizer(self.__lR)
            trainStep = trainStep.minimize(self.loss)
        return trainStep

    def __getHardThresholdOp(self):
        W, B, Z, _ = self.protoNNObj.getModelMatrices()
        self.W_th = tf.compat.v1.placeholder(tf.float32, name='W_th')
        self.B_th = tf.compat.v1.placeholder(tf.float32, name='B_th')
        self.Z_th = tf.compat.v1.placeholder(tf.float32, name='Z_th')
        with tf.compat.v1.name_scope('hard-threshold-assignments'):
            # hard_thrsd_W = W.assign(self.W_th)
            # hard_thrsd_B = B.assign(self.B_th)
            # hard_thrsd_Z = Z.assign(self.Z_th)
            # Code changes for tf 1.11
            hard_thrsd_W = tf.compat.v1.assign(W, self.W_th)
            hard_thrsd_B = tf.compat.v1.assign(B, self.B_th)
            hard_thrsd_Z = tf.compat.v1.assign(Z, self.Z_th)
            hard_thrsd_op = tf.group(hard_thrsd_W, hard_thrsd_B, hard_thrsd_Z)
        return hard_thrsd_op

    def train(self, batchSize, totalEpochs, sess,
              x_train, x_val, y_train, y_val, noInit=False,
              redirFile=None, printStep=10, valStep=3):
        '''
        Performs dense training of ProtoNN followed by iterative hard
        thresholding to enforce sparsity constraints.

        batchSize: Batch size per update
        totalEpochs: The number of epochs to run training for. One epoch is
            defined as one pass over the entire training data.
        sess: The Tensorflow session to use for running various graph
            operators.
        x_train, x_val, y_train, y_val: The numpy array containing train and
            validation data. x data is assumed to in of shape [-1,
            featureDimension] while y should have shape [-1, numberLabels].
        noInit: By default, all the tensors of the computation graph are
        initialized at the start of the training session. Set noInit=False to
        disable this behaviour.
        printStep: Number of batches between echoing of loss and train accuracy.
        valStep: Number of epochs between evolutions on validation set.
        '''
        d, d_cap, m, L, gamma = self.protoNNObj.getHyperParams()
        assert batchSize >= 1, 'Batch size should be positive integer'
        assert totalEpochs >= 1, 'Total epochs should be positive integer'
        assert x_train.ndim == 2, 'Expected training data to be of rank 2'
        assert x_train.shape[1] == d, 'Expected x_train to be [-1, %d]' % d
        assert x_val.ndim == 2, 'Expected validation data to be of rank 2'
        assert x_val.shape[1] == d, 'Expected x_val to be [-1, %d]' % d
        assert y_train.ndim == 2, 'Expected training labels to be of rank 2'
        assert y_train.shape[1] == L, 'Expected y_train to be [-1, %d]' % L
        assert y_val.ndim == 2, 'Expected validation labels to be of rank 2'
        assert y_val.shape[1] == L, 'Expected y_val to be [-1, %d]' % L

        # Numpy will throw asserts for arrays
        if sess is None:
            raise ValueError('sess must be valid Tensorflow session.')

        trainNumBatches = int(np.ceil(len(x_train) / batchSize))
        valNumBatches = int(np.ceil(len(x_val) / batchSize))
        x_train_batches = np.array_split(x_train, trainNumBatches)
        y_train_batches = np.array_split(y_train, trainNumBatches)
        x_val_batches = np.array_split(x_val, valNumBatches)
        y_val_batches = np.array_split(y_val, valNumBatches)
        if not noInit:
            sess.run(tf.compat.v1.global_variables_initializer())
        X, Y = self.X, self.Y
        W, B, Z, _ = self.protoNNObj.getModelMatrices()
        for epoch in range(totalEpochs):
            for i in range(len(x_train_batches)):
                batch_x = x_train_batches[i]
                batch_y = y_train_batches[i]
                feed_dict = {
                    X: batch_x,
                    Y: batch_y
                }
                sess.run(self.trainStep, feed_dict=feed_dict)
                if i % printStep == 0:
                    loss, acc = sess.run([self.loss, self.accuracy],
                                         feed_dict=feed_dict)
                    msg = "Epoch: %3d Batch: %3d" % (epoch, i)
                    msg += " Loss: %3.5f Accuracy: %2.5f" % (loss, acc)
                    print(msg, file=redirFile)

            # Perform Hard thresholding
            if self.sparseTraining:
                W_, B_, Z_ = sess.run([W, B, Z])
                fd_thrsd = {
                    self.W_th: utils.hardThreshold(W_, self.__sW),
                    self.B_th: utils.hardThreshold(B_, self.__sB),
                    self.Z_th: utils.hardThreshold(Z_, self.__sZ)
                }
                sess.run(self.__hthOp, feed_dict=fd_thrsd)

            if (epoch + 1) % valStep  == 0:
                acc = 0.0
                loss = 0.0
                for j in range(len(x_val_batches)):
                    batch_x = x_val_batches[j]
                    batch_y = y_val_batches[j]
                    feed_dict = {
                        X: batch_x,
                        Y: batch_y
                    }
                    acc_, loss_ = sess.run([self.accuracy, self.loss],
                                           feed_dict=feed_dict)
                    acc += acc_
                    loss += loss_
                acc /= len(y_val_batches)
                loss /= len(y_val_batches)
                print("Test Loss: %2.5f Accuracy: %2.5f" % (loss, acc))

