import tensorflow as tf
import numpy as np


class ProtoNNTrainer:
    def __init__(self, protoNNObj, regW, regB, regZ,
                 sparcityW, sparcityB, sparcityZ,
                 learningRate, X, Y, lossType='l2'):
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
        self.__lossType = lossType
        self.__validInit = False
        self.__validInit = self.__validateInit()
        self.__protoNNOut = protoNNObj(X)
        self.loss = self.__lossGraph()
        self.trainStep = self.__trainGraph()
        self.accuracy = self.__accGraph()
        '''
        assert for sparcity and dimensions of X
        and Y
        '''

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
                loss_0 = tf.nn.l2_loss(self.Y - pnnOut)
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

    def __accGraph(self):
        with tf.name_scope('protonn-accuracy'):
            predictions = tf.argmax(self.__protoNNOut, 1)
            target = tf.argmax(self.Y, 1)
            correctPrediction = tf.equal(predictions, target)
            acc = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
        return acc

    def train(self, batchSize, totalEpochs, sess,
              x_train, x_test, y_train, y_test,
              noInit=False, redirFile=None):
        '''
        Dense training of ProtoNN
        noInit: if not to perform initialization
        TODO: Implement dense - IHT - sparse
        '''
        d, d_cap, m, L, gamma = self.protoNNObj.getHyperParams()
        assert batchSize >= 1, 'Batch size should be positive integer'
        assert totalEpochs >= 1, 'Total epochs should be psotive integer'
        assert x_train.ndim == 2, 'Expected training data to be of rank 2'
        assert x_train.shape[1] == d, 'Expected x_train to be [-1, %d]' % d
        assert x_test.ndim == 2, 'Expected testing data to be of rank 2'
        assert x_test.shape[1] == d, 'Expected x_test to be [-1, %d]' % d
        assert y_train.ndim == 2, 'Expected training labels to be of rank 2'
        assert y_train.shape[1] == L, 'Expected y_train to be [-1, %d]' % L
        assert y_test.ndim == 2, 'Expected testing labels to be of rank 2'
        assert y_test.shape[1] == L, 'Expected y_test to be [-1, %d]' % L

        # Numpy will throw asserts for arrays
        if sess is None:
            raise ValueError('sess must be valid tensorflow session.')

        trainNumBatches = int(np.ceil(len(x_train) / batchSize))
        testNumBatches = int(np.ceil(len(x_test) / batchSize))
        x_train_batches = np.array_split(x_train, trainNumBatches)
        y_train_batches = np.array_split(y_train, trainNumBatches)
        x_test_batches = np.array_split(x_test, testNumBatches)
        y_test_batches = np.array_split(y_test, testNumBatches)
        if not noInit:
            sess.run(tf.global_variables_initializer())
        X, Y = self.X, self.Y
        for epoch in range(totalEpochs):
            for i in range(len(x_train_batches)):
                batch_x = x_train_batches[i]
                batch_y = y_train_batches[i]
                feed_dict = {
                    X: batch_x,
                    Y: batch_y
                }
                sess.run(self.trainStep, feed_dict=feed_dict)
                if i % 10 == 0:
                    loss, acc = sess.run([self.loss, self.accuracy],
                                         feed_dict=feed_dict)
                    msg = "Epoch: %3d Batch: %3d" % (epoch, i)
                    msg += " Loss: %3.5f Accuracy: %2.5f" % (loss, acc)
                    print(msg, file=redirFile)
            if (epoch + 1) % 3 == 0:
                acc = 0.0
                loss = 0.0
                for j in range(len(x_test_batches)):
                    batch_x = x_test_batches[j]
                    batch_y = y_test_batches[j]
                    feed_dict = {
                        X: batch_x,
                        Y: batch_y
                    }
                    acc_, loss_ = sess.run([self.accuracy, self.loss],
                                           feed_dict=feed_dict)
                    acc += acc_
                    loss += loss_
                acc /= len(y_test_batches)
                loss /= len(y_test_batches)
                print("Test Loss: %2.5f Accuracy: %2.5f" % (loss, acc))



