'''
ProtoNN Graph construction
'''
import numpy as np
import tensorflow as tf
from scipy.spatial import distance

class ProtoNN:
    def __init__(self, inputDimension, projectionDimension, numPrototypes,
                 numOutputLabels, gamma,
                 W = None, B = None, Z = None):
        '''
        inputDimension: input data dimension. A [-1, inputDimension] matrix
            is expected as part of __call__ method.
        projectionDimension: (model paramter)
        numPrototypes: (model parameter)
        numOutputLabels: The number of output labels or classes
        W, B, Z: initial value to use for W, B, Z
        Expected Dimensions:
        W   d x d_cap
        B   d_cap x m
        Z   L x m
        X   n x d
        '''
        with tf.name_scope('protoNN') as ns:
            self.__nscope = ns
        self.__d = inputDimension
        self.__d_cap = projectionDimension
        self.__m = numPrototypes
        self.__L = numOutputLabels

        self.__np_gamma = gamma
        print ("Init value of gamma : ",gamma)
        self.__X = None
        print ("Inside ProtoNN init : ",type(gamma))

        self.__inW = W
        self.__inB = B
        self.__inZ = Z
        self.__inGamma = gamma
        # The tf.Variables
        self.W, self.B, self.Z = None, None, None
        self.gamma = None

        self.__validInit = False
        self.__initWBZ()
        self.__initGamma()
        self.__validateInit()
        self.protoNNOut = None
        self.predictions = None
        self.accuracy = None

    def __validateInit(self):
        self.__validInit = False
        errmsg = "Dimensions mismatch! Should be W[d, d_cap]"
        errmsg += ", B[d_cap, m] and Z[L, m]"
        d, d_cap, m, L, _ = self.getHyperParams()
        assert self.W.shape[0] == d, errmsg
        assert self.W.shape[1] == d_cap, errmsg
        assert self.B.shape[0] == d_cap, errmsg
        assert self.B.shape[1] == m, errmsg
        #assert self.Z.shape[0] == L, errmsg
        #assert self.Z.shape[1] == m, errmsg
        self.__validInit = True

    def __initWBZ(self):
        with tf.name_scope(self.__nscope):
            W = self.__inW
            if W is None:
                W = tf.random_normal_initializer()
                W = W([self.__d, self.__d_cap])
            self.W = tf.Variable(W, name='W', dtype=tf.float32)

            B = self.__inB
            if B is None:
                B = tf.random_uniform_initializer()
                B = B([self.__d_cap, self.__m])
            self.B = tf.Variable(B, name='B', dtype=tf.float32)

            Z = self.__inZ
            if Z is None:
                Z = tf.random_normal_initializer()
                #One dimension of Z will be equal to the data dimension, with a bias term.
                #Z = Z([self.__L, self.__m])
                Z = Z([self.__d, self.__m])
            Z = tf.Variable(Z, name='Z', dtype=tf.float32)
            self.Z = Z
        return self.W, self.B, self.Z

    def updateGamma(self,new_gamma=0.0,sess=None,x_train=None):
        print ("--------------------------------------")
        '''
        Call a function that takes in two tensors and performs numpy operations on them.
        The arrays taken in are B and X and W.

        W : shape [data_dimension , projectionDimension]
        B : shape [projectionDimension , num_prototypes]
        X : shape [None , data_dimension]

        X*W : shape [None , projectionDimension]
        B_hat : shape [num_prototypes , projectionDimension]
        '''
        with tf.name_scope(self.__nscope):
            W , B , Z , _ = self.getModelMatrices()
            matrixList = sess.run([W, B, Z])
            B = matrixList[1]
            B = np.transpose(B)
            print ("B : ",B.shape)
            print ("X : ",x_train.shape)
            print ("W : ",matrixList[0].shape)
            x_train = np.matmul(x_train,matrixList[0])
            print ("XW : ",x_train.shape)
            #print (B)
            #----------------------------------#
            cdist_ = distance.cdist(B,x_train,'minkowski',p=1)
            print ("Shape of cdist : ",cdist_.shape)
            new_gamma = np.mean(np.median(cdist_,axis=1))
            new_gamma = 1 / (2.5 * new_gamma)
            print ("Updated Value of gamma : ",new_gamma)

            #----------------------------------#
            self.__np_gamma = new_gamma
            self.__inGamma = new_gamma
            print ("Before Updation ,  Gamma Value inside updateGamma : ",sess.run(self.gamma))
            update_Gamma = tf.assign(self.gamma,tf.cast(tf.constant(new_gamma),np.float32))
            sess.run(update_Gamma)
            print ("Updated Gamma Value inside updateGamma : ",sess.run(self.gamma))
            print ("---------------------------------------")

    def __initGamma(self):
        with tf.name_scope(self.__nscope):
            print ("Inside initGamma")
            print ("Value of self.__inGamma : ",self.__inGamma)
            gamma = self.__inGamma
            '''
            Set a variable to 'trainable=False' if it's weights shouldn't be modified.
            '''
            #self.gamma = tf.constant(gamma, name='gamma')
            self.gamma = tf.Variable(gamma, name='gamma',trainable=False)

    def getModelSize(self, bytesPerVar = 4):
        # TODO: Use Bonsai's util method
        sZ, sW, sB = self.sparcity_Z, self.sparcity_W, self.sparcity_B
        d, d_cap, m, L = self.getHyperParams()
        nnzZ = np.ceil(int(m * L) * sZ)
        nnzW = np.ceil(int(d * d_cap) * sW)
        nnzB = np.ceil(int(d_cap * m) * sB)
        if sZ < 0.5:
            nnzZ *= 2
        if sW < 0.5:
            nnzW *= 2
        if sB < 0.5:
            nnzB *= 2
        return (nnzZ + nnzW + nnzB) * bytesPerVar

    def getHyperParams(self):
        d = self.__d
        dcap = self.__d_cap
        m = self.__m
        L = self.__L
        return d, dcap, m, L, self.gamma

    def getModelMatrices(self):
        return self.W, self.B, self.Z, self.gamma

    def __call__(self, X, Y=None):
        '''
        Returns a protoNN graph
        Returned y is of dimension [-1, numLabels]

        X is [-1, d]
        Y is [-1, numLabels] . Y is optional; if provided will beused
            to create an accuracy computation operator.
        '''
        # This should never execute
        assert self.__validInit is True, "Initialization failed!"
        if self.protoNNOut is not None:
            return self.protoNNOut

        W, B, Z, gamma = self.W, self.B, self.Z, self.gamma
        with tf.name_scope(self.__nscope):
            WX = tf.matmul(X, W)
            # Convert WX to tensor so that broadcasting can work
            dim = [-1, WX.shape.as_list()[1], 1]
            WX = tf.reshape(WX, dim)
            dim = [1, B.shape.as_list()[0], -1]
            B = tf.reshape(B, dim)
            l2sim = B - WX
            l2sim = tf.pow(l2sim, 2)
            l2sim = tf.reduce_sum(l2sim, 1, keep_dims=True)
            self.l2sim = l2sim
            gammal2sim = (-1 * gamma * gamma) * l2sim
            M = tf.exp(gammal2sim)
            '''
            To get the shape, as a list of ints.
            Append an extra dimension to reshape Z.

            dim = [1] + Z.shape.as_list()
            Z = tf.reshape(Z, dim)
            print ("Shape of Z : ",Z.shape.as_list())
            print ("Shape of M : ",M.shape.as_list())
            y = tf.multiply(Z, M)
            print ("Shape of y : ",y.shape.as_list())
            y = tf.reduce_sum(y, 2, name='protoNNScoreOut')
            self.protoNNOut = y
            print ("Check")

            self.predictions = tf.argmax(y, 1, name='protoNNPredictions')
            if Y is not None:
                target = tf.argmax(Y, 1)
                correctPrediction = tf.equal(self.predictions, target)
                acc = tf.reduce_mean(tf.cast(correctPrediction, tf.float32),
                        name='protoNNAccuracy')
            '''
            print ("Shape of Z , before Dim : ",Z.shape.as_list())
            #dim = [1] + Z.shape.as_list()
            print ("X : ",X.shape.as_list())
            print ("Z : ",Z.shape.as_list())
            print ("M : ",M.shape.as_list())
            Z = tf.matmul(X,Z)
            print ("Z , After multiplication : ",Z.shape.as_list())
            dim = [1] + Z.shape.as_list()
            '''
            This fix is necessary, to ensure that a dimension can be added before a 'None' dimension.
            It returns a scalar value.
            '''
            dim[1] = tf.shape(Z)[0]
            print ("Dim : ",dim)

            #Renormalize back the Ws.
            Z = tf.reshape(Z,dim)
            #Reshape the dimensions , so that it is now of the shape , [None , 1, 50]
            Z = tf.transpose(Z,[1,0,2])
            print ("Z , ( after reshape) After multiplication : ",Z.shape.as_list())
            #Returns a tensor of the same dimensions of X.
            y = tf.multiply(Z,M)

            print ("Shape of y before reduce sum : ",y.shape.as_list())
            y = tf.reduce_sum(y, 2, name='protoNNScoreOut')

            #Divide by reduce_sum(M) , to renormalize 'W' again.
            y = tf.divide(y,tf.reduce_sum(M,2,name='Renormalize'))
            #print ("Test : ",tf.reduce_sum(M,2,name='Renormalize').shape.as_list() )
            print ("Shape of y ",y.shape.as_list())
            self.protoNNOut = y

            self.predictions = self.protoNNOut
            #self.predictions = tf.argmax(y, 1, name='protoNNPredictions')
            if Y is not None:
                #target = tf.argmax(Y, 1)
                #correctPrediction = tf.equal(self.predictions, target)
                #acc = tf.reduce_mean(tf.cast(correctPrediction, tf.float32),
                                     #name='protoNNAccuracy')
                #self.accuracy = acc
                print ("Acc : ",Y.shape.as_list() )
                print ("Predictions : ",self.predictions.shape.as_list())
                self.accuracy = tf.metrics.mean_absolute_error(Y,self.predictions)
                #self.accuracy = tf.metrics.mean_absolute_error(tf.reshape(self.Y,[-1,1]),tf.reshape(self.score,[-1,1]))
        return y

    def getPredictionsOp(self):
        return self.predictions

    def getAccuracyOp(self):
        msg = "Accuracy operator not defined in graph. Did you provide Y as an"
        msg += " argument to _call_?"
        assert self.accuracy is not None, msg
        return self.accuracy
