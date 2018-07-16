import numpy as np
import tensorflow as tf
import pandas as pd
import shutil
import time
import sys
import os
from edgeml.utils import getConfusionMatrix, printFormattedConfusionMatrix
from edgeml.utils import getPrecisionRecall
from edgeml.utils import getMacroMicroFScore
from edgeml.utils import getMacroPrecisionRecall
from edgeml.utils import getMicroPrecisionRecall


def getUpdateIndexList(newY, oldY, numSubinstance, numOutput):
    '''
    returns the index of the bags that
    had an update after certain policy
    update.
    The said policy transforms oldY to 
    newY.
    -1, numsubinstance, numLabel
    '''
    assert newY.ndim == 3
    assert oldY.ndim == 3
    assert newY.shape[0] == oldY.shape[0]
    assert newY.shape[1] == oldY.shape[1]
    assert newY.shape[2] == oldY.shape[2]
    assert newY.shape[1] == numSubinstance
    assert newY.shape[2] == numOutput
    
    updateIndexList = []
    newY__ = np.argmax(newY, axis=2)
    oldY__ = np.argmax(oldY, axis=2)
    newY__ = np.reshape(newY__, [-1])
    oldY__ = np.reshape(oldY__, [-1])
    total = np.equal(newY__, oldY__)
    total = len(newY__) - np.sum(total.astype(int))
    for i in range(newY.shape[0]):
        rowNew = newY[i, :, :]
        rowOld = oldY[i, :, :]
        rowNew = np.argmax(rowNew, axis=1)
        rowOld = np.argmax(rowOld, axis=1)
        rowDiff = rowNew - rowOld
        if np.sum(rowDiff) != 0:
            updateIndexList.append(i)
    return updateIndexList, total

def getPosExample(numSamples, numWaveSamples,
                  mu=0.0, sigma=0.01, addNoise=True,
                  prefixLimit=None, prefix=None):
    '''
    Returns a positive time series example
    '''
    x = np.zeros(numSamples)
    sin = np.sin(np.arange(numWaveSamples)*(2/numWaveSamples)*np.pi)
    if prefix is not None:
        indx = prefix
    elif prefixLimit is not None:
        indx = np.random.randint(min(prefixLimit, numSamples- numWaveSamples))
    else:
        indx = np.random.randint(numSamples- numWaveSamples)
    x[indx:indx + numWaveSamples] = sin[:]
    if addNoise:
        noise = np.random.normal(mu, sigma, len(x))
        x += noise
    return x

def getNegExample(numSamples, numWaveSamples,
                   mu=0.0, sigma=0.01, addNoise=True,
                  prefixLimit=None, prefix=None):
    '''
    Returns a negative time series example
    '''
    x = np.zeros(numSamples)
    cos = np.cos(np.arange(numWaveSamples)*(2/numWaveSamples)*np.pi + np.pi/2)
    if prefix is not None:
        indx = prefix
    elif prefixLimit is not None:
        indx = np.random.randint(min(prefixLimit, numSamples- numWaveSamples))
    else:
        indx = np.random.randint(numSamples- numWaveSamples)
    x[indx:indx + numWaveSamples] = cos[:]
    if addNoise:
        noise = np.random.normal(mu, sigma, len(x))
        x += noise
    return x


def unstackAndBag(x, subinstanceWidth, subinstanceStride, numSubinstance, numFeats, stride, numTimeSteps):
    assert x.ndim == 1
    subinstList = []
    # Divide into list of subinstance
    start = 0 
    while True:
        vec = x[start:start + subinstanceWidth]
        if len(vec) < subinstanceWidth:
            ar = np.zeros(subinstanceWidth)
            ar[:len(vec)] = vec[:]
            vec = ar
        subinstList.append(vec)
        if start + subinstanceWidth >= len(x):
            break
        start += subinstanceStride
        
    assert len(subinstList) == numSubinstance,"%d %d" % (len(subinstList), numSubinstance)
    # Weird condition, should not happen
    # because of controlled setting
    if len(subinstList) <= 0:
        return None
    
    # Calculate total number of steps
    #int(np.ceil((SUBINSTANCE_WIDTH - NUM_FEATS) / STEP_STRIDE) + 1)
    totalSteps = ((subinstanceWidth - numFeats) / stride)
    if totalSteps <= 0:
        totalSteps = 0
    totalSteps = np.ceil(totalSteps) + 1
    assert(totalSteps == numTimeSteps)
    bag = []
    for subinst in subinstList:
        currIndex = 0
        outList = []
        while True:
            featVec = subinst[currIndex:currIndex + numFeats]
            if len(featVec) < numFeats:
                ar = np.zeros(numFeats)
                ar[0:len(featVec)] = featVec[:]
                featVec = ar
            outList.append(featVec)
            if currIndex + numFeats >= len(subinst):
                break
            currIndex += stride
        assert len(outList) == numTimeSteps, "%d %d" % (len(outList), numTimeSteps)
        bag.append(np.array(outList))
    return bag
        


def recoverTimeStep(x, numFeats, stepStride, expectedNumSamples):
    '''
    x should be [numTimeSteps, numFeats]
    Remember that there will be some zero padding at the end
    due to recovery.
    '''
    assert(x.ndim == 2)
    recovered = []
    recovered.extend(np.reshape(x[0, :], [-1]).tolist())
    numNewElements = stepStride
    assert numNewElements > 0
    for featVec in x[1:]:
        newElements = featVec[-1*numNewElements:]
        newElements = np.reshape(newElements, [-1])
        recovered.extend(newElements.tolist())
    # greter than because of zero padding
    assert(len(recovered) >= expectedNumSamples)
    return np.array(recovered[:expectedNumSamples])


def recoverBag(x, subinstanceWidth, subinstanceStride,
               expectedNumSamples, numFeats, stepStride):
    '''
    x should be [numSubinstance, numTimeSteps, numFeats]
    '''
    assert(x.ndim == 3)
    subinstList = np.array([recoverTimeStep(inst,
                            numFeats, stepStride, subinstanceWidth) for inst in x])
    recovered = []
    recovered.extend(np.reshape(subinstList[0, :], [-1]).tolist())
    numNewElements = subinstanceStride
    assert numNewElements > 0
    for featVec in subinstList[1:]:
        newElements = featVec[-1*numNewElements:]
        newElements = np.reshape(newElements, [-1])
        recovered.extend(newElements.tolist())
    # greter than because of zero padding
    assert(len(recovered) >= expectedNumSamples)
    return np.array(recovered[:expectedNumSamples])




def plotPredictions(pred, curve, ax, subinstanceStride, subinstanceWidth, windowWidth):
    X = [min(subinstanceWidth + subinstanceStride * p, windowWidth)
        for p in range(len(pred))]
    ax.scatter(X, pred, c='r')
    ax.plot(curve)

def transformIndexList(indexList, subinstanceWidth, subinstanceStride):
    '''
    You get a 1-D array of index of the first positive dicriminatory signature
    in a wave and you have to map it to the index of the first subinstance
    containing that positive
    '''
    assert(indexList.ndim == 1)
    val = (np.ceil( (indexList - subinstanceWidth) / subinstanceStride)).astype(int)
    val[val < 0] = 0
    return val
    
            
class NetworkV2:
    def __init__(self, numSubinstance, numFeats, numTimeSteps,
                 numHidden, numFC, numOutput, prefetchNum=100,
                 useCudnn=False, useDropout=False,
                 useEmbeddings=False):
        assert(numOutput >= 2)
        ## Parameters
        self.numSubinstance = numSubinstance
        self.numOutput = numOutput
        self.numFeats = numFeats
        self.numTimeSteps = numTimeSteps
        self.numHidden = numHidden
        self.numFC = numFC
        self.prefetchNum = prefetchNum
        self.useCudnn = useCudnn
        self.useDropout = useDropout
        self.useEmbeddings = useEmbeddings
        self.lossList = None
        ## Operations
        # Note that operations only execute once per batch.
        # Better to use train/inferrence methods
        # Raw outputs
        self.output = None
        self.pred = None
        self.l2Loss = None
        self.softmaxLoss = None
        self.lossOp = None
        self.predictionClass = None
        self.dataset_init = None
        self.embedded_word_ids = None
        # Accuracy with respect to belief label (not true label)
        self.accTilda = None 
        
        ## Placeholders
        # X is a bag and Y is label on instances in bag
        self.X = None
        self.Y = None
        self.batchSize = None
        self.numEpochs = None
        self.keep_prob = None
        
        ## Network
        self.B1 = None
        self.W1 = None
        self.B2 = None
        self.W2 = None
        self.cell = None 
        self.LSTMVars = None
        self.varList = None
        self.word_embeddings = None
        
        # Private variables
        self.train_step_model = None
        self.sess = None
        self.__saver = None
        self.__dataset_next = None
        # Validity flags
        self.__graphCreated = False

    def __createInputPipeline(self):
        '''
        The painful process of figuring out how this worked without any documentation.
        https://groups.google.com/a/tensorflow.org/forum/#!msg/discuss/SXWDjrz5kZw/Oj1PO_RnBQAJ
        https://stackoverflow.com/questions/47064693/tensorflow-data-api-prefetch
        https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
        https://stackoverflow.com/questions/47403407/is-tensorflow-dataset-api-slower-than-queues
        https://stackoverflow.com/questions/48777889/tf-data-api-how-to-efficiently-sample-small-patches-from-images
        https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
        
        Don't forget to use prefetch operation. 
        
        Apparently this is how savable iterators are to be used
        Can't figure out how 
        https://www.tensorflow.org/api_docs/python/tf/contrib/data/make_saveable_from_iterator
        
        Stackoverflow
        https://stackoverflow.com/questions/46917588/restoring-a-tensorflow-model-that-uses-iterators/49236050#49236050
        '''
        assert self.__graphCreated is False
        dim = [None, self.numSubinstance, self.numTimeSteps, self.numFeats]
        if self.useEmbeddings:
            dim = [None, self.numSubinstance, self.numTimeSteps]
        self.X = tf.placeholder(tf.float32, dim,  name='inpX')
        self.Y = tf.placeholder(tf.float32,
                                [None, self.numSubinstance, self.numOutput], name='inpY')
        self.batchSize = tf.placeholder(tf.int64, name='batchSize')
        self.numEpochs= tf.placeholder(tf.int64, name='numEpochs')
        dataset_x_target = tf.data.Dataset.from_tensor_slices(self.X)
        dataset_y_target = tf.data.Dataset.from_tensor_slices(self.Y)
        dataset_target = tf.data.Dataset.zip((dataset_x_target, dataset_y_target)).repeat(self.numEpochs)
        dataset_target = dataset_target.batch(self.batchSize).prefetch(self.prefetchNum)
        dataset_iterator_target = tf.data.Iterator.from_structure(dataset_target.output_types,
                                                                dataset_target.output_shapes)
        dataset_next_target = dataset_iterator_target.get_next()
        dataset_init_target = dataset_iterator_target.make_initializer(dataset_target, name='dataset_init')
        self.dataset_init = dataset_init_target
        self.__dataset_next = dataset_next_target

    def __getRNNOut(self, x, zeroStateShape):
        '''
        x: [num_timestep, batch_size, num_feats]
        '''
        if self.useCudnn is True:
            # Does not support forget bias due to sum bug
            assert self.useCudnn is False, 'Cudnn support not complete. This argument is depricated and will be removed'
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            self.cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=self.numHidden, name='cudnnCell')
            outputs, states = self.cell(x)
        else:
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.numHidden, forget_bias=1.0, name='cell')
            state = self.cell.zero_state(zeroStateShape, tf.float32)
            wrapped_cell = self.cell
            if self.useDropout is True:
                self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
                wrapped_cell = tf.contrib.rnn.DropoutWrapper(self.cell,
                                                             input_keep_prob=self.keep_prob,
                                                             output_keep_prob=self.keep_prob)
                    
            outputs, states = tf.nn.static_rnn(wrapped_cell, x, dtype=tf.float32)
        return outputs, states
            
    def __createForwardGraph(self, X):
        assert self.__graphCreated is False
        self.W1 = tf.Variable(tf.random_normal([self.numHidden, self.numFC]), name='W1')
        self.B1 = tf.Variable(tf.random_normal([self.numFC]), name='B1')
        self.W2 = tf.Variable(tf.random_normal([self.numFC, self.numOutput]), name="W2")
        self.B2 = tf.Variable(tf.random_normal([self.numOutput]), name='B2')
        zeroStateShape = tf.shape(X)[0]
        # Reshape into 3D such that the first dimension is -1 * numSubinstance
        # where each numSubinstance segment corresponds to one bag
        # then shape it back in into 4D
        x = tf.reshape(X, [-1, self.numTimeSteps, self.numFeats])
        x = tf.unstack(x, num=self.numTimeSteps, axis=1)
        # Get the LSTM output
        outputs, states = self.__getRNNOut(x, zeroStateShape)
        ret = tf.add(tf.matmul(outputs[-1], self.W1), self.B1)
        ret = tf.add(tf.matmul(ret, self.W2), self.B2)
        # Convert back to bag form
        with tf.name_scope("final_output"):
            ret = tf.reshape(ret, [-1, self.numSubinstance, self.numOutput], name='bag_output')
            self.output = ret
            self.pred = tf.nn.softmax(self.output, axis=2, name='softmax_pred')
            self.predictionClass = tf.argmax(self.pred, axis=2, name='predicted_classes')
            
        varList = [self.W1, self.B1, self.W2, self.B2]
        self.LSTMVars = self.cell.variables
        varList.extend(self.LSTMVars)
        self.varList = varList
    
    def __createLossGraph(self, X, Y, alpha, beta):
        assert self.__graphCreated is False
        # pred of dim [-1, numSubinstance, numOutputs]
        diff = (self.output - Y)
        diffL = (self.pred - Y)
        logits = tf.reshape(self.output, [-1, self.numOutput])
        labels = tf.reshape(Y, [-1, self.numOutput])
        # Regular softmax
        softmax1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
        # A mask that selects only the negative sets
        negInstanceMask = tf.reshape(tf.cast(tf.argmin(Y, axis=2), dtype=tf.float32), [-1])
        # Additional penalty for misprediction on negative set
        softmax2 = tf.reduce_mean(negInstanceMask *  tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
        l2Loss = tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2)
        self.softmaxLoss = tf.add(softmax1, alpha * softmax2)
        self.softmaxLoss = tf.add(self.softmaxLoss, beta * l2Loss, name='xentropy-loss')
        equal = tf.equal(tf.argmax(self.pred, axis=2), tf.argmax(Y, axis=2))
        self.accTilda = tf.reduce_mean(tf.cast(equal, tf.float32), name='acc_tilda')
        
    def __createTrainGraph(self, stepSize, loss, redirFile):
        with tf.name_scope("gradient"):
            lossOp = self.l2Loss
            if loss != 'l2':
                print("Using softmax loss", file=redirFile)
                lossOp = self.softmaxLoss
            else:
                assert ("l2 not supported")
                print("Usign untested L2 loss", file=redirFile)
                lossOp = self.l2Loss

            assert self.train_step_model is None
            assert lossOp is not None
            tst = tf.train.AdamOptimizer(stepSize).minimize(lossOp)
            self.train_step_model = tst
            self.lossOp = lossOp
            tf.add_to_collection("train_step", self.train_step_model)
            tf.add_to_collection("loss_op", self.lossOp)

    def __retrieveEmbeddings(self, X, embeddings_init, trainable):
        w2v = tf.constant(embeddings_init, dtype=tf.float32)
        vocabulary_size, embedding_size = embeddings_init.shape[0], embeddings_init.shape[1]
        assert embedding_size == self.numFeats
        self.word_embeddings = tf.get_variable("word_embeddings", initializer=w2v, trainable=trainable)
        self.embedded_word_ids = tf.nn.embedding_lookup(self.word_embeddings, X, name='embedding_lookup_op')
        return self.embedded_word_ids

    def createGraph(self, stepSize, alpha=0.0, beta=0.0, loss='smx',
                    trainEmbeddings=False, embeddings_init=None, redirFile=None):
        assert self.__graphCreated is False
        self.__createInputPipeline()
        X, Y = self.__dataset_next
        if self.useEmbeddings is True:
            assert embeddings_init is not None
            X = tf.cast(X, dtype=tf.int32)
            X = self.__retrieveEmbeddings(X, embeddings_init, trainEmbeddings)

        self.__createForwardGraph(X)
        self.__createLossGraph(X, Y, alpha, beta)
        self.__createTrainGraph(stepSize, loss, redirFile)
        self.__graphCreated = True
    
    def runOpList(self, opList, x, y = None, batch_size = 1000):
        if self.useEmbeddings is False:
            assert (x.ndim == 4)
            assert (x.shape[1] == self.numSubinstance)
            assert (x.shape[2] == self.numTimeSteps)
            assert (x.shape[3] == self.numFeats)
        else:
            assert x.ndim == 3
            assert x.shape[1] == self.numSubinstance
            assert x.shape[2] == self.numTimeSteps

        assert (self.sess != None)
        # TODO: Figure out a better way of doing this. With two iterators?
        if y is None:
            y = np.zeros([x.shape[0], self.numSubinstance, self.numOutput])
        _feed_dict = {self.X: x, self.Y: y, self.batchSize: batch_size, self.numEpochs: 1}
        outputList = []
        self.sess.run(self.dataset_init, feed_dict = _feed_dict)
        while True:
            try:
                if self.useDropout is False:
                    out = self.sess.run(opList)
                else:
                    out = self.sess.run(opList, feed_dict={self.keep_prob: 1.0})
                outputList.append(out)
                
            except tf.errors.OutOfRangeError:
                break
        return outputList
    
    def inference(self, x, batch_size):
        '''
        returns raw out and softmax out
        '''
        if self.useEmbeddings is False:
            assert (x.ndim == 4)
            assert (x.shape[1] == self.numSubinstance)
            assert (x.shape[2] == self.numTimeSteps)
            assert (x.shape[3] == self.numFeats)
        else:
            assert x.ndim == 3
            assert x.shape[1] == self.numSubinstance
            assert x.shape[2] == self.numTimeSteps

        assert (self.sess != None)
        # TODO: Figure out a better way of doing this. With two iterators?
        y = np.zeros([x.shape[0], self.numSubinstance, self.numOutput])
        _feed_dict = {self.X: x, self.Y: y, self.batchSize: batch_size, self.numEpochs: 1}
        outputList = []
        predictionList = []
        predictedClassList = []
        self.sess.run(self.dataset_init, feed_dict = _feed_dict)
        while True:
            try:
                if self.useDropout is False:
                    out, pred, pclass = self.sess.run([self.output, self.pred, self.predictionClass])
                else:
                    out, pred, pclass = self.sess.run([self.output, self.pred, self.predictionClass],
                                                      feed_dict={self.keep_prob: 1.0})
                outputList.extend(out)
                predictionList.extend(pred)
                predictedClassList.extend(pclass)
            except tf.errors.OutOfRangeError:
                break
        return np.array(outputList), np.array(predictionList), np.array(predictedClassList)

    def checkpointModel(self, modelPrefix, max_to_keep=5, global_step=1000, redirFile=None):
        saver = self.__saver
        if self.__saver is None:
            saver = tf.train.Saver(max_to_keep=max_to_keep, save_relative_paths=True)
            self.__saver = saver
        sess = self.sess
        assert(sess is not None)
        saver.save(sess, modelPrefix, global_step=global_step)
        print('Model saved to %s, global_step %d' % (modelPrefix, global_step), file=redirFile)
    
    def importModelTF(self, modelPrefix, global_step=1000, redirFile=None):
        assert self.__saver is None
        assert self.useCudnn is False, 'CudnnLSTM restore not supported yet'
        if self.sess is None:
            self.sess = tf.Session()
        
        metaname = modelPrefix + '-%d.meta' % global_step
        basename = os.path.basename(metaname)
        fileList = os.listdir(os.path.dirname(modelPrefix))
        fileList = [x for x in fileList if x.startswith(basename)]
        assert len(fileList) is 1, "%r \n %s" % (fileList, os.path.dirname(modelPrefix))
        chkpt = basename + '/' + fileList[0]
        
        saver = tf.train.import_meta_graph(metaname)
        metaname = metaname[:-5]
        saver.restore(self.sess, metaname)
        print('Restoring %s' % metaname, file=redirFile)
        graph = tf.get_default_graph()
        
        # Restore placeholders
        self.X = graph.get_tensor_by_name("inpX:0")
        self.Y = graph.get_tensor_by_name("inpY:0")
        self.batchSize = graph.get_tensor_by_name("batchSize:0")
        self.numEpochs = graph.get_tensor_by_name("numEpochs:0")
        if self.useDropout:
            self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
        if self.useEmbeddings is True:
            self.word_embeddings = graph.get_tensor_by_name("word_embeddings:0")
        
        # Restore operations
        self.output = graph.get_tensor_by_name("final_output/bag_output:0")
        self.pred = graph.get_tensor_by_name("final_output/softmax_pred:0")
        self.predictionClass = graph.get_tensor_by_name("final_output/predicted_classes:0")
        self.train_step_model = tf.get_collection("train_step")[0]
        self.lossOp = tf.get_collection("loss_op")[0]
        self.accTilda = graph.get_tensor_by_name("acc_tilda:0")
        if self.useEmbeddings is True:
            self.embedded_word_ids = graph.get_operation_by_name('embedding_lookup_op')
        
        # Creating datset
        assert self.dataset_init is None
        self.dataset_init = graph.get_operation_by_name('dataset_init')
        assert self.dataset_init is not None
        
        # Restore model parameters
        self.B1 = graph.get_tensor_by_name('B1:0')
        self.W1 = graph.get_tensor_by_name('W1:0')
        self.W2 = graph.get_tensor_by_name('W2:0')
        self.B2 = graph.get_tensor_by_name('B2:0')
        kernel = graph.get_tensor_by_name("rnn/cell/kernel:0")
        bias = graph.get_tensor_by_name("rnn/cell/bias:0")
        self.LSTMVars = [kernel, bias]
        self.varList = [self.W1, self.B1, self.W2, self.B2]
        self.varList.extend(self.LSTMVars)
        self.__graphCreated = True
        return graph
    
        def exportNPY(self, outFolder=None):
        W1, B1, W2, B2 = self.W1, self.B1, self.W2, self.B2
        lstmKernel, lstmBias = self.LSTMVars
        W1, B1, W2, B2, lstmKernel, lstmBias = self.sess.run([W1, B1, W2, B2, lstmKernel, lstmBias])
        FCW = np.matmul(W2.T, W1.T)
        FCB = np.matmul(W2.T, B1) + B2
        lstmKernel = lstmKernel.T
        lstmBias = lstmBias.T
        if outFolder is None:
            return lstmKernel, lstmBias, FCW, FCB
        lstmKernel_f = outFolder + '/' + 'lstmKernel.npy'
        lstmBias_f = outFolder + '/' + 'lstmBias.npy'
        FCB_f = outFolder + '/' + 'fcb.npy'
        FCW_f = outFolder + '/' + 'fcw.npy'
        assert os.path.isdir(outFolder)
        assert os.path.isfile(lstmKernel_f) is False
        assert os.path.isfile(lstmBias_f) is False
        assert os.path.isfile(FCB_f) is False
        assert os.path.isfile(FCW_f) is False
        np.save(lstmKernel_f, lstmKernel)
        np.save(lstmBias_f, lstmBias)
        np.save(FCW_f, FCW)
        np.save(FCB_f, FCB)
        return lstmKernel, lstmBias, FCW, FCB

    def __trainingSetup(self, reuse, gpufrac, redirFile):
        if self.sess is None:
            if gpufrac is not None:
                assert (gpufrac >= 0)
                assert (gpufrac <= 1)
                print('GPU Fraction: %f' % gpufrac, file = redirFile)
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpufrac)
                self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            else:
                print('GPU Fraction: 1.0', file=redirFile)
                self.sess = tf.Session()
        else:
            print("Reusing previous session", file=redirFile)

        if not reuse:
            init = tf.global_variables_initializer()
            self.sess.run(init)
        else:
            print("Reusing previous init", file=redirFile)

                
    def trainModel(self, x_train, y_train, x_test, y_test,
                   trainingParams, redirFile=None,
                   reuse=False, gpufrac=None):
        
        assert self.__graphCreated is True
        self.__trainingSetup(reuse, gpufrac, redirFile)
        batch_size= trainingParams['batch_size']
        if self.useDropout:
            keep_prob = trainingParams['keep_prob']
        lossOp = self.lossOp
        
        max_epochs = trainingParams['max_epochs']
        num_batches = int(np.ceil(len(x_train) / batch_size))
        if self.lossList is None:
            self.lossList = []
        train_step = self.train_step_model
        
        currentBatch = 0
        print("Executing %d epochs" % max_epochs, file=redirFile)
        self.sess.run(self.dataset_init,
                      feed_dict={self.X: x_train, self.Y: y_train, self.batchSize: batch_size,
                                 self.numEpochs: max_epochs})
        while True:
            try:
                if currentBatch % 15 == 0:
                    if self.useDropout is True:
                        _, acc, loss = self.sess.run([train_step, self.accTilda, lossOp],
                                                    feed_dict = {self.keep_prob: keep_prob})
                    else:
                        _, acc, loss = self.sess.run([train_step, self.accTilda, lossOp])
                    self.lossList.append(loss)
                    epoch = int(currentBatch / num_batches)
                    tmp = int(currentBatch % max(num_batches, 1))
                    print("\rEpoch %3d Batch %5d (%5d) Loss %2.5f Accuracy %2.5f " %
                          (epoch, tmp, currentBatch, loss, acc), end='', file=redirFile)
                else:
                    if self.useDropout is True:
                        self.sess.run(train_step, feed_dict = {self.keep_prob: keep_prob})                       
                    else:
                        self.sess.run(train_step)                       
                    ed = time.time()
                currentBatch += 1
            except tf.errors.OutOfRangeError:
                break
        print(file=redirFile)
         

                
def getLengthScores(Y_predicted, val=1):
    '''
    Returns an matrix which contains the length of the longest positive
    subsequence of val ending at that index.
    Y_predicted: [-1, numSubinstance] Is the instance level class
        labels.
    '''
    scores = np.zeros(Y_predicted.shape)
    for i, bag in enumerate(Y_predicted):
        for j, instance in enumerate(bag):
            prev = 0
            if j > 0:
                prev = scores[i, j-1]
            if instance == val:
                scores[i, j] = prev + 1
            else:
                scores[i, j] = 0
    return scores


def bagPrediction(Y_predicted, minSubsequenceLen = 4, numClass = 2,
                  redirFile = None):
    '''
    Similar to bagStats
    '''
    assert(Y_predicted.ndim == 2)
    scoreList = []
    for x in range(1, numClass):
        scores = getLengthScores(Y_predicted, val=x)
        length = np.max(scores, axis=1) 
        scoreList.append(length)
    scoreList = np.array(scoreList)
    scoreList = scoreList.T
    assert(scoreList.ndim == 2)
    assert(scoreList.shape[0] == Y_predicted.shape[0])
    assert(scoreList.shape[1] == numClass - 1)
    length = np.max(scoreList, axis=1)
    assert(length.ndim == 1)
    assert(length.shape[0] == Y_predicted.shape[0])
    predictionIndex = (length >= minSubsequenceLen)
    prediction = np.zeros((Y_predicted.shape[0]))
    labels = np.argmax(scoreList, axis=1) + 1
    prediction[predictionIndex] = labels[predictionIndex]
    return prediction, scoreList


def bagStats(Y_predicted, Y_true, Y_bag=None,
             minSubsequenceLen = 4, numClass=2,
             redirFile = None):
    '''
    Returns bag level statistics given instance level predictions

    A bag is considered to belong to a non-zero class if
    minSubsequenceLen is satisfied. Otherwise, it is assumed
    to belong to class 0. class 0 is negative by default.
    
    Y_predicted is the predicted instance level results
    [-1, numsubinstance]
    Y True is the correct instance level label
    [-1, numsubinstance]
    '''
    assert(Y_predicted.ndim == 2)
    assert(Y_true.ndim == 2)

    scoreList = []
    for x in range(1, numClass):
        scores = getLengthScores(Y_predicted, val=x)
        length = np.max(scores, axis=1) 
        scoreList.append(length)
    scoreList = np.array(scoreList)
    scoreList = scoreList.T
    assert(scoreList.ndim == 2)
    assert(scoreList.shape[0] == Y_predicted.shape[0])
    assert(scoreList.shape[1] == numClass - 1)
    if Y_bag is None:
        Y_bag = Y_true[:, 0] 
    length = np.max(scoreList, axis=1)
    assert(length.ndim == 1)
    assert(length.shape[0] == Y_predicted.shape[0])
    predictionIndex = (length >= minSubsequenceLen)
    prediction = np.zeros((Y_predicted.shape[0]))
    labels = np.argmax(scoreList, axis=1) + 1
    prediction[predictionIndex] = labels[predictionIndex]
    assert(len(Y_bag) == len(prediction))
    correct = (prediction == Y_bag).astype('int')
    acc = np.mean(correct)
    prediction = prediction.astype('int')
    cmatrix = getConfusionMatrix(prediction, Y_bag, numClass) 
    return acc, cmatrix


def analysisModel(predictions, trueLabels,
                  Y_bag, numSubinstance, plt,
                  plot=True, redirFile=None):
    '''
    some basic analysis on predictions and true labels
    predictions [-1, numsubinstance]
    trueLabels [-1, numsubinstance]

    WARNING: DOES NOT SUPPORT MULTICLASS
    '''
    bagAccList = []
    preList = []
    recList = []
    fscoreList = []
    for i in range(1, numSubinstance + 1):
        trueAcc, cmatrix = bagStats(predictions,
                                    trueLabels,
                                    minSubsequenceLen=i, Y_bag=Y_bag)
        pre, rec = getPrecisionRecall(cmatrix)
        bagAccList.append(trueAcc)
        preList.append(pre)
        recList.append(rec)
        fscore = 2 * pre * rec
        denom = pre + rec
        if denom == 0:
            denom = 1
        fscore /= (denom)
        fscoreList.append(fscore)

    if plot:
        ax.ylim(0., 1)
        ax.plot(np.arange(1, len(bagAccList) + 1),
                bagAccList, linestyle='--', marker='o', label='accuracy')
        ax.plot(np.arange(1, len(bagAccList) + 1),
                preList, linestyle='--', marker='o', label='precision')
        ax.plot(np.arange(1, len(bagAccList) + 1),
                recList, linestyle='--', marker='o', label='recall')
        ax.plot(np.arange(1, len(bagAccList) + 1),
                fscoreList, linestyle='--', marker='o', label='fscore')
        ax.legend(loc='lower right')
    df = pd.DataFrame({
        'len': np.arange(1, numSubinstance + 1),
        'precision' : preList,
        'recall': recList,
        'fscore': fscoreList,
        'acc': bagAccList
    })
    df.set_index('len')
    df = df[['len', 'acc', 'fscore', 'precision', 'recall']]
    print(df, file=redirFile)
    print("Max accuracy %f at subsequencelength %d"
          % (np.max(bagAccList), np.argmax(bagAccList) + 1), file=redirFile)
    print("Max fscore %f at subsequencelength %d"
          % (np.max(fscoreList), np.argmax(fscoreList) + 1), file=redirFile)
    idx = np.argmax(fscoreList)
    print("Precision at subsequencelength %d: %f"
          % (idx + 1, preList[idx]), file=redirFile)
    print("Recall at subsequencelength %d: %f"
          % (idx + 1, recList[idx]), file=redirFile)
    predictionsFlat = np.reshape(predictions, [-1])
    trueLabelsFlat = np.reshape(trueLabels, [-1])
    indx = (trueLabelsFlat == 0)
    print("Fraction false alarm %f (%d/%d) " % (np.mean(predictionsFlat[indx]),
          np.sum(predictionsFlat[indx]), len(np.where(indx)[0])), file=redirFile)
    return np.argmax(fscoreList) + 1
 

def analysisModelMultiClass(predictions, trueLabels,
                            Y_bag, numSubinstance, 
                            numClass, redirFile=None,
                            verbose=False):
    '''
    some basic analysis on predictions and true labels
    This is the multiclass version
    predictions [-1, numsubinstance] is the instance level prediction
    trueLabels [-1, numsubinstance] is the instance level true label
        This is used as bagLabel if bag labels no provided.
    verbose: Prints verbose data frame. Includes additionally, precision
        and recall information.
        
    In the 2 class setting, precision, recall and f-score for
    class 1 is also printed.
    '''
    assert (predictions.ndim == 2)
    assert (predictions.shape[1] == numSubinstance)
    assert (trueLabels.ndim == 2)
    assert (trueLabels.shape[1] == numSubinstance)
    assert (Y_bag.ndim == 1)
    assert (len(predictions) == len(trueLabels))
    assert (len(Y_bag) == len(predictions))
    pholder = [0.0] * numSubinstance
    df = pd.DataFrame()
    df['len'] = np.arange(1, numSubinstance + 1)
    df['acc'] = pholder
    df['macro-fsc'] = pholder
    df['macro-pre'] = pholder
    df['macro-rec'] = pholder

    df['micro-fsc'] = pholder
    df['micro-pre'] = pholder
    df['micro-rec'] = pholder
    colList = []
    colList.append('acc') 
    colList.append('macro-fsc')
    colList.append('macro-pre')
    colList.append('macro-rec')

    colList.append('micro-fsc')
    colList.append('micro-pre')
    colList.append('micro-rec')
    for i in range(0, numClass):
        pre = 'pre_%02d' % i
        rec = 'rec_%02d' % i
        df[pre] = pholder
        df[rec] = pholder
        colList.append(pre)
        colList.append(rec)

    for i in range(1, numSubinstance + 1):
        trueAcc, cmatrix = bagStats(predictions, trueLabels,
                                    numClass=numClass,
                                    minSubsequenceLen=i,
                                    Y_bag=Y_bag, redirFile = redirFile)
        df.iloc[i-1, df.columns.get_loc('acc')] = trueAcc

        macro, micro = getMacroMicroFScore(cmatrix)
        df.iloc[i-1, df.columns.get_loc('macro-fsc')] = macro
        df.iloc[i-1, df.columns.get_loc('micro-fsc')] = micro

        pre, rec = getMacroPrecisionRecall(cmatrix)
        df.iloc[i-1, df.columns.get_loc('macro-pre')] = pre
        df.iloc[i-1, df.columns.get_loc('macro-rec')] = rec 

        pre, rec = getMicroPrecisionRecall(cmatrix)
        df.iloc[i-1, df.columns.get_loc('micro-pre')] = pre
        df.iloc[i-1, df.columns.get_loc('micro-rec')] = rec 
        for j in range(numClass):
            pre, rec = getPrecisionRecall(cmatrix, label=j) 
            pre_ = df.columns.get_loc('pre_%02d' % j)
            rec_ = df.columns.get_loc('rec_%02d' % j)
            df.iloc[i-1, pre_ ] = pre
            df.iloc[i-1, rec_ ] = rec

    df.set_index('len')
    # Comment this line to include all columns
    colList = ['len', 'acc', 'macro-fsc', 'macro-pre', 'macro-rec']
    colList += ['micro-fsc', 'micro-pre', 'micro-rec']
    if verbose:
        for col in df.columns:
            if col not in colList:
                colList.append(col)
    if numClass == 2:
        precisionList = df['pre_01'].values
        recallList = df['rec_01'].values
        denom = precisionList + recallList
        denom[denom == 0] = 1
        numer = 2 * precisionList * recallList
        f_ = numer / denom
        df['fscore_01'] = f_
        colList.append('fscore_01')
        
    df = df[colList]
    with pd.option_context('display.max_rows', 100,
                           'display.max_columns', 100,
                           'expand_frame_repr', True):
        print(df, file=redirFile)

    idx = np.argmax(df['acc'].values)
    val = np.max(df['acc'].values)
    print("Max accuracy %f at subsequencelength %d" % (val, idx + 1), file=redirFile)
    
    val = np.max(df['micro-fsc'].values)
    idx = np.argmax(df['micro-fsc'].values) 
    print("Max micro-f %f at subsequencelength %d" % (val, idx + 1), file=redirFile)
    val = df['micro-pre'].values[idx]
    print("Micro-precision %f at subsequencelength %d" % (val, idx + 1), file=redirFile)
    val = df['micro-rec'].values[idx]
    print("Micro-recall %f at subsequencelength %d" % (val, idx + 1), file=redirFile)

    idx = np.argmax(df['macro-fsc'].values)
    val = np.max(df['macro-fsc'].values)
    print("Max macro-f %f at subsequencelength %d" % (val, idx + 1), file=redirFile)
    val = df['macro-pre'].values[idx]
    print("macro-precision %f at subsequencelength %d" % (val, idx + 1), file=redirFile)
    val = df['macro-rec'].values[idx]
    print("macro-recall %f at subsequencelength %d" % (val, idx + 1), file=redirFile)
    if numClass == 2 and verbose:
        idx = np.argmax(df['fscore_01'].values)
        val = np.max(df['fscore_01'].values)
        print('Max fscore %f at subsequencelength %d' % (val, idx + 1), file=redirFile)
        print('Precision %f at subsequencelength %d' % (df['pre_01'].values[idx], idx + 1), file=redirFile)
        print('Recall %f at subsequencelength %d' % (df['rec_01'].values[idx], idx + 1), file=redirFile)
    predictionsFlat = np.reshape(predictions, [-1])
    trueLabelsFlat = np.reshape(trueLabels, [-1])
    indx1 = (trueLabelsFlat == 0)
    indx2 = (predictionsFlat != 0)
    indx = indx1.astype('int') + indx2.astype('int')
    # Both true
    indx = (indx == 2)
    predictionsFlat = np.zeros(len(predictionsFlat)) 
    predictionsFlat[indx] = 1
    num = np.sum(predictionsFlat[indx])
    denom = len(np.where(indx1)[0])
    frac = 0.0
    if denom != 0:
        frac = num / denom
    print("Fraction false alarm %f (%d/%d) " % (frac, num, denom), file=redirFile)
    return df



def analyseAudio(file, doCopy=False, subinstanceWidth=10000,
                 subinstanceStride=1000, redirFile=None):
    # Move file to staging area
    # /lstmExperiments/tmp_audio
    if doCopy:
        dst = '/home/t-dodenn/Work/lstmExperiments/temp_audio/'
        shutil.copy(file, dst) 
    print(subinstanceWidth, subinstanceStride, file=redirFile)
    # Read the file and print size and what not
    sampleRate, x = r.read(file)
    print("numsamles: %d" % len(x), file=redirFile)
    numZeros = 26032 - len(x)
    print("num zeros: %d" % numZeros, file=redirFile)
    print("duration: %fs" % (len(x)/ sampleRate), file=redirFile)
    w = max(0, numZeros - subinstanceWidth)
    i = np.ceil((w) / subinstanceStride) * subinstanceStride 
    print("First non-zero window %d  [%d, %d]" % (i/subinstanceStride, i, i + subinstanceWidth), file=redirFile)
    print("Number of non-zeros in that window %d" % ( i + subinstanceWidth - numZeros), file=redirFile)


def plotLenHist(predictions, bagLabels, ax, sns):
    '''
    Plots a histogram of the longest continuous subsequences
    '''
    scores = getLengthScores(predictions)
    index = (bagLabels ==1 )
    scores = np.max(scores, axis=1)
    ax.set_xlabel('Longest continuous')
    sns.distplot(scores[index], ax=ax)
    
    
def plotProbability(res, x, ax):
    '''
    Plots probability of negative class
    given the result and index into the result
    res : [-1, numSubinstance, 2]
    x: index into res
    '''
    ax.plot(res[x, :, 0], '--', label='negative')
    ax.plot(res[x, :, 1], '--', label='positive')
    ax.plot(res[x, :, 0] - res[x, :, 1],  marker='o',label='difference')
    ax.legend(loc='lower right')
    

def updateYLabel(currentY, predictions, bagLabel, minSubsequenceLen, numClass, epsilonL=1, epsilonR=1):
    '''
    CurrentY [-1, numsubinstance, numClass]
    bagLabel [-1, numClass]
    minSubsequenceLen: A bag is considered to be predicted as positive
        iff `minSubsequenceLen` of instances are classified as positive.
        In such cases, every other instance outside this subsequenceLen
        +- epsilon is labeled as negative
    epsilon: an integer. Everything outside the minimumSubsequenceLen - epsilon
        to minimumSubsequenceLen + epsilon  is labeled as negative
    '''
    # Can only handle binary
    assert(numClass == 2)
    scores = getLengthScores(predictions)
    scoresMax = np.max(scores, axis=1)
    scoresArg = np.argmax(scores, axis=1)
    startIndex = np.zeros(len(scoresArg))
    startIndex[:] = -1
    index = (scoresMax != 0)
    startIndex[index] = scoresArg[index] - scoresMax[index] + 1
    # StartIndex has -1 for examples which have 0 positives
    index1 = (startIndex == -1).astype(int)
    index2 = (bagLabel == 0).astype(int)
    index3 = (index1 + index2)
    index3 = (index3 == 0)
    newY = np.array(currentY)
    startIndex = startIndex.astype(int)
    for i in range(len(newY)):
        if index3[i] == True:
            newY[i, :, 0] = 1
            newY[i, :, 1] = 0
            start = max(0, startIndex[i] - epsilonL)
            end = min(len(newY[i]), scoresArg[i] + 1 + epsilonR)
            newY[i, start:end, 1] = 1
            newY[i, start:end, 0] = 0
    return newY


def updateYLabel2(currentY, softmaxOut, bagLabel, numClass=2, negativeProb=-0.25, maxSamplesLen=4):
    '''
    CurrentY: [-1, numsubinstance, numClass]
    softmaxOut: [-1, numsubinstance, numClass]
    bagLabel [-1]
    Uses the softmax output from the previous run 
    A continuous prefix and/or suffix of negative is labeled as a negative
    if the sequence length is less than maxSamples. A sample is determined
    as negative if prob(neg) - prob(pos) >= negativeProb
    '''
    assert(numClass == 2)
    index = (bagLabel == 1)
    indexList = np.where(bagLabel)[0]
    negativeProbList = softmaxOut[:, :, 0] - softmaxOut[:, :, 1]
    newY = np.array(currentY)
    for i in indexList:
        currSample = negativeProbList[i]
        prefixLen = 0
        suffixLen = 0
        for prob in currSample:
            if prob >= negativeProb:
                prefixLen += 1
            else:
                break
        if prefixLen > maxSamplesLen:
            prefixLen = 0
        for prob in (reversed(currSample)):
            if prob >= negativeProb:
                suffixLen += 1
            else:
                break
        if suffixLen > maxSamplesLen:
            suffixLen = 0
        # Update prefix set
        newY[i, :, 0] = 0
        newY[i, :, 1] = 1
        if prefixLen != 0:
            newY[i, :prefixLen, 0] = 1
            newY[i, :prefixLen, 1] = 0
            
        if suffixLen != 0:
            newY[i, -suffixLen:, 0] = 1
            newY[i, -suffixLen:, 1] = 0
    return newY



def updateYPolicy3(currentY, softmaxOut, bagLabel, minNegativeProb,
                   updatesPerCall, maxAllowedUpdates, numClass):
    '''
    CurrentY: [-1, numsubinstance, numClass]
    softmaxOut: [-1, numsubinstance, numClass]
    bagLabel [-1]
    minNegativeProb: A instance predicted as negative is labeled as
        negative iff prob. negative >= minNegativeProb
    updatePerCall: At most number of updates to per function call 
    maxAllowedUpdates: Total updates on positive bag cannot exceede
        maxAllowedUpdate

    Uses the softmax output from the previous run
    This policy incrementally increases the prefix/suffix of negative labels 
    in currentY.
    An instance is labelled as a negative if:
        1. All the instances preceding it in case of a prefix and
          all instances succeeding it in case of a continuous prefix
          and/or suffix of negative is labeled as a negative.
        2. The probability of the instance being negative > negativeProb.
        3. The instance is indeed predicted as negative (i.e. prob class 0 is max)
        4. If the sequence length is less than maxSamples.
        
    All four conditions must hold.
    In case of a tie between instances near the suffix and prefix, the one with
    maximum probability is updated. If probabilities are same, then the left
    prefix is updated.

    CLASS 0 is assumed to be negative class
    '''
    assert currentY.ndim == 3
    assert softmaxOut.ndim == 3
    assert bagLabel.ndim == 1
    assert len(currentY) == len(softmaxOut)
    assert len(softmaxOut) == len(bagLabel)
    numSubinstance = currentY.shape[1]
    assert maxAllowedUpdates < numSubinstance
    assert softmaxOut.shape[1] == numSubinstance
    
    index = (bagLabel != 0)
    indexList = np.where(bagLabel)[0]
    newY = np.array(currentY)
    for i in indexList:
        currLabel = currentY[i]
        currProbabilities = softmaxOut[i]
        prevPrefix = 0
        prevSuffix = 0
        for inst in currLabel:
            if np.argmax(inst) == 0:
                prevPrefix += 1
            else:
                break
        for inst in reversed(currLabel):
            if np.argmax(inst) == 0:
                prevSuffix += 1
            else:
                break
        assert (prevPrefix + prevSuffix <= maxAllowedUpdates)
        leftIdx = int(prevPrefix)
        rightIdx = numSubinstance - int(prevSuffix) - 1
        possibleUpdates = min(updatesPerCall, maxAllowedUpdates - prevPrefix - prevSuffix)
        while (possibleUpdates > 0):
            assert leftIdx < numSubinstance
            assert leftIdx >= 0
            assert rightIdx < numSubinstance
            assert rightIdx >= 0
            leftLbl = np.argmax(currProbabilities[leftIdx])
            leftProb = np.max(currProbabilities[leftIdx])
            rightLbl = np.argmax(currProbabilities[rightIdx])
            rightProb = np.max(currProbabilities[rightIdx])
            if (leftLbl != 0 and rightLbl !=0):
                break
            elif (leftLbl == 0 and rightLbl != 0):
                if leftProb >= minNegativeProb:
                    newY[i, leftIdx, :] = 0
                    newY[i, leftIdx, 0] = 1
                    leftIdx += 1
                else:
                    break
            elif (leftLbl != 0 and rightLbl == 0):
                if rightProb >= minNegativeProb:
                    newY[i, rightIdx, :] = 0
                    newY[i, rightIdx, 0] = 1
                    rightIdx -= 1
                else:
                    break
            elif leftProb >= rightProb:
                if leftProb >= minNegativeProb:
                    newY[i, leftIdx, :] = 0
                    newY[i, leftIdx, 0] = 1
                    leftIdx += 1
                else:
                    break
            elif rightProb > leftProb:
                if rightProb >= minNegativeProb:
                    newY[i, rightIdx, :] = 0
                    newY[i, rightIdx, 0] = 1
                    rightIdx -= 1
                else:
                    break
            possibleUpdates -= 1
    return newY


def updateYPolicy4(currentY, softMaxOut, bagLabel, numClasses, k):
    '''
    currentY: [-1, numsubinstance, numClass]
    softmaxOut: [-1, numsubinstance, numClass]
    bagLabel [-1]
    k: minimum length of continuous non-zero examples
    
    Check which is the longest continuous label for each bag
    If this label is the same as the bagLabel, and if the length is at least k:
        find all the strings with this longest length
        apart from the string having maximum summation of probabilities for that class label, label all other instances as 0
    '''
    assert currentY.ndim == 3
    assert k <= currentY.shape[1]
    assert k > 0
    # predicted label for each instance is max of softmax
    predictedLabels = np.argmax(softMaxOut, axis=2)
    scoreList = []
    # classScores[i] is a 2d array where a[j,k] is the longest string of consecutive class labels i in bag j ending at instance k
    classScores = [-1]
    for i in range(1, numClasses):
        scores = getLengthScores(predictedLabels, val=i)
        classScores.append(scores)
        length = np.max(scores, axis=1) 
        scoreList.append(length)
    scoreList = np.array(scoreList)
    scoreList = scoreList.T
    # longestContinuousClass[i] is the class label having longest substring in bag i
    longestContinuousClass = np.argmax(scoreList, axis=1) + 1
    # longestContinuousClassLength[i] is length of longest class substring in bag i
    longestContinuousClassLength = np.max(scoreList, axis=1)
    
    assert longestContinuousClass.ndim == 1
    assert longestContinuousClass.shape[0] == bagLabel.shape[0]
    assert longestContinuousClassLength.ndim == 1
    assert longestContinuousClassLength.shape[0] == bagLabel.shape[0]
    
    newY = np.array(currentY)
    index = (bagLabel != 0)
    indexList = np.where(index)[0]
    
    # iterate through all non-zero bags
    for i in indexList:
        lcc = longestContinuousClass[i] # longest continuous class for this bag
        lccl = int(longestContinuousClassLength[i]) # length of longest continuous class for this bag
        if lcc != bagLabel[i]: # if bagLabel is not the same as longest continuous class, don't update
            continue
        if lccl < k: # we check for longest string to be at least k
            continue
        lengths = classScores[lcc][i] 
        assert np.max(lengths) == lccl
        possibleCandidates = np.where(lengths == lccl)[0]
        sumProbsAcrossLongest = {} # stores (candidateIndex, sum of probabilities over window for this index) pairs
        for candidate in possibleCandidates:
            sumProbsAcrossLongest[candidate] = 0.0
            for j in range(0, lccl): # sum the probabilities over the continuous substring
                sumProbsAcrossLongest[candidate] += softMaxOut[i, candidate-j, lcc]
        
        # we want only the one with maximum sum of probabilities; sort dict by value
        sortedProbs = sorted(sumProbsAcrossLongest.items(), key=lambda x: x[1], reverse=True)
        bestCandidate = sortedProbs[0][0]
        # apart from (bestCanditate-lcc,bestCandidate] label everything else as 0
        newY[i, :, :] = 0
        newY[i, :, 0] = 1
        newY[i, bestCandidate-lccl+1:bestCandidate+1, 0] = 0
        newY[i, bestCandidate-lccl+1:bestCandidate+1, lcc] = 1
    
    return newY


def getPrefixSuffixList(softmaxOut, bagLabel, negativeProb):
    index = (bagLabel == 1)
    indexList = np.where(bagLabel)[0]
    negativeProbList = softmaxOut[:, :, 0] - softmaxOut[:, :, 1]
    prefixList = []
    suffixList = []
    for i in indexList:
        currSample = negativeProbList[i]
        prefixLen = 0
        suffixLen = 0
        for prob in currSample:
            if prob >= negativeProb:
                prefixLen += 1
            else:
                break
        for prob in (reversed(currSample)):
            if prob >= negativeProb:
                suffixLen += 1
            else:
                break
        prefixList.append(prefixLen)
        suffixList.append(suffixLen)
    return prefixList, suffixList
