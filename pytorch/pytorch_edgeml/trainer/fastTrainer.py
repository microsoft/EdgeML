# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import sys
import torch
import torch.nn as nn
import pytorch_edgeml.utils as utils
from pytorch_edgeml.graph.rnn import *
import numpy as np


class FastTrainer:

    def __init__(self, FastObj, numClasses, sW=1.0, sU=1.0,
                 learningRate=0.01, outFile=None, device=None):
        '''
        FastObj - Can be either FastRNN or FastGRNN or any of the RNN cells 
        in graph.rnn with proper initialisations
        numClasses is the # of classes
        sW and sU are the sparsity factors for Fast parameters
        batchSize is the batchSize
        learningRate is the initial learning rate
        '''
        self.FastObj = FastObj

        self.sW = sW
        self.sU = sU

        self.numClasses = numClasses
        self.inputDims = self.FastObj.input_size
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.learningRate = learningRate

        if outFile is not None:
            self.outFile = open(outFile, 'w')
        else:
            self.outFile = sys.stdout

        if self.sW > 0.99 and self.sU > 0.99:
            self.isDenseTraining = True
        else:
            self.isDenseTraining = False

        self.assertInit()
        self.numMatrices = self.FastObj.num_weight_matrices
        self.totalMatrices = self.numMatrices[0] + self.numMatrices[1]

        self.optimizer = self.optimizer()

        self.RNN = BaseRNN(self.FastObj).to(self.device)

        self.FC = nn.Parameter(torch.randn(
            [self.FastObj.output_size, self.numClasses])).to(self.device)
        self.FCbias = nn.Parameter(torch.randn(
            [self.numClasses])).to(self.device)

        self.FastParams = self.FastObj.getVars()

    def classifier(self, feats):
        '''
        Can be raplaced by any classifier
        TODO: Make this a separate class if needed
        '''
        return torch.matmul(feats, self.FC) + self.FCbias

    def computeLogits(self, input):
        '''
        Compute graph to unroll and predict on the FastObj
        '''
        if self.FastObj.cellType == "LSTMLR":
            feats, _ = self.RNN(input)
            logits = self.classifier(feats[:, -1])
        else:
            feats = self.RNN(input)
            logits = self.classifier(feats[:, -1])

        return logits, feats[:, -1]

    def optimizer(self):
        '''
        Optimizer for FastObj Params
        '''
        optimizer = torch.optim.Adam(
            self.FastObj.parameters(), lr=self.learningRate)

        return optimizer

    def loss(self, logits, labels):
        '''
        Loss function for given FastObj
        '''
        loss = utils.crossEntropyLoss(logits, labels)

        return loss

    def accuracy(self, logits, labels):
        '''
        Accuracy fucntion to evaluate accuracy when needed
        '''
        correctPredictions = (logits.argmax(dim=1) == labels.argmax(dim=1))
        accuracy = torch.mean(correctPredictions.float())

        return accuracy

    def assertInit(self):
        err = "sparsity must be between 0 and 1"
        assert self.sW >= 0 and self.sW <= 1, "W " + err
        assert self.sU >= 0 and self.sU <= 1, "U " + err

    def runHardThrsd(self):
        '''
        Function to run the IHT routine on FastObj
        '''
        self.thrsdParams = []
        thrsdParams = []
        for i in range(0, self.numMatrices[0]):
            thrsdParams.append(
                utils.hardThreshold(self.FastParams[i].data.cpu(), self.sW))
        for i in range(self.numMatrices[0], self.totalMatrices):
            thrsdParams.append(
                utils.hardThreshold(self.FastParams[i].data.cpu(), self.sU))
        for i in range(0, self.totalMatrices):
            self.FastParams[i].data = torch.FloatTensor(
                thrsdParams[i]).to(self.device)
        for i in range(0, self.totalMatrices):
            self.thrsdParams.append(torch.FloatTensor(
                np.copy(thrsdParams[i])).to(self.device))

    def runSparseTraining(self):
        '''
        Function to run the Sparse Retraining routine on FastObj
        '''
        self.reTrainParams = []
        for i in range(0, self.totalMatrices):
            self.reTrainParams.append(
                utils.copySupport(self.thrsdParams[i],
                                  self.FastParams[i].data))
        for i in range(0, self.totalMatrices):
            self.FastParams[i].data = self.reTrainParams[i]

    def getModelSize(self):
        '''
        Function to get aimed model size
        '''
        totalnnZ = 0
        totalSize = 0
        hasSparse = False
        for i in range(0, self.numMatrices[0]):
            nnz, size, sparseFlag = utils.countnnZ(self.FastParams[i], self.sW)
            totalnnZ += nnz
            totalSize += size
            hasSparse = hasSparse or sparseFlag

        for i in range(self.numMatrices[0], self.totalMatrices):
            nnz, size, sparseFlag = utils.countnnZ(self.FastParams[i], self.sU)
            totalnnZ += nnz
            totalSize += size
            hasSparse = hasSparse or sparseFlag
        for i in range(self.totalMatrices, len(self.FastParams)):
            nnz, size, sparseFlag = utils.countnnZ(self.FastParams[i], 1.0)
            totalnnZ += nnz
            totalSize += size
            hasSparse = hasSparse or sparseFlag

        # Replace this with classifier class call
        nnz, size, sparseFlag = utils.countnnZ(self.FC, 1.0)
        totalnnZ += nnz
        totalSize += size
        hasSparse = hasSparse or sparseFlag

        nnz, size, sparseFlag = utils.countnnZ(self.FCbias, 1.0)
        totalnnZ += nnz
        totalSize += size
        hasSparse = hasSparse or sparseFlag

        return totalnnZ, totalSize, hasSparse

    def saveParams(self, currDir):
        '''
        Function to save Parameter matrices
        '''
        if self.numMatrices[0] == 1:
            np.save(os.path.join(currDir, "W.npy"),
                    self.FastParams[0].data.cpu())
        elif self.FastObj.wRank is None:
            if self.numMatrices[0] == 2:
                np.save(os.path.join(currDir, "W1.npy"),
                        self.FastParams[0].data.cpu())
                np.save(os.path.join(currDir, "W2.npy"),
                        self.FastParams[1].data.cpu())
            if self.numMatrices[0] == 3:
                np.save(os.path.join(currDir, "W1.npy"),
                        self.FastParams[0].data.cpu())
                np.save(os.path.join(currDir, "W2.npy"),
                        self.FastParams[1].data.cpu())
                np.save(os.path.join(currDir, "W3.npy"),
                        self.FastParams[2].data.cpu())
            if self.numMatrices[0] == 4:
                np.save(os.path.join(currDir, "W1.npy"),
                        self.FastParams[0].data.cpu())
                np.save(os.path.join(currDir, "W2.npy"),
                        self.FastParams[1].data.cpu())
                np.save(os.path.join(currDir, "W3.npy"),
                        self.FastParams[2].data.cpu())
                np.save(os.path.join(currDir, "W4.npy"),
                        self.FastParams[3].data.cpu())
        elif self.FastObj.wRank is not None:
            if self.numMatrices[0] == 2:
                np.save(os.path.join(currDir, "W1.npy"),
                        self.FastParams[0].data.cpu())
                np.save(os.path.join(currDir, "W2.npy"),
                        self.FastParams[1].data.cpu())
            if self.numMatrices[0] == 3:
                np.save(os.path.join(currDir, "W.npy"),
                        self.FastParams[0].data.cpu())
                np.save(os.path.join(currDir, "W1.npy"),
                        self.FastParams[1].data.cpu())
                np.save(os.path.join(currDir, "W2.npy"),
                        self.FastParams[2].data.cpu())
            if self.numMatrices[0] == 4:
                np.save(os.path.join(currDir, "W.npy"),
                        self.FastParams[0].data.cpu())
                np.save(os.path.join(currDir, "W1.npy"),
                        self.FastParams[1].data.cpu())
                np.save(os.path.join(currDir, "W2.npy"),
                        self.FastParams[2].data.cpu())
                np.save(os.path.join(currDir, "W3.npy"),
                        self.FastParams[3].data.cpu())
            if self.numMatrices[0] == 5:
                np.save(os.path.join(currDir, "W.npy"),
                        self.FastParams[0].data.cpu())
                np.save(os.path.join(currDir, "W1.npy"),
                        self.FastParams[1].data.cpu())
                np.save(os.path.join(currDir, "W2.npy"),
                        self.FastParams[2].data.cpu())
                np.save(os.path.join(currDir, "W3.npy"),
                        self.FastParams[3].data.cpu())
                np.save(os.path.join(currDir, "W4.npy"),
                        self.FastParams[4].data.cpu())

        idx = self.numMatrices[0]
        if self.numMatrices[1] == 1:
            np.save(os.path.join(currDir, "U.npy"),
                    self.FastParams[idx + 0].data.cpu())
        elif self.FastObj.uRank is None:
            if self.numMatrices[1] == 2:
                np.save(os.path.join(currDir, "U1.npy"),
                        self.FastParams[idx + 0].data.cpu())
                np.save(os.path.join(currDir, "U2.npy"),
                        self.FastParams[idx + 1].data.cpu())
            if self.numMatrices[1] == 3:
                np.save(os.path.join(currDir, "U1.npy"),
                        self.FastParams[idx + 0].data.cpu())
                np.save(os.path.join(currDir, "U2.npy"),
                        self.FastParams[idx + 1].data.cpu())
                np.save(os.path.join(currDir, "U3.npy"),
                        self.FastParams[idx + 2].data.cpu())
            if self.numMatrices[1] == 4:
                np.save(os.path.join(currDir, "U1.npy"),
                        self.FastParams[idx + 0].data.cpu())
                np.save(os.path.join(currDir, "U2.npy"),
                        self.FastParams[idx + 1].data.cpu())
                np.save(os.path.join(currDir, "U3.npy"),
                        self.FastParams[idx + 2].data.cpu())
                np.save(os.path.join(currDir, "U4.npy"),
                        self.FastParams[idx + 3].data.cpu())
        elif self.FastObj.uRank is not None:
            if self.numMatrices[1] == 2:
                np.save(os.path.join(currDir, "U1.npy"),
                        self.FastParams[idx + 0].data.cpu())
                np.save(os.path.join(currDir, "U2.npy"),
                        self.FastParams[idx + 1].data.cpu())
            if self.numMatrices[1] == 3:
                np.save(os.path.join(currDir, "U.npy"),
                        self.FastParams[idx + 0].data.cpu())
                np.save(os.path.join(currDir, "U1.npy"),
                        self.FastParams[idx + 1].data.cpu())
                np.save(os.path.join(currDir, "U2.npy"),
                        self.FastParams[idx + 2].data.cpu())
            if self.numMatrices[1] == 4:
                np.save(os.path.join(currDir, "U.npy"),
                        self.FastParams[idx + 0].data.cpu())
                np.save(os.path.join(currDir, "U1.npy"),
                        self.FastParams[idx + 1].data.cpu())
                np.save(os.path.join(currDir, "U2.npy"),
                        self.FastParams[idx + 2].data.cpu())
                np.save(os.path.join(currDir, "U3.npy"),
                        self.FastParams[idx + 3].data.cpu())
            if self.numMatrices[1] == 5:
                np.save(os.path.join(currDir, "U.npy"),
                        self.FastParams[idx + 0].data.cpu())
                np.save(os.path.join(currDir, "U1.npy"),
                        self.FastParams[idx + 1].data.cpu())
                np.save(os.path.join(currDir, "U2.npy"),
                        self.FastParams[idx + 2].data.cpu())
                np.save(os.path.join(currDir, "U3.npy"),
                        self.FastParams[idx + 3].data.cpu())
                np.save(os.path.join(currDir, "U4.npy"),
                        self.FastParams[idx + 4].data.cpu())

        if self.FastObj.cellType == "FastGRNN":
            np.save(os.path.join(currDir, "Bg.npy"),
                    self.FastParams[self.totalMatrices].data.cpu())
            np.save(os.path.join(currDir, "Bh.npy"),
                    self.FastParams[self.totalMatrices + 1].data.cpu())
            np.save(os.path.join(currDir, "zeta.npy"),
                    self.FastParams[self.totalMatrices + 2].data.cpu())
            np.save(os.path.join(currDir, "nu.npy"),
                    self.FastParams[self.totalMatrices + 3].data.cpu())
        elif self.FastObj.cellType == "FastRNN":
            np.save(os.path.join(currDir, "B.npy"),
                    self.FastParams[self.totalMatrices].data.cpu())
            np.save(os.path.join(currDir, "alpha.npy"), self.FastParams[
                    self.totalMatrices + 1].data.cpu())
            np.save(os.path.join(currDir, "beta.npy"),
                    self.FastParams[self.totalMatrices + 2].data.cpu())
        elif self.FastObj.cellType == "UGRNNLR":
            np.save(os.path.join(currDir, "Bg.npy"),
                    self.FastParams[self.totalMatrices].data.cpu())
            np.save(os.path.join(currDir, "Bh.npy"),
                    self.FastParams[self.totalMatrices + 1].data.cpu())
        elif self.FastObj.cellType == "GRULR":
            np.save(os.path.join(currDir, "Br.npy"),
                    self.FastParams[self.totalMatrices].data.cpu())
            np.save(os.path.join(currDir, "Bg.npy"),
                    self.FastParams[self.totalMatrices + 1].data.cpu())
            np.save(os.path.join(currDir, "Bh.npy"),
                    self.FastParams[self.totalMatrices + 2].data.cpu())
        elif self.FastObj.cellType == "LSTMLR":
            np.save(os.path.join(currDir, "Bf.npy"),
                    self.FastParams[self.totalMatrices].data.cpu())
            np.save(os.path.join(currDir, "Bi.npy"),
                    self.FastParams[self.totalMatrices + 1].data.cpu())
            np.save(os.path.join(currDir, "Bc.npy"),
                    self.FastParams[self.totalMatrices + 2].data.cpu())
            np.save(os.path.join(currDir, "Bo.npy"),
                    self.FastParams[self.totalMatrices + 3].data.cpu())

        np.save(os.path.join(currDir, "FC.npy"), self.FC.data.cpu())
        np.save(os.path.join(currDir, "FCbias.npy"), self.FCbias.data.cpu())

    def train(self, batchSize, totalEpochs, Xtrain, Xtest, Ytrain, Ytest,
              decayStep, decayRate, dataDir, currDir):
        '''
        The Dense - IHT - Sparse Retrain Routine for FastCell Training
        '''
        fileName = str(self.FastObj.cellType) + 'Results_pytorch.txt'
        resultFile = open(os.path.join(dataDir, fileName), 'a+')
        numIters = int(np.ceil(float(Xtrain.shape[0]) / float(batchSize)))
        totalBatches = numIters * totalEpochs

        counter = 0
        trimlevel = 15
        ihtDone = 0
        maxTestAcc = -10000
        if self.isDenseTraining is True:
            ihtDone = 1
            maxTestAcc = -10000
        header = '*' * 20
        self.timeSteps = int(Xtest.shape[1] / self.inputDims)
        Xtest = Xtest.reshape((-1, self.timeSteps, self.inputDims))
        Xtrain = Xtrain.reshape((-1, self.timeSteps, self.inputDims))

        for i in range(0, totalEpochs):
            print("\nEpoch Number: " + str(i), file=self.outFile)

            if i % decayStep == 0 and i != 0:
                self.learningRate = self.learningRate * decayRate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learningRate

            shuffled = list(range(Xtrain.shape[0]))
            np.random.shuffle(shuffled)
            trainAcc = 0.0
            trainLoss = 0.0
            numIters = int(numIters)
            for j in range(0, numIters):

                if counter == 0:
                    msg = " Dense Training Phase Started "
                    print("\n%s%s%s\n" %
                          (header, msg, header), file=self.outFile)

                k = shuffled[j * batchSize:(j + 1) * batchSize]
                batchX = Xtrain[k]
                batchY = Ytrain[k]

                self.optimizer.zero_grad()
                logits, _ = self.computeLogits(batchX.to(self.device))
                batchLoss = self.loss(logits, batchY.to(self.device))
                batchAcc = self.accuracy(logits, batchY.to(self.device))
                batchLoss.backward()
                self.optimizer.step()

                del batchX, batchY

                trainAcc += batchAcc.item()
                trainLoss += batchLoss.item()

                if (counter >= int(totalBatches / 3.0) and
                        (counter < int(2 * totalBatches / 3.0)) and
                        counter % trimlevel == 0 and
                        self.isDenseTraining is False):
                    self.runHardThrsd()
                    if ihtDone == 0:
                        msg = " IHT Phase Started "
                        print("\n%s%s%s\n" %
                              (header, msg, header), file=self.outFile)
                    ihtDone = 1
                elif ((ihtDone == 1 and counter >= int(totalBatches / 3.0) and
                       (counter < int(2 * totalBatches / 3.0)) and
                       counter % trimlevel != 0 and
                       self.isDenseTraining is False) or
                        (counter >= int(2 * totalBatches / 3.0) and
                            self.isDenseTraining is False)):
                    self.runSparseTraining()
                    if counter == int(2 * totalBatches / 3.0):
                        msg = " Sprase Retraining Phase Started "
                        print("\n%s%s%s\n" %
                              (header, msg, header), file=self.outFile)
                counter += 1

            trainLoss /= numIters
            trainAcc /= numIters
            print("Train Loss: " + str(trainLoss) +
                  " Train Accuracy: " + str(trainAcc),
                  file=self.outFile)

            logits, _ = self.computeLogits(Xtest.to(self.device))
            testLoss = self.loss(logits, Ytest.to(self.device)).item()
            testAcc = self.accuracy(logits, Ytest.to(self.device)).item()

            if ihtDone == 0:
                maxTestAcc = -10000
                maxTestAccEpoch = i
            else:
                if maxTestAcc <= testAcc:
                    maxTestAccEpoch = i
                    maxTestAcc = testAcc
                    self.saveParams(currDir)

            print("Test Loss: " + str(testLoss) +
                  " Test Accuracy: " + str(testAcc), file=self.outFile)
            self.outFile.flush()

        print("\nMaximum Test accuracy at compressed" +
              " model size(including early stopping): " +
              str(maxTestAcc) + " at Epoch: " +
              str(maxTestAccEpoch + 1) + "\nFinal Test" +
              " Accuracy: " + str(testAcc), file=self.outFile)
        print("\n\nNon-Zeros: " + str(self.getModelSize()[0]) +
              " Model Size: " + str(float(self.getModelSize()[1]) / 1024.0) +
              " KB hasSparse: " + str(self.getModelSize()[2]) + "\n",
              file=self.outFile)

        resultFile.write("MaxTestAcc: " + str(maxTestAcc) +
                         " at Epoch(totalEpochs): " +
                         str(maxTestAccEpoch + 1) +
                         "(" + str(totalEpochs) + ")" + " ModelSize: " +
                         str(float(self.getModelSize()[1]) / 1024.0) +
                         " KB hasSparse: " + str(self.getModelSize()[2]) +
                         " Param Directory: " +
                         str(os.path.abspath(currDir)) + "\n")

        print("The Model Directory: " + currDir + "\n")

        # output the tensorflow model
        # model_dir = os.path.join(currDir, "model")
        # os.makedirs(model_dir, exist_ok=True)

        resultFile.close()
        self.outFile.flush()
        if self.outFile is not sys.stdout:
            self.outFile.close()
