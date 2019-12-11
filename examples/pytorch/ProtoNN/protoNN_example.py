# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import print_function
import sys
import os
import numpy as np
from edgeml_pytorch.trainer.protoNNTrainer import ProtoNNTrainer
from edgeml_pytorch.graph.protoNN import ProtoNN
import edgeml_pytorch.utils as utils
import helpermethods as helper
import torch

def main():
    # change cuda:0 to cuda:gpu_id for using particular gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = helper.getProtoNNArgs()
    # Get hyper parameters
    DATA_DIR = config.data_dir
    PROJECTION_DIM = config.projection_dim
    NUM_PROTOTYPES = config.num_prototypes
    REG_W = config.rW
    REG_B = config.rB
    REG_Z = config.rZ
    SPAR_W = config.sW
    SPAR_B = config.sB
    SPAR_Z = config.sZ
    LEARNING_RATE = config.learning_rate
    NUM_EPOCHS = config.epochs
    BATCH_SIZE = config.batch_size
    PRINT_STEP = config.print_step
    VAL_STEP = config.val_step
    OUT_DIR = config.output_dir

    # Load data
    train = np.load(DATA_DIR + '/train.npy')
    test = np.load(DATA_DIR + '/test.npy')
    x_train, y_train = train[:, 1:], train[:, 0]
    x_test, y_test = test[:, 1:], test[:, 0]
    # Convert y to one-hot
    minval = min(min(y_train), min(y_test))
    numClasses = max(y_train) - min(y_train) + 1
    numClasses = max(numClasses, max(y_test) - min(y_test) + 1)
    numClasses = int(numClasses)
    y_train = helper.to_onehot(y_train, numClasses, minlabel=minval)
    y_test = helper.to_onehot(y_test, numClasses, minlabel=minval)
    dataDimension = x_train.shape[1]

    W, B, gamma = helper.getGamma(config.gamma, PROJECTION_DIM, dataDimension,
                                  NUM_PROTOTYPES, x_train)

    # Setup input and train protoNN
    protoNN = ProtoNN(dataDimension, PROJECTION_DIM,
                      NUM_PROTOTYPES, numClasses,
                      gamma, W=W, B=B).to(device)

    trainer = ProtoNNTrainer(protoNN, REG_W, REG_B, REG_Z,
                             SPAR_W, SPAR_B, SPAR_Z,
                             LEARNING_RATE, lossType='xentropy', device=device)
    # Train the protoNN object
    trainer.train(BATCH_SIZE, NUM_EPOCHS, x_train, x_test,
                  y_train, y_test, printStep=PRINT_STEP, valStep=VAL_STEP)

    # Print some summary metrics
    x_, y_= (torch.Tensor(x_test)).to(device), (torch.Tensor(y_test)).to(device)

    logits = protoNN.forward(x_)
    _, predictions = torch.max(logits, dim=1)
    _, target = torch.max(y_, dim=1)
    acc, count = trainer.accuracy(predictions, target)
    #Model needs to be on cpu for numpy operations below
    protoNN = protoNN.cpu()
    W, B, Z, gamma  = protoNN.getModelMatrices()
    matrixList = [W, B, Z]
    matrixList = [x.detach().numpy() for x in matrixList]
    sparcityList = [SPAR_W, SPAR_B, SPAR_Z]
    nnz, size, sparse = helper.getModelSize(matrixList, sparcityList)
    print("Final test accuracy", acc)
    print("Model size constraint (Bytes): ", size)
    print("Number of non-zeros: ", nnz)
    nnz, size, sparse = helper.getModelSize(matrixList, sparcityList,
                                            expected=False)
    print("Actual model size: ", size)
    print("Actual non-zeros: ", nnz)
    print("Saving model matrices to: ", OUT_DIR)
    np.save(OUT_DIR + '/W.npy', matrixList[0])
    np.save(OUT_DIR + '/B.npy', matrixList[1])
    np.save(OUT_DIR + '/Z.npy', matrixList[2])
    np.save(OUT_DIR + '/gamma.npy', gamma)


if __name__ == '__main__':
    main()
