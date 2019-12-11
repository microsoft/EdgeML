# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import helpermethods
import numpy as np
import sys
from edgeml_pytorch.trainer.bonsaiTrainer import BonsaiTrainer
from edgeml_pytorch.graph.bonsai import Bonsai
import torch


def main():
    # change cuda:0 to cuda:gpuid for specific allocation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Fixing seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Hyper Param pre-processing
    args = helpermethods.getArgs()

    sigma = args.sigma
    depth = args.depth

    projectionDimension = args.proj_dim
    regZ = args.rZ
    regT = args.rT
    regW = args.rW
    regV = args.rV

    totalEpochs = args.epochs

    learningRate = args.learning_rate

    dataDir = args.data_dir

    outFile = args.output_file

    (dataDimension, numClasses, Xtrain, Ytrain, Xtest, Ytest,
     mean, std) = helpermethods.preProcessData(dataDir)

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

    if args.batch_size is None:
        batchSize = np.maximum(100, int(np.ceil(np.sqrt(Ytrain.shape[0]))))
    else:
        batchSize = args.batch_size

    useMCHLoss = True

    if numClasses == 2:
        numClasses = 1

    currDir = helpermethods.createTimeStampDir(dataDir)

    helpermethods.dumpCommand(sys.argv, currDir)
    helpermethods.saveMeanStd(mean, std, currDir)

    # numClasses = 1 for binary case
    bonsaiObj = Bonsai(numClasses, dataDimension,
                       projectionDimension, depth, sigma).to(device)

    bonsaiTrainer = BonsaiTrainer(bonsaiObj,
                                  regW, regT, regV, regZ,
                                  sparW, sparT, sparV, sparZ,
                                  learningRate, useMCHLoss, outFile, device)

    bonsaiTrainer.train(batchSize, totalEpochs,
                        torch.from_numpy(Xtrain.astype(np.float32)),
                        torch.from_numpy(Xtest.astype(np.float32)),
                        torch.from_numpy(Ytrain.astype(np.float32)),
                        torch.from_numpy(Ytest.astype(np.float32)),
                        dataDir, currDir)
    sys.stdout.close()


if __name__ == '__main__':
    main()
