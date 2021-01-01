# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import helpermethods
import torch
import numpy as np
import sys
from edgeml_pytorch.graph.rnn import *
from edgeml_pytorch.trainer.fastTrainer import FastTrainer


def main():
    # change cuda:0 to cuda:gpuid for specific allocation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Fixing seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Hyper Param pre-processing
    args = helpermethods.getArgs()

    dataDir = args.data_dir
    cell = args.cell
    inputDims = args.input_dim
    batch_first = args.batch_first
    hiddenDims = args.hidden_dim

    totalEpochs = args.epochs
    learningRate = args.learning_rate
    outFile = args.output_file
    batchSize = args.batch_size
    decayStep = args.decay_step
    decayRate = args.decay_rate

    wRank = args.wRank
    uRank = args.uRank

    sW = args.sW
    sU = args.sU

    update_non_linearity = args.update_nl
    gate_non_linearity = args.gate_nl

    (dataDimension, numClasses, Xtrain, Ytrain, Xtest, Ytest,
     mean, std) = helpermethods.preProcessData(dataDir)

    assert dataDimension % inputDims == 0, "Infeasible per step input, " + \
        "Timesteps have to be integer"

    currDir = helpermethods.createTimeStampDir(dataDir, cell)

    helpermethods.dumpCommand(sys.argv, currDir)
    helpermethods.saveMeanStd(mean, std, currDir)

    if cell == "FastGRNN":
        FastCell = FastGRNNCell(inputDims, hiddenDims,
                                gate_nonlinearity=gate_non_linearity,
                                update_nonlinearity=update_non_linearity,
                                wRank=wRank, uRank=uRank)
    elif cell == "FastGRNNCUDA":
        FastCell = FastGRNNCUDACell(inputDims, hiddenDims,
                                    gate_nonlinearity=gate_non_linearity,
                                    update_nonlinearity=update_non_linearity,
                                    wRank=wRank, uRank=uRank)
    elif cell == "FastRNN":
        FastCell = FastRNNCell(inputDims, hiddenDims,
                               update_nonlinearity=update_non_linearity,
                               wRank=wRank, uRank=uRank)
    elif cell == "UGRNN":
        FastCell = UGRNNLRCell(inputDims, hiddenDims,
                               update_nonlinearity=update_non_linearity,
                               wRank=wRank, uRank=uRank)
    elif cell == "GRU":
        FastCell = GRULRCell(inputDims, hiddenDims,
                             update_nonlinearity=update_non_linearity,
                             wRank=wRank, uRank=uRank)
    elif cell == "LSTM":
        FastCell = LSTMLRCell(inputDims, hiddenDims,
                              update_nonlinearity=update_non_linearity,
                              wRank=wRank, uRank=uRank)
    else:
        sys.exit('Exiting: No Such Cell as ' + cell)

    FastCellTrainer = FastTrainer(FastCell, numClasses, sW=sW, sU=sU,
                                  learningRate=learningRate, outFile=outFile,
                                  device=device, batch_first=batch_first)

    FastCellTrainer.train(batchSize, totalEpochs,
                          torch.from_numpy(Xtrain.astype(np.float32)),
                          torch.from_numpy(Xtest.astype(np.float32)),
                          torch.from_numpy(Ytrain.astype(np.float32)),
                          torch.from_numpy(Ytest.astype(np.float32)),
                          decayStep, decayRate, dataDir, currDir)


if __name__ == '__main__':

    main()
