# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import numpy as np
from edgeml_pytorch.graph.rnn import *

class RNNPool(nn.Module):
    def __init__(self, nRows, nCols, nHiddenDims,
                     nHiddenDimsBiDir, inputDims):
        super(RNNPool, self).__init__()
        self.nRows = nRows
        self.nCols = nCols
        self.inputDims = inputDims
        self.nHiddenDims = nHiddenDims
        self.nHiddenDimsBiDir = nHiddenDimsBiDir

        self._build()

    def _build(self):

        self.cell_rnn = FastGRNN(self.inputDims, self.nHiddenDims, gate_nonlinearity="sigmoid",
                                update_nonlinearity="tanh", zetaInit=100.0, nuInit=-100.0,
                                batch_first=False, bidirectional=False)

        self.cell_bidirrnn = FastGRNN(self.nHiddenDims, self.nHiddenDimsBiDir, gate_nonlinearity="sigmoid",
                                update_nonlinearity="tanh", zetaInit=100.0, nuInit=-100.0,
                                batch_first=False, bidirectional=True, is_shared_bidirectional=True)


    def static_single(self,inputs, hidden, batch_size):

        outputs = self.cell_rnn(inputs, hidden[0], hidden[1])
        return torch.split(outputs[-1], split_size_or_sections=batch_size, dim=0)

    def forward(self,inputs,batch_size):
        ## across rows

        row_timestack = torch.cat(torch.unbind(inputs, dim=3),dim=0) 

        stateList = self.static_single(torch.stack(torch.unbind(row_timestack,dim=2)),
                        (torch.zeros(1, batch_size * self.nRows, self.nHiddenDims).to(torch.device("cuda")),
                        torch.zeros(1, batch_size * self.nRows, self.nHiddenDims).to(torch.device("cuda"))),batch_size)       

        outputs_cols = self.cell_bidirrnn(torch.stack(stateList),
                        torch.zeros(2, batch_size, self.nHiddenDimsBiDir).to(torch.device("cuda")),
                        torch.zeros(2, batch_size, self.nHiddenDimsBiDir).to(torch.device("cuda")))


        ## across columns
        col_timestack = torch.cat(torch.unbind(inputs, dim=2),dim=0)

        stateList = self.static_single(torch.stack(torch.unbind(col_timestack,dim=2)),
                        (torch.zeros(1, batch_size * self.nRows, self.nHiddenDims).to(torch.device("cuda")),
                        torch.zeros(1, batch_size * self.nRows, self.nHiddenDims).to(torch.device("cuda"))),batch_size)

        outputs_rows = self.cell_bidirrnn(torch.stack(stateList),
                        torch.zeros(2, batch_size, self.nHiddenDimsBiDir).to(torch.device("cuda")),
                        torch.zeros(2, batch_size, self.nHiddenDimsBiDir).to(torch.device("cuda")))



        output = torch.cat([outputs_rows[-1],outputs_cols[-1]],1)

        return output
