'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT license.

genDataHeader.py generates data.h
'''

import numpy as np
import pandas as pd
from dataFileTemplate import populateDataFileTemplate
import sys


def loadTLCMatrices(dfolder):
    '''
    Loads Matrices B, W and Z from TLC format
    '''
    # W is stored as d_cap x d
    df = pd.read_csv(dfolder + 'W', sep='\t', header=None)
    W = np.matrix(df)
    d_cap = W.shape[0]

    # B is stored as d_cap x m
    df = pd.read_csv(dfolder + 'B', sep='\t', header=None)
    B = np.matrix(df)
    m = B.shape[1]
    assert(d_cap == B.shape[0])

    # Z is stored as L x m
    df = pd.read_csv(dfolder + 'Z', sep='\t', header=None)
    Z = np.matrix(df)
    assert(Z.shape[1] == m)

    return W, B, Z


def createDataFile(W, B, Z, gamma, outfile='data.h'):
    '''
    Exports the provided matrices loaded, into a temp file
    so that they can be directly copied onto the MKR1000
    board's data.h file.
    Required input matrix dimensions:
    W: d_cap x d
    B: d_cap x m
    Z: L x m

    Created output dimensions:
    W: d_cap x d
    B: m x d_cap (i.e column major)
    Z: m x L (i.e. column major)
    '''
    valueDict = {}
    valueDict['gamma'] = gamma
    d_cap = W.shape[0]
    d = W.shape[1]
    valueDict['featDim'] = '%d' % (d)
    valueDict['ldDim'] = '%d' % (d_cap)
    WStr = '\n\t\t'
    for i in range(0, d_cap):
        for j in range(0, d):
            WStr += str(W[i, j]) + ','
        WStr += '\n\t\t'
    valueDict['ldProjectionMatrix'] = WStr[:-1]

    assert(B.shape[0] == d_cap)
    m = B.shape[1]
    valueDict['numPrototypes'] = '%d' % (m)
    BStr = '\n\t\t'
    for i in range(0, m):
        for j in range(0, d_cap):
            # Column major (j, i)
            BStr += '%f' % (B[j, i]) + ','
        BStr += '\n\t\t'
    valueDict['prototypeMatrix'] = BStr[:-1]
    assert(Z.shape[1] == m)
    L = Z.shape[0]
    valueDict['numLabels'] = L
    ZStr = '\n\t\t'
    for i in range(0, m):
        for j in range(0, L):
            # Columns major
            ZStr += '%f' % (Z[j, i]) + ','
        ZStr += '\n\t\t'
    valueDict['prototypeLabelMatrix'] = ZStr[:-2]
    template = populateDataFileTemplate(valueDict)
    fin = open(outfile, 'w')
    fin.write(template)
    fin.close()



def automaticExport():
    # Copy W, B, Z from Debug
    gamma = sys.argv[1]
    dfolder = './'
    W, B, Z = loadTLCMatrices(dfolder)
    # Create a new file with gamma in it
    createDataFile(W, B, Z, gamma, 'data.h')


if __name__ == '__main__':
    automaticExport()
