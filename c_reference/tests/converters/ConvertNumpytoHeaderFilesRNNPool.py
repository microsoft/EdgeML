# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import argparse
import os
import sys

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def loadTestData(dataDir):
    test_input = np.load(dataDir + "/test_input.npy")
    test_output = np.load(dataDir + "/test_output.npy")
    return test_input, test_output


def loadModel(modelDir):
    model = {
        "W1": np.transpose(np.load(modelDir + "/W1.npy")),
        "W2": np.transpose(np.load(modelDir + "/W2.npy")),
        "U1": np.transpose(np.load(modelDir + "/U1.npy")),
        "U2": np.transpose(np.load(modelDir + "/U2.npy")),
        "Bg1": np.load(modelDir + "/Bg1.npy"),
        "Bh1": np.load(modelDir + "/Bh1.npy"),
        "Bg2": np.load(modelDir + "/Bg2.npy"),
        "Bh2": np.load(modelDir + "/Bh2.npy"),
        "zeta1": sigmoid(np.load(modelDir + "/zeta1.npy")),
        "nu1": sigmoid(np.load(modelDir + "/nu1.npy")),
        "zeta2": sigmoid(np.load(modelDir + "/zeta2.npy")),
        "nu2": sigmoid(np.load(modelDir + "/nu2.npy")),
    }
    
    return model



def getArgs():
    '''
    Function to parse arguments for FastCells
    '''
    parser = argparse.ArgumentParser(
        description='HyperParams for Fast(G)RNN inference')
    
    parser.add_argument('-mdir', '--model-dir', required=True,
                        help='Model directory containing' +
                        'FastRNN or FastGRNN model')

    parser.add_argument('-rnn1oF', '--rnn1-out-file', default="None",
                        help='Give a output file for the model to dump' +
                        'default: stdout')

    parser.add_argument('-rnn2oF', '--rnn2-out-file', default="None",
                        help='Give a output file for the model to dump' +
                        'default: stdout')

    parser.add_argument('-doF', '--data-out-file', default="None",
                        help='Give a output file for the data to dump' +
                        'default: stdout')
    
    return parser.parse_args()


def saveReadableModel(modelDir, model):
    if os.path.isdir(modelDir + '/ReadableModel') is False:
        try:
            os.mkdir(modelDir + '/ReadableModel')
        except OSError:
            print("Creation of the directory %s failed" %
                  modelDir + '/ReadableModel')
    currDir = modelDir + '/ReadableModel'

    np.savetxt(currDir + "/W1.txt",
               np.reshape(model["W1"], [1, -1]), delimiter=',')
    np.savetxt(currDir + "/W2.txt",
               np.reshape(model["W2"], [1, -1]), delimiter=',')
    np.savetxt(currDir + "/U1.txt",
               np.reshape(model["U1"], [1, -1]), delimiter=',')
    np.savetxt(currDir + "/U2.txt",
               np.reshape(model["U2"], [1, -1]), delimiter=',')
    np.savetxt(currDir + "/Bg1.txt",
               np.reshape(model["Bg1"], [1, -1]), delimiter=',')
    np.savetxt(currDir + "/Bh1.txt",
               np.reshape(model["Bh1"], [1, -1]), delimiter=',')
    np.savetxt(currDir + "/Bg2.txt",
               np.reshape(model["Bg2"], [1, -1]), delimiter=',')
    np.savetxt(currDir + "/Bh2.txt",
               np.reshape(model["Bh2"], [1, -1]), delimiter=',')
    np.savetxt(currDir + "/zeta1.txt",
               np.reshape(model["zeta1"], [1, -1]), delimiter=',')
    np.savetxt(currDir + "/nu1.txt",
               np.reshape(model["nu1"], [1, -1]), delimiter=',')
    np.savetxt(currDir + "/zeta2.txt",
               np.reshape(model["zeta2"], [1, -1]), delimiter=',')
    np.savetxt(currDir + "/nu2.txt",
               np.reshape(model["nu2"], [1, -1]), delimiter=',')

    return currDir


def convertMatrixToVecString(mat):
    mat = str(np.reshape(mat, [1, -1])[0, :].tolist())
    mat = '{' + mat[1:-1] + '}'
    return mat


def main():
    args = getArgs()
    rnn1OutFile = args.rnn1_out_file
    rnn2OutFile = args.rnn2_out_file
    
    rnn1OutFile = open(rnn1OutFile, 'w')
    rnn2OutFile = open(rnn2OutFile, 'w')

    model = loadModel(args.model_dir)


    inputDims = model["W1"].shape[1]
    hiddenDims1 = model["W2"].shape[1]
    hiddenDims2 = model["U2"].shape[1]
    

    currDir = saveReadableModel(args.model_dir, model)
    
    print ("#define HIDDEN_DIMS1 8\n", file=rnn1OutFile)

    print("static float W1[INPUT_DIMS * HIDDEN_DIMS1] = " + convertMatrixToVecString(model['W1']) + ";", file=rnn1OutFile)
   
    print("static float U1[HIDDEN_DIMS1 * HIDDEN_DIMS1] = " + convertMatrixToVecString(model['U1']) + ";", file=rnn1OutFile)
    
    print("static float Bg1[HIDDEN_DIMS1] = " + convertMatrixToVecString(model['Bg1']) + ";", file=rnn1OutFile)
    print("static float Bh1[HIDDEN_DIMS1] = " +
          convertMatrixToVecString(model['Bh1']) + ";\n", file=rnn1OutFile)

    print("static float sigmoid_zeta1 = " + str(model['zeta1'][0][0]) + ";", file=rnn1OutFile)
    print("static float sigmoid_nu1 = " + str(model['nu1'][0][0]) + ";\n", file=rnn1OutFile)


    print("static float W2[HIDDEN_DIMS1 * HIDDEN_DIMS2] = " + convertMatrixToVecString(model['W2']) + ";", file=rnn2OutFile)
   
    print("static float U2[HIDDEN_DIMS2 * HIDDEN_DIMS2] = " + convertMatrixToVecString(model['U2']) + ";\n", file=rnn2OutFile)
    
    print("static float Bg2[HIDDEN_DIMS2] = " + convertMatrixToVecString(model['Bg2']) + ";", file=rnn2OutFile)
    print("static float Bh2[HIDDEN_DIMS2] = " +
          convertMatrixToVecString(model['Bh2']) + ";\n", file=rnn2OutFile)

    print("static float sigmoid_zeta2 = " + str(model['zeta2'][0][0]) + ";", file=rnn2OutFile)
    print("static float sigmoid_nu2 = " + str(model['nu2'][0][0]) + ";\n", file=rnn2OutFile)    


    

    rnn1OutFile.flush()
    rnn1OutFile.close()
    rnn2OutFile.flush()
    rnn2OutFile.close()
 
main()