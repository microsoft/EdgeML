#! /usr/bin/env python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# This file is a fixer that fixes the inaccuracy created 
# during multiple training instances of the same model  .

import numpy as np
import os, sys
import argparse
import re


def parse():
    parser = argparse.ArgumentParser(description='Modify SeeDot input file')
    parser.add_argument('--seedot_file', type=str,metavar='',
                        help='path .sd file (including file name)')
    parser.add_argument('--model_dir', type=str,metavar='',
                        help='path to model files directory')
    parser.add_argument('--dataset_dir', type=str,metavar='',
                        help='path to data files directory (the directory with train.npy and test.npy)')
    parser.add_argument("-n", "--numOutputs", type=int, metavar='',
                        help='The number of outputs that the model under consideration produces', default=1)
    parser.add_argument('--normalise_data', action='store_true',
                    help='Normalise the input train and test files.')

    return parser.parse_args()


def readModelWeights(model_dir, dataset_dir, numOutputs, normalise_data):
    filelist = os.listdir(os.path.join(os.getcwd(), model_dir))
    cur_dir = os.getcwd()
    os.chdir(model_dir)
    filelist = [x for x in filelist if x[-4:] == '.npy']
    weight_min_max_dict = {}
    for filename in filelist:
        f = np.load(filename).flatten()
        if (len(f) == 1):
            m1 = 1.0/(1.0 + np.exp(-1*f[0]))
            weight_min_max_dict[filename[:-4]] = [m1]
        else:
            m1 = np.min(f)
            m2 = np.max(f)
            weight_min_max_dict[filename[:-4]] = [m1, m2]
    
    os.chdir(cur_dir)
    os.chdir(dataset_dir)

    train = np.load("train.npy")
    Xtrain = train[:, numOutputs:]
    
    test = np.load("test.npy")
    Xtest = test[:, numOutputs:]
    
    if normalise_data:
        mean = np.mean(Xtrain, 0)
        std = np.std(Xtrain, 0)
        std[std[:] < 0.000001] = 1
        
        Xtrain = (Xtrain - mean) / std
        Xtest = (Xtest - mean) / std

    m1 = np.min(Xtrain)
    m2 = np.max(Xtrain)

    m1 = min(m1, np.min(Xtest))
    m2 = min(m2, np.max(Xtest))
    weight_min_max_dict['X'] = [m1, m2]
    
    if normalise_data:
        train[:, numOutputs:] = Xtrain
        test[:, numOutputs:] = Xtest
        
        np.save("train.npy", train)
        np.save("test.npy", test)

    os.chdir(cur_dir)

    return weight_min_max_dict

def getVar(line, weights_dict):
    replace = False
    new_line = None
    if line.count('=') == 1:
        left, right = line.split('=')
        left = left.lstrip().rstrip()
        var = left.split(' ')[-1].split('\t')[-1]
        right = right.lstrip().rstrip()
        if var in weights_dict.keys():
            replace = True
            weights = weights_dict[var]
            if len(weights) == 1:
                new_line = "let " + var + " = " + "%.20f"%(weights[0]) + " in"
            else:
                shape = line[line.find('('):line.find(')')+1]
                new_line = "let " + var + " = " + shape + " in ["  +\
                         "%.20f"%(weights[0]) + ", " + "%.20f"%(weights[1]) + "] in"
    return  replace, new_line


def writeToInputDotSD(file, dir):
    os.chdir(dir)
    f = open("input.sd", "w")

    for i in range(len(file)):
        f.write(file[i] + "\n")
    f.close()


def run(args):
    input_file = open(args.seedot_file).read().split("\n")
    
    model_weights_dict = readModelWeights(args.model_dir, args.dataset_dir, args.numOutputs, args.normalise_data)
    
    for i in range(len(input_file)):
        line = input_file[i]
        replace, new_line = getVar(line, model_weights_dict)
        if replace:
            input_file[i] = new_line
            # print(line + " | " + new_line)
    writeToInputDotSD(input_file, args.model_dir)



if __name__ == '__main__':
    args = parse()
    run(args)
