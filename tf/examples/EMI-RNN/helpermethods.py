# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
# 
# Helper methods to generate training, validation and test splits from the UCI HAR dataset. 
# Each split consists of a separate set of users.
# Reference : https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition

import numpy as np
import os

def generateIndicesForSplits(path='./HAR/UCI HAR Dataset/train/subject_train.txt'):
    f = open(path)
    subjects = []
    for line in f:
        subject = line.strip().split()
        subjects.append(int(subject[0]))
    subjects = np.array(subjects)

    # get unique subjects
    numSubjects = np.unique(subjects)

    # shuffle amongst train subjects so that difficult/easy subjects spread in both val and train
    np.random.shuffle(numSubjects)

    l = len(numSubjects)

    splitRatio = 0.1
    valSplit = int(l * splitRatio + 1)

    valSubjects = numSubjects[:valSplit]
    trainSubjects = numSubjects[valSplit:]

    trainSubjectIndices = []
    valSubjectIndices = []

    for i, subject in enumerate(subjects):
        if subject in trainSubjects:
            trainSubjectIndices.append(i)
        elif subject in valSubjects:
            valSubjectIndices.append(i)
        else:
            raise Exception("some bug in your code")

    # assert that train/val different
    for x in trainSubjectIndices:
        assert x not in valSubjectIndices

    trainSubjectIndices = np.array(trainSubjectIndices)
    valSubjectIndices = np.array(valSubjectIndices)

    # shuffle more, so that readings not grouped by a subject
    # therefore, no need to shuffle after slicing from read dataset, as we are shuffling here
    idx = np.arange(len(trainSubjectIndices))
    np.random.shuffle(idx)
    trainSubjectIndices = trainSubjectIndices[idx]

    idx = np.arange(len(valSubjectIndices))
    np.random.shuffle(idx)
    valSubjectIndices = valSubjectIndices[idx]

    assert len(trainSubjectIndices) + len(valSubjectIndices) == len(subjects)

    return trainSubjectIndices, valSubjectIndices

def readData(extractedDir):
    INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
    ]

    # Output classes to learn how to classify
    LABELS = [
        "WALKING", 
        "WALKING_UPSTAIRS", 
        "WALKING_DOWNSTAIRS", 
        "SITTING", 
        "STANDING", 
        "LAYING"
    ] 
    DATASET_PATH = extractedDir + "/UCI HAR Dataset/"
    TRAIN = "train/"
    TEST = "test/"
    # Load "X" (the neural network's training and testing inputs)

    def load_X(X_signals_paths):
        X_signals = []
        
        for signal_type_path in X_signals_paths:
            file = open(signal_type_path, 'r')
            # Read dataset from disk, dealing with text files' syntax
            X_signals.append(
                [np.array(serie, dtype=np.float32) for serie in [
                    row.replace('  ', ' ').strip().split(' ') for row in file
                ]]
            )
            file.close()
        
        return np.transpose(np.array(X_signals), (1, 2, 0))

    X_train_signals_paths = [
        DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
    ]
    X_test_signals_paths = [
        DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
    ]

    x_train_val_combined = load_X(X_train_signals_paths)
    x_test = load_X(X_test_signals_paths)


    # Load "y" (the neural network's training and testing outputs)

    def load_y(y_path):
        file = open(y_path, 'r')
        # Read dataset from disk, dealing with text file's syntax
        y_ = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]], 
            dtype=np.int32
        )
        file.close()
        
        # Substract 1 to each output class for friendly 0-based indexing 
        return y_ - 1

    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + TEST + "y_test.txt"

    y_train_val_combined = load_y(y_train_path)
    y_test = load_y(y_test_path)

    return x_train_val_combined, y_train_val_combined, x_test, y_test

def one_hot(y, numOutput):
    y = np.reshape(y, [-1])
    ret = np.zeros([y.shape[0], numOutput])
    for i, label in enumerate(y):
        ret[i, label] = 1
    return ret


def generateData(extractedDir):
    x_train_val_combined, y_train_val_combined, x_test, y_test = readData(extractedDir)
    timesteps = x_train_val_combined.shape[-2]
    feats = x_train_val_combined.shape[-1]

    trainSubjectIndices, valSubjectIndices = generateIndicesForSplits()

    x_train = x_train_val_combined[trainSubjectIndices]
    y_train = y_train_val_combined[trainSubjectIndices]
    x_val = x_train_val_combined[valSubjectIndices]
    y_val = y_train_val_combined[valSubjectIndices]

    # normalization
    x_train = np.reshape(x_train, [-1, feats])
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    # normalize train
    x_train = x_train - mean
    x_train = x_train / std
    x_train = np.reshape(x_train, [-1, timesteps, feats])

    # normalize val
    x_val = np.reshape(x_val, [-1, feats])
    x_val = x_val - mean
    x_val = x_val / std
    x_val = np.reshape(x_val, [-1, timesteps, feats])

    # normalize test
    x_test = np.reshape(x_test, [-1, feats])
    x_test = x_test - mean
    x_test = x_test / std
    x_test = np.reshape(x_test, [-1, timesteps, feats])

    # shuffle test, as this was remaining
    idx = np.arange(len(x_test))
    np.random.shuffle(idx)
    x_test = x_test[idx]
    y_test = y_test[idx]

    # one-hot encoding of labels
    numOutput = 6
    y_train = one_hot(y_train, numOutput)
    y_val = one_hot(y_val, numOutput)
    y_test = one_hot(y_test, numOutput)
    extractedDir += '/'
    try:
        os.mkdir(extractedDir + 'RAW')
    except OSError:
        exit("Could not create %s" % extractedDir + 'RAW')
    np.save(extractedDir + "RAW/x_train", x_train)
    np.save(extractedDir + "RAW/y_train", y_train)
    np.save(extractedDir + "RAW/x_test", x_test)
    np.save(extractedDir + "RAW/y_test", y_test)
    np.save(extractedDir + "RAW/x_val", x_val)
    np.save(extractedDir + "RAW/y_val", y_val)
    return extractedDir

def loadData(dirname):
    x_train = np.load(dirname + '/' + 'x_train.npy')
    y_train = np.load(dirname + '/' + 'y_train.npy')
    x_test = np.load(dirname + '/' + 'x_test.npy')
    y_test = np.load(dirname + '/' + 'y_test.npy')
    x_val = np.load(dirname + '/' + 'x_val.npy')
    y_val = np.load(dirname + '/' + 'y_val.npy')
    return x_train, y_train, x_test, y_test, x_val, y_val


def bagData(X, Y, subinstanceLen, subinstanceStride):
    '''
    Takes x and y of shape
    [-1, 128, 9] and [-1, 6] respectively and converts it into bags of instances.
    returns [-1, numInstance, ]
    '''
    numClass = 6
    numSteps = 128
    numFeats = 9
    assert X.ndim == 3
    assert X.shape[1] == numSteps
    assert X.shape[2] == numFeats
    assert subinstanceLen <= numSteps
    assert subinstanceLen > 0
    assert subinstanceStride <= numSteps
    assert subinstanceStride >= 0
    assert len(X) == len(Y)
    assert Y.ndim == 2
    assert Y.shape[1] == numClass
    x_bagged = []
    y_bagged = []
    for i, point in enumerate(X[:, :, :]):
        instanceList = []
        start = 0
        end = subinstanceLen
        while True:
            x = point[start:end, :]
            if len(x) < subinstanceLen:
                x_ = np.zeros([subinstanceLen, x.shape[1]])
                x_[:len(x), :] = x[:, :]
                x = x_
            instanceList.append(x)
            if end >= numSteps:
                break
            start += subinstanceStride
            end += subinstanceStride
        bag = np.array(instanceList)
        numSubinstance = bag.shape[0]
        label = Y[i]
        label = np.argmax(label)
        labelBag = np.zeros([numSubinstance, numClass])
        labelBag[:, label] = 1
        x_bagged.append(bag)
        label = np.array(labelBag)
        y_bagged.append(label)
    return np.array(x_bagged), np.array(y_bagged)


def makeEMIData(subinstanceLen, subinstanceStride, sourceDir, outDir):
    x_train, y_train, x_test, y_test, x_val, y_val = loadData(sourceDir)
    x, y = bagData(x_train, y_train, subinstanceLen, subinstanceStride)
    np.save(outDir + '/x_train.npy', x)
    np.save(outDir + '/y_train.npy', y)
    print('Num train %d' % len(x))
    x, y = bagData(x_test, y_test, subinstanceLen, subinstanceStride)
    np.save(outDir + '/x_test.npy', x)
    np.save(outDir + '/y_test.npy', y)
    print('Num test %d' % len(x))
    x, y = bagData(x_val, y_val, subinstanceLen, subinstanceStride)
    np.save(outDir + '/x_val.npy', x)
    np.save(outDir + '/y_val.npy', y)
    print('Num val %d' % len(x))
