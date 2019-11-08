'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT license.

featureExtraction.py extracts features from labelled data files.

ASSUMPTIONS:

- Negatives in general have to be specified by 1 and should
be at least 400 wide. In fact all non zero classes have to be
at least 400 wide. Any more, and I can't gaurentee anything.
This is for things like pertubations and all to work properly

- Do not change multi-threads to multi-process. It will not 
work with the current formulation.
'''
import pandas as pd
import time
import numpy as np
import re
import threading
import os


labelledFileList = [
    # Should not contain files with only noise.
    # They are delt with separately - allNoiseFileList.
    
    'foo.csv',
]

allNoiseFileList = [
    # Files containing only noise - walking, climbing stairs, etc.
    # Note - This requires raw files, *NOT* labelled files

    'bar.csv',

    
]

# Not tested for more than 1 thread
NUM_THREADS = 1

def collapseLabel(dataFrame, hyperParams):
    '''
    if [start, end) is == 1
    then end - 1 is set to 1 and everything else
    is zero
    '''
    pertubations = hyperParams['pertubations']
    windowWidth = hyperParams['windowWidth']
    assert('mlabel' in dataFrame.columns)
    dataFrame['label'] = 0
    labelIndex = dataFrame.columns.get_loc('label')
    startIndex = 0
    endIndex = windowWidth
    allVals = dataFrame['mlabel'].values
    while endIndex < len(dataFrame):
        # check if all values in current window
        # are non zero and equal
        currVals = allVals[startIndex:endIndex]
        pureWindow = True
        base = currVals[0]
        if base == 0:
            pureWindow = False
        else:
            tot = np.sum(currVals)
            # not fool proof but good enough
            if tot != base * windowWidth:
                pureWindow = False
        if pureWindow:
            minInd = max(0, endIndex - 1 - pertubations)
            maxInd = min(endIndex + pertubations, len(dataFrame))
            dataFrame.iloc[minInd:maxInd, labelIndex] = base
        startIndex += 1
        endIndex += 1
    return dataFrame


def indicesMaxMin(__list, hyperParams):
    '''
    This function is called by longestPosNegFeatures()
    This takes a 400 long list of values and returns the 
    index of longest max and min values, along with the 
    measure of longest such instances.
    '''
    i = imax = imin = maxval = minval = maxCount = minCount = 0
    thresholdCount = 3
    windowWidth = hyperParams['windowWidth']
    length = len(__list)
    assert(length == windowWidth)
    while(i < length):
        if(__list[i] > 0.62):
            maxCount = 0
            postemp = i
            while(i < length and __list[i] > 0.62):
                i = i + 1
                maxCount = maxCount + 1
            if(maxCount > maxval):
                maxval = maxCount
                imax = postemp
        elif(i < length and __list[i] < 0.32):
            minCount = 0
            negtemp = i
            while(i < length and __list[i] < 0.32):
                i = i + 1
                minCount = minCount + 1
            if(minCount > minval):
                minval = minCount
                imin = negtemp
        else:
            i = i + 1
    if(maxval < thresholdCount):
        imax = -1
        maxval = 0
    if(minval < thresholdCount):
        imin = -1
        minval = 0
    maxmin = [imax, maxval, minval, imin]
    return maxmin


def longestPosNegFeatures(dataFrame, hyperParams):
    '''
    This function calls the entire data frame, uses
    indicesMaxMin() function to iteratively obtain the
    features.
    '''
    assert('norm_gy' in dataFrame.columns)
    windowWidth = hyperParams['windowWidth']
    windowStride = hyperParams['windowStride']
    assert(windowStride == 1)
    dataFrame['longestPosEdge'] = 0
    dataFrame['longestNegEdge'] = 0
    dataFrame['longestPosCount'] = 0
    dataFrame['longestNegCount'] = 0
    colIndex = dataFrame.columns.get_loc('norm_gy')
    colIndexPos = dataFrame.columns.get_loc('longestPosEdge')
    colIndexNeg = dataFrame.columns.get_loc('longestNegEdge')
    colIndexPosCount = dataFrame.columns.get_loc('longestPosCount')
    colIndexNegCount = dataFrame.columns.get_loc('longestNegCount')
    startWindow = 0
    endWindow = windowWidth
    lengthDF = len(dataFrame)
    while endWindow < lengthDF:
        values = dataFrame.iloc[startWindow:endWindow, colIndex].values
        listMaxMinIndex = indicesMaxMin(values, hyperParams)
        dataFrame.iloc[endWindow - 1, colIndexPos] = listMaxMinIndex[0]
        dataFrame.iloc[endWindow - 1, colIndexNeg] = listMaxMinIndex[-1]
        dataFrame.iloc[endWindow - 1, colIndexPosCount] = listMaxMinIndex[1]
        dataFrame.iloc[endWindow - 1, colIndexNegCount] = listMaxMinIndex[-2]
        startWindow = startWindow + windowStride
        endWindow = startWindow + windowWidth
    return dataFrame


def binningFeatures(dataFrame, hyperParams):
    rawColumns = hyperParams['rawColumns']
    normColumns = ['norm_' + x for x in rawColumns]
    for col in normColumns:
        assert (col in normColumns)
    windowWidth = hyperParams['windowWidth']
    numbins = hyperParams['numHistogramBins']
    for col in normColumns:
        for i in range(0, numbins):
            dataFrame['bin_' + str(i) + '_' + col] = 0
    oldBinBoundaries = None
    binBoundaries = []
    for col in normColumns:
        colIndex = dataFrame.columns.get_loc(col)
        startWindow = 0
        endWindow = windowWidth
        values = dataFrame.iloc[startWindow:endWindow, colIndex]
        binF, binBoundaries = np.histogram(values,
                                           bins=numbins,
                                           range=(0.0, 1.0))
        if oldBinBoundaries is None:
            oldBinBoundaries = binBoundaries
        else:
            for i in range(0, len(binBoundaries)):
                assert binBoundaries[i] == oldBinBoundaries[i]
        tot = 0
        for i in range(0, numbins):
            colname = 'bin_' + str(i) + '_' + col
            colIndex = dataFrame.columns.get_loc(colname)
            dataFrame.iloc[endWindow - 1, colIndex] = binF[i]
            tot += binF[i]

    def findIndex(val, boundaryDict):
        for i in range(1, len(boundaryDict)):
            if val < boundaryDict[i]:
                return i - 1
        # too big, goes to last bin
        return len(boundaryDict) - 2

    lengthDF = len(dataFrame)
    for col in normColumns:
        allValues = dataFrame[col].values
        freqDict = {}
        for i in range(0, numbins):
            colname = 'bin_' + str(i) + '_' + col
            colIndex = dataFrame.columns.get_loc(colname)
            freqDict[colname] = [0] * (lengthDF)
            val = dataFrame.iloc[windowWidth - 1, colIndex]
            freqDict[colname][windowWidth - 1] = val

        startWindow = 0
        endWindow = windowWidth
        while endWindow < lengthDF:
            for i in range(0, numbins):
                colname = 'bin_' + str(i) + '_' + col
                freqDict[colname][endWindow] = freqDict[colname][endWindow - 1]

            valueSub = allValues[startWindow]
            valueAdd = allValues[endWindow]
            inS = findIndex(valueSub, oldBinBoundaries)
            colIndex = 'bin_' + str(inS) + '_' + col
            freqDict[colIndex][endWindow] -= 1
            inA = findIndex(valueAdd, oldBinBoundaries)
            colIndex = 'bin_' + str(inA) + '_' + col
            freqDict[colIndex][endWindow] += 1
            startWindow += 1
            endWindow += 1
        for i in range(int(windowWidth - 1), lengthDF):
            tot = 0
            for key in freqDict:
                tot += freqDict[key][i]
            assert tot == 400, i

        for key in freqDict:
            colIndex = dataFrame.columns.get_loc(key)
            dataFrame.iloc[windowWidth:,
                           colIndex] = freqDict[key][windowWidth:]

    return dataFrame


def normalizeDF(df, hyperParams):
    '''
    min max and mean for each of the raw columns should
    be provided.
    '''
    rawColumns = hyperParams['rawColumns']
    for col in rawColumns:
        temp = 'norm_' + col
        df[temp] = 0
    minMaxDict = hyperParams['minMaxDict']
    for col in rawColumns:
        temp = 'norm_' + col
        assert(temp in df.columns)
        _min_ = minMaxDict[col]['min']
        _max_ = minMaxDict[col]['max']
        assert(_max_ - _min_ is not 0)
        df[temp] = (df[col] - _min_) / (_max_ - _min_)
    return df


def featureExtractor(dataFrame, hyperParams, isDebug,
                     collapse = True):
    '''
    Pick one file at a time
    Make basic assertions
    make sure infile is part of the data
    Start sliding by 400 wide window from the top
    and come down while extracting featuers.
    '''
    assert('mlabel' in dataFrame.columns)
    assert('infile' in dataFrame.columns)
    assert('millis' in dataFrame.columns)
    assert(len(dataFrame.infile.unique()) == 1)
    assert(dataFrame.mlabel.unique()[0] in [0, 1, 3, 4, 5, 7, 9])
    assert(hyperParams['windowStride'] == 1)
    # STEP 0: Loading the data
    # The cleaning process should ensure that the data is sorted. The following
    # assertion will verify this.
    assertMsg = 'Monotonicity in millis not maintained.'
    i = 1
    while i < len(dataFrame.millis):
        assert dataFrame.millis[i - 1] <= dataFrame.millis[i], assertMsg
        i += 1
    # STEP 1: Normalization
    oldColOrder = dataFrame.columns
    dataFrame = normalizeDF(dataFrame, hyperParams)
    for i in range(0, len(oldColOrder)):
        assert(dataFrame.columns[i] == oldColOrder[i])
    # STEP 2: Binning Feature
    oldColOrder = dataFrame.columns
    dataFrame = binningFeatures(dataFrame, hyperParams)
    for i in range(0, len(oldColOrder)):
        assert(dataFrame.columns[i] == oldColOrder[i])
    # STEP 3: Index of longest positive and longest negative
    dataFrame = longestPosNegFeatures(dataFrame, hyperParams)
    # STEP 4: Collapse labels
    if collapse:
        oldColOrder = dataFrame.columns
        dataFrame = collapseLabel(dataFrame, hyperParams)
        for i in range(0, len(oldColOrder)):
            assert(dataFrame.columns[i] == oldColOrder[i])
    else:
        dataFrame = dataFrame.iloc[400:]
        dataFrame['label'] = 0
        dataFrame.iloc[:, dataFrame.columns.get_loc('label')] = 1


def threadFeatureExtractor(df, hyperParams, isDebug,
                           collapse=True, NUM_THREADS = 1):
    '''
    Takes the given data frame and breaks it down into 
    smaller chunks
    '''
    # split dataframe
    print("\n Starting splitting of data frame")
    numDataFrames = NUM_THREADS
    dataFrames = np.array_split(df, numDataFrames)
    threads = [None for i in range(numDataFrames)]
    print("Beginning Spawning Threads")
    # begin threads
    for threadCount in range(0, numDataFrames):
        tempDf = dataFrames[threadCount]
        threads[threadCount] = threading.Thread(target=featureExtractor, args=(tempDf, hyperParams, isDebug,))
    print("Starting Threads")
    for threadCount in range(0, numDataFrames):
        print("Starting thread:", threadCount)
        threads[threadCount].start()
    print("Joining Threads")
    for threadCount in range(0, numDataFrames):
        threads[threadCount].join()
        print("Joined thread: ", threadCount)
    print("Concatenating dataframes")
    # Concatinating Dataframes
    df = pd.concat(dataFrames)
    return df


def main(inputFolder, outputFolder, fileList,
         isDebug=False, collapse=True):
    rawColumns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    hyperParams = {
        'windowWidth': 400,
        'windowStride': 1,
        'numHistogramBins': 20,
        'rawColumns': rawColumns,
        # for N pertubations (including original), set below value to N/2
        # (True - N, True + N]
        'pertubations': 5,
        'minMaxDict': {
            'ax': {'min': -16384, 'max': 16384},
            'ay': {'min': -16384, 'max': 16384},
            'az': {'min': -16384, 'max': 16384},
            'gx': {'min': -512, 'max': 512},
            'gy': {'min': -2048, 'max': 2048},
            'gz': {'min': -512, 'max': 512},
        },
    }
    for key in hyperParams:
        print('%15s: %s' % (key, str(hyperParams[key])))

    startTime = time.time()
    currentFile = 1
    oldColOrder = None
    for __file in fileList:
        inpFile = inputFolder + '/' + __file
        msg = '\rFile: %3d/%-3d ' % (currentFile, len(fileList))
        msg += "(%2.2f%%) %-20s" % ((currentFile / len(fileList) * 100), __file)
        print(msg, end = '')
        currentFile += 1

        df = pd.read_csv(inpFile)
        df['infile'] = __file
        ret = threadFeatureExtractor(df, hyperParams, isDebug, collapse, NUM_THREADS)
        print("Feature extractor done")
        if oldColOrder is None:
            oldColOrder = ret.columns
        for i in range(0, len(oldColOrder)):
            assert(oldColOrder[i] == ret.columns[i])
        outputName = outputFolder + '/' + __file[:-4] + '_extracted.csv'
        print("Starting to write to csv")
        ret.to_csv(outputName, index=False)
    endTime = time.time()
    print("\nElapsed feature Extraction: %ds" % (endTime - startTime))


def exportTLCTrainTest(inputFolder, outputFolder, fileList):
    dataFrame = None
    for __file in fileList:
        inpFile = inputFolder + '/' + __file[:-4] + '_extracted.csv'
        ret = pd.read_csv(inpFile)
        if dataFrame is None:
            dataFrame = ret
        else:
            dataFrame = pd.concat([dataFrame, ret])
    dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)
    binCol = [x for x in dataFrame.columns if x.startswith('bin')]
    allCol = ['label', 'longestPosEdge', 'longestPosCount',
              'longestNegCount', 'longestNegEdge'] + binCol
    df = dataFrame[allCol]
    df = df[df.label != 0]
    df = df.reset_index(drop=True)
    # Train test split
    traindf = df.sample(frac=0.8, random_state=42)
    testdf = df.drop(traindf.index)
    traindf.to_csv(outputFolder + '/' + '_train.csv', index=False)
    testdf.to_csv(outputFolder + '/' + '_test.csv', index=False)


def labelAllAsNoise(inputFolder, outputFolder, fileList):
    for f in fileList:
        df = pd.read_csv(inputFolder + '/' + f)
        df['mlabel'] = 0
        colIndex = df.columns.get_loc('mlabel')
        df.iloc[:, colIndex] = 1
        df.iloc[0, colIndex] = 0
        outName = f[:-4] + '_labelled.csv'
        df.to_csv(outputFolder + '/' + outName, index=False)


def processNoiseData(allNoiseFileList):
    '''
    WARNING: This takes RAW files and labels them. This does not take
    labeled files like the other methods
    '''
    rawSource = './data/raw_data/'
    labeledOutput = './data/labelled_data/'
    if not os.path.exists('data/labelled_data'):
        os.mkdir('data/labelled_data')
    extractedOutput = './data/extracted_data/'
    if not os.path.exists('data/extracted_data'):
        os.mkdir('data/extracted_data')
    labelAllAsNoise(rawSource,
                    labeledOutput, allNoiseFileList)
    labelledFileList = [x[:-4] + '_labelled.csv' for x in allNoiseFileList]
    main(labeledOutput, extractedOutput,
         labelledFileList, isDebug=False, collapse=False)


def processLabelledData(labelledFileList):
    if not os.path.exists('data/extracted_data'):
        os.mkdir('data/extracted_data')
    labelledFileList = [x[:-4] + '_labelled.csv' for x in labelledFileList]
    main('./data/labelled_data/', './data/extracted_data/',
         labelledFileList, isDebug=False, collapse=True)


if __name__ == '__main__':
    print("Starting Feature Extraction")
    # The below 2 methods help in feature extraction.
    processLabelledData(labelledFileList)
    # processNoiseData(allNoiseFileList)
    # The below method generates train and test split.
    filesForExport = [x[:-4] + '_labelled.csv' for x in labelledFileList]
    filesForExport += [x[:-4] + '_labelled.csv' for x in allNoiseFileList]
    print("Generating training and test data from %d files" % 
                                    (len(filesForExport)))
    exportTLCTrainTest('./data/extracted_data/',
                       './', filesForExport)
    print("Done!")
