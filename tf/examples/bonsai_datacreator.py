"""
Will iterate through a directory, and convert all the csv files, to a train and test npy arrays, of the same name, as that of the files.

Usage :
Given a directory, containing tsv/csv files, this script will iterate through each folder, and build the train and test npy, arrays.
After building data in this format, run the bonsai_folderscaller.py file.

Command line argument to be given, is the name of the directory.
"""

import os
import sys
import numpy as np
import pickle
import pandas as pd
os.chdir("../../"+sys.argv[1]+"/")
dir_contents = os.listdir(".")
split_dict = {}

for file in (dir_contents):
    print ("\n\n")
    print ("-------------------------------------")
    print (file,"\n\n")
    if("csv" in file):
        df = pd.read_csv(file,sep=",")
        print ("Reading file : ",file)
        print ("Number of NULL values present : ",np.sum(df.isnull().sum(axis=0)))
        print ("Shape of the df : ",df.shape)
        #Calculate the split.
        split = int(0.8*df.shape[0])
        split_dict[file] = split
        train = df.values[:split,:]
        test = df.values[split:,:]
        print (train.shape,test.shape)
        os.mkdir(str(file.split(".")[0]))
        np.save(str(file.split(".")[0])+"/train.npy",train)
        np.save(str(file.split(".")[0])+"/test.npy",test)
print ("Saving the dictionary of splits.")
pickle.dump(split_dict,open("split_dict.pkl","wb"))
