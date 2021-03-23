#! /usr/bin/env python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import scipy.io.wavfile as r
import glob
from python_speech_features import fbank
import os
from os import listdir
import argparse

np.random.seed(42)

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--datapath", type=str, 
					metavar='', 
					help="Location where the data folders are stored.")
parser.add_argument("-s", "--save_dir", type=str, 
					metavar='', 
					help="Location to store the processed dataset.")
args = parser.parse_args()

datapath=os.path.abspath(args.datapath)
save_dir=os.path.abspath(args.save_dir)

test_list=datapath+'/testing_list.txt'
valid_list=datapath+'/validation_list.txt'
sample_rate=16000
nfilt=32

classes=["yes", "no", "up", "down", "left","right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four",
	"five", "six", "seven", "eight", "nine","bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila",
	"tree", "wow"]

with open(valid_list) as f:
	valid_lines=f.readlines()

with open(test_list) as f:
	test_lines=f.readlines()

all_files=[]
all_data=[]
all_data_cls=[]
feats_processed=[]
i=0
for cl in classes:
	files=glob.glob(datapath+'/'+cl+'/*.wav')
	for f in files:
		_,x = r.read(f)
		x_new=np.zeros([16000])
		x_new[0:x.shape[0]]=x
		all_data.append(x_new)
		all_files.append(f)
		all_data_cls.append(i)
	i=i+1	
	print(cl+" read")

i=0
for wav in all_data:
    temp,_ = fbank(wav,nfilt=nfilt,winfunc=np.hamming)
    feat=np.log(temp)
    feat=np.reshape(feat,(temp.shape[0]*temp.shape[1]))
    feat=np.append(all_data_cls[i],feat[:])
    feats_processed.append(feat)
    i=i+1
train_data=[]
valid_data=[]
test_data=[]
i=0
for j in feats_processed:
	a=all_files[i].split('/')
	name=a[-2]+'/'+a[-1]+'\n'
	if(name in valid_lines):
		valid_data.append(j)
	elif(name in test_lines):
		test_data.append(j)
	else:
		train_data.append(j)
	i=i+1

train_data=np.array(train_data)
test_data=np.array(test_data)
valid_data=np.array(valid_data)
indx = np.random.permutation(len(train_data))
indx=indx.astype(int)
train_data=train_data[indx]
indx = np.random.permutation(len(test_data))
indx=indx.astype(int)
test_data=test_data[indx]
indx = np.random.permutation(len(valid_data))
indx=indx.astype(int)
valid_data=valid_data[indx]

np.save(save_dir+'/train.npy',train_data)
np.save(save_dir+'/test.npy',test_data)
np.save(save_dir+'/valid.npy',valid_data)
F=open(save_dir+'/specs.txt','w') 
F.write("dims: "+str(train_data.shape[1]-1)+"\n")
F.write("Train points: "+str(train_data.shape[0])+"\n")
F.write("Valid points: "+str(valid_data.shape[0])+"\n")
F.write("Test points: "+str(test_data.shape[0])+"\n")
F.write("Num Classes : "+str(len(classes))+"\n")

