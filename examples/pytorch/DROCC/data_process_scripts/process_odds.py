import os
import numpy as np
from scipy.io import loadmat
import argparse

parser = argparse.ArgumentParser(description='Preprocess Dataset from ODDS Repository')
parser.add_argument('-d', '--data_path', type=str, default='./arrhythmia.mat')
parser.add_argument('-o', '--output_path', type=str, default='.')
args = parser.parse_args()

dataset = loadmat(args.data_path)

data = np.concatenate((dataset['X'], dataset['y']), axis=1)

test = data[data[:,-1] == 1]
num_normal_samples_test = test.shape[0]

normal = data[data[:,-1] == 0]
np.random.shuffle(normal)

test = np.concatenate((test, normal[:num_normal_samples_test]), axis=0)

train = normal[num_normal_samples_test:]
train_data = train[:,:-1]
# DROCC requires normal data to be labelled 1
train_labels = np.ones(train_data.shape[0])

test_data = test[:,:-1]
# DROCC requires normal data to be labelled 1 and anomalies 0
test_labels = np.concatenate((
        np.zeros(num_normal_samples_test), np.ones(num_normal_samples_test)),
        axis=0)

np.save(os.path.join(args.output_path,'train_data.npy'), train_data)
np.save(os.path.join(args.output_path,'train_labels.npy'), train_labels)
np.save(os.path.join(args.output_path,'test_data.npy'), test_data)
np.save(os.path.join(args.output_path,'test_labels.npy'), test_labels)