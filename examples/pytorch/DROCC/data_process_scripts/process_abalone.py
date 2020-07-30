import os
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Preprocess Abalone Data')
parser.add_argument('-d', '--data_path', type=str, default='./abalone.data')
parser.add_argument('-o', '--output_path', type=str, default='.')
args = parser.parse_args()

data = pd.read_csv(args.data_path, header=None, sep=',')

data = data.rename(columns={8: 'y'})

data['y'].replace([8, 9, 10], -1, inplace=True)
data['y'].replace([3, 21], 0, inplace=True)
data.iloc[:, 0].replace('M', 0, inplace=True)
data.iloc[:, 0].replace('F', 1, inplace=True)
data.iloc[:, 0].replace('I', 2, inplace=True)

test = data[data['y'] == 0]
num_normal_samples_test = test.shape[0]

normal = data[data['y'] == -1].sample(frac=1)

test_data = np.concatenate((test.drop('y', axis=1), normal[:num_normal_samples_test].drop('y', axis=1)), axis=0)
train = normal[num_normal_samples_test:]
train_data = train.drop('y', axis=1).values
train_labels = train['y'].replace(-1, 1)
test_labels = np.concatenate((test['y'], normal[:num_normal_samples_test]['y'].replace(-1, 1)), axis=0)

np.save(os.path.join(args.output_path,'train_data.npy'), train_data)
np.save(os.path.join(args.output_path,'train_labels.npy'), train_labels)
np.save(os.path.join(args.output_path,'test_data.npy'), test_data)
np.save(os.path.join(args.output_path,'test_labels.npy'), test_labels)