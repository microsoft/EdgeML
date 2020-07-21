import os
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, default='./data.csv')
parser.add_argument('-o', '--output_path', type=str, default='.')
args = parser.parse_args()

data = pd.read_csv(args.data_path)

data['y'] = data['y'].replace(1, 0)

data['y'] = data['y'].replace([2, 3, 4, 5], 1)


test = data[data['y'] == 0]
normal = data[data['y'] == 1].sample(frac=1).reset_index(drop=True)

test = pd.concat([test, normal.iloc[:2300]])

normal = normal.iloc[2300:]

normal = normal.drop(['y', 'Unnamed: 0'], axis=1)
np.save(os.path.join(args.output_path, 'train.npy'), normal.values)

test = test.drop('Unnamed: 0', axis=1)
test = test.sample(frac=1).reset_index(drop=True)

labels = test['y'].values

test = test.drop('y', axis=1).values
np.save(os.path.join(args.output_path, 'test_data.npy'), test)
np.save(os.path.join(args.output_path, 'test_labels.npy'), labels)

