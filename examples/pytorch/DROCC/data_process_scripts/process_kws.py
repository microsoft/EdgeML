import numpy as np

train_f = np.load('train_seven.npz')['features'] # containing only the class marvin
others_f = np.load('other_seven.npz')['features'] # containing classes other than marvin

np.random.shuffle(train_f)
np.random.shuffle(others_f)

len_train = 0.8 * len(train_f)
len_test = len(train_f) - len_train

data = train_f[:len_train]
np.save('train.npy', data)

test_data = np.concatenate((train_f[len_train:], others_f[len_t:len_train+len_test]), axis=0)
labels = [1] * len_test + [0] * len_test
np.save('test_data.npy', test_data)
np.save('test_labels.npy', labels)
