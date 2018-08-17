# creates x_train, y_train, x_test, y_test, x_val, y_val npy files. val set is created after cutting out a user provided fraction from train set.

import numpy as np
import os

f = open('train/subject_train.txt')
subjects = []
for line in f:
	subject = line.strip().split()
	subjects.append(int(subject[0]))
subjects = np.array(subjects)

# get unique subjects
numSubjects = np.unique(subjects)
print ("Unique subjects are", numSubjects)

# shuffle amongst train subjects so that difficult/easy subjects spread in both val and train
np.random.seed(42)
np.random.shuffle(numSubjects)
print ("Unique subjects after shuffling are", numSubjects)

l = len(numSubjects)

splitRatio = 0.1
valSplit = int(l*splitRatio + 1)

valSubjects = numSubjects[:valSplit]
trainSubjects = numSubjects[valSplit:]

print ("Train subjects are", trainSubjects)
print ("Val subjects are", valSubjects)

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
np.random.seed(42)
np.random.shuffle(idx)
trainSubjectIndices = trainSubjectIndices[idx]

idx = np.arange(len(valSubjectIndices))
np.random.seed(42)
np.random.shuffle(idx)
valSubjectIndices = valSubjectIndices[idx]

print (len(trainSubjectIndices), len(valSubjectIndices))
assert len(trainSubjectIndices) + len(valSubjectIndices) == len(subjects)

# painfully read files
def readData():
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
	DATASET_PATH = ""
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


x_train_val_combined, y_train_val_combined, x_test, y_test = readData()
timesteps = x_train_val_combined.shape[-2]
feats = x_train_val_combined.shape[-1]

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
np.random.seed(42)
np.random.shuffle(idx)
x_test = x_test[idx]
y_test = y_test[idx]

# one-hot encoding of labels
numOutput = 6

one_hot = np.zeros([y_train.shape[0], numOutput])
for i, label in enumerate(y_train):
    one_hot[i][label[0]] = 1
y_train = one_hot

one_hot = np.zeros([y_test.shape[0], numOutput])
for i, label in enumerate(y_test):
    one_hot[i][label[0]] = 1
y_test = one_hot

one_hot = np.zeros([y_val.shape[0], numOutput])
for i, label in enumerate(y_val):
    one_hot[i][label[0]] = 1
y_val = one_hot

print("x_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print("x_test shape", x_test.shape)
print("y_test shape", y_test.shape)
print("x_val shape", x_val.shape)
print("y_val shape", y_val.shape)

# Since for us, validation set is test set and test set is validation set
# We pass run train and test sets ( so test is val )
np.save("x_train", x_train)
np.save("y_train", y_train)
np.save("x_test", x_test)
np.save("y_test", y_test)
np.save("x_val", x_val)
np.save("y_val", y_val)

