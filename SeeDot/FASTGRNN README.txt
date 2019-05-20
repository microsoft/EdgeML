
1.	Normalization: The mean and std are dumped seperately and the user is expected to normalize the dataset on their own. Instead, absorb mean and std into the train and the test set.

2.	zeta and nu are actually sigmoid(zeta) and sigmoid(nu) respectively.

3.	0-indexed labels instead of 1-indexed labels.
