# Bonsai

[Bonsai](../../docs/publications/Bonsai.pdf) is a novel tree based algorithm for efficient prediction on IoT devices – such as those based on the Arduino Uno board having an 8 bit ATmega328P microcontroller operating at 16 MHz with no native floating point support, 2 KB RAM and 32 KB read-only flash.

Bonsai maintains prediction accuracy while minimizing model size and prediction costs by: 

        (a) Developing a tree model which learns a single, shallow, sparse tree with powerful nodes 
        (b) Sparsely projecting all data into a low-dimensional space in which the tree is learnt
        (c) Jointly learning all tree and projection parameters

Experimental results on multiple benchmark datasets demonstrate that Bonsai can make predictions in milliseconds even on slow microcontrollers, can fit in KB of memory, has lower battery consumption than all other algorithms while achieving prediction accuracies that can be as much as 30% higher than state-of-the-art methods for resource-efficient machine learning. Bonsai is also shown to generalize to other resource constrained settings beyond IoT by generating significantly better search results as compared to Bing’s L3 ranker when the model size is restricted to 300 bytes.

## Algorithm

Bonsai learns a balanced tree of user speciﬁed height `h`.

The parameters that need to be learnt include:

        (a) Z: the sparse projection matrix; 
        (b) θ = [θ1,...,θ2h−1]: the parameters of the branching function at each internal node
        (c) W = [W1,...,W2h+1−1] and V = [V1,...,V2h+1−1]:the predictor parameters at each node

We formulate a joint optimization problem to train all the parameters using the following three phase training routine:

        (a) Unconstrained Gradient Descent: Train all the parameters without having any Budget Constraint
        (b) Iterative Hard Thresholding (IHT): Applies IHT constantly while training
	    (c) Training with constant support: After the IHT phase the support(budget) for the parameters is fixed and are trained

We use simple Batch Gradient Descent as the solver with Armijo rule as the step size selector.

## Prediction

When given an input feature vector X, Bonsai gives the prediction as follows :

        (a) We project the data onto a low dimensional space by computing x^ = Zx
        (b) The final bonsai prediction score is the non linear scores (wx^ * tanh(sigma*vx^) ) predicted by each of the individual nodes along the path traversed by the Bonsai tree

## Usage

BonsaiTrain

    ./BonsaiTrain [Options] DataFolder
    Options:

    -F    : [Required] Number of features in the data.
    -C    : [Required] Number of Classification Classes/Labels.
    -nT   : [Required] Number of training examples.
    -nE   : [Required] Number of examples in test file.
    -f    : [Optional] Input format. Takes two values [0 and 1]. 0 is for libsvm_format(default), 1 is for tab/space separated input.

    -P   : [Optional] Projection Dimension. (Default: 10 Try: [5, 20, 30, 50]) 
    -D   : [Optional] Depth of the Bonsai tree. (Default: 3 Try: [2, 4, 5])
    -S   : [Optional] sigma = parameter for sigmoid sharpness  (Default: 1.0 Try: [3.0, 0.05, 0.005] ).

    -lW  : [Optional] lW = regularizer for predictor parameter W  (Default: 0.0001 Try: [0.01, 0.001, 0.00001]).
    -lT  : [Optional] lTheta = regularizer for branching parameter Theta  (Default: 0.0001 Try: [0.01, 0.001, 0.00001]).
    -lV  : [Optional] lV = regularizer for predictor parameter V  (Default: 0.0001 Try: [0.01, 0.001, 0.00001]).
    -lZ  : [Optional] lZ = regularizer for projection parameter Z  (Default: 0.00001 Try: [0.001, 0.0001, 0.000001]).

    Use Sparsity Params to vary your model size for a given tree depth and projection dimension
    -sW  : [Optional] lambdaW = sparsity for predictor parameter W  (Default: For Binaay 1.0 else 0.2 Try: [0.1, 0.3, 0.4, 0.5]).
    -sT  : [Optional] lambdaTheta = sparsity for branching parameter Theta  (Default: For Binary 1.0 else 0.2 Try: [0.1, 0.3, 0.4, 0.5]).
    -sV  : [Optional] lambdaV = sparsity for predictor parameters V  (Default: For Binary 1.0 else 0.2 Try: [0.1, 0.3, 0.4, 0.5]).
    -sZ  : [Optional] lambdaZ = sparsity for projection parameters Z  (Default: 0.2 Try: [0.1, 0.3, 0.4, 0.5]).

    -I   : [Optional] [Default: 42 Try: [100, 30, 60]] Number of passes through the dataset.
	-B   : [Optional] Batch Factor [Default: 1 Try: [2.5, 10, 100]] Float Factor to multiply with sqrt(ntrain) to make the batch_size = min(max(100, B*sqrt(nT)), nT).
    DataFolder : [Required] Path to folder containing data with filenames being 'train.txt' and 'test.txt' in the folder."
    
    Note - Both libsvm_format and Space/Tab separated format can be either Zero or One Indexed in labels. To use Zero Index enable ZERO_BASED_IO flag in config.mk and recompile Bonsai

BonsaiPredict:

    ./BonsaiPredict [Options]

    Options:
    -f    : [Required] Input format. Takes two values [0 and 1]. 0 is for libsvmFormat(default), 1 is for tab/space separated input.
    -N    : [Required] Number of data points in the test data.
    -D    : [Required] Directory of data with test.txt present in it.
    -M    : [Required] Directory of the Model (loadableModel and loadableMeanStd).

## Data Format    
    
    (a) "train.txt" is train data file with label followed by features, "test.txt" is test data file with label followed by features
    (b) They can be either in libsvm_format or a simple tab/space separated format
    (c) Try to shuffle the "train.txt" file before feeding it in. Ensure that all instances of a single class are not together

## Running on USPS-10

Following the instructions in the [common readme](../README.md) will give you a binaries for BonsaiTrain and BonsaiPredict along with a folder called usps10 with train and test datasets.

For running Training separately followed by prediction
```bash
sh run_BonsaiTrain_usps10.sh

The script prints the path of the model-dir

use "ln -s <model-dir> current_model" to set a soft alias(shortcut) if you wish to run on that model or you choose <model-dir> as per your wish so as to use it in BonsaiPredict on usps10

sh run_BonsaiPredict_usps10.sh
```
This should give you output as described in the next section. Test accuracy will be about 94.07% with the specified parameters.

## Output

The DataFolder will have a new forlder named "BonsaiResults" with the following files in it:

    (a) A directory for each run with the signature hrs_min_sec_day_month with the following in it:
        (1) loadableModel - Char file which can be directly loaded using the inbuilt load model functions
        (2) loadableMeanStd - Char file which can be directly loaded using inbuilt load mean-var functions
        (3) predClassAndScore - File with Prediction Score and Predicted Class for each Data point in the test set
        (4) runInfo - File with the hyperparameters for that run of Bonsai along with Test Accuracy and Total NonZeros in the model
        (5) timerLog - Created on using the `TIMER` flag. This file stores proc time and wall time taken to execute various function calls in the code. Indicates the degree of parallelization and is useful for identifying bottlenecks to optimize the code. On specifying the `CONCISE` flag, timing information will only be printed if running time is higher than a threshold specified in `src/common/timer.cpp`
        (6) Params - A directory with readable files with Z, W, V, Theta, Mean and Std
    (b) A file resultDump which has consolidated results and map to the respective run directory

##  Notes
    (a) You can load an pretrained model and continue further training on it using one of the BonsaiTrainer constructors. Please look into BonsaiTrainer.cpp for details
    (b) You can load the pretrained model and the test data by setting -nT 0 and then export model and construct predictor for the same. Look into BonsaiTrainer.cpp for details
    (c) As of now, there is no support to Multi Label Classification, Ranking and Regression in Bonsai
    (d) Model Size = 8*totalNonZeros Bytes. 4 Bytes to store index and 4 Bytes to store value to store a sparse model
    (e) We do not provide support for Cross-Validation, support exists only for Train-Test. The user can write a bash wrapper to perform Cross-Validation.
    (f) Currently, Bonsai is being compiled with MKL_SEQ_LDFLAGS, one can change to MKL_PAR_FLAGS if interested
    
