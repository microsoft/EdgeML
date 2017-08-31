# Bonsai

[Bonsai](publications/Bonsai.pdf) is a novel tree based algorithm for efficient prediction on IoT devices – such as those based on the Arduino Uno board having an 8 bit ATmega328P microcontroller operating at 16 MHz with no native floating point support, 2 KB RAM and 32 KB read-only flash.

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

    ./Bonsai [Options] DataFolder
    Options:

    -F    : [Required] Number of features in the data.
    -C    : [Required] Number of Classification Classes/Labels.
    -nT   : [Required] Number of training examples.
    -nE   : [Required] Number of examples in test file.
    -O    : [Optional] Flag to Indicate if Labels are One Indexed (Default set to 0)
    -f    : [Optional] Input format. Takes two values [0 and 1]. 0 is for libsvm_format(default), 1 is for tab/space separated input.

    -P   : [Optional] Projection Dimension. (Default: 10 Try: [5, 20, 30, 50]) 
    -D   : [Optional] Depth of the Bonsai tree. (Default: 3 Try: [2, 4, 5])
    -S   : [Optional] sigma = parameter for sigmoid sharpness  (Default: 1.0 Try: [3.0, 0.05, 0.005] ).

    -lW  : [Optional] lambda_W = regularizer for classifier parameter W  (Default: 0.0001 Try: [0.01, 0.001, 0.00001]).
    -lT  : [Optional] lambda_Theta = regularizer for kernel parameter Theta  (Default: 0.0001 Try: [0.01, 0.001, 0.00001]).
    -lV  : [Optional] lambda_V = regularizer for kernel parameters V  (Default: 0.0001 Try: [0.01, 0.001, 0.00001]).
    -lZ  : [Optional] lambda_Z = regularizer for kernel parameters Z  (Default: 0.00001 Try: [0.001, 0.0001, 0.000001]).

    Use Sparsity Params to vary your model size for a given tree depth and projection dimension
    -sW  : [Optional] sparsity_W = regularizer for classifier parameter W  (Default: For Binaray 1.0 else 0.2 Try: [0.1, 0.3, 0.4, 0.5]).
    -sT  : [Optional] sparsity_Theta = regularizer for kernel parameter Theta  (Default: For Binaray 1.0 else 0.2 Try: [0.1, 0.3, 0.4, 0.5]).
    -sV  : [Optional] sparsity_V = regularizer for kernel parameters V  (Default: For Binaray 1.0 else 0.2 Try: [0.1, 0.3, 0.4, 0.5]).
    -sZ  : [Optional] sparsity_Z = regularizer for kernel parameters Z  (Default: 0.2 Try: [0.1, 0.3, 0.4, 0.5]).

    -I   : [Optional] [Default: 40 Try: [100, 30, 60]] Number of passes through the dataset.
	-B   : [Optional] Batch Factor [Default: 1 Try: [2.5, 10, 100]] Float Factor to multiply with sqrt(ntrain) to make the batch_size = min(max(100, B*sqrt(nT)), nT).
    DataFolder : [Required] Path to folder containing data with filenames being 'train.txt' and 'test.txt' in the folder."
    
    Note - libsvm_format can be either Zero or One Indexed in labels. Space/Tab separated format has to be Zero indexed in labels by design
      

## Data Format    
    
    (a) "train.txt" is train data file with label followed by features, "test.txt" is test data file with label followed by features
    (b) They can be either in libsvm_format or a simple tab/space separated format
    (c) Try to shuffle the "train.txt" file before feeding it in. Ensure that all instances of a single class are not together

## Running on USPS-10

Following the instructions in the [common readme](README.md) will give you a binary for Bonsai and a folder called usps10 with train and test datasets.
Now run the script
```bash
sh run_Bonsai_usps10.sh
```
This should give you output as described in the next section. Test accuracy will be about 94.07% with the specified parameters.

## Output

The DataFolder will have a new forlder named Results with the following files in it:

    (a) A directory for each run with the signature hrs_min_sec_day_month with the following in it:
        (1) loadableModel - Char file which can be directly loaded using the inbuilt load model functions
        (2) loadableMeanVar - Char file which can be directly loaded using inbuilt load mean-var functions
        (3) predClassAndScore - File with Prediction Score and Predicted Class for each Data point in the test set
        (4) runInfo - File with the hyperparameters for that run of Bonsai along with Test Accuracy and Total NonZeros in the model
        (5) Params - A directory with readable files with Z, W, V, Theta, Mean and Variance
    (b) A file resultDump which has consolidated results and map to the respective run directory

##  Notes
    (a) As of now, there is no support to Multi Label Classification, Ranking and Regression in Bonsai
    (b) Model Size = 8*totalNonZeros Bytes. 4 Bytes to store index and 4 Bytes to store value to store a sparse model
    (c) We do not provide support for Cross-Validation, support exists only for Train-Test. The user can write a bash wrapper to perform Cross-Validation.
    
