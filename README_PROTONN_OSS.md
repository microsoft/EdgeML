# ProtoNN: Compressed and accurate KNN for resource-constrained devices ([paper](publications/ProtoNN.pdf))
ProtoNN is an algorithm developed for binary, multiclass and multilabel supervised learning. ProtoNN models are time and memory efficient and are thus ideal for resource-constrained scenarios like Internet of Things (IoT). 

## Overview of algorithm
Suppose a single data-point has **dimension** $$D$$. Suppose also that the total number of **classes** is $$L$$. For the most basic version of ProtoNN, there are 2 more user-defined hyper-parameters: the **projection dimension** $$d$$ and the **number of prototypes** $$m$$. 

- ProtoNN learns 3 parameter matrices:
    - A **projection matrix** $$W$$ of dimension $$(d,\space D)$$ that projects the datapoints to a small dimension $$d$$.
    - A **prototypes matrix** $$B$$ that learns $$m$$ prototypes in the projected space, each $$d$$-dimensional. $$B = [B_1,\space B_2, ... \space B_m]$$.
    - A **prototype labels matrix** $$Z$$ that learns $$m$$ label vectors for each of the prototypes to allow a single prototype to represent multiple labels. Each prototype label is $$L$$-dimensional. $$Z = [Z_1,\space Z_2, ... \space Z_m]$$.

- By default, these matrices are dense. However, for high model-size compression, we need to learn sparse versions of the above matrices. The user can restrict the **sparsity of these matrices using the parameters**: $$\lambda_W$$, $$\lambda_B$$ and $$\lambda_Z$$.
    - $$||W||_0 < \lambda_W \cdot size(W) = \lambda_W \cdot d \cdot D$$
    - $$||B||_0 < \lambda_B \cdot size(B) = \lambda_B \cdot d \cdot m$$
    - $$||Z||_0 < \lambda_Z \cdot size(Z) = \lambda_Z \cdot L \cdot m$$ 

- ProtoNN also assumes an **RBF-kernel parametrized by a single parameter:** $$\gamma$$, which can be inferred heuristically from data, or be specified by the user.

More details about the ProtoNN prediction function, the training algorithm, and pointers on how to tune hyper-parameters are suspended to the end of this Readme for better readability. 


## Running
Follow the instructions on the main Readme to compile and create an executable `ProtoNN` 
##### A sample execution with 10-class USPS
After creating the executable, we download a sample dataset: **USPS10**. Instructions for this can be found on the main README. To execute ProtoNN on this dataset, run the following script in bash:
```bash
sh run_ProtoNN_usps10.sh
```
This should give you output on screen as described in the output section. The final test accuracy will be about 93.4 with the specified parameters. 

##### Loading a new dataset
A folder (say `foo`) is required to hold the dataset. `foo` must contain two files: train.txt and test.txt, that hold the training and testing data respectively. The dataset should be in one of the following two formats: 
- **Tab-separated (tsv)**: This is only supported for multiclass and binary datasets, not multilabel ones. The file should have $$N$$ rows and $$D+1$$ columns, where $$N$$ is the number of data-points and $$D$$ is the dimensionality of each point. Columns should be separated by _tabs_. The first column contains the label, which must be a natural number between $$1$$ and $$L$$. The rest of the $$D$$ columns contain the data which are real numbers.
- **Libsvm format**: See https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/. The labels should be between $$1$$ and $$L$$, and the indices should be between $$1$$ and $$D$$. The sample **USPS-10** dataset uses this format. 

The following flag in `config.mk` changes this behavior: 
    
    ZERO_BASED_IO: The expected label range becomes 0 ... (L-1), and the expected feature range becomes 0 ... (D-1). 
The number of lines in train and test data files, the dimension of the data, and the number of labels will _not_ be inferred automatically. They must be specified as described below. 

##### Specifying parameters and executing
To specify hyper-parameters for ProtoNN as well as metadata such as the location of the dataset, input format, etc., one has to write a bash script akin to the sample script at `run_ProtoNN_usps10.sh`. 

Once ProtoNN is compiled, we execute it via this script: 
```bash
sh run_ProtoNN_usps10.sh
```
This bash script is a config file as well as an execution script. There are a number of hyper-parameters to specify, so we split them into categories as described below. Consult `run_ProtoNN_usps10.sh` for a sample. The value in the bracket indicates the command line flag used to set the given hyperparameter. 

##### Input-output parameters
- **-C**: Problem format. Specify one of:
    - 0 (binary)
    - 1 (multiclass)
    - 2 (multilabel)
- **-I**: Input directory for the data, referred to above as `foo`.
- **-F**: Input format for data (described above):
    - 0 (libsvm format)
    - 1 (tab-separated format)
- **-P**: Option to load a predefined model. [**Default:** 0]. Specify as 1 if pre-loading initial values of matrices $$W$$, $$B$$, $$Z$$. One can use this option to initialize with the output of a previous run, or with SLEEC, LMNN, etc. All three matrices, should be present in the data input directory `foo` in tsv format. The values of the parameters $$d$$, $$D$$, $$L$$ will _not_ be inferred, and must be specified correctly in the rest of the fields. The filenames and dimensions of the matrices should be as follows: 
    - $$W$$: Filename: "W". Dimension: ($$d$$, $$D$$). 
    - $$B$$: Filename: "B". Dimension: ($$d$$, $$m$$). 
    - $$Z$$: Filename: "Z". Dimension: ($$L$$, $$m$$).
    - $$\gamma$$: Filename: "gamma". A single number representing the RBF kernel parameter.
	

##### Data-dependent parameters
- **-r**: Number of training points.
- **-e**: Number of testing points.
- **-D**: The original dimension of the data.
- **-l**: Number of classes.

##### ProtoNN hyper-parameters (required)
- **-d**: Projection dimension (the dimension into which the data is projected). [**Default:** $$15$$]
- **Specify only one of the -m and the -k flags:**
    - **-m**: Number of Prototypes. On specifying this parameter, the initialization of prototypes is done by clustering the training data in projected space. [**Default:** $$20$$]
    - **-k**: Number of Prototypes Per Class. On specifying this parameter, initialization of prototypes is done by performing k-means clustering separately for each class, to identify $$k$$ different prototypes for each class. Thus, $$m$$ is automatically set to $$L\cdot k$$.

##### ProtoNN hyper-parameters (optional)
- **-W**: Projection sparsity ($$\lambda_W$$). [**Default:** $$1.0$$]
- **-B**: Prototype sparsity. [**Default:** $$1.0$$]
- **-Z**: Label Sparsity. [**Default:** $$1.0$$]
- **-g**: GammaNumerator.
    - On setting GammaNumerator, the RBF kernel parameter $$\gamma$$ is set as;
    - $$\gamma = (2.5 \cdot GammaNumerator)/(median(||B_j,W - X_i||_2^2))$$
    - **Default:** $$1.0$$
- **-N**: Normalization. Specify one of: 
    - **Default**: 0  (no normalization)
    - 1 (min-max normalization wherein each feature is linearly scaled to lie with 0 and 1)
    - 2 (l2-normalization wherein each data-point is normalized to unit l2-norm)
- **-R**: A random number seed which can be used to re-generate previously obtained experimental results. [**Default:** $$42$$]

##### ProtoNN optimization hyper-parameters (optional)
- **-b**: Batch size for mini-batch stochastic gradient descent. [**Default:** $$1024$$]
- **-T**: Total number of optimization iterations. [**Default:** $$20$$]
- **-E**: Number of epochs (complete see-through's) of the data for each iteration, and each parameter. [**Default:** $$20$$] 

##### Final Execution
The script in this section combines all the specified hyper-parameters to create an execution command. This command is printed to stdout, and then executed.
Most users should copy this section directly to all their ProtoNN execution scripts without change. We provide a single option here that is commented out by default: 
- **gdb --args**: Run ProtoNN with given hyper-parameters in debug mode using gdb. 

## Disclaimers
- The training data is not automatically shuffled in the code. If possible, **pre-shuffle** the data before passing to ProtoNN. For instance, all examples of a single class should not occur consecutively.
- **Normalization**: Ideally, the user should provide **standardized** (Mean-Variance normalized) data. If this is not possible, use one of the normalization options that we provide. The code may be unstable in the absence of normalization.
- The results on various datasets as reported in the ProtoNN paper were using **Gradient Descent** as the optimization algorithm, whereas this repository uses **Stochastic Gradient Descent**. It is possible that the results don't match exactly. We will publish an update to this repository with Gradient Descent implemented. 
- We do not provide support for **Cross-Validation**, only **Train-Test** style runs. The user can write a bash wrapper to perform Cross-Validation. 

## Interpreting the output
- The following information is printed to **std::cout**: 
    - The chosen value of $$\gamma$$.
    - **Training, testing accuracy, and training objective value**, thrice for each iteration, once after optimizing each parameter. For multilabel problems, **prec@1** is output instead.  
    - On enabling the `VERBOSE` flag in `config.mk`, additional informative output is printed to stdout. 

- **Errors and warnings** are printed to **std::cerr**. 

- Additional **parameter dumps**, **timer logs** and other **debugging logs** will be placed in the input folder `foo`. Hence, the user should have read-write permissions on `foo` (use chmod if necessary). 
    -  On execution, a folder called `ProtoNNResults` is created in `foo`. The results folder will have another folder whose name will indicate to the user the list of parameters with which the run was instantiated. In this folder, upto **6 files** will be created depending on which flags are set in `config.mk`: 
    - **diagnosticLog**: Created on using the `LOGGER` or `LIGHT_LOGGER` flags. This file stores logging information such as the call flow of ProtoNN and the min, max, norms of various matrices. This is mainly for debugging/optimization purposes and requires a more detailed understanding of the code to interpret. It may contain useful information if your code did not run as expected. **The log file is populated synchronously while the ProtoNN optimization is executing.** 
    - **timerLog**: Created on using the `TIMER` flag. This file stores proc time and wall time taken to execute various function calls in the code. Indicates the degree of parallelization and is useful for identifying bottlenecks to optimize the code. On specifying the `CONCISE` flag, timing information will only be printed if running time is higher than a threshold specified in `src/common/timer.cpp`.
    - **runInfo**: This file contains the hyperparameters and meta-information for the respective instantiation of ProtoNN. It also shows you the exact bash script call that was made, which is helpful for reproducing results purposes. Additionally, the training, testing accuracy and objective value at the end of each iteration is printed in a readable format. **This file is created at the end of the ProtoNN optimization.**
    - **W, B, Z**: These files contain the learnt parameter matrices $$W$$, $$B$$ and $$Z$$ in human-readable tsv format. The dimensions of storage are $$(d, D)$$, $$(d, m)$$ and $$(L, m)$$ respectively. **These files are created at the end of the ProtoNN optimization.**
    - **gamma**: This file contains a single number, the chosen value of $$\gamma$$, the RBF kernel parameter.

The files **W, B, Z**, and **gamma** can be copied to `foo` to continue training of ProtoNN by initializing with these previously learned matrices. Use the **-P** option for this (see above). On doing so, the starting train/test accuracies should match the final accuracy as specified in the runInfo file. 

## Choosing hyperparameters
##### Model size as a function of hyperparameters
The user presented with a model-size budget has to make a decision regarding the following 5 hyper-parameters: 
- The projection dimension $$d$$
- The number of prototypes $$m$$
- The 3 sparsity parameters: $$\lambda_W$$, $$\lambda_B$$, $$\lambda_Z$$
 
Each parameter requires the following number of non-zero values for storage:
- $$S_W: min(1, 2\lambda_W) \cdot d \cdot D$$
- $$S_B: min(1, 2\lambda_B) \cdot d \cdot m$$
- $$S_Z: min(1, 2\lambda_Z) \cdot L \cdot m$$

The factor of 2 is for storing the index of a sparse matrix, apart from the value at that index. Clearly, if a matrix is more than 50% dense ($$\lambda > 0.5$$), it is better to store the matrix as dense instead of incurring the overhead of storing indices along with the values. Hence the minimum operator. 
Suppose each value is a single-precision floating point (4 bytes), then the total space required by ProtoNN is $$4\cdot(S_W + S_B + S_Z)$$. This value is computed and output to screen on running ProtoNN. 

##### Pointers on choosing hyperparameters
Choosing the right hyperparameters may seem to be a daunting task in the beginning but becomes much easier with a little bit of thought. To get an idea of default parameters on some sample datasets, see the ([paper](publications/protonn.pdf)). Few rules of thumb:
-- $$S_B$$ is typically small, and hence $$\lambda_B \approx 1.0$$. 
-- One can set $$m$$ to $$min(10\cdot L, 0.01\cdot numTrainingPoints)$$, and $$d$$ to $$15$$ for an initial experiment. Typically, you want to cross-validate for $$m$$ and $$d$$. 
-- Depending on $$L$$ and $$D$$, $$S_W$$ or $$S_Z$$ is the biggest contributors to model size. $$\lambda_W$$ and $$\lambda_Z$$ can be adjusted accordingly or cross-validated for. 

## ProtoNN for Extreme Multilabel Learning (XML)
[XML](http://manikvarma.org/downloads/XC/XMLRepository.html) refers to a difficult class of multi-label learning problems where the number of labels is large (ranging from a few thousands to a few millions). ProtoNN has been written to be compatible with these datasets (still in beta). This mode can be enabled by the `XML` flag. 

## Formal details
##### Prediction function
ProtoNN predicts on a new test-point in the following manner. For a test-point $$X$$, ProtoNN computes the following $$L$$ dimensional score vector:
$$Y_{score}=\sum_{j=0}^{m}\space \left(RBF_\gamma(W\cdot X,B_j)\cdot Z_j\right)$$, where
$$RBF_\gamma (U, V) = exp\left[-\gamma^2||U - V||_2^2\right]$$
The prediction label is then $$\space max(Y_{score})$$. 

##### Training 
While training, we are presented with training examples $$X_1, X_2, ... X_n$$ along with their label vectors $$Y_1, Y_2, ... Y_n$$ respectively. $$Y_i$$ is an L-dimensional vector that is $$0$$ everywhere, except the component to which the training point belongs, where it is $$1$$.  For example, for a $$3$$ class problem, for a data-point that belongs to class $$2$$, $$Y=[0, 1, 0]$$. 
We optimize the $$l_2$$-square loss over all training points as follows:  $$\sum_{i=0}^{n} = ||Y_i-\sum_{j=0}^{m}\space \left(exp\left[-\gamma^2||W\cdot X_i - B_j||^2\right]\cdot Z_j\right)||_2^2$$. 
While performing stochastic gradient descent, we hard threshold after each gradient update step to ensure that the three memory constraints (one each for $$\lambda_W, \lambda_B, \lambda_Z$$) are satisfied by the matrices $$W$$, $$B$$ and $$Z$$. 

