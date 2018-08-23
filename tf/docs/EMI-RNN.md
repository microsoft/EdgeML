# Early Multi-Instance Recurrent Neural Network

This document aims to elaborate on certain aspects and advanced use cases
supported in the implementation of EMI-RNN algorithm that is part of the EdgeML
Tensorflow library. This document **does not** seek to be a comprehensive
documentation of the EMI-RNN code base.

For a quick and dirty 'getting started' example please refer to
`tf/examples/EMI-RNN` directory.

![MIML Formulation of Bags and Instances](img/MIML_illustration.png)

In this work, a typical time series data-point is divided into a *bag* of
*instances*, as explained in the associated publication. The image illustrates
a single time series data point divided into individual instances $Z_{i, j}$.

1. **Bag**: We use the term *bag* to mean the ordered set of all instances
   obtained from a single time series data point. The bag comprising of the
instances illustrated in the above figure is $\chi_i$.
2. **Instance or Sub-Instance**: We use the term *instance* and *sub-instance*
   interchangibly to mean the *instances* that comprise a bag.

From an implementation perspective, a data point, originally a 2-D
`numpy.array` object of shape `[Number of RNN time steps, Number of features]`,
after dividing into *instances*, becomes an 3-D array of shape `[Number of
instances in a bag, Number of RNN time steps in each instance, Number of
featuers]`. For example, consider a time series data-point that has 128 time
steps and 6 features at each time step. The shape is [128, 6]. Lets say, we
divide this data point into instances of width 48 (time steps), with
consecutive instances separated by 16 (time steps) - we would obtain 6 instance
(0-48, 16-64, 32-80, 48-96, 64-112, 80-128) and the shape of the bag would be
[6, 48, 6].

## Data Preparation

[Typical RNN
models](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/recurrent_network.ipynb),
by Tensorflow convention, uses a 3 dimensional tensor to hold input data. This
tensor is of shape `[number of examples, number of time steps, number of
features]`. To incorporate the notion of *bags* and *sub-instances*, we extend
this by adding an additional fourth dimension, as explained previously. This
makes our input data shape - `[number of bags, number of sub-instances, number
of time steps, number of features]`. Additionally, the typical shape of the
one-hot encoded label tensor - `[number of examples, number of outputs]` is
extended to incorporate sub-instance level labels, thus making it `[number of
bags, number of sub-instances, number of output classes]`. The label of each
sub-instance is set initialized to equal the label of the entire bag.

Thus, the EMI-RNN implementation expects the train data to be of shape,

    [Num. of examples, Num. of instances, Num. of timestep, Num. features]
    
Further, the label information is expected to be one hot encoded and of the
shape,

    [Num. of examples, Num. of instances, Num. classes]

As a concrete end to end example, please refer to
`tf/examples/EMI-RNN/fetch_har.py`.

## Training

![An illustration of the parts of the computation graph](img/3PartsGraph.png)

The EMI-RNN algorithm consists of a graph construction phase and a training
phase. 

### Graph Construction

The *EMI-RNN* computation graph is constructed out of the following three
mutually disjoint parts:

1. `EMI_DataPipeline`: An efficient data input pipeline using the Tensorflow
   Dataset API. This module ingests data compatible with EMI-RNN and provides
two iterators for a batch of input data $x$ and label $y$. We do not support
feed dict based data input methods and assume that $x$ and $y$ are iterables.
2. `EMI_RNN`: The 'abstract' `EMI-RNN` class defines the methods and attributes
   required for the forward computation graph. This module expects two Dataset
API iterators for $x$-batch and $y$-batch as inputs (for example, from the
EMI_DataPipeline) and constructs the forward computation graph based on them.
Users are free to define their own implementations of `EMI_RNN` with arbitrary
forward computation graphs. All implementations of `EMI_RNN` are expected and
assumed to provide an `EMI_RNN.output` attribute - the Tensor/Operation with
the forward computation outputs. The following implementations of `EMI_RNN` are
provided:
  - `EMI_LSTM` 
  - `EMI_GRU`
  - `EMI_FastRNN`
  - `EMI_FastGRNN`
  - `EMI_UGRNN`

3. `EMI_Trainer`: An instance of `EMI_Trainer` class which defines the loss
functions and the training routine. This expects the `output` attribute from an
`EMI-RNN` implementation as input and attaches loss functions and training
routines to it. Currently, L2-loss and Softmax cross entropy loss is supported
without any regularization terms, natively. Users can modify these loss
operators by providing appropriate arguments to `EMI_Trainer`. Please refer to
the `EMI_Trainer` doc-string for more information.

To build the computation graph, we create one instance of all the above,
initialize them with appropriate arguments and then connect them together by
invoking their respective `__call__` methods.

### Training

All of the details of the training process is abstracted away by the
`EMI_Driver` class, specifically by `EMI_Driver.run()` method. Nevertheless, an
elaboration of the procedure is provided here.

The training algorithm can be described as follows:

```
Train_EMI_RNN:
    Required:
		X: Train data
		Y: Train labels
		EMI_Graph: A complete training graph
		updatePolicy: An update policy that will update the instace lables
			after each training rounds. 
		NUM_ROUNDS: Number of rounds of training
    
    curr_Y = Y
	for round in range(NUM_ROUNDS): 
		minimize_loss(EMI_graph, X, curr_Y)
		curr_Y = updatePolicy(EMI_RNN(X))
```

- `minimize_loss`: This function used above is a complete training procedure of
  `EMI_RNN` till convergence. The training is done against the current instance
level label information, `curr_Y`. The implementation of `minimize_loss`
checkpoints models in between the training phase and picks up the model with
the best validation accuracy at the end. That is, a form of early stopping is
implemented. The granularity or frequency of this check pointing process is
controlled by the `numIterations` parameter of `EMI_Driver.run()`. A checkpoint
is created after each iteration. Each iteration consists of a fixed number of
epochs specified by `numEpochs` argument to `EMI_Driver.run()`. Hence, the
total epochs for a round is `numIterations x numEpochs`. Note that the maximum
allowed checkpoints is restricted to 1000. This can be controlled by the
`max_to_keep` argument in `EMI_Driver`.
- `updatePolicy`: At the end of a training round, the label information of the
  training data is updated using an `updatePolicy`. Two update policies are
provided, one based on pruning negative examples and the other based on picking
up the top-k non negative examples for some fixed k. The policy to use is
specified by the `updatePolicy` argument to `EMI_Driver.run()`. The pruning
based update policy tries to identify the instances containing the signatures
by identifying and pruning out negative instances at either extremes of a bag.
The top-k policy picks a subsequence of positively predicted instance of length
at least k, and labels all other instances as negatives. Arguments for finer
control of both the policies are documented as part of `EMI_Driver`.

### Sessions in EMI_Driver

Since, the training procedure uses `tf.Saver` to save and restore models
multiple times during the training procedure, `EMI_Driver` requires close
control of the computation graph and the `tf.Session` running this graph.
Hence, sessions are handled by `EMI_Driver` internally. If access to the
current `tf.Session` is required, it should be obtained through
`EMI_Driver.getCurrentSession()` method *everytime*. Improper access of
sessions can lead to references to nodes in the computation graph becoming
invalid upon the graph being reset internally.

### Restoring trained models

It is possible to restore a trained model into a session from its checkpoint.
`EMI_Driver` exposes an easy to use way of achieving this through
`loadSavedGraphToNewSession` method. 

To use this method, first construct a new computation graph as you would
normally do and setup `EMI_Driver` with this computation graph. Then you can
call `loadSavedGraphToNewSession` method with the checkpoint of the model you
want to restore.

Additionally, for advanced use cases that requires more control over the graph
construction and restoring process, restoring a checkpoint directly during the
graph construction process itself is supported. This can be achieved by loading
the graph using `utils.GraphManager` and then passing this graph as
arguments `__init__` of `DataPipeline`, `EMI_RNN` and `EMI_Trainer`.

Finally, as an experimental feature, restoring model parameters from numpy
matrices is also supported. This is achieved by attaching `tf.assign`
operations to all the model tensors. Please have a look at `addAssignOps`
method of `DataPipeline`, `EMI_RNN` and `EMI_Trainer` for more information.

Please refer to `tf/examples/02_emi_lstm_initialization_and_restoring.npy` for
example usages.

## Evaluating the  trained model

### Accuracy

Since the trained model predicts on a input with fewer time steps while our test
data has labels for longer inputs (i.e. bag level labels), evaluating the
accuracy of the trained model is not straightforward. We perform the
evaluation as follows:

1. Divide the test data also into sub-instances; similar to what was done for
the train data.
2. Obtain sub-instance level predictions for each bag in the test data.
3. Obtain bag level predictions from sub-instance level predictions. For this,
we use our estimate of the length of the signature to estimate the expected
number of sub-instances that would be non negative - $k$ illustrated in the
figure. If a bag has $k$ consecutive sub-instances with the same label, that
becomes the label of the bag. All other bags are labeled negative.
4. Compare the predicted bag level labels with the known bag level labels in
test data.

### Early Savings

Early prediction is accomplished by defining an early prediction policy method.
This method receives the prediction at each step of the learned RNN for a
sub-instance as input and is expected to return a predicted class and the
0-indexed step at which it made this prediction.  Please refer to the
`tf/examples/EMI-RNN` for concrete examples of the same.
