# Bonsai

Bonsai ([paper](http://proceedings.mlr.press/v70/kumar17a/kumar17a.pdf)) 
is a novel tree based algorithm for for efficient prediction on IoT devices – 
such as those based on the Arduino Uno board having an 8 bit ATmega328P microcontroller operating 
at 16 MHz with no native floating point support, 2 KB RAM and 32 KB read-only flash.

    Bonsai maintains prediction accuracy while minimizing model size and prediction costs by: 
        (a) developing a tree model which learns a single, shallow, sparse tree with powerful nodes; 
        (b) sparsely projecting all data into a low-dimensional space in which the tree is learnt; 
        (c) jointly learning all tree and projection parameters.

Experimental results on multiple benchmark datasets demonstrate that Bonsai can make predictions in milliseconds even on slow microcontrollers, 
can fit in KB of memory, has lower battery consumption than all other algorithms while achieving prediction accuracies that can be as much as 
30% higher than state-of-the-art methods for resource-efficient machine learning.

Bonsai is also shown to generalize to other resource constrained settings beyond IoT 
by generating significantly better search results as compared to Bing’s L3 ranker when the model size is restricted to 300 bytes.

## Algorithm

Bonsai learns a balanced tree of user speciﬁed height `h`.

    The parameters that need to be learnt include: 
        (a) Z: the sparse projection matrix; 
        (b) θ = [θ1,...,θ2h−1]: the parameters of the branching function at each internal node;
        (c) W = [W1,...,W2h+1−1] and V = [V1,...,V2h+1−1]:the predictor parameters at each node

We formulate a joint optimization problem to train all the parameters using a training routine which is as follows.

    It has 3 Phases:
        (a) Unconstrained Gradient Descent: Train all the parameters without having any Budget Constraint
        (b) Iterative Hard Thresholding (IHT): Applies IHT constantly while training
        (c) Training with constant support: After the IHT phase the support(budget) for the parameters is fixed and are trained
We use simple Batch Gradient Descent as the solver with Armijo rule as the step size selector.

## Prediction

When given an input fearure vector X, Bonsai gives the prediction as follows :

    (a) We project the data onto a low dimensional space by computing x^ = Zx.
    (b) The final bonsai prediction score is the sum of the non linear scores ( wx^ * tanh(sigma*vx^) ) predicted by each of the individual nodes along the path traversed by the Bonsai tree.


## Parameters and HyperParameters

    pd   : Projection Dimension. (Default: 10 Try: [5, 20, 30, 50]) 
    td   : Depth of the Bonsai tree. (Default: 3 Try: [2, 4, 5])
    s    : sigma = parameter for sigmoid sharpness  (Default: 1.0 Try: [3.0, 0.05, 0.005] ).

    rw   : lambda_W = regularizer for classifier parameter W  (Default: 0.0001 Try: [0.01, 0.001, 0.00001]).
    rTheta  : lambda_Theta = regularizer for kernel parameter Theta  (Default: 0.0001 Try: [0.01, 0.001, 0.00001]).
    rv   : lambda_V = regularizer for kernel parameters V  (Default: 0.0001 Try: [0.01, 0.001, 0.00001]).
    rz   : lambda_Z = regularizer for kernel parameters Z  (Default: 0.00001 Try: [0.001, 0.0001, 0.000001]).

    Use Sparsity Params to vary your model Size
    sw  : sparsity_W = sparsity for classifier parameter W  (Default: For Binaray 1.0 else 0.2 Try: [0.1, 0.3, 0.4, 0.5]).
    sTheta  : sparsity_Theta = sparsity for kernel parameter Theta  (Default: For Binaray 1.0 else 0.2 Try: [0.1, 0.3, 0.4, 0.5]).
    sv  : sparsity_V = sparsity for kernel parameters V  (Default: For Binaray 1.0 else 0.2 Try: [0.1, 0.3, 0.4, 0.5]).
    sz  : sparsity_Z = sparsity for kernel parameters Z  (Default: 0.2 Try: [0.1, 0.3, 0.4, 0.5]).

    iter   : [Default: 40 Try: [100, 30, 60]] Number of passes through the dataset.
