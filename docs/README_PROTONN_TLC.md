# ProtoNN: Compressed and accurate KNN for resource-constrained devices

ProtoNN ([paper](http://manikvarma.org/pubs/gupta17.pdf)) has been developed for machine learning applications where the intended footprint of the ML model is small. ProtoNN models have memory requirements that are several orders of magnitude lower than other modern ML algorithms. At prediction time, ProtoNN is fast, precise, and accurate. 

One example of a ubiquitous real-world application where such a model is desirable are resource-scarce devices such as an Internet of Things (IoT) sensor. To make real-time predictions locally on IoT devices, without connecting to the cloud, we need models that are just a few kilobytes large. ProtoNN shines in this setting, beating all other algorithms by a significant margin. 

## The model
Suppose a single data-point is D-dimensional. Suppose also that there are a total of L labels to predict. 

ProtoNN learns 3 parameters:
- A projection matrix W of dimension (d,\space D) projects the datapoints to a small dimension d
- m prototypes in the projected space, each d-dimensional: B = [B_1,\space B_2, ... \space B_m]
- m label vectors for each of the prototypes to allow a single prototype to store information for multiple labels, each L-dimensional: Z = [Z_1,\space Z_2, ... \space Z_m]

ProtoNN also assumes an RBF-kernel parametrized by a single parameter \gamma. Each of the three matrices are trained to be sparse. The user can specify the maximum proportion of entries that can be non-zero in each of these matrices using the parameters \lambda_W, \lambda_B and \lambda_Z:
- ||W||_0 < \lambda_W \cdot size(W)
- ||B||_0 < \lambda_B \cdot size(B)
- ||Z||_0 < \lambda_Z \cdot size(Z) 

## Effect of various parameters
The user presented with a model-size budget has to make a decision regarding the following 5 parameters: 
- The projection dimension d
- The number of prototypes m
- The 3 sparsity parameters: \lambda_W, \lambda_B, \lambda_Z
 
Each parameter requires the following number of non-zero values for storage:
- S_W: min(1, 2\lambda_W) \cdot d \cdot D
- S_B: min(1, 2\lambda_B) \cdot d \cdot m
- S_Z: min(1, 2\lambda_Z) \cdot L \cdot m

The factor of 2 is for storing the index of a sparse matrix, apart from the value at that index. Clearly, if a matrix is more than 50% dense (\lambda > 0.5), it is better to store the matrix as dense instead of incurring the overhead of storing indices along with the values. Hence the minimum operator. 
Suppose each value is a single-precision floating point (4 bytes), then the total space required by ProtoNN is 4\cdot(S_W + S_B + S_Z).

## Prediction
Given these parameters, ProtoNN predicts on a new test-point in the following manner. For a test-point X, ProtoNN computes the following L dimensional score vector:
Y_{score}=\sum_{j=0}^{m}\space \left(RBF_\gamma(W\cdot X,B_j)\cdot Z_j\right), where
RBF_\gamma (U, V) = exp\left[-\gamma^2||U - V||_2^2\right]
The prediction label is then \space max(Y_{score}). 

## Training 
While training, we are presented with training examples X_1, X_2, ... X_n along with their label vectors Y_1, Y_2, ... Y_n respectively. Y_i is an L-dimensional vector that is 0 everywhere, except the component to which the training point belongs, where it is 1.  For example, for a 3 class problem, for a data-point that belongs to class 2, Y=[0, 1, 0]. 

We optimize the l_2-square loss over all training points as follows:  \sum_{i=0}^{n} = ||Y_i-\sum_{j=0}^{m}\space \left(exp\left[-\gamma^2||W\cdot X_i - B_j||^2\right]\cdot Z_j\right)||_2^2. 
While performing stochastic gradient descent, we hard threshold after each gradient update step to ensure that the three memory constraints (one each for \lambda_W, \lambda_B, \lambda_Z) are satisfied by the matrices W, B and Z. 


## Parameters
- Projection Dimension (d): this is the dimension into which the data is projected
- Clustering Init: This option specifies whether the initialization for the prototypes is performed by clustering the entire training data (OverallKmeans), or clustering data-points belonging to different classes separately (PerClassKmeans). 
- Num Prototypes (m): This is the number of prototypes. This parameter is only used if Clustering Init is specified as OverallKmeans. 
- Num Prototypes Per Class (k): This is the number of prototypes per class. This parameter is only used if Clustering Init is specified as PerClassKmeans. On using it, m becomes L\cdot k where L is the number of classes. 
- gammaNumerator:
    - On setting gammaNumerator, the RBF kernel parameter \gamma is set as;
    - \gamma = (2.5 \cdot gammaNumerator)/(median(||B_j,W - X_i||_2^2))
- sparsity parameters (described in detail above): Projection sparsity (\lambda_W), Prototype Sparsity (\lambda_B), Label Sparsity(\lambda_Z).
- Batch size: Batch size for mini-batch stochastic gradient descent.
- Number of iterations: total number of optimization iterations.
- Epochs: Number of see-through's of the data for each iteration, and each parameter. 
- Seed: A random number seed which can be used to re-generate previously obtained experimental results. 
