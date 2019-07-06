---
layout: blogpost
title: Bonsai&#58; Strong, Shallow and Sparse Non-linear Tree Based Classifier
postdate: 04 September, 2018
author: Aditya Kusupati
---

### TL;DR
Bonsai is a new tree model for supervised learning tasks such as binary and
multi-class classification, regression, ranking, etc. Bonsai learns a single,
shallow, sparse tree with powerful predictors at internal and leaf nodes. This
allows Bonsai to achieve state-of-the-art prediction accuracies while making
predictions efficiently in microseconds to milliseconds (depending on processor
speed) using models that fit in a few KB of memory. Bonsai can be trained in
the cloud or on your laptop, but can then make predictions locally on tiny
resource-constrained devices without needing cloud connectivity.

Bonsai has been deployed successfully on microcontrollers tinier than a grain
of rice such as the ARM Cortex M0 with just 2 KB RAM. Bonsai can also make
predictions accurately and efficiently on the tiniest of IoT boards such as the
Arduino Pro Mini based on an 8 bit Atmel ATmega328P microcontroller operating
at 8 MHz without any floating point support in hardware, with 2 KB RAM and 32
KB read-only flash memory. Bonsai can also fit in the L1 cache of processors
found in mobiles, tablets, laptops, and servers for low-latency applications.


Bonsai can also be useful for switching to a smaller, cheaper and more
energy-efficient form factor such as from a Raspberry Pi 3 to an Arduino Pro
Mini. Finally, Bonsai also generalizes to other resource-constrained scenarios
beyond the Internet of Things and can be used on laptops, servers and the cloud
for low-latency applications and to bring down energy consumption and operating
costs. [Bonsai end-to-end script](https://github.com/Microsoft/EdgeML/tree/master/tf/examples/Bonsai).

### Introduction, Motivation, and Ideas:
##### Tree algorithms:
Tree algorithms are general and can be used for classification, regression, ranking and other problems commonly found in the IoT setting. Even more importantly, they are ideally suited to IoT applications as they can achieve good prediction accuracies with prediction times and energies that are logarithmic in the number of training points. Unfortunately, they do not directly fit on tiny IoT devices as their space complexity is linear rather than logarithmic. 
In particular, learning shallow trees, or aggressively pruning deep trees or large ensembles, to fit in just a 
few KB often leads to poor prediction accuracy.

Given the capabilities of Tree Algorithms in general, Bonsai targets three key points to make them 
work in IoT scenario.

##### The main components:
There are *3* vital ideas in Bonsai:

-  Bonsai learns a single, shallow, sparse tree so as to reduce model size but with powerful nodes for accurate prediction.
-  Both internal and leaf nodes in Bonsai together make non-linear predictions. They contribute to the final prediction similar to an ensemble.
-  Bonsai learns a sparse matrix which projects all data points into a low-dimensional space in which the tree is learned.

##### Stronger nodes in the Tree:
Both leaf and internal nodes of Bonsai use two *1-vs-all* classifiers combined in a non-linear fashion to make the nodes much 
powerful than the regular tree-based algorithms.

Assume that $$\hat{\v x}$$ is the input for a given node. Each node has two matrices (*1-vs-all* classifiers) $$\v W \ \& \ \v V$$. 
The prediction ($$\v p$$) at each node is ($$\sigma$$ is a scalar and acts the sigmoid sharpness paramter for $$\tanh$$):
<div>
$$
\v p = \v W \hat{\v x}\odot \tanh(\sigma\v V \hat{\v x}) 
$$
</div>

One can find any optimal non-linear combination if deemed right for the task. Hence this is tunable.

##### Contribution of internal and leaf nodes to prediction:
Bonsai’s overall prediction for a point is the sum of the individual node predictions along the path traversed by the point. Path-based prediction allows Bonsai to accurately learn non-linear decision boundaries while sharing parameters along paths to further reduce model size.

If $$\v p_{i}$$ is the prediction from $$i^{th}$$ node of Bonsai. *0* indexed at the root and the children follow the numbering of *2i+1* and *2i+2*. As Bonsai is a balanced binary tree, every path to the leaf node has the same no: of 
nodes ie., the depth of the tree.

The final prediction of Bonsai comes from the aggregate from internal nodes and leaf node in the path taken by the point. 

Final prediction $$y(\v x)$$ is given by:
<div>
$$
y(\v x) = \sum_{i=0}^{k-1}I_i(\v x)\v W_{i}\hat{\v x}\odot \tanh(\sigma\v V_{i} \hat{\v x})
$$
</div>

$$I_i(\v x)$$ is the indicator function which states if the node lies in the path of the data point or not. $$k$$ is the total no: of nodes in the Bonsai tree. 

The indicator function is simulated by simple branching hyperplane at each internal node. Each internal node has branching hyperplane $$\v \theta_{i}$$.
At each internal node the point passes through it chooses the next node/child based on the sign of the scalar ie., $$\v \theta_{i}^\top \hat{\v x}$$. 
Hence the path is determined by following the data point using the branching hyper-planes at each internal node.

##### Sparse projection matrix to work in low-dimensional space:
Bonsai learns a sparse matrix ($$\v Z$$) which projects all data points into a low-dimensional space in which the tree is learned. 
This allows Bonsai to fit in a few KB of flash. Furthermore, the sparse projection is implemented in a streaming 
fashion thereby allowing Bonsai to tackle IoT applications where even a single feature vector might not fit in 2 KB of RAM.

So the $$\hat{\v x}$$ being discussed till now is actually generated using the sparse projection matrix $$\v Z$$. The projection dimension is 
generally very small when compared to the actual dimentionality of $$\v x$$, there by helping to learn paramter matrices in very low-dimensional space 
which inturn help in lower compute and model size.
<div>
$$
\hat{\v x} = \v Z \v x
$$
</div>

### Bonsai Training:
Rather than learning the Bonsai tree node by node in a greedy fashion, all nodes are learned jointly, along with the sparse projection matrix, so as to optimally allocate memory budgets to each node while maximizing prediction accuracy. Even though the Bonsai in the current formulation is non-differentiable (non-backpropable) 
it can be made back propagation friendly or differentiable using an annealed Indicator function which starts as a soft indicator function and finally converges to the aforementioned hard indicator ie., $$\v \theta_{i}^\top \hat{\v x}$$. This can be modeled using 

<div>
$$
I_i(\hat{\v x}) = I_{(\lceil{\frac{i}{2}}\rceil-1)}(\hat{\v x})\frac{1 + ((-1)^{i+1})\tanh(\sigma_{I}\v \theta_{i}^\top \hat{\v x})}{2}\\
I_{0}(\hat{\v x}) = 1
$$
</div>

For any node $$i$$ apart from the root node, the parent node is indexed by $$(\lceil{\frac{i}{2}}\rceil-1)$$. So this formulation depends on 
$$\sigma_{I}$$ value for the softness/hardness of the indicator/branching function. Essentially, this formulation suggests it is a 
probabilistic weight given to each node depending on the path the point takes. $$\sigma_{I}$$ is a scalar which is updated heuristically over training routine finally culminating at a very high positive value ensuring a hard indicator function. This is empirical and can be set anyway one wishes it to converge to hard branching function.

After making Bonsai differentiable, we employ a *3* phase joint training routine for all the parameters to ensure better performance 
while maintaining the required budget constraints.

##### Three phase training routine:
The training routine has 3 phases:

-  Dense Training Phase
-  Sparse Training with IHT to find optimal support
-  Sparse Re-training on a fixed support

In the first stage of the training, Bonsai is trained for one-third epochs with the model using non-convex optimizers. 
This stage of optimization ignores the sparsity constraints on the parameters and learns the dense parameter matrices.

Bonsai is next trained for the next third of the epochs using a non-convex optimizer, projecting the parameters onto the space of sparse low-rank matrices after every few batches while maintaining support between two consecutive projection steps. This stage, using non-convex optimizers with Iterative Hard Thresholding (IHT), helps Bonsai identify the correct support for parameters $$(\v W_{i},\v V_{i})$$.

Lastly, Bonsai is trained for the last third of the epochs with non-convex optimizer while freezing the support set of the parameters. Early stopping is often deployed in stages (II) and (III) to obtain the best models within budget constraints and this acts as a regularizer.

This training routine was developed during the time of DSD-training and uses much higher sparsity constraints and shows that the routine can 
maintaining the performance while reducing the memory and compute footprint.

##### Final Optimisation - Integer arithmetic
Standard floating point operations are 4x more expensive than 1-byte integer operations on edge-devices without Floating point unit. This most of the times will result in slow inferences. In most Bonsai, the expensive step is the floating point operations and especially the exponentiation in the non-linearity (tanh). This can be circumvented using the piece-wise linear approximations for the non-linearities and using them in the models instead of the original ones during training and thereby during prediction. 

The approximate or quantized non-linearities are most of the times simple conditionals and integer operations and when trained as part of the model
maintain the accuracies and the final model after byte-quantization will be tailor-made for all integer operations.

### Results:

One can have a look at the results in the the [ICML 2017](http://manikvarma.org/pubs/kumar17.pdf) publication as well as the [poster](https://github.com/Microsoft/EdgeML/wiki/files/BonsaiPoster.pdf) and [presentation](https://github.com/Microsoft/EdgeML/wiki/files/BonsaiResults.pptx) ([video](https://vimeo.com/237274524))

##### Auxiliary observations:
The join training of the projection matrix and the three-phase training are very vital and actually, make a significant difference if the projection matrix was replaced by PCA and the sparsification was done after the complete training. You can find a table on this as part of the paper.

### Conclusions:
The paper proposed an alternative IoT paradigm, centric
to the device rather than the cloud, where ML models run
on tiny IoT devices without necessarily connecting to the
cloud thereby engendering local decision-making capabilities.
The Bonsai tree learner was developed towards this
end and demonstrated to be fast, accurate, compact and
energy-efficient at prediction time. Bonsai was deployed
on the Arduino Uno board as it could fit in a few KB of
flash required only 70 bytes of writable memory for binary
classification and 500 bytes for a 62 class problem,
handled streaming features and made predictions in milliseconds
taking only milliJoules of energy. Bonsai’s prediction
accuracies could be as much as 30% higher as compared
to state-of-the-art resource-efficient ML algorithms
for a fixed model size and could even approach and outperform
those of uncompressed models taking many MB of
RAM. Bonsai achieved these gains by developing a novel
model based on a single, shallow, sparse tree learned in a
low-dimensional space. Predictions made by both internal
and leaf nodes and the sharing of parameters along
paths allowed Bonsai to learn complex non-linear decision
boundaries using a compact representation.

### Code and Usage:
The code is public and is part of the [Microsoft EdgeML Repository](https://github.com/Microsoft/EdgeML). The Bonsai Graph can be found as part of `tf.edgeml.graph.bonsai` and can be used in a plug and play fashion in place of any classifier in tensorflow as `tf.edgeml.graph.bonsai.Bonsai`. 
Bonsai graph has multiple arguments for fine-tuning and appropriate selection of the hyperparameters for the given problem.
The `tf.edgeml.trainer.bonsaiTrainer.BonsaiTrainer` takes in the created Bonsai graph object and run the 3-phase training routine to ensure optimal compression. 
Bonsai is packaged as a single end-point user script and can be found along with example on a public dataset as part of [Bonsai](https://github.com/Microsoft/EdgeML/tree/master/tf/examples/Bonsai).
