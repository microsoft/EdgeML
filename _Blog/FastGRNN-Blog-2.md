---
layout: blogpost
title: Fast(G)RNN - Fast, Accurate, Stable and Tiny (Gated) Recurrent Neural Network (Part II)
postdate: 04 September, 2018
author: Aditya Kusupati
---

### FastGRNN

##### Why move away from FastRNN to FastGRNN?
Even after stable training, the expressiveness of FastRNN falls short of state-of-the-art gated architectures in performance. 
This is due to having the same scalar gate (dumb & constant gate) for all the hidden units over all the timesteps. The ideal scenario, like in LSTM/GRU is to have an input dependent and per hidden unit scalar gate for having more expressivity. FastGRNN is created incorporating all the things discussed along with a much lower memory and compute footprint (45x).

##### Base architecture: FastGRNN-LSQ
As mentioned in the earlier section the base architecture created is addressed as *FastGRNN-LSQ*. The choice of architecture is very intuitive and tries to reuse as much information as possible and have minimal memory and compute footprint.

FastGRNN-LSQ essentially uses the self-information from the update equation and reuses it in the gate with a different bias vector and a non-linearity. This ensures the compute is shared for both the update and the gate equations. The final hidden state update ie., the linear combination of gated update equation and the previously hidden state, ensure that the architecture is expressive enough to match the 
performance of state-of-the-art gated architectures.

The update equation $$\tilde {\v h}_{t}$$ and the gate equation $$\v z_{t}$$, therefore share the memory and compute. 
The final update equation $$\v h_{t}$$ can be interpreted as the gate acting as the forget gate for the previous 
hidden state and the affine transformed $$(1 - \v z_{t})$$ ie., $$(\zeta(1 - \v z_{t}) + \nu)$$ improves the model's 
capability and expressivity and helps it achieve the last few points of accuracy to beat or be on par with LSTM/GRU.

The paper shows that FastGRNN-LSQ is either better or in the leagues of LSTM/GRU across benchmark datasets while being up to 4x smaller than LSTM/GRU. 
Given that we have a model which is powerful, small and elegant, can we make it very small and make it work on edge-devices?

##### FastGRNN Compression.
We use three components Low-Rank (L), Sparsity (S) and Quantization (Q) as part of our compression routine. 

Low-Rank parameterisation of weight matrices:
<div>
$$
\vec{W} = \vec{W}^1 (\vec{W}^2)^\top,\ \vec{U} = \vec{U}^1 (\vec{U}^2)^\top
$$
</div>

$$\v W^1 \ \& \ \v W^2$$ are two low-rank matrices of the same rank used to re-parameterize $$\v W$$.
$$\v U^1 \ \& \ \v U^2$$ are two low-rank matrices of the same rank used to re-parameterize $$\v U$$.

Sparse weight matrices ($$s_w \ \& \ s_u$$ are the no:of non-zeros):
<div>
$$
\|\vec{W}^i\|_0 \leq s_w^i,\ \|\vec{U}^i\|_0 \leq s_u^i,\ i=\{1,2\}
$$
</div>

Byte Quantization: The parameter matrices are trained with 4-byte floats and are finally quantized to 1-byte integers. 
This directly gives a 4x compression and if done right (using approximate piecewise non-linearities) will result in pure integer arithmetic on edge-devices without floating point unit. Note that one can use much more effective compression pipelines like Deep Compression over and above this to achieve further compression. For example, clustering of weights and generation of codebooks can result in up to 2-3x compression on FastGRNN.

##### Training routine to induce compression
The training routine has 3 phases:

-  Dense Training Phase
-  Sparse Training with IHT to find optimal support
-  Sparse Re-training on a fixed support

In the first stage of the training, FastGRNN is trained for one-third epochs with the model using non-convex optimizers. 
This stage of optimization ignores the sparsity constraints on the parameters and learns a low-rank representation of the parameters.

FastGRNN is next trained for the next third of the epochs using a non-convex optimizer, projecting the parameters onto the space of sparse low-rank matrices after every few batches while maintaining support between two consecutive projection steps. This stage, using non-convex optimizers with Iterative Hard Thresholding (IHT), helps FastGRNN identify the correct support for parameters $$(\vec{W}^i,\vec{U}^i)$$.

Lastly, FastGRNN is trained for the last third of the epochs with non-convex optimizer while freezing the support set of the parameters. Early stopping is often deployed in stages (II) and (III) to obtain the best models within budget constraints and this acts as a regularizer.

This training routine was developed during the time of DSD-training and uses much higher sparsity constraints and shows that the routine can 
maintaining the performance while reducing the memory and compute footprint.

##### Final Optimisation - Integer arithmetic
Standard floating point operations are 4x more expensive than 1-byte integer operations on edge-devices without Floating point unit. This most of the times will result in slow inferences. In most of the RNN architectures, the expensive step is floating point operations and especially the exponentiation in the non-linearities like tanh and sigmoid. This can be circumvented using the piece-wise linear approximations for the non-linearities and using them in the models instead of the original ones during training and thereby during prediction. 

The approximate or quantized non-linearities are most of the times simple conditionals and integer operations and when trained as part of the model
maintain the accuracies and the final model after byte-quantization will be tailor-made for all integer operations.

### Results

##### Datasets
<p align="center">
  <img src="{{ site.baseurl }}/img/algorithms/fastgrnn/datasets.png" height="100%" width="100%">
</p>

**Google-12 & Google-30:** Google Speech Commands dataset contains 1 second long utterances of 30 short words (30 classes) sampled at 16KHz. Standard log Mel-filter-bank featurization with 32 filters over a window size of 25ms and stride of 10ms gave 99 timesteps of 32 filter responses for a 1-second audio clip. For the 12 class version, 10 classes used in Kaggle's [Tensorflow Speech Recognition challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge) were used and remaining two classes were noise and background sounds (taken randomly from remaining 20 short word utterances). Both the datasets were zero mean - unit variance normalized during training and prediction.

**Wakeword-2:** Wakeword-2 consists of 1.63 second long utterances sampled at 16KHz. This dataset was featurized in the same way as the Google Speech Commands dataset and led to 162 timesteps of 32 filter responses. The dataset was zero mean - unit variance normalized during training and prediction.

**[HAR-2](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones):** Human Activity Recognition (HAR) dataset was collected from an accelerometer and gyroscope on a Samsung Galaxy S3 smartphone. The features available on the repository were directly used for experiments. The 6 activities were merged to get the binarized version. The classes {Sitting, Laying, Walking_Upstairs} and {Standing, Walking, Walking_Downstairs} were merged to obtain the two classes. The dataset was zero mean - unit variance normalized during training and prediction.

**[DSA-19](https://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities):** This dataset is based on Daily and Sports Activity (DSA) detection from a resource-constrained IoT wearable device with 5 Xsens MTx sensors having accelerometers, gyroscopes and magnetometers on the torso and four limbs. The features available on the repository were used for experiments. The dataset was zero mean - unit variance normalized during training and prediction.

**Yelp-5:**  Sentiment Classification dataset based on the [text reviews](https://www.yelp.com/dataset/challenge). The data consists of 500,000 train points and 500,000 test points from the first 1 million reviews. Each review was clipped or padded to be 300 words long. The vocabulary consisted of 20000 words and 128-dimensional word embeddings were jointly trained with the network.

**Penn Treebank:** 300 length word sequences were used for word level language modeling task using Penn Treebank (PTB) corpus. The vocabulary consisted of 10,000 words and the size of trainable word embeddings was kept the same as the number of hidden units of architecture.

**Pixel-MNIST-10:** Pixel-by-pixel version of the standard [MNIST-10 dataset](http://yann.lecun.com/exdb/mnist/). The dataset was zero mean - unit variance normalized during training and prediction.


##### Accuracy and Model Size Comparision:
Apart from the tables in the paper, have a look at the charts below to see that:

-  FastGRNN is at most 1.13% lower than the state-of-the-art but it can be up to 45x smaller.
-  FastGRNN-LSQ has almost similar performance as state-of-the-art with up to 4.5x smaller size.
-  FastRNN is better all the Unitary techniques in 6 of the 8 datasets.
-  Spectral RNN is the best unitary technique.
-  FastGRNN is up to 45x smaller than unitary techniques is always higher in performance.


<p align="center">
  <img src="{{ site.baseurl }}/img/algorithms/fastgrnn/accVsGated.png" height="100%" width="100%">
</p>

<p align="center">
  <img src="{{ site.baseurl }}/img/algorithms/fastgrnn/accVsUnitary.png" height="100%" width="100%">
</p>

Penn Treebank Language Modelling with 1-layer RNN:

The results suggest that FastGRNN, FastGRNN-LSQ, and FastRNN all have higher train perplexity scores while having good test perplexity scores. This might suggest that the proposed architectures are avoiding overfitting due to lesser parameters in general.
<p align="center">
  <img src="{{ site.baseurl }}/img/algorithms/fastgrnn/PTB.png" height="60%" width="60%">
</p>

##### Model Size vs Accuracy in 0-64KB:
Both FastGRNN and FastGRNN-LSQ are always the best possible models in the regime and this resonates with edge devices due to their RAM and flash limitations.
<p align="center">
  <img src="{{ site.baseurl }}/img/algorithms/fastgrnn/accVsModelSize.png" height="100%" width="100%">
</p>

##### Analysis of each component of compression in FastGRNN:
The effect of Low-rank is surprising as FastGRNN-SQ generally gains accuracy over 
FastGRNN-LSQ and thereby gives the required boost to nullify the loss of accuracy due to the other two components ie., sparsity and quantization. Sparsity and quantization result in a slight drop in accuracy and together account up to 1.5% drop in performance.
<p align="center">
  <img src="{{ site.baseurl }}/img/algorithms/fastgrnn/effectCompression.png" height="100%" width="100%"> 
</p>

##### The edge device deployment and inference times:
The models were deployed on two popular IoT boards Arduino MKR1000 and Arduino Due. 
Unfortunately, the experiments on Arduino UNO were not possible as no other models except FastGRNN were small enough to be burnt onto the flash (32KB) and need more working RAM (2KB). 

FastGRNN-Q is the model without quantization and no integer arithmetic. Given the boards 
don't have Floating Point Unit, one can observe that FastGRNN (the model with all integer arithmetic and quantized weights) 
is 4x faster during prediction on the edge-device. FastGRNN was 25-45x faster than UGRNN 
(smallest state-of-the-art gated RNN) and 57-132x faster than Spectral RNN (best and one of the smaller Unitary RNN)

<p align="center">
  <img src="{{ site.baseurl }}/img/algorithms/fastgrnn/edgePrediction.png" height="100%" width="100%">
</p>

### Conclusion
This work studies the FastRNN algorithm for addressing the issues of inaccurate training and inefficient prediction in RNNs. FastRNN develops a peephole connection architecture with the addition of two extra scalar parameters to address this problem. It then builds on FastRNN to develop a novel gated architecture, FastGRNN, which reuses RNN matrices in the gating unit. Further compression in the model size of FastGRNN is achieved by allowing the parameter matrices to be low-rank, sparse and quantized. The performance of FastGRNN and FastRNN is benchmarked on several datasets and are shown to achieve state-of-the-art accuracies whilst having up to 45x smaller model as compared to leading gated RNN techniques.

### Code and Usage:
The code is public and is part of the [Microsoft EdgeML Repository](https://github.com/Microsoft/EdgeML). The new cells FastRNNCell and FastGRNNCell can be found as part of `tf.edgeml.graph.rnn` and can be used in a plug and play fashion in place of any inbuilt Tensorflow RNN Cell as `tf.edgeml.graph.rnn.FastRNNCell` and `tf.edgeml.graph.rnn.FastGRNNCell`. 
Both the cells have multiple arguments for fine-tuning and appropriate selection of the hyperparameters for the given problem.
The `tf.edgeml.trainer.fastTrainer.FastTrainer` takes in the created cell object and run the 3-phase training routine to ensure optimal compression. Note that 
even though FastGRNN is the architecture that uses the 3-phase training, as it is independent of the architecture, the current code supports 
the trainer for both FastGRNN and FastRNN ensuring that even FastRNN can be compressed further if required.
Both of these are packaged as a single end-point user script and can be found along with example on a public dataset as part of [FastCells](https://github.com/Microsoft/EdgeML/tree/master/tf/examples/FastCells).
