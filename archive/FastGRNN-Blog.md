---
layout: blogpost
title: Fast(G)RNN - Fast, Accurate, Stable and Tiny (Gated) Recurrent Neural Network
postdate: 04 September, 2018
author: Aditya Kusupati
---

### TL;DR
*FastRNN* and *FastGRNN*, two RNN architectures (cells), together called
*FastCells*, are developed to address the twin RNN limitations of inaccurate/unstable
training and inefficient prediction. 

FastRNN provably stabilizes the RNN training which usually suffers from the infamous vanishing and exploding gradient
problems. 

FastGRNN extends over and above FastRNN and learns low-rank, sparse and quantized weight matrices whilst having novel, elegant and expressive Gated RNN update equations. This allows FastGRNN to achieve state-of-the-art 
prediction accuracies while making predictions in microseconds to milliseconds 
(depending on processor speed) using models that fit in a few KB of memory. 

FastGRNN is up to **45x** smaller and faster (inference on edge-devices) than
state-of-the-art RNN architectures (LSTM/GRU) whilst maintaining accuracies
on various benchmark datasets and FastRNN has provably stable training and
better performance when compared to the Unitary architectures which try to solve the same.

Fast(G)RNN can be trained in the cloud or on your laptop, but can then make 
predictions locally on tiny resource-constrained devices without needing cloud connectivity.
While the NIPS'18 publication talks only about 3-component compression in FastGRNN, the same pipeline can be seamlessly used for any RNN architecture and in our open source release, we extend the same to FastRNN. [FastCells end-to-end script](https://github.com/Microsoft/EdgeML/tree/master/tf/examples/FastCells).


### Introduction and Motivation
*FastRNN* and *FastGRNN* architectures were particularly inspired to tackle and get rid of three major problems:

- Costly and expert feature engineering techniques like FFT for ML classifiers in Time-series regime.
- Unstable RNN Training due to vanishing and exploding gradient problem.
- Expensive RNN models and compute footprint for inference on edge-devices.

##### Solution to expensive feature engineering - Deep Learning? and Why RNNs?
The major motivation comes from the fact that most of the feature engineering techniques 
involved in the Time-series classification are expensive like FFT, which is 
the bottleneck if the edge-device doesn't have DSP support, and also involve the investment of 
experts' time to understand and craft the ideal features for the task.

Deep Learning has proved over last few years that one can incorporate featurization 
as part of the model cost and try to learn compact models which can be used on raw data 
while having the state-of-the-art accuracies.

Recurrent Neural Networks (RNNs) are the compact Neural Network models that have 
capacity to harness temporal information in a time-series setting and make effective/accurate 
predictions.

<div>
$$
\v h_{t} = \tanh(\vec{W} \v x_t + \vec{U} \v h_{t-1} + \v b_h).
$$
</div>

$$\v W$$ is the input-to-hidden state transition matrix, $$\v U$$ is the hidden-to-hidden state transition matrix.
$$\v x_t$$ is the input at timestep $$t$$, $$\v h_{t}$$ is the hidden state at the end of timestep $$t$$. Note that 
the total number of timesteps is $$T$$ and the classification is done using a simple FC-layer on $$\v h_T$$.

Spending several weeks on finding the appropriate features for the [Gesture Pod] along 
with data visualization, revealed that multi-modal sensor data obtained from IoT devices 
can be modeled as time-series and hence RNNs might be the right way to circumvent the issues 
and painstaking feature engineering.

##### Solution to Ineffective/Unstable RNN Training - FastRNN 
Simple RNNs are plagued with inaccurate/unstable training for longer time-series sequences 
and it has been the reason for the advent of complex yet more stable and expressive models like

- *Gated Architectures*: LSTM, GRU, UGRNN etc.,
- *Unitary Architectures*: Unitary RNN, Orthogonal RNN, Spectral RNN etc.,

These architectures, however, have their own drawbacks and will be addressed as we go along.

FastRNN stabilizes the entire RNN training using at most *2* scalars. FastRNN is not a novel 
architecture but a simple improvement over the existing Leaky Units along with a rigorous analysis 
which show the stability of training along with generalization and convergence bounds.

<div>
$$
\vec{\tilde{h}}_{t}= \sigma(\vec{W} \v x_t+\vec{U}\v h_{t-1}+\v b_h), \\ 
\v h_{t} = \alpha \vec{\tilde{h}}_{t}+\beta \v h_{t-1}, \\
0 < \alpha, \beta < 1, \\
\sigma \in \text{ [relu, sigmoid or tanh]}.
$$
</div>

$$\v W$$ is the input-to-hidden state transition matrix, $$\v U$$ is the hidden-to-hidden state transition matrix.
$$\v x_t$$ is the input at timestep $$t$$, $$\v h_{t}$$ is the hidden state at the end of timestep $$t$$. 
$$\alpha$$ and $$\beta$$ are trainable parameters and generally parameterized using sigmoid function to ensure that 
they lie between $$0 \ \& \ 1$$.
Note that the total number of timesteps is $$T$$ and the classification is done using a simple FC-layer on $$\v h_T$$.

<p align="center">
  <img src="{{ site.baseurl }}/img/algorithms/fastgrnn/FastRNN.png">
</p>

##### Solution to expensive RNN models and inference on edge-devices - FastGRNN
Even though FastRNN stabilizes the RNN training, the model's expressivity is limited and relies on constant attention (scalar gating) for the new information and the running memory/ context across time-steps and across the hidden-units. 
This led to the creation of a Gated architecture named FastGRNN, which while being as accurate as 
state-of-the-art RNN models (LSTM/GRU) but is 45x smaller and faster (on edge-devices).

FastGRNN inherently consists of *3* components of compression over the base architecture:

- Low-Rank parameterization of weight matrices $$\v W$$ and $$\v U$$ (**L**)
- Sparse parameter matrices (**S**)
- Byte Quantized weights in parameter matrices (**Q**)

The base architecture without any compression is hereby referred to as *FastGRNN-LSQ* (read as minus LSQ). 
FastGRNN-LSQ is *4x* smaller and faster than LSTM for inference and has very small compute overhead when compared to Simple RNN. 

<div>
$$
\v z_t= \sigma(\vec{W} \v x_t+\vec{U}\v h_{t-1}+\v b_z), \\ 
\vec{\tilde{h}}_{t}= \tanh(\vec{W} \v x_t+\vec{U}\v h_{t-1}+\v b_h), \\ 
\v h_t=  \left(\zeta(\v 1-\v z_t) + \nu \right)\odot\vec{\tilde{h}}_{t}+\v z_t\odot \v h_{t-1},\\
0 < \zeta, \nu < 1, \\
\sigma \in \text{ [relu, sigmoid or tanh]}.
$$
</div>

$$\v W$$ is the input-to-hidden state transition matrix, $$\v U$$ is the hidden-to-hidden state transition matrix.
$$\v x_t$$ is the input at timestep $$t$$, $$\v h_{t}$$ is the hidden state at the end of timestep $$t$$. 
$$\vec{\tilde{h}}_{t}$$ is the simple RNN update equation and $$\v z_t$$ is the gate equation. 
$$\zeta$$ and $$\nu$$ are trainable parameters and generally parameterized using sigmoid function to ensure that 
they lie between $$0 \ \& \ 1$$.
Note that the total number of timesteps is $$T$$ and the classification is done using a simple FC-layer on $$\v h_T$$.
<p align="center">
  <img src="{{ site.baseurl }}/img/algorithms/fastgrnn/FastGRNN.png">
</p>

Upon using the 3 compression components it culminates at FastGRNN which is 45x smaller and faster (on edge-device) than 
the state-of-the-art LSTM/GRU models whilst maintaining similar accuracies.

### Mathematical arguments for RNN and FastRNN Training

##### Why is RNN training unstable?
Simple RNNs eventhough are theoretically powerful, their training faces great threats from 
vanishing and exploding gradient problem. Note that simple RNN is a special case of FastRNN when 
$$\alpha = 1 \ \& \  \beta = 0 $$
Mathematically the gradients of the weight matrices 
of RNN with respect to the loss function $$L$$ for binary classification where $$\v v$$ is the final classifier and $$T$$ total sequence length:

<div>
$$
\frac{\partial L}{\partial \v U} =  \sum_{t=0}^{T} \v D_t \left(\prod_{k = t}^{T-1}( \v U^\top \v D_{k+1} ) \right)(\nabla_{\v h_T}L)\v h_{t-1}^\top,\\
\frac{\partial L}{\partial \v W} =  \sum_{t=0}^{T} \v D_t \left(\prod_{k = t}^{T-1}( \v U^\top \v D_{k+1} ) \right)(\nabla_{\v h_T}L)\v x_{t}^\top,\\
\frac{\partial L}{\partial \v v} = \frac{(-\v v^\top \v h_T)y\exp{(-\v v ^\top \v h_T)}}{1+\exp{(-\v v ^\top \v h_T)}} \v h_T,\\
\nabla_{\v h_T}L = c(\v \theta)(-\v  v^\top \v h_T) \v v, c(\v \theta) = \frac{1}{1+\exp{(\v v^\top \v h_T)}}\\
\vec{D}_{k} = \text{diag}(\tanh'(\vec{W}\vec{x}_k+\vec{U}\vec{h}_{k-1}+\vec{b}_h))\\
$$  
</div>

Note that the critical term in above gradient is:

<div>
$$
M(\v U)=\prod_{k = t}^{T-1}( \v U^\top \v D_{k+1} )
$$
</div>

Even though $$\v D_{k+1}$$ is bounded, the $$M(\v U)$$ term is still a 
repeated multiplication of similar matrices. On looking at the eigenvalue decomposition followed by the repeated multiplication, one can observe that the singular values are exponentiated and this results in gradient vanishing in the smallest eigenvector direction while having gradient explode along the largest eigenvector direction. Also the term $$M(\v U)$$ 
can be very ill-conditioned, which summarises the above two concerns. These two potentially lead to no change in train loss or result in NaN for the usual Non-Convex Optimizers. 

<div>
$$
\kappa(M(\v U)) \leq (\max_k \frac{\|\v U^\top \v D_{k+1}\|}{\lambda_{min}(\v U^\top \v D_{k+1})})^{T-t} \\
\lambda_{min} \ \text{is the smallest singular value}
$$
</div>

Classically, the gradient explosion problem is tackled in various ways and the most famous one being gradient clipping, but this doesn't address the vanishing gradient problem. 
Unitary architectures claim to solve both the problems by using re-parameterization of $$\v U$$ matrix.

##### Do Unitary methods address this issue completely?
Unitary architectures rely on Unitary parameterization of $$\v U$$ matrix so as to stabilize training. 
Unitary parameterization implies that during the entire training phase $$\v U$$ is forced to be 
Unitary either by re-parameterization (most of the time expensive) or applying transforms during the gradient update (very expensive). 

There are a couple of papers which rely on hard unitary constraints like 
Unitary RNN (Arjosky et al., ICML 2016), Full Capacity Unitary RNN (Wisdom et al., NIPS 2016) and (Mhammedi et al., ICML 2017). 
The hard unitary constraints restrict the solution space drastically and potentially miss out the right weight matrices. To tackle this, soft unitary constraints were enforced, to span a larger space like 
Factorized RNN (Vorontsov et al., ICML 2017), Spectral RNN (Zhang et al., ICML 2018) and Kronecker Recurrent Units (Jose et al., ICML 2018).

The soft constraints improve the performance and training times, but their solution space is still restricted and needs very extensive grid search to find the appropriate hyper-parameters. Even with all of this, they still fall short of the state-of-the-art Gated RNNs like LSTM/GRU.
Their re-parameterization forces them to have higher hidden-units to reach considerably good performance and thereby increasing the model sizes 
making them unfit for Edge-devices.

As unitary matrices only focus on $$\v U$$, the fact that the coupled term $$(\v U^\top \v D_{k+1})$$ is the culprit is often overlooked. 
The coupled term actually prone to gradient vanishing when the non-linearity has gradients less than *1* (Ex: tanh and sigmoid). 
The stabilization solution provided by these methods is still flawed and still can suffer from vanishing gradient problem while effectively 
dodging the exploding gradient.

##### How does FastRNN stabilize RNN training?
As discussed earlier FastRNN is a very simple extension of Leaky Units, FastRNN has both $$\alpha \ \& \  \beta$$ trainable, which are already prevalent in the literature. 
But the mathematical analysis which at the end shows that at most *2* scalars are enough to stabilize RNN training is not extensively studied and perhaps FastRNN is the first one to do so. Note: Simple RNN is a special case of FastRNN where 
$$\alpha = 1 \ \& \  \beta = 0 $$.

Let us look at the gradient and the $$M(\v U)$$ terms in case of FastRNN with respect to the loss function $$L$$ for binary classification where $$\v v$$ is the final classifier and $$T$$ total sequence length:

<div>
$$
\frac{\partial L}{\partial \v U} = \alpha \sum_{t=0}^{T} \v D_t \left(\prod_{k = t}^{T-1}(\alpha \v U^\top \v D_{k+1} + \beta \v I) \right)(\nabla_{\v h_T}L)\v h_{t-1}^\top,\\
\frac{\partial L}{\partial \v W} = \alpha \sum_{t=0}^{T} \v D_t \left(\prod_{k = t}^{T-1}(\alpha \v U^\top \v D_{k+1} + \beta \v I) \right)(\nabla_{\v h_T}L)\v x_{t}^\top,\\
\frac{\partial L}{\partial \v v} = \frac{(-\v v^\top \v h_T)y\exp{(-\v v ^\top \v h_T)}}{1+\exp{(-\v v ^\top \v h_T)}} \v h_T, \\ 
\nabla_{\v h_T}L = c(\v \theta)(-\v  v^\top \v h_T) \v v, c(\v \theta) = \frac{1}{1+\exp{(\v v^\top \v h_T)}}\\
\vec{D}_{k} = \text{diag}(\sigma'(\vec{W}\vec{x}_k+\vec{U}\vec{h}_{k-1}+\vec{b}_h))\\
$$
</div>

The critical term of the gradient:
<div>
$$
M(\v U)=\prod_{k = t}^{T-1}(\alpha \v U^\top \v D_{k+1} + \beta \v I)
$$
</div>

Most of the times one can use the fact that $$\beta \approx 1 - \alpha$$ and use a single parameter to stabilize the entire 
training. Looking at the bounds on largest singular value (2-norm), smallest singular value and condition number of $$M(\v U)$$:

<div>
$$
\lambda_{max}(M(\v U)) \leq(\beta+\alpha\max_k \|\v U^\top \v D_{k+1}\|)^{T-t}, \\
\lambda_{min}(M(\v U)) \geq(\beta+\alpha\max_k \|\v U^\top \v D_{k+1}\|)^{T-t}, \\
\kappa(M(\v U))\leq \frac{(1+\frac{\alpha}{\beta}\max_k \|\v U^\top \v D_{k+1}\|)^{T-t}}{(1-\frac{\alpha}{\beta}\max_k \|\v U^\top \v D_{k+1}\|)^{T-t}}.
$$
</div>

All the above terms start to well-behave with the simple theoretical setting of $$\beta=1-\alpha$$ and $$\alpha=\frac{1}{T \max_k \|\v U^T\v D_{k+1}\|}$$.
One can observe that this setting leads to:

<div>
$$
\lambda_{max}(M(\v U)) = \bigO{1}, \\
\lambda_{min}(M(\v U)) = \Om{1}, \\
\kappa(M(\v U)) = \bigO{1}.
$$
</div>

This shows that FastRNN can probably avoid both vanishing and exploding gradient problems while ensuring the $$M(\v U)$$ term is well conditioned. Empirically, FastRNN training ensures $$\alpha$$ behaves as predicted and is $$\alpha = \bigO{1/T}$$. FastRNN outperforms most of the Unitary methods across most of the benchmark datasets used 
across speech, NLP, activity and image classification as shown in the paper.

##### Any guarantees for FastRNN in terms of Convergence and Generalization?
It turns out that FastRNN has polynomial bounds in terms of length of the time-series sequence ($$T$$). Using the 
theoretical setting that $$\alpha = \bigO{1/T}$$ one can show that by using randomized stochastic gradient descent with no 
assumptions on the data distributions, FastRNN converges to a stationary point in upper bound of $$\Om{T^{6}}$$ iterations while the 
same analysis for simple RNN reveals an exponential upper bound.

Coming to the generalization error, while assuming that the loss function is 1-Lipschitz, FastRNN has the scaling polynomial in $$T$$ ie., $$\bigO{\alpha T}$$ 
whereas same analysis for simple RNN reveals an exponential dependency. Note that the upper bounds are being compared and comparing and proving of stronger lower bounds is an open problem that needs to be answered to analyze this further.


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
Both FastGRNN and FastGRNN-LSQ are always the best possible models in the regime and this resonates with Edge-devices due to their RAM and flash limitations.
<p align="center">
  <img src="{{ site.baseurl }}/img/algorithms/fastgrnn/accVsModelSize.png" height="100%" width="100%">
</p>

##### Analysis of each component of compression in FastGRNN:
The effect of Low-rank is surprising as FastGRNN-SQ generally gains accuracy over 
FastGRNN-LSQ and thereby gives the required boost to nullify the loss of accuracy due to the other two components ie., sparsity and quantization. Sparsity and quantization result in a slight drop in accuracy and together account up to 1.5% drop in performance.
<p align="center">
  <img src="{{ site.baseurl }}/img/algorithms/fastgrnn/effectCompression.png" height="100%" width="100%"> 
</p>

##### The Edge-device deployment and inference times:
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
