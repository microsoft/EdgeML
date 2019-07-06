---
layout: blogpost
title: Fast(G)RNN - Fast, Accurate, Stable and Tiny (Gated) Recurrent Neural Network (Part I)
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
making them unfit for edge devices.

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

### FastGRNN & Results
These two are part of the Part II of the Fast(G)RNN Blog. Link to [Part II]({{ site.baseurl }}/Blog/fastgrnn-blog-2).
Code: [FastCells](https://github.com/Microsoft/EdgeML/tree/master/tf/examples/FastCells).
