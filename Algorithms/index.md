---
layout: default
title: Algorithms and Tools
permalink: /Algorithms
---

The algorithms that are part of EdgeML are written in Tensorflow and PyTorch for Python.
They are hosted on [GitHub](https://github.com/Microsoft/EdgeML/).
Additionally, the repository also provides fast and scalable C++
implementations of Bonsai and ProtoNN. 

Usecases:
- **Bonsai** & **ProtoNN**: Can be used for traditional machine learning tasks with pre-computed features like gesture recongition ([Gesturepod](https://microsoft.github.io/EdgeML/Projects/GesturePod/instructable.html)), activity detection, image classification. They can also be used to replace bulky traditonal classifiers like fully connected layers, RBF-SVMs etc., in ML pipleines.
- **EMI-RNN** & **FastGRNN**: These complementary techniques can be applied on time-series classification tasks which require the models to learn new feature representations such as wakeword detection ([Key-word spotting](https://microsoft.github.io/EdgeML/Projects/WakeWord/instructable.html)), sentiment classification, activity recognition. FastGRNN can be used as a cheaper alternative to LSTM and GRU in deep learning pipleines while EMI-RNN provides framework for computational savings using multi-instance learning.
- **SeeDot**: 

A very brief introduction of these algorithms and tools is provided below.

1. **Bonsai**: *Bonsai* is a shallow and strong non-linear tree based classifier which is designed to solve traditional ML problem with 2KB sized models.

<span>
    [<a href="https://github.com/microsoft/EdgeML/tree/master/tf/examples/EMI-RNN">Code</a>]
    [<a href="https://youtu.be/l7PlPbWSbcc">Video</a>]
</span>

2. **ProtoNN**: *ProtoNN* is a prototype based k-nearest neighbors (kNN) classifier which is designed to solve traditional ML problem with 2KB sized models.
3. **EMI-RNN**: Training routine to recover critical signature from time series data for faster and accurate RNN predictions. EMI-RNN helps in speeding-up RNN inference up to 72x when compared to traditional implementations.
4. **FastRNN** & **FastGRNN**: Fast, Accurate, Stable and Tiny (Gated) RNN Cells which can be used instead of LSTM and GRU. FastGRNN can be up to 35x smaller and faster than LSTM and GRU while solving time series classification problems with models with size less than 10KB.
5. **SeeDot**: Floating-point to fixed-point quantization tool including a new language and compiler.

All the above algorithms and tools are aimed at enabling machine learning inference on the edge devices which form the back-bone for the Internet of Things (IoT).



Links to appropriate resources for each of the algorithms and tools:
1. **Bonsai** (ICML 2017) - [Paper](http://manikvarma.org/pubs/kumar17.pdf) | [Bibtex](http://manikvarma.org/pubs/selfbib.html#Kumar17) | [Cpp code](https://github.com/microsoft/EdgeML/tree/master/cpp) | [Tensorflow example](https://github.com/microsoft/EdgeML/tree/master/tf/examples/Bonsai) | [PyTorch example](https://github.com/microsoft/EdgeML/tree/master/pytorch/examples/Bonsai) | [Blog](https://blogs.microsoft.com/ai/ais-big-leap-tiny-devices-opens-world-possibilities/).
2. **ProtoNN** (ICML 2017) - [Paper](http://manikvarma.org/pubs/gupta17.pdf) | [Bibtex](http://manikvarma.org/pubs/selfbib.html#Gupta17) | [Cpp code](https://github.com/microsoft/EdgeML/tree/master/cpp) | [Tensorflow example](https://github.com/microsoft/EdgeML/tree/master/tf/examples/ProtoNN) | [PyTorch example](https://github.com/microsoft/EdgeML/tree/master/pytorch/examples/ProtoNN) | [Blog](https://blogs.microsoft.com/ai/ais-big-leap-tiny-devices-opens-world-possibilities/).
3. **EMI-RNN** (NeurIPS 2018) - [Paper](http://www.prateekjain.org/publications/all_papers/DennisPSJ18.pdf) | [Bibtex](https://dkdennis.xyz/static/emi-rnn-nips18-bibtex.html) | [Tensorflow example](https://github.com/microsoft/EdgeML/tree/master/tf/examples/EMI-RNN) | [PyTorch example](https://github.com/microsoft/EdgeML/tree/master/pytorch/examples/EMI-RNN).
4. **FastRNN** & **FastGRNN** (NeurIPS 2018) - [Paper](http://manikvarma.org/pubs/kusupati18.pdf) | [Bibtex](http://manikvarma.org/pubs/selfbib.html#Kusupati18) | [Tensorflow example](https://github.com/microsoft/EdgeML/tree/master/tf/examples/FastCells) | [PyTorch example](https://github.com/microsoft/EdgeML/tree/master/pytorch/examples/FastCells) | [Blog](https://www.microsoft.com/en-us/research/blog/fast-accurate-stable-and-tiny-breathing-life-into-iot-devices-with-an-innovative-algorithmic-approach/).
5. **SeeDot** (PLDI 2019) - [Paper](http://www.sridhargopinath.in/wp-content/uploads/2019/06/pldi19-SeeDot.pdf) | [Bibtex](https://dblp.org/rec/bibtex/conf/pldi/GopinathGSS19) | [Code](https://github.com/microsoft/EdgeML/tree/master/Tools/SeeDot).

 
<!-- Bonsai enables
high prediction accuracy while minimizing model size and prediction costs by a)
learning a single, shallow, sparse tree with powerful nodes, b) sparsely
projecting data into a low-dimensional space and c) jointly learning the tree
and projection parameters.

Get started with Bonsai through
<a style="color:var(--ms-green);"
href="https://github.com/Microsoft/EdgeML/tree/master/tf/examples/Bonsai">examples
</a>. Learn more about Bonsai from our
<a
href="http://manikvarma.org/pubs/kumar17.pdf"
style="color:var(--ms-green);">ICML '17 publication</a>.


## ProtoNN

*ProtoNN* is a multi-class classification algorithm, inspired by k-Nearest
Neighbor (kNN). Models generated by ProtoNN have several orders lowers storage
and prediction complexity. This is enabled by a) learning a small number of
prototypes to represent the entire training set, b) sparse low dimensional
projection of data and c) joint discriminative learning of the projection and
prototypes.

Get started with ProtoNN through
<a style="color:var(--ms-green);"
href="https://github.com/Microsoft/EdgeML/tree/master/tf/examples/ProtoNN">examples
</a>. Learn more about ProtoNN from our
<a
href="https://github.com/Microsoft/EdgeML/blob/master/docs/publications/ProtoNN.pdf"
style="color:var(--ms-green);">ICML '17 publication</a>.


## EMI-RNN

*EMI-RNN* is a Multiple Instance learning formulation for time-series data.
Early Multi Instance (EMI) RNN exploits the fact that a) *signature* of a
particular class is a small fraction of the overall data and b) class
signatures tend to be discernible early-on
to learn a model that not only enables early prediction but also improves
accuracy.

Get started with EMI-RNN through
<a style="color:var(--ms-green);"
href="https://github.com/Microsoft/EdgeML/tree/master/tf/examples/EMI-RNN">examples
</a>. Learn more about EMI-RNN from our
<a
href="https://github.com/Microsoft/EdgeML/blob/master/docs/publications/emi-rnn-nips18.pdf"
style="color:var(--ms-green);">NIPS '18 publication</a>.



## FastRNN and FastGRNN

*FastRNN* and *FastGRNN* are two novel RNN architectures (together called Fast
Cells) designed to address the twin RNN limiations of inaccurate training and 
inefficient prediction. FastRNN provably stabilizes the RNN training which 
usually suffers from vanishing and exploding gradients. FastGRNN is a gated RNN 
extended over FastRNN, that learns low-rank, sparse and quantized weight matrices 
resulting in models that are up to **35x** smaller and faster for inference compared 
to LSTM/GRU without compromising prediction accuracies.

Get started with Fast Cells through
<a style="color:var(--ms-green);"
href="https://github.com/Microsoft/EdgeML/tree/master/tf/examples/FastCells">examples.</a>
Learn more about Fast Cells from our
<a
href="http://manikvarma.org/pubs/kusupati18.pdf"
style="color:var(--ms-green);">NIPS '18 publication</a>. -->
