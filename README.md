## Edge Machine Learning

This repository provides code for machine learning algorithms for edge devices
developed at [Microsoft Research
India](https://www.microsoft.com/en-us/research/project/resource-efficient-ml-for-the-edge-and-endpoint-iot-devices/). 

Machine learning models for edge devices need to have a small footprint in
terms of storage, prediction latency, and energy. One example of a ubiquitous
real-world application where such models are desirable is resource-scarce
devices and sensors in the Internet of Things (IoT) setting. Making real-time
predictions locally on IoT devices without connecting to the cloud requires
models that fit in a few kilobytes.

This repository contains algorithms that shine in this setting in terms of both model size and compute, namely:
 - **Bonsai**: Strong and shallow non-linear tree based classifier.
 - **ProtoNN**: **Proto**type based k-nearest neighbors (k**NN**) classifier. 
 - **EMI-RNN**: Training routine to recover the critical signature from time series data for faster and accurate RNN predictions.
 - **FastRNN & FastGRNN - FastCells**: **F**ast, **A**ccurate, **S**table and **T**iny (**G**ated) RNN cells.
 - **SeeDot**: Floating-point to fixed-point quantization tool.
 
These algorithms can train models for classical supervised learning problems
with memory requirements that are orders of magnitude lower than other modern
ML algorithms. The trained models can be loaded onto edge devices such as IoT
devices/sensors, and used to make fast and accurate predictions completely
offline.

The `tf` directory contains code, examples and scripts for all these algorithms
in TensorFlow. The `pytorch` directory contains code, examples and scripts for all these algorithms
in PyTorch. The `cpp` directory has training and inference code for Bonsai and
ProtoNN algorithms in C++. Please see install/run instruction in the Readme
pages within these directories. The `applications` directory has code/demonstrations
of applications of the EdgeML algorithms. The `Tools/SeeDot` directory has the
quantization tool to generate fixed-point inference code.  

For details, please see our [wiki
page](https://github.com/Microsoft/EdgeML/wiki/) and our ICML'17 publications
on [Bonsai](docs/publications/Bonsai.pdf) and
[ProtoNN](docs/publications/ProtoNN.pdf) algorithms, NeurIPS'18 publications on
[EMI-RNN](docs/publications/emi-rnn-nips18.pdf) and
[FastGRNN](docs/publications/FastGRNN.pdf), PLDI'19 publication on 
[SeeDot](docs/publications/SeeDot.pdf).


Core Contributors:
  - [Aditya Kusupati](https://adityakusupati.github.io/)
  - [Ashish Kumar](https://ashishkumar1993.github.io/)
  - [Chirag Gupta](https://aigen.github.io/)
  - [Don Dennis](https://dkdennis.xyz)
  - [Harsha Vardhan Simhadri](http://harsha-simhadri.org)
  - [Shishir Patil](https://shishirpatil.github.io/)
  - [Sridhar Gopinath](http://www.sridhargopinath.in/)

We welcome contributions, comments, and criticism. For questions, please [email
us](mailto:edgeml@microsoft.com).

[People](https://github.com/Microsoft/EdgeML/wiki/People/) who have contributed
to this
[project](https://www.microsoft.com/en-us/research/project/resource-efficient-ml-for-the-edge-and-endpoint-iot-devices/).

If you use the EdgeML library in your projects or publications, please do cite us using the following BibTex:

```
@software{edgeml01,
   author = {{Dennis, Don Kurian and Gopinath, Sridhar and Gupta, Chirag and
      Kumar, Ashish and Kusupati, Aditya and Patil, Shishir G and Simhadri, Harsha Vardhan}},
   title = {{EdgeML: Machine Learning for resource-constrained edge devices}},
   url = {https://github.com/Microsoft/EdgeML},
   version = {0.1},
}
```

### Microsoft Open Source Code of Conduct This project has adopted the
[Microsoft Open Source Code of
Conduct](https://opensource.microsoft.com/codeofconduct/). For more information
see the [Code of Conduct
FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact
[opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional
questions or comments.
