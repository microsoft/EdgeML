## The Edge Machine Learning library

This repository provides code for machine learning algorithms for edge devices
developed at [Microsoft Research
India](https://www.microsoft.com/en-us/research/project/resource-efficient-ml-for-the-edge-and-endpoint-iot-devices/). 

Machine learning models for edge devices need to have a small footprint in
terms of storage, prediction latency, and energy. One instance of where such 
models are desirable is resource-scarce devices and sensors in the Internet 
of Things (IoT) setting. Making real-time predictions locally on IoT devices 
without connecting to the cloud requires models that fit in a few kilobytes.

### Contents
Algorithms that shine in this setting in terms of both model size and compute, namely:
 - **Bonsai**: Strong and shallow non-linear tree based classifier.
 - **ProtoNN**: **Proto**type based k-nearest neighbors (k**NN**) classifier. 
 - **EMI-RNN**: Training routine to recover the critical signature from time series data for faster and accurate RNN predictions.
 - **FastRNN & FastGRNN - FastCells**: **F**ast, **A**ccurate, **S**table and **T**iny (**G**ated) RNN cells.

These algorithms can train models for classical supervised learning problems
with memory requirements that are orders of magnitude lower than other modern
ML algorithms. The trained models can be loaded onto edge devices such as IoT
devices/sensors, and used to make fast and accurate predictions completely
offline.

A tool that adapts models trained by above algorithms to be inferred by fixed point arithmetic.
 - **SeeDot**: Floating-point to fixed-point quantization tool.

Applications demonstrating usecases of these algorithms.

### Organization
 - The `edgeml_tf` directory contains the graphs and models in TensorFlow,
	and `examples/tf` contains examples and scripts that illustrate their usage.
 - The `edgeml` directory contains the graphs and models in PyTorch,
	and `examples/pytorch` contains examples and scripts that illustrate their usage.
 - The `cpp` directory has training and inference code for Bonsai and
	ProtoNN algorithms in C++. Please see install/run instruction in the Readme
	pages within these directories.
 - The `applications` directory has code/demonstrations of applications of the EdgeML algorithms. 
 - The `Tools/SeeDot` directory has the quantization tool to generate fixed-point inference code.  

### Details and project pages
For details, please see our
 [project page](https://microsoft.github.io/EdgeML/),
 [wiki](https://github.com/Microsoft/EdgeML/wiki/), and
 [Microsoft Research page](https://www.microsoft.com/en-us/research/project/resource-efficient-ml-for-the-edge-and-endpoint-iot-devices/).
our ICML'17 publications on [Bonsai](docs/publications/Bonsai.pdf) and
[ProtoNN](docs/publications/ProtoNN.pdf) algorithms, 
NeurIPS'18 publications on [EMI-RNN](docs/publications/emi-rnn-nips18.pdf) and
[FastGRNN](docs/publications/FastGRNN.pdf),
and PLDI'19 publication on [SeeDot](docs/publications/SeeDot.pdf).


Checkout the [ELL](https://github.com/Microsoft/ELL) project which can
provide optimized binaries for some of the ONNX models trained by this library.

### Contributors:
Code for algorithms, applications and tools here was contributed by:
  - [Don Dennis](https://dkdennis.xyz)
  - [Sridhar Gopinath](http://www.sridhargopinath.in/)
  - [Chirag Gupta](https://aigen.github.io/)
  - [Ashish Kumar](https://ashishkumar1993.github.io/)
  - [Aditya Kusupati](https://adityakusupati.github.io/)
  - [Shishir Patil](https://shishirpatil.github.io/)
  - [Harsha Vardhan Simhadri](http://harsha-simhadri.org)

[People](https://github.com/Microsoft/EdgeML/wiki/People/) who have contributed to this project. 
New contributors welcome.

Please [email us](mailto:edgeml@microsoft.com) your comments, criticism, and questions.

If you use software from this library in your work, please cite us using this BibTex entry:

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
