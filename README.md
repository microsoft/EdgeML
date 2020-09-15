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
 - **Shallow RNN**: A meta-architecture for training RNNs that can be applied to streaming data.
 - **FastRNN & FastGRNN - FastCells**: **F**ast, **A**ccurate, **S**table and **T**iny (**G**ated) RNN cells.
 - **DROCC**: **D**eep **R**obust **O**ne-**C**lass **C**lassfiication for training robust anomaly detectors.

These algorithms can train models for classical supervised learning problems
with memory requirements that are orders of magnitude lower than other modern
ML algorithms. The trained models can be loaded onto edge devices such as IoT
devices/sensors, and used to make fast and accurate predictions completely
offline.

A tool that adapts models trained by above algorithms to be inferred by fixed point arithmetic.
 - **SeeDot**: Floating-point to fixed-point quantization tool.

Applications demonstrating usecases of these algorithms:
 - **GesturePod**: Gesture recognition pipeline for microcontrollers.
 - **MSC-RNN**: Multi-scale cascaded RNN for analyzing Radar data.

### Organization
 - The `tf` directory contains the `edgeml_tf` package which specifies these architectures in TensorFlow,
   and `examples/tf` contains sample training routines for these algorithms.
 - The `pytorch` directory contains the `edgeml_pytorch` package which specifies these architectures in PyTorch,
   and `examples/pytorch` contains sample training routines for these algorithms.
 - The `cpp` directory has training and inference code for Bonsai and ProtoNN algorithms in C++.
 - The `applications` directory has code/demonstrations of applications of the EdgeML algorithms. 
 - The `tools/SeeDot` directory has the quantization tool to generate fixed-point inference code.  

Please see install/run instructions in the README pages within these directories.

### Details and project pages
For details, please see our
 [project page](https://microsoft.github.io/EdgeML/), 
 [Microsoft Research page](https://www.microsoft.com/en-us/research/project/resource-efficient-ml-for-the-edge-and-endpoint-iot-devices/),
the ICML '17 publications on [Bonsai](/docs/publications/Bonsai.pdf) and
[ProtoNN](/docs/publications/ProtoNN.pdf) algorithms, 
the NeurIPS '18 publications on [EMI-RNN](/docs/publications/emi-rnn-nips18.pdf) and
[FastGRNN](/docs/publications/FastGRNN.pdf),
the PLDI '19 publication on [SeeDot compiler](/docs/publications/SeeDot.pdf),
the UIST '19 publication on [Gesturepod](/docs/publications/GesturePod-UIST19.pdf), 
the BuildSys '19 publication on [MSC-RNN](/docs/publications/MSCRNN.pdf),
the NeurIPS '19 publication on [Shallow RNNs](/docs/publications/Sha-RNN.pdf),
and the ICML '20 publication on [DROCC](/docs/publications/drocc.pdf).


Also checkout the [ELL](https://github.com/Microsoft/ELL) project which can
provide optimized binaries for some of the ONNX models trained by this library.

### Contributors:
Code for algorithms, applications and tools contributed by:
  - [Don Dennis](https://dkdennis.xyz)
  - [Yash Gaurkar](https://github.com/mr-yamraj/)
  - [Sridhar Gopinath](http://www.sridhargopinath.in/)
  - [Sachin Goyal](https://saching007.github.io/)
  - [Chirag Gupta](https://aigen.github.io/)
  - [Moksh Jain](https://github.com/MJ10)
  - [Ashish Kumar](https://ashishkumar1993.github.io/)
  - [Aditya Kusupati](https://adityakusupati.github.io/)
  - [Chris Lovett](https://github.com/lovettchris)
  - [Shishir Patil](https://shishirpatil.github.io/)
  - [Oindrila Saha](https://github.com/oindrilasaha)
  - [Harsha Vardhan Simhadri](http://harsha-simhadri.org)

[Contributors](https://microsoft.github.io/EdgeML/People) to this project. New contributors welcome.

Please [email us](mailto:edgeml@microsoft.com) your comments, criticism, and questions.

If you use software from this library in your work, please use the BibTex entry below for citation.

```
@software{edgeml03,
   author = {{Dennis, Don Kurian and Gaurkar, Yash and Gopinath, Sridhar and Goyal, Sachin 
              and Gupta, Chirag and Jain, Moksh and Kumar, Ashish and Kusupati, Aditya and 
              Lovett, Chris and Patil, Shishir G and Saha, Oindrila and Simhadri, Harsha Vardhan}},
   title = {{EdgeML: Machine Learning for resource-constrained edge devices}},
   url = {https://github.com/Microsoft/EdgeML},
   version = {0.3},
}
```

### Microsoft Open Source Code of Conduct This project has adopted the
[Microsoft Open Source Code of
Conduct](https://opensource.microsoft.com/codeofconduct/). For more information
see the [Code of Conduct
FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact
[opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional
questions or comments.
