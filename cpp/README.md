## Edge Machine Learning - C++ Library

This library consists of two machine learning algortihms **Bonsai** and **ProtoNN** implemented in C++ for speed and scalability.

### Requirements
* Linux:
  * gcc version 5.4. Other gcc versions above 5.0 could also work.
  * We developed the code on Ubuntu 16.04LTS. Other linux versions could also work.
  * You can either use the Makefile in the root, or cmake via the build directory (see below).
  
* Windows 10:
  * Visual Studio 2015. Use cmake (see below).
  * For Anniversary Update or later, one can use the Windows Subsystem for Linux, and the instructions for Linux build. 

* On both Linux and Windows 10, you need an implementation of BLAS, sparseBLAS and vector math calls.
  We link with the implementation provided by the [Intel(R) Math Kernel Library](https://software.intel.com/en-us/mkl).
  Please download later versions (2017v3+) of MKL as far as possible.
  The code can be made to work with other math libraries with a few modifications.

### Building using Makefile

After cloning this repository, set compiler and flags appropriately in `config.mk`. Then execute the following in bash:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<MKL_PATH>:<EDGEML_ROOT>
make -Bj
```
Typically, MKL_PATH = /opt/intel/mkl/lib/intel64_lin/, and EDGEML_ROOT is '.'.

This will build four executables _BonsaiTrain_, _BonsaiPredict_, _ProtoNNTrain_ and _ProtoNNPredict_ in <EDGEML_ROOT>.
Sample data to try these executables is not included in this repository, but instructions to do so are given below. 

### Building using CMake

For Linux, in the <EDGEML_ROOT> directory:

```bash
mkdir build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<MKL_PATH>
cd build
cmake ..
make -Bj
```

For Windows 10, in the <EDGEML_ROOT> directory, modify `CMakeLists.txt` file to change <MKL_ROOT> by changing the
line 
```set(MKL_ROOT "<MKL_ROOT>")```

Then, generate Visual Studio 2015 solution using:

```mkdir build
cd build
cmake -G "Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Release ..
```
Finally, open `EdgeML.sln` in VS2015, build and run.

For both Linux and Windows10, cmake builds will generate four executables _BonsaiTrain_, _BonsaiPredict_, _ProtoNNTrain_ and _ProtoNNPredict_ in <EDGEML_ROOT>.

### Download a sample dataset
Follow the bash commands given below to download a sample dataset, USPS10, to the root of the repository. Bonsai and ProtoNN come with sample scripts to run on the usps10 dataset. EDGEML_ROOT is defined in the previous section. 

```bash
cd <EDGEML_ROOT>
mkdir usps10
cd usps10
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2
bzip2 -d usps.bz2
bzip2 -d usps.t.bz2
mv usps train.txt
mv usps.t test.txt
mkdir ProtoNNResults
cd <EDGEML_ROOT>
```
This will create a sample train and test dataset, on which
you can train and test Bonsai and ProtoNN algorithms. As specified, we create an output folder for ProtoNN. Bonsai on the other hand creates its own output folder. 
For instructions to actually run the algorithms, see [Bonsai Readme](docs/README_BONSAI_OSS.md) and [ProtoNN Readme](docs/README_PROTONN_OSS.ipynb).

### Makefile flags
You could change the behavior of the code by setting these flags in `config.mk` and rebuilding with `make -Bj` when building with the default Makefile in <EDGEML_ROOT>. When building with CMake, change these flags in `CMakeLists.txt` in <EDGEML_ROOT>. All these flags can be set for both ProtoNN and Bonsai.
The following are supported currently by both ProtoNN and Bonsai. 

    SINGLE/DOUBLE:  Single/Double precision floating-point. Single is most often sufficient. Double might help with reproducibility.
    ZERO_BASED_IO:  Read datasets with 0-based labels and indices instead of the default 1-based. 
    TIMER:          Timer logs. Print running time of various calls.
    CONCISE:        To be used with TIMER to limit the information printed to those deltas above a threshold.

The following currently only change the behavior of ProtoNN, but one can write corresponding code for Bonsai. 
 
    LOGGER:         Debugging logs. Currently prints min, max and norm of matrices.
    LIGHT_LOGGER:   Less verbose version of LOGGER. Can be used to track call flow. 
    XML:            Enable training with large sparse datasets with many labels. This is in beta.
    VERBOSE:        Print additional informative output to stdout.
    DUMP:           Dump models after each optimization iteration instead of just in the end.
    VERIFY:         Legacy verification code for comparison with Matlab version.
    
Additionally, there is one of two flags that has to be set in the Makefile: 
    
    MKL_PAR_LDFLAGS: Linking with parallel version of MKL.
    MKL_SEQ_LDFLAGS: Linking with sequential version of MKL.

### Microsoft Open Source Code of Conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

Copyright (c) Microsoft Corporation. All rights reserved. 

Licensed under the MIT license.