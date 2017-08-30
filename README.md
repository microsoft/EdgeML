## Edge Machine Learning

### Requirements
- Linux. We developed the code on Ubuntu 16.04LTS.
  The code can also compiled in Windows with Visual Studio 2015, but this release does not include makefile. 
- gcc version 5.4. Other gcc versions above 5.0 could also work.
- [Intel(R) Math Kernel Library](https://software.intel.com/en-us/mkl). We use BLAS, sparseBLAS and VML routines. 

### Building
After cloning this reposirory, do:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<MKL_PATH>:<EDGEML_ROOT>
make -j
```
Typically, MKL_PATH = /opt/intel/mkl/lib/intel64_lin/, and EDGEML_ROOT is '.'.

This will build two executables _Bonsai_ and _ProtoNN_. 

### Download a sample dataset

```bash
mkdir usps10
cd usps10
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2
bzip2 -d usps.bz2
bzip2 -d usps.t.bz2
mv usps train.txt
mv usps.t test.txt
cd <EDGEML_ROOT>
```
This will create a sample test set. You can now train and test Bonsai and ProtoNN algorithms on this datasets.
For detailed instructions, see [Bonsai Readme](README_BONSAI_OSS.md) and [ProtoNN Readme](README_PROTONN_OSS.md).

### Makefile flags
You could change the behavior of the code by setting these flags in `config.mk` and rebuilding with `make -Bj`. 

SINGLE
TIMER
LOGGER
LIGHT_LOGGER
VERBOSE
MKL_PAR/SEQ

### Microsoft Open Source Code of Conduct
This project has adopted the [Microsoft Open Source Code of
Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct
FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com)
with any additional questions or comments.
