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

### Microsoft Open Source Code of Conduct
This project has adopted the [Microsoft Open Source Code of
Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct
FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com)
with any additional questions or comments.
