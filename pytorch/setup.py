import setuptools #enables develop
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from edgeml_pytorch.utils import findCUDA

if findCUDA() is not None:
    setuptools.setup(
        name='fastgrnn_cuda',
        ext_modules=[
            CUDAExtension('fastgrnn_cuda', [
                'edgeml_pytorch/cuda/fastgrnn_cuda.cpp',
                'edgeml_pytorch/cuda/fastgrnn_cuda_kernel.cu',
            ]),
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )

setuptools.setup(
    name='edgeml',
    version='0.3.0',
    description='PyTorch code for ML algorithms for edge devices developed at Microsoft Research India.',
    author_email="edgeml@microsoft.com",
    packages=['edgeml_pytorch', 'edgeml_pytorch.trainer', 'edgeml_pytorch.graph'],
    license='MIT License',
    long_description=open('README.md').read(),
    url='https://github.com/Microsoft/EdgeML',
)
