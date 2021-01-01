import setuptools #enables develop
import os

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
