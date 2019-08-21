import setuptools #enables develop

setuptools.setup(
    name='edgeml',
    version='0.2.2',
    description='PyTorch code for ML algorithms for edge devices developed at Microsoft Research India.',
    author_email="edgeml@microsoft.com",
    packages=['edgeml_pytorch']
    license='MIT License',
    long_description=open('edgeml_pytorch/README.md').read(),
    url='https://github.com/Microsoft/EdgeML',
)
