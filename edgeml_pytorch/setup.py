import setuptools #enables develop

setuptools.setup(
    name='edgeml',
    version='0.2.2',
    description='PyTorch code for Ml algorithms for edge devices developed at Microsoft Research India.',
    author_email="edgeml@microsoft.com",
    packages=setuptools.find_packages(),
    license='MIT License',
    long_description=open('README.md').read(),
    url='https://github.com/Microsoft/EdgeML',
)
