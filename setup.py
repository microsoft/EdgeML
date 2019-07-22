import setuptools #enables develop

setuptools.setup(
    name='edgeml',
    version='0.2',
    description='machine learning algorithms for edge devices developed at Microsoft Research India.',
    packages=setuptools.find_packages(),
    license='MIT License',
    long_description=open('License.txt').read(),
    url='https://github.com/Microsoft/EdgeML',
)
