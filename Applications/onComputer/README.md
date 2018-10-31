GesturePod *(Simulation)*
=========================

## Table of Contents

- [About](#about)
- [Quick Start](#quick-start)
- [Docs](#docs)
- [Also](#also)

## About

GesturePod is a plug-and-play gesture recognition device. 

As the GesturePod involves Hardware - in this quick simulation, we have stored the sensor data in .txt file. Data from this file is used for prediction. You can now play around with GesturePod on your system!

## Quick Start

1. Installation requires [Git](https://git-scm.com/), [make](https://www.gnu.org/software/make/), and [g++](https://gcc.gnu.org/).

2. Clone the repo and navigate to this folder
	```
	git clone ...
	cd EdgeML/Applications/GesturePod/onComputer
	```
3. Compile and build the code
	```
	make
	```
4. Run..!
	```
	./gesturepodsim
	```
5. By default the program runs for data that has *Double Taps*.

	To try different gestures, you could change the ```DATA_FILE``` in ```src/main.cpp:23```. 
	A more exhaustive data set is released as a part of our [TechReport](https://www.microsoft.com/en-us/research/uploads/prod/2018/05/dataTR_v1.tar-5b058a4590168.gz)

## Docs
Checkout our Blog post *Coming Soon..!*

## Also
Make your own GesturePod. Get the code [here](https://github.com/Microsoft/EdgeML/tree/master/Applications/GesturePod/onMKR1000)