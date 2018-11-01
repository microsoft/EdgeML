GesturePod *(Gesture Recognition Pipeline)*
=========================

## Table of Contents

- [About](#about)
- [Quick Start](#quick-start)
- [Docs](#docs)
- [Also](#also)

## About

GesturePod is a plug-and-play, gesture recognition device. 

As the GesturePod involves hardware, this implements GesturePod's gesture recognition pipeline on your computer system. We have stored sensor data collected from the GesturePod. This pre-stored sensor data is used for gesture recognition. You can now play around with GesturePod on your computer!

## Quick Start

1. Installation requires 
	- [git](https://git-scm.com/), 
	- [make](https://www.gnu.org/software/make/)
	- [g++](https://gcc.gnu.org/).

2. Clone the repo and navigate to this directory
	```
	git clone https://github.com/Microsoft/EdgeML
	cd EdgeML/Applications/GesturePod/onComputer
	```
3. Compile and build the code
	```
	make
	```
4. Run
	```
	./gesturepodsim
	```
5. By default the program runs for data that has the following gesture: *Double Taps*.

	To try different gestures, you could change the ```DATA_FILE``` in ```src/main.cpp:23```. 
	A more exhaustive data set is released as a part of our [TechReport](https://www.microsoft.com/en-us/research/uploads/prod/2018/05/dataTR_v1.tar-5b058a4590168.gz)

## Docs
Checkout our Blog post *Coming Soon..!*

## Also
Make your own GesturePod. Get the code [here](https://github.com/Microsoft/EdgeML/tree/master/Applications/GesturePod/onMKR1000).