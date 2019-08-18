GesturePod *(Gesture Recognition Pipeline)*
=========================

## Table of Contents

- [About](#about)
- [Quick Start](#quick-start)

## About

GesturePod is a plug-and-play, gesture recognition device. 

This repository isolates the gesture recognition pipeline from the Arduino
MKR1000 implementation. All additional modules, for instance, BLE
communication, interface with the IMU are not included. We provide sample IMU
sensor data, which is used for prediciton. This repository is intended to be
used as a playground to understand the gesture recognition pipeline. It is a
bare-minimum implementation of gesture recognition. Additionally, this can be
used as a basis for porting GesturePod to other  platforms.

## Quick Start

1. Installation requires 
	- [git](https://git-scm.com/) 
	- [make](https://www.gnu.org/software/make/)
	- [g++](https://gcc.gnu.org/)

2. Clone the repo and navigate to this directory
	```
	git clone https://github.com/microsoft/EdgeML.git
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
5. Currently the data is read from ```./data/taps.h```. This data file has
   sensor readings collected from GesturePod, while *Double Taps* gesture was
   performed.

GesturePod data set is released as a part of our
[TechReport](https://www.microsoft.com/en-us/research/uploads/prod/2018/05/dataTR_v1.tar-5b058a4590168.gz)


Checkout our Blog post *Coming Soon...*
