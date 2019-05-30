GesturePod *(for MKR1000)*
=========================

## Table of Contents

- [About](#about)
- [Quick Start](#quick-start)
- [Docs](#docs)
- [Also](#also)
- [Dependencies](#dependencies)

## About

[GesturePod](https://1drv.ms/u/s!AjDloPaG_l0Et7Ikid1voOVFuI116Q) is a plug-and-play, gesture recognition device. GesturePod can be clamped onto any white-cane used by persons with Visually Impairment. Simple and natural gestures performed on the cane triggers corresponding activities on the phone. The gesture recognition system on the cane is powered by (real time and robust) Machine Learning (ML) based classifier [ProtoNN](https://github.com/Microsoft/EdgeML/blob/master/docs/publications/ProtoNN.pdf). This repository contains the code for recognizing the gestures and communicating the gesture over Bluetooth Low Energy (BLE). The pertrained model can recognize 5 gestures - Double Tap, Right Twist, Left Twist, Twirl and Double Swipe with an accuracy of 99.87.


## Quick Start

1. Clone the repo and navigate to this folder
	```
	git clone https://github.com/microsoft/EdgeML.git
	cd Applications/GesturePod/onMKR1000
	```
2. Refer to this tutorial _(Coming Soon ..!)_ to build the Hardware and setup external dependencies - [Cortex M0+ Board support](https://www.hackster.io/charifmahmoudi/arduino-mkr1000-getting-started-08bb4a).
		
3. Compile, Build and Upload
	Select the ```COM``` port to which MKR1000 is connected.
	Upload the code.

4. Launch the ```COM``` port and perform gestures on the cane to get the results of live predictions.

## Docs
Blog post _Coming soon ..!_

## Simulation
Make your own GesturePod. Don't have time? No issues, checkout the simulation on your computer with preloaded sensor values _Coming Soon ..!_

## Dependencies
To communicate with the MPU6050, we use [jrowberg's](https://github.com/jrowberg/i2cdevlib) ```i2cdevlib``` library.  Last tested with commit [900b8f9](https://github.com/jrowberg/i2cdevlib/tree/900b8f959e9fa5c3126e0301f8a61d45a4ea99cc).
