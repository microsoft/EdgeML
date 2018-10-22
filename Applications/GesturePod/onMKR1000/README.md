GesturePod *(on MKR1000)*
=========================

## Table of Contents

- [About](#about)
- [Quick Start](#quick-start)
- [Docs](#docs)
- [Also](#also)
- [Dependencies](#dependencies)

## About

GesturePod is a plug-and-play, gesture recognition device. GesturePod can be clamped onto any white-cane used by persons with Visually Impairment. Simple and natural gestures performed on the cane triggers corresponding activities on the phone. This gesture recognition system on the cane powered by (real time robust Machine Learning (ML) based classifier [ProtoNN](https://github.com/Microsoft/EdgeML/blob/master/docs/publications/ProtoNN.pdf). 


Here in this code base you will find gesture prediction code that runs on a Cortex M0+ microcontroller. 

## Quick Start

1. Clone the repo and navigate to this folder
	```
	git clone ...
	cd Applications/GesturePod/onMKR1000
	```
2. Refer to this tutorial _(Coming Soon ..!)_ to build the Hardware and setup external dependencies - [Cortex M0+ Board support](https://www.hackster.io/charifmahmoudi/arduino-mkr1000-getting-started-08bb4a).
		
3. Compile, Build and Upload
	Select the ```COM``` port to which MKR1000 is connected.
	Upload the code.

4. Launch the ```COM``` port and perform gestures on the cane to get the results of live predictions.

## Docs
Blog post _Coming soon ..!_

## Also
Make your own GesturePod.
Don't have time? No issues, checkout the code on your computer with preloaded sensor values _Coming Soon ..!_

## Dependencies
MPU6050, I2C dev and helper 3d math libraries from [jrowberg](https://github.com/jrowberg/i2cdevlib). Last tested with commit [900b8f9](https://github.com/jrowberg/i2cdevlib/tree/900b8f959e9fa5c3126e0301f8a61d45a4ea99cc).
