---
layout: default
title: Bonsai - Strong, Shallow and Sparse Non-linear Tree
---

Bonsai is a new tree model for supervised learning tasks such as binary and
multi-class classification, regression, ranking, etc. Bonsai learns a single,
shallow, sparse tree with powerful predictors at internal and leaf nodes. This
allows Bonsai to achieve state-of-the-art prediction accuracies while making
predictions efficiently in microseconds to milliseconds (depending on processor
speed) using models that fit in a few KB of memory. Bonsai can be trained in
the cloud or on your laptop, but can then make predictions locally on tiny
resource-constrained devices without needing cloud connectivity.

Bonsai has been deployed successfully on microcontrollers tinier than a grain
of rice such as the ARM Cortex M0 with just 2 KB RAM. Bonsai can also make
predictions accurately and efficiently on the tiniest of IoT boards such as the
Arduino Pro Mini based on an 8 bit Atmel ATmega328P microcontroller operating
at 8 MHz without any floating point support in hardware, with 2 KB RAM and 32
KB read-only flash memory. Bonsai can also fit in the L1 cache of processors
found in mobiles, tablets, laptops, and servers for low-latency applications.

Bonsai is most useful for IoT scenarios where it is not advantageous to
transmit sensor readings (or features) to the cloud and predictions need to be
made locally on-device due to:

- Poor bandwidth or lack of connectivity
- Low-latency requirements where predictions need to be made very quickly and
	there isn’t enough time to transmit data to the cloud and get back a prediction
- Concerns about privacy and security where the data should not leave the
	device
- Low-energy requirements where data cannot be transmitted to the cloud so as
	to enhance battery life


Bonsai can also be useful for switching to a smaller, cheaper and more
energy-efficient form factor such as from a Raspberry Pi 3 to an Arduino Pro
Mini. Finally, Bonsai also generalizes to other resource-constrained scenarios
beyond the Internet of Things and can be used on laptops, servers and the cloud
for low-latency applications and to bring down energy consumption and operating
costs.

Please see our [ICML 2017 paper](/Microsoft/EdgeML/wiki/files/BonsaiPaper.pdf)
for more details about the model and algorithm and our [Getting
Started](/Microsoft/EdgeML/wiki/GettingStarted) section for instructions on how
to use Bonsai.

