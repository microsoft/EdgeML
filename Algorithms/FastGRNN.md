---
layout: default
title: Fast(G)RNN - Fast, Accurate, Stable and Tiny (Gated) Recurrent Neural Network
---

*FastRNN* and *FastGRNN*, two RNN architectures (cells), together called
*FastCells*, are developed to address the twin RNN limitations of inaccurate
training and inefficient prediction. FastRNN provably stabilizes the RNN
training which usually suffers from the famous vanishing and exploding gradient
problems. FastGRNN learns low-rank, sparse and quantized weight matrices whilst
having elegant and provably stable Gated RNN update equations. This allows
FastGRNN to achieve state-of-the-art prediction accuracies while making
predictions in microseconds to milliseconds (depending on processor speed)
using models that fit in a few KB of memory. Fast(G)RNN can be trained in the
cloud or on your laptop, but can then make predictions locally on tiny
resource-constrained devices without needing cloud connectivity.

FastGRNN is upto **45x** smaller and faster (inference on edge-devices) than
state-of-the-art RNN architectures like LSTM/GRU whilst maintaining accuracies
on various benchmark datasets and FastRNN has provably stable training and
better performance when compared to Unitary architectures proposed to tackle
the unstable training.

### FastRNN
![FastRNN]({{ site.baseurl }}/img/algorithms/fastgrnn/FastRNN.png)
![FastRNN Equation]({{ site.baseurl }}/img/algorithms/fastgrnn/FastRNN_eq.png)

### FastGRNN
![FastGRNN Base Architecture]({{ site.baseurl }}/img/algorithms/fastgrnn/FastGRNN.png)
![FastGRNN Base Equation]({{ site.baseurl }}/img/algorithms/fastgrnn/FastGRNN_eq.png)

FastGRNN has been deployed successfully on microcontrollers tinier than a grain
of rice such as the ARM Cortex M0+ with just 2 KB RAM. Bonsai can also make
predictions accurately and efficiently on the tiniest of IoT boards such as the
Arduino MKR1000 based on a 32-bit low power ARM Cortex M0+ microcontroller
without any floating point support in hardware, with 32 KB RAM and 256 KB
read-only flash memory. FastGRNN can also fit in the L1 cache of processors
found in mobiles, tablets, laptops, and servers for low-latency applications.

Most of the IoT sensor readings are multi-modal and have inherent/latent
temporal dependence. Using RNNs will eliminate the expensive and time-consuming
feature extraction and engineering, thereby incorporating that as part of the
model. RNNs are shown to be the state-of-the-art in various
Time-series/Temporal tasks over the last few years.

FastGRNN is most useful for IoT scenarios where it is not advantageous to
transmit sensor readings (or features) to the cloud and predictions need to be
made locally on-device due to:    

- Poor bandwidth or lack of connectivity
- Low-latency requirements where predictions need to be made very quickly and
    there isn’t enough time to transmit data to the cloud and get back a
prediction
- Concerns about privacy and security where the data should not leave the
    device
- Low-energy requirements where data cannot be transmitted to the cloud so as
    to enhance battery life


FastGRNN can also be useful for switching to a smaller, cheaper and more
energy-efficient form factor such as from a Raspberry Pi 3 to an Arduino Pro
Mini. Finally, Fast(G)RNN also generalizes to other resource-constrained
scenarios beyond the Internet of Things and can be used on laptops, servers and
the cloud for low-latency applications and to bring down energy consumption and
operating costs. 

Apart from the resource-constrained scenarios, Fast(G)RNN is proved to be a
smaller yet powerful replacement of expensive LSTM/GRU in various benchmark
tasks like Sentiment Classification, Language Modelling, and Image
Classification. This shows that the architectures have a wider impact and reach
spanning NLP, Image and Time-series tasks.

Please see our [NIPS 2018
paper](/Microsoft/EdgeML/wiki/files/FastGRNNPaper.pdf) for more details about
the model and algorithm and our [Getting
Started](/Microsoft/EdgeML/wiki/GettingStarted) section for instructions on how
to use Fast(G)RNN.

