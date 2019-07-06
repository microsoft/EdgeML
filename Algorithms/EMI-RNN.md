---
layout: default
title: Early Multi-Instance Recurrent Neural Network 
---

Deploying sequential data classification modules on tiny devices is challenging as
predictions over sliding windows of data need to be invoked continuously at a
high frequency. Each of these predictors themselves are expensive as they
evaluate large models over long windows of data. In this paper, we address this
challenge by exploiting the following two observations about classification
tasks arising in typical IoT related applications: (a) the "signature" of a
particular class (e.g. an audio keyword) typically occupies a small fraction of
the overall data, and (b) class signatures tend to discernible early-on in the
data. We propose a method that exploits these observations by using a multiple
instance learning formulation along with an early prediction technique to learn
a model that can achieve better accuracy compared to baseline models, while
reducing the computation by a large fraction. For instance, on an audio keyword
detection benchmark our model improves standard LSTM model’s accuracy by up to
1.5% while decreasing the computation cost by more than 60%.
