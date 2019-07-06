---
layout: default
title: ProtoNN - Compressed Accurate K-Nearest Neighbour
---

ProtoNN is a multi-class classification algorithm inspired by k-Nearest
Neighbor (KNN) but has several orders lower storage and prediction complexity.
ProtoNN models can be deployed even on devices with puny storage and
computational power (e.g. an Arduino UNO with 2kB RAM) to get excellent
prediction accuracy. ProtoNN derives its strength from three key ideas: a)
learning a small number of prototypes to represent the entire training set, b)
sparse low dimensional projection of data, c) joint discriminative learning of
the projection and prototypes with explicit model
size constraint.

$$
(\vec{h}_{t} = \tanh({W}_h x_t + {U}_h h_{t-1} + b_h).) \\
(h_{t} = \tanh({W}_h x_t + {U}_h h_{t-1} + b_h).)
$$
