# SeeDot Architecture

This document descibes the averall architecture of the SeeDot quantization tool. 

SeeDot is run by runing the `SeeDot-dev.py` python scipt. 

Running SeeDot using default arguments, i.e the call
```
    python SeeDot-dev.py
```
is euivalent to running the [ProtoNN](https://github.com/microsoft/EdgeML/blob/master/docs/publications/ProtoNN.pdf) algorithm with `fixed-point` encoding on the `cifar-binary` dataset for `x86` target device; i.e the call 
```
    python SeeDot-dev.py -a protonn -v fixed -d cifar-binary -n 1 -t x86 -m acc -l error.
```

## Walkthrough

In this text, we will discuss the execution of SeeDot through
###


