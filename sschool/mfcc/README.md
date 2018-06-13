# MFCC Featurizer

This library presents a way of computing MFCC feature vectors for audio data. It uses Hans-Kristian Arntzen's [muFFT](https://github.com/Themaister/muFFT) libray for efficent FFT computation.

To compile, please download the [muFFT](https://github.com/Themaister/muFFT) library from github and place the contents in a directory names `muFFT`. To create an object `mfcc.o` to statically link against, run

    make mfcc.o


Example usage is provided in `testmfcc.c`. Build the example using 

    make testmfcc


Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT license.
