# MFCC Featurizer

This library presents a way of compouting MFCC feature vectors for audio data. It uses Hans-Kristian Arntzen's [muFFT](https://github.com/Themaister/muFFT) libray for efficent FFT computation.

To compile, please download the [muFFT](https://github.com/Themaister/muFFT) library from github and place it in the root folder as `muFFT`. Then run

    make mfcc.o

To create an object `mfcc.o` against which you can link statically.

Example usage is provided in `testmfcc.c`. Use 

    make testmfcc

To build this example against the `mfcc.o` object file.

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT license.