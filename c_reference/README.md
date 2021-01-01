# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

This folder consists of `C` reference code for some `EdgeML` operators, 
and sample parameter sets and input/output traces to verify them.
The code is intended to compile with `GCC` on `Linux` machines,
and is to be adapted as needed for other embedded platforms.

## Directory Structure

The `EdgeML/c_reference/` directory is broadly structured into the following sub-directories:

- **include/**: Contains the header files for various lower level operators and layers.
- **models/**: Contains the optimized source code and header files for various models built by stiching together different layers and operators. Also contains the layer weights and hyper-parameters for the corresponding models as well (stored using `Git LFS`). (**Note:** Cloning the repo without installing `Git LFS` would fail to clone the actual headers. It's recommended to follow instructions on setting up `LFS` from [here](https://git-lfs.github.com/) before cloning.)
- **src/**: Contains the optimized source code files for various lower level operators and layers.
- **tests/**: Contains extensive test cases for individual operators and layers, as well as the implemented models. The executables are generated in the main directory itself, while the test scripts and their configurations can be accessed in the appropriate sub-directories.

## Compiling

Run `make` inside the `EdgeML/c_reference/` directory to compile the entire project at once. Alternatively, run `make clean` to discard the previously generated object files and executables. By default, the directory is compiled with loop unrolling and shift operations turned off.

## Running

Head to `c_reference/tests/` directory and execute the test script of your choice. Test patches (wherever required) are currently not included because of license restrictions. Please open an issue / refer to an existing issue for the same.
