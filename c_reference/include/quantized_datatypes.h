// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __QUANTIZED_DATATYPES_H__
#define __QUANTIZED_DATATYPES_H__

#include <stdint.h>

// Macros for input type.
typedef int8_t Q7_T;
typedef int16_t Q15_T;
typedef int32_t Q31_T;
typedef int64_t Q63_T;
// Macro for unsigned iterator type.
typedef uint32_t ITER_T;
// Macro for signed iterator type.
typedef int32_t S_ITER_T;
// Macro for scale variable type.
typedef int32_t SCALE_T;
// Macro for max value of input type.
#define Q15_TMAX 32767
// Macro for min value of input type.
#define Q15_TMIN -32768

#endif
