// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __QUANTIZED_DATATYPES_H__
#define __QUANTIZED_DATATYPES_H__

#include <stdint.h>

// Macros for input type.
typedef int8_t Q7_T;
typedef int16_t Q15_T;
typedef int32_t Q31_T;
// Macro for unsigned iterator type.
typedef uint16_t ITER_T;
// Macro for signed iterator type.
typedef int16_t S_ITER_T;
// Macros for scale variable type.
#ifdef SHIFT
  typedef uint8_t SCALE_T;
  typedef uint8_t L_SCALE_T;
#else
  typedef int16_t SCALE_T;
  typedef int32_t L_SCALE_T;
#endif
// Macro for max value of input type.
#define Q15_TMAX 32767
// Macro for min value of input type.
#define Q15_TMIN -32768

#endif
