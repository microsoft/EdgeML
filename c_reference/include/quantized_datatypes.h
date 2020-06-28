// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __QUANTIZED_DATATYPES_H__
#define __QUANTIZED_DATATYPES_H__

#include <stdint.h>

// Macro for input type.
typedef int16_t INT_T;
// Macro for signed iterator type.
typedef int16_t S_ITER_T;
// Macro for unsigned iterator type.
#ifdef CMSISDSP
  typedef uint32_t ITER_T;
#else
  typedef uint16_t ITER_T;
#endif
// Macro for intermediate buffer type.
typedef int32_t INTM_T;
// Macros for scale variable type.
#ifdef SHIFT
  typedef uint8_t SCALE_T;
  typedef uint8_t L_SCALE_T;
#elif defined CMSISDSP
  typedef int8_t SCALE_T;
  typedef uint8_t L_SCALE_T;
#else
  typedef int16_t SCALE_T;
  typedef int32_t L_SCALE_T;
#endif
// Macro for max value of input type.
#define INT_TMAX 32767
// Macro for min value of input type.
#define INT_TMIN -32768

#endif
