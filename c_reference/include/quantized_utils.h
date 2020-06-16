// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __QUANTIZED_UTILS_H__
#define __QUANTIZED_UTILS_H__

#include <math.h>
#include <stdint.h>

// Macro for scale variable type.
#ifdef SHIFT
  #define MYSCL int16_t
#else
  #define MYSCL int32_t
#endif

// Macro for input type.
#define MYINT int16_t
// Macro for iterator type.
#define MYITE uint16_t
// Macro for intermediate buffer type.
#define MYINM int32_t

// Functions for input type relational comparison.
MYINT min(MYINT a, MYINT b);
MYINT max(MYINT a, MYINT b);

// Functions for calculating quantized operations and activations.
void v_q_add(const MYINT* const vec1, const MYINT* const vec2, MYITE len,
             MYINT* const ret, MYSCL scvec1, MYSCL scvec2, MYSCL scret);
void v_q_sub(const MYINT* const vec1, const MYINT* const vec2, MYITE len,
             MYINT* const ret, MYSCL scvec1, MYSCL scvec2, MYSCL scret);
void v_q_hadamard(const MYINT* const vec1, const MYINT* const vec2, MYITE len,
                  MYINT* const ret, MYSCL scvec1, MYSCL scvec2);
void v_q_sigmoid(const MYINT* const vec, MYITE len, MYINT* const ret, MYINT div, MYINT add,
                 MYINT sigmoid_limit, MYSCL scale_in, MYSCL scale_out);
void v_q_tanh(const MYINT* const vec, MYITE len, MYINT* const ret,
              MYSCL scale_in, MYSCL scale_out);
// Function for adding a scalar to every element of a vector.
void v_q_scalar_add(MYINT scalar, const MYINT* const vec, MYITE len,
                    MYINT* const ret, MYSCL scscalar, MYSCL scvec, MYSCL scret);
// Function for subtracting every element of a vector from a scalar.
// The resultant vector has elements C_{i} = A - B_{i}.
void v_q_scalar_sub(MYINT scalar, const MYINT* const vec, MYITE len,
                    MYINT* const ret, MYSCL scscalar, MYSCL scvec, MYSCL scret);
// Function for multiplying a scalar to every element of a vector.
void v_q_scalar_mul(MYINT scalar, const MYINT* const vec, MYITE len,
                    MYINT* const ret, MYSCL scscalar, MYSCL scvec);
// Function for multiplying a matrix with a vector.
void m_q_mulvec(const MYINT* const mat, const MYINT* const vec, MYITE nrows,
                MYITE ncols, MYINT* const ret, MYSCL scmat, MYSCL scvec,
                MYITE H1, MYITE H2);

#endif