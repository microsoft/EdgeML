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

// Functions for input type relational comparison.
// Here q16 denotes quantization to int16_t.
MYINT q16_sigmoid(MYINT x);
MYINT q16_tanh(MYINT x);

// Functions for calculating quantized activations.
// Currently these makes use of 16-bit quantization only.
void m_q_sigmoid(const MYINT* const A, MYITE nrows, MYITE ncols,
                 MYINT* const B);
void m_q_tanh(const MYINT* const A, MYITE nrows, MYITE ncols,
                   MYINT* const B);

// Function for reversing the order of rows in an input matrix.
void m_reverse(const MYINT* const A, MYINT* const B, MYITE nrows, MYITE ncols);

// Function for adding a scalar to every element of a matrix.
void m_q_scalar_add(MYINT A, const MYINT* const B, MYINT* const C,
                    MYITE nrows, MYITE ncols, MYSCL scA, MYSCL scB, MYSCL scC);
// Function for subtracting every element of a matrix from a scalar.
// The resultant matrix has elements C_{i, j} = A - B_{i, j}.
void m_q_scalar_sub(MYINT A, const MYINT* const B, MYINT* const C, MYITE nrows,
                    MYITE ncols, MYSCL scA, MYSCL scB, MYSCL scC);
// Function for multiplying a scalar to every element of a matrix.
void m_q_scalar_mul(MYINT A, const MYINT* const B, MYINT* const C, MYITE nrows,
                    MYITE ncols, MYSCL scA, MYSCL scB, MYSCL scC);
// Function for computing the Hadamard product of two matrices.
void m_q_hadamard(const MYINT* const A, const MYINT* const B, MYINT* const C,
                  MYITE nrows, MYITE ncols, MYSCL scA, MYSCL scB, MYSCL scC);

// Function for computing the element-wise sum of two matrices.
void m_q_add(const MYINT* const A, const MYINT* const B, MYINT* const C,
             MYITE nrows, MYITE ncols, MYSCL scA, MYSCL scB, MYSCL scC);
// Function for computing the matrix multiplication of two matrices.
void m_q_mul(const MYINT* const A, const MYINT* const B, MYINT* const C,
             MYITE nrows, MYITE nmid, MYITE ncols, MYSCL scA, MYSCL scB,
             MYSCL scC);

#endif