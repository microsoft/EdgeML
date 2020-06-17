// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __QUANTIZED_UTILS_H__
#define __QUANTIZED_UTILS_H__

#include <math.h>
#include "quantized_datatypes.h"

// Function for saturating the input to the required format.
// This function isn't used currently because of SeeDot generated scales
// ensuring the overflows aren't a possibility.
inline MYINT saturate(MYINM inp) {
    if (inp > MYINTMAX){
        return (MYINT)MYINTMAX;
    } else if (inp < MYINTMIN) {
        return (MYINT)MYINTMIN;
    } else {
        return (MYINT)inp;
    }
}

// Functions for calculating quantized operations and activations.
// Function for computing the element-wise addition between two vectors.
void v_q_add(const MYINT* const vec1, const MYINT* const vec2, MYITE len,
             MYINT* const ret, MYSCL scvec1, MYSCL scvec2, MYSCL scret);
// Function for computing the element-wise difference between two vectors.
void v_q_sub(const MYINT* const vec1, const MYINT* const vec2, MYITE len,
             MYINT* const ret, MYSCL scvec1, MYSCL scvec2, MYSCL scret);
// Function for computing the Hadamard product between two vectors.
void v_q_hadamard(const MYINT* const vec1, const MYINT* const vec2, MYITE len,
                  MYINT* const ret, MYSCL scvec1, MYSCL scvec2);
// Function for computing the Sigmoid activation on the input vector.
void v_q_sigmoid(const MYINT* const vec, MYITE len, MYINT* const ret, MYINT div, MYINT add,
                 MYINT sigmoid_limit, MYSCL scale_in, MYSCL scale_out);
// Function for computing the TanHyperbolic activation on the input vector.
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
