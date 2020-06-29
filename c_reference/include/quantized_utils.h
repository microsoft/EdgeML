// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __QUANTIZED_UTILS_H__
#define __QUANTIZED_UTILS_H__

#include <math.h>
#include "quantized_datatypes.h"

// Function for saturating the input to the required format.
// This function isn't used currently because of SeeDot generated scales
// ensuring the overflows aren't a possibility.
inline INT_T saturate(INTM_T inp) {
    if (inp > INT_TMAX){
        return (INT_T)INT_TMAX;
    } else if (inp < INT_TMIN) {
        return (INT_T)INT_TMIN;
    } else {
        return (INT_T)inp;
    }
}

// This function is used to provide a truncation of input to a specific
// range within the ReLU operation.
inline INTM_T q_relu(INTM_T inp, INTM_T limit) {
    if (inp > limit){
        return limit;
    } else if (inp < 0) {
        return 0;
    } else {
        return inp;
    }
}

// Functions for calculating quantized operations and activations.
// Function for computing TreeSum from a given vector holding intermediate results.
void v_q_treesum(INTM_T* const vec, ITER_T len, SCALE_T H1, SCALE_T H2);
// Function for computing the element-wise addition between two vectors.
void v_q_add(const INT_T* const vec1, const INT_T* const vec2, ITER_T len,
             INT_T* const ret, SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret);
// Function for computing the element-wise difference between two vectors.
void v_q_sub(const INT_T* const vec1, const INT_T* const vec2, ITER_T len,
             INT_T* const ret, SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret);
// Function for computing the Hadamard product between two vectors.
void v_q_hadamard(const INT_T* const vec1, const INT_T* const vec2, ITER_T len,
                  INT_T* const ret, SCALE_T scvec1, SCALE_T scvec2);
// Function for computing the Sigmoid activation on the input vector.
void v_q_sigmoid(const INT_T* const vec, ITER_T len, INT_T* const ret, INT_T div,
                 INT_T add, INT_T sigmoid_limit, SCALE_T scale_in, SCALE_T scale_out);
// Function for computing the TanHyperbolic activation on the input vector.
void v_q_tanh(const INT_T* const vec, ITER_T len, INT_T* const ret,
              SCALE_T scale_in, SCALE_T scale_out);
// Function for adding a scalar to every element of a vector.
void v_q_scalar_add(INT_T scalar, const INT_T* const vec, ITER_T len,
                    INT_T* const ret, SCALE_T scscalar, SCALE_T scvec, SCALE_T scret);
// Function for subtracting every element of a vector from a scalar.
// The resultant vector has elements C_{i} = A - B_{i}.
void v_q_scalar_sub(INT_T scalar, const INT_T* const vec, ITER_T len,
                    INT_T* const ret, SCALE_T scscalar, SCALE_T scvec, SCALE_T scret);
// Function for multiplying a scalar to every element of a vector.
void v_q_scalar_mul(INT_T scalar, const INT_T* const vec, ITER_T len,
                    INT_T* const ret, SCALE_T scscalar, SCALE_T scvec);
// Function for multiplying a matrix with a vector.
void m_q_mulvec(const INT_T* const mat, const INT_T* const vec, ITER_T nrows,
                ITER_T ncols, INT_T* const ret, SCALE_T scmat, SCALE_T scvec,
                SCALE_T H1, SCALE_T H2);
//Function for multiplying matrices
void sp_mat_mul(const INT_T *Aidx, const INT_T *Aval, INT_T **B, INT_T *C, INT_T K, \
                INT_T shrA, INT_T shrB, INT_T shrC);
//Function for getting the index largest element in array
void arg_max(INT_T *A, INT_T len, INT_T *index);
//Function for transposing the matrix
void transpose(INT_T *A, INT_T *B, INT_T I, INT_T J);
//Function for performing addition or substraction on 4D circle
void add_or_sub_cir_4D(INT_T *A, const INT_T *B, INT_T *X, INT_T N, INT_T H, INT_T W, INT_T C, INT_T shrA, INT_T shrB, INT_T shrC, uint8_t add);
//Function for performing the addtion and substraction on 2D circle
void add_or_sub_cir_2D(INT_T *A, const INT_T *B, INT_T *X, INT_T H, INT_T W, INT_T shrA, INT_T shrB, INT_T shrC, uint8_t add);
//Function for replacing the negative data items to zero
void relu_4D_2D(INT_T *A, INT_T N, INT_T H, INT_T W, INT_T C);
//Function for scaling the matrix
void exp_scale(INT_T *A, INT_T I, INT_T J, INT_T shrA, INT_T shrB, INT_T *B);
//Function for performing division, addition, scaling of the signal
void sigmoid(INT_T *A, INT_T I, INT_T J, INT_T div, INT_T add, INT_T sigmoid_limit, \
             INT_T scale_in, INT_T scale_out, INT_T *B);
//Function for adjusting the scale down
void adjust_scale_shr(INT_T *A, INT_T I, INT_T J, INT_T K, INT_T L, INT_T scale);
//Function for adjusting scale up
void adjust_scale_shl(INT_T *A, INT_T I, INT_T J, INT_T K, INT_T L, INT_T scale);
//Function for reversing the data item
void Reverse2(INT_T *A, INT_T axis, INT_T I, INT_T J, INT_T *B);
void maxpool(INT_T *A, INT_T *B, INT_T N, INT_T H, INT_T W, INT_T C, INT_T FH, \
             INT_T FW, INT_T strideH, INT_T strideW, INT_T HPADL, INT_T HPADR, \
            INT_T WPADL, INT_T WPADR);
//Function for performing conv
void conv(INT_T *A, const INT_T *B, INT_T *C, INT_T *tmp,                   \
          INT_T N, INT_T H, INT_T W, INT_T CI, INT_T HF,                    \
          INT_T WF, INT_T CO, INT_T shrA, INT_T shrB, INT_T H1, INT_T H2);
//Function for performing convolution
void convolution(INT_T *A, const INT_T *B, INT_T *C, INT_T *tmp, 
                 INT_T N, INT_T H, INT_T W, INT_T CIN, INT_T HF,        \
                 INT_T WF, INT_T CINF, INT_T COUTF, INT_T HOUT,         \
                 INT_T WOUT, INT_T HPADL, INT_T HPADR, INT_T WPADL,     \
                 INT_T WPADR, INT_T HSTR, INT_T WSTR, INT_T HDL,        \
                 INT_T WDL, INT_T G, INT_T shrA, INT_T shrB, INT_T H1,  \
                 INT_T H2);
//Function to perform transpose of matrix
void transpose(INT_T *A, INT_T *B, INT_T I, INT_T J);

#endif
