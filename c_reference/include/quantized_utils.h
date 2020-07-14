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
// Function for subtracting every element of a vector B from a scalar A.
// The resultant vector has elements C_{i} = A - B_{i}.
void v_q_scalar_sub(INT_T scalar, const INT_T* const vec, ITER_T len,
                    INT_T* const ret, SCALE_T scscalar, SCALE_T scvec, SCALE_T scret);
// Function for subtracting a scalar B from every element of a vector A.
// The resultant vector has elements C_{i} = A_{i} - B.
void v_q_sub_scalar(const INT_T* const vec, INT_T scalar, ITER_T len,
                    INT_T* const ret, SCALE_T scvec, SCALE_T scscalar, SCALE_T scret);
// Function for multiplying a scalar to every element of a vector.
void v_q_scalar_mul(INT_T scalar, const INT_T* const vec, ITER_T len,
                    INT_T* const ret, SCALE_T scscalar, SCALE_T scvec);
/**
 * @brief Finds the index of largest element in a vector.
 * @param[in]       vec       pointer to input vector
 * @param[in]       len       length of the vector
 * @param[out]      ret       pointer variable storing the index of the largest element in the vector
 * @return          none
 * @example         vec       = {12, 24, 54, 1, 2, 10}
 *                  *ret      = 2
 */
void v_q_argmax(const INT_T* const vec, ITER_T len, ITER_T* const ret);
/**
 * @brief Replace any negative element present in the vector withs zero.
 * @param[in, out]  vec       pointer to vector on which element-wise ReLU operation is to be applied
 * @param[in]       len       length of the input vector
 * @return          none
 * @example         vec       = {1324, -5453, 3454, -3435, 8789}
 *                  len       = 4
 *                  vec       = {1324, 0, 3454, 0, 8789}
 */
void v_q_relu(INT_T* const vec, ITER_T len);
/**
 * @brief Computes exponentiation of all elements in the vec (interpreted as a floating-point value) to the base e and stores the result in ret.
 * @param[in]       vec       pointer to vector whose exponential scaling is to be performed
 * @param[in]       len       length of the vector
 * @param[in]       scvec     scaling factor for input vector
 * @param[in]       scret     scaling factor for output vector
 * @param[out]      ret       pointer to the output vector
 * @return          none
 * @example         vec       = {13, 54, 34, 35, 87}
 *                  len       = 5
 *                  scvec     = 8
 *                  scret     = 8
 *                  ret       = {40, 6832, 560, 635, 29493}
 */
void v_q_exp(const INT_T* const vec, ITER_T len, INT_T* const ret,
             SCALE_T scvec, SCALE_T scret);
/**
 * @brief Performs element-wise up-scaling on a vector.
 * @param[in, out]  vec       pointer to the vector on which up-scaling is to be performed
 * @param[in]       len       length of the vector
 * @param[in]       scvec     scaling factor of the vector
 * @return          none
 * @example         vec       = {423, -987, -2342, 1232}
 *                  len       = 4
 *                  scvec     = 10
 *                  mat       = {4230, -9870, -23420, 12320}
 */
void v_q_scale_up(INT_T* const vec, ITER_T len, SCALE_T scvec);
/**
 * @brief Performs element-wise down-scaling on a vector.
 * @param[in, out]  vec       pointer to the vector on which down-scaling is to be performed
 * @param[in]       len       length of the vector
 * @param[in]       scvec     scaling factor of the vector
 * @return          none
 * @example         vec       = {4232, -9879, -2342, 1232}
 *                  len       = 4
 *                  scvec     = 37
 *                  mat       = {114, -267, -63, 33}
 */
void v_q_scale_down(INT_T* const vec, ITER_T len, SCALE_T scvec);

/**
 * @brief Performs the transpose on the input matrix.
 * @param[in]       mat       pointer to the input matrix which is to be transposed
 * @param[in]       nrows     number of rows of output matrix
 * @param[in]       ncols     number of columns of output matrix
 * @param[out]      ret       pointer to the output matrix which will hold the transpose
 * @return          none
 * @example         mat       = { {1, 2},
 *                                {4, 5} }
 *                  ret       = { {1, 4},
 *                                {2, 5} }
 *
 * @example         mat       = { {1, 2, 3},
 *                                {4, 5, 6} }
 *                  ret       = { {1,  4},
 *                                {2,  5},
 *                                {3,  6} }
 */
void m_q_transpose(const INT_T* const mat, ITER_T nrows, ITER_T ncols,
                   INT_T* const ret);
/**
 * @brief Performs the row-order or the column-order reversal of the input matrix.
 * @param[in]       mat       pointer to the matrix on which reversal is to be performed
 * @param[in]       nrows     number of rows of the input matrix
 * @param[in]       ncols     number of columns of the input matrix
 * @param[in]       axis      axis of reversal; 0 for reversal along rows and 1 for reversal along columns
 * @param[out]      mat_out   pointer to the output matrix
 * @return          none
 * @example         mat       = { {1, 2},
 *                                {4, 5} }
 *                  nrows     = 2
 *                  ncols     = 2
 *                  axis      = 0
 *                  ret       = { {4, 5},
 *                                {1, 2} }
 */
void m_q_reverse(const INT_T* const mat, ITER_T nrows, ITER_T ncols,
                 ITER_T axis, INT_T* const ret);
/**
 * @brief Performs the column-wise addition of a bias term to the input matrix.
 * dim(mat) = dim(ret) = [nrows][ncols]; dim(vec) = [ncols].
 * @param[in]       mat       pointer to the input matrix on which addition is to be performed
 * @param[in]       vec       pointer to the bias vector which is to be added
 * @param[in]       nrows     number of rows of the input matrix
 * @param[in]       ncols     number of columns of the input matrix
 * @param[out]      ret       pointer to the output matrix
 * @param[in]       scmat     scaling factor for the input matrix
 * @param[in]       scvec     scaling factor for the bias vector
 * @param[in]       scret     scaling factor for the output matrix
 * @return          none
 * @example         mat       = {1324, 5453, 3454, 3435, 8789, 3411, 5412, 8934}
 *                  vec       = {8452, 2341, 9383, 2353}
 *                  nrows     = 4
 *                  ncols     = 2
 *                  ret       = {2775, 3311, 4072, 2305, 6507, 2290, 5051, 5055}
 *                  scmat     = 1
 *                  scvec     = 2
 *                  scret     = 2
 */
void m_q_add_vec(const INT_T* const mat, const INT_T* const vec,
                 ITER_T nrows, ITER_T ncols, INT_T* const ret,
                 SCALE_T scmat, SCALE_T scvec, SCALE_T scret);
/**
 * @brief Performs the column-wise subtraction of a bias term from the input matrix.
 * dim(mat) = dim(ret) = [nrows][ncols]; dim(vec) = [ncols].
 * @param[in]       mat       pointer to the input matrix from which subtraction is to be performed
 * @param[in]       vec       pointer to the bias vector which is to be subtracted
 * @param[in]       nrows     number of rows of the input matrix
 * @param[in]       ncols     number of columns of the input matrix
 * @param[out]      ret       pointer to the output matrix
 * @param[in]       scmat     scaling factor for the input matrix
 * @param[in]       scvec     scaling factor for the bias vector
 * @param[in]       scret     scaling factor for the output matrix
 * @return          none
 * @example         mat       = {1324, 5453, 3454, 3435, 8789, 3411, 5412, 8934}
 *                  vec       = {8452, 2341, 9383, 2353}
 *                  nrows     = 4
 *                  ncols     = 2
 *                  ret       = {-1451, 2141, -618, 1129, 2281, 1120, 361, 3879}
 *                  scmat     = 1
 *                  scvec     = 2
 *                  scret     = 2
 */
void m_q_sub_vec(const INT_T* const mat, const INT_T* const vec,
                 ITER_T nrows, ITER_T ncols, INT_T* const ret,
                 SCALE_T scmat, SCALE_T scvec, SCALE_T scret);
// Function for multiplying a matrix with a vector.
void m_q_mulvec(const INT_T* const mat, const INT_T* const vec, ITER_T nrows,
                ITER_T ncols, INT_T* const ret, SCALE_T scmat, SCALE_T scvec,
                SCALE_T H1, SCALE_T H2);
/**
 * @brief Performs sparse matrix multiplication of a matrix and a vector.
 * mat_indices and mat_values combined are a sparse representation; dim(mat_values) = [ndims], dim(mat_indices) = [ndims + ncols].
 * mat_values[i] is the i^th non-zero value of the input matrix, and mat_indices[i] encodes the location of mat_values[i].
 * Number of zeroes before Aidx[i] : row of Aval[i]
 * Aidx[i] + ... + Aidx[l] where l is the largest value less than i such that A[idx] = 0 : column of Aval[i]
 * @param[in]       mat_indices  pointer to input matrix which evaluates to matrix A
 * @param[in]       mat_values   pointer to input matrix which evaluates to matrix A
 * @param[in]       vec          pointer to the input vector
 * @param[in]       ndims        dimension of mat_values matrix
 * @param[out]      ret          pointer to the output matrix
 * @param[in]       scmat        scale factor of the input matrix
 * @param[in]       scvec        scale factor of the input vector
 * @param[in]       scret        scale factor of the output matrix
   @return          none
 * @example         mat_indices  = {1, 2, 3, 4, 5, 6, 7, 0}
 *                  mat_values   = {10, 20, 30, 40, 50, 60, 70, 80}
 *                  vec          = {1, 2}
 *                  ndims        = 2
 *                  scmat        = 1
 *                  scvec        = 2
 *                  scret        = 4
 *                  ret          = {1, 2, 8, 5, 6, 7, 8, 0}
 */
void m_q_sparse_mulvec(const INT_T* const mat_indices, const INT_T* const mat_values,
                       const INT_T* const vec, ITER_T ndims, INT_T* const ret,
                       SCALE_T scmat, SCALE_T scvec, SCALE_T scret);

/**
 * @brief Performs the channel-wise addition of a bias term to the input tensor.
 * dim(mat) = dim(ret) = [nbatches][nrows][ncols][nchannels]; dim(vec) = [nchannels].
 * @param[in]       mat       pointer to the input tensor on which addition is to be performed
 * @param[in]       vec       pointer to the bias vector which is to be added
 * @param[in]       nbatches  number of batches of the input tensor
 * @param[in]       nrows     number of rows of the input tensor
 * @param[in]       ncols     number of columns of the input tensor
 * @param[in]       nchannels number of channels of the input tensor
 * @param[out]      ret       pointer to the output tensor
 * @param[in]       scmat     scaling factor for the input tensor
 * @param[in]       scvec     scaling factor for the bias vector
 * @param[in]       scret     scaling factor for the output tensor
 * @return          none
 * @example         mat       = { { {1324, 5453}, {3454, 3435} },
 *                                { {8789, 3411}, {5412, 8934} } },
 *                              { { {6895, 1211}, {6790, 5425} },
 *                                { {8976, 4539}, {9348, 9321} } }
 *                  vec       = {8452, 2341}
 *                  nrows     = 4
 *                  ncols     = 2
 *                  ret       = { { {2775, 3311}, {3840, 2302} },
 *                                { {6507, 2290}, {4819, 5052} } },
 *                              { { {5560, 1190}, {5508, 3297} },
 *                                { {6601, 2854}, {6787, 5245} } }
 *                  scmat     = 1
 *                  scvec     = 2
 *                  scret     = 2
 */
void t_q_add_vec(const INT_T* const mat, const INT_T* const vec,
                 ITER_T nbatches, ITER_T nrows, ITER_T ncols,
                 ITER_T nchannels, INT_T* const ret, SCALE_T scmat,
                 SCALE_T scvec, SCALE_T scret);
/**
 * @brief Performs the channel-wise subtraction of a bias term from the input tensor.
 * dim(mat) = dim(ret) = [nbatches][nrows][ncols][nchannels]; dim(vec) = [nchannels].
 * @param[in]       mat       pointer to the input tensor from which subtraction is to be performed
 * @param[in]       vec       pointer to the bias vector which is to be subtracted
 * @param[in]       nbatches  number of batches of the input tensor
 * @param[in]       nrows     number of rows of the input tensor
 * @param[in]       ncols     number of columns of the input tensor
 * @param[in]       nchannels number of channels of the input tensor
 * @param[out]      ret       pointer to the output tensor
 * @param[in]       scmat     scaling factor for the input tensor
 * @param[in]       scvec     scaling factor for the bias vector
 * @param[in]       scret     scaling factor for the output tensor
 * @return          none
 * @example         mat       = { { {1324, 5453}, {3454, 3435} },
 *                                { {8789, 3411}, {5412, 8934} } },
 *                              { { {6895, 1211}, {6790, 5425} },
 *                                { {8976, 4539}, {9348, 9321} } }
 *                  vec       = {8452, 2341}
 *                  nrows     = 4
 *                  ncols     = 2
 *                  ret       = { { {-1451, 2141}, {-386, 1132} },
                                  { {2281, 1120}, {593, 3882} } },
                                { { {1334, 20}, {1282, 2127} },
                                  { {2375, 1684}, {2561, 4075} } }
 *                  scmat     = 1
 *                  scvec     = 2
 *                  scret     = 2
 */
void t_q_sub_vec(const INT_T* const ten, const INT_T* const vec,
                 ITER_T nbatches, ITER_T nrows, ITER_T ncols,
                 ITER_T nchannels, INT_T* const ret, SCALE_T scmat,
                 SCALE_T scvec, SCALE_T scret);

/**
 * @brief Computes the maxpool of mat_in and stores the result in mat_out
 * Dimensions:  mat_in, mat_out are matrices, dim(mat_in) = dim(mat_out) = [N][H][W][C]
 * @param[in]       mat_in       pointer to input matrix
 * @param[out]      mat_out      pointer to output matrix
 * @param[in]       N            dimension of matrices
 * @param[in]       H            dimension of matrices
 * @param[in]       W            dimension of matrices
 * @param[in]       C            dimension of matrices
 * @param[in]       FH           FH, FW : Size of filter amongst which max is taken
 * @param[in]       FW           FH, FW : Size of filter amongst which max is taken
 * @param[in]       strideH      Convolution horizontal stride
 * @param[in]       strideW      Convolution vertical stride
 * @param[in]       HPADL        Thickness of padding on top of the image
 * @param[in]       HPADR        Thickness of padding on bottom of the image
 * @param[in]       WPADL        Thickness of padding on left of the image
 * @param[in]       WPADR        Thickness of padding on right of the image
 * @param[in]
 * @return          none
 * @example         mat_in       = {
 *                                    1, 2,
 *                                    3, 4,
 *
 *                                    5, 6,
 *                                    7, 8,
 *
 *
 *                                    9, 10,
 *                                    11, 12,
 *
 *                                    13, 14,
 *                                    15, 16
 *                                 }
 *
 *                  N,H,W,C      = 2, 2, 2, 2
 *                  FW, FH       = 2, 2
 *                  strideH      = 1
 *                  strideW      = 1
 *                  HPADL, HPADR = 1, 1
 *                  WPADL, WPADR = 1, 1
 *                  mat_out      = {7, 8,
 *                                  9, 10,
 *
 *                                  11, 12,
 *                                  13, 14,
 *
 *                                  15, 16,
 *                                  15, 16,
 *
 *                                  15, 16,
 *                                  15, 16}
 */
void q_maxpool(const INT_T* const mat, ITER_T nbatches, ITER_T nrows, ITER_T ncols,
               ITER_T nchannels, ITER_T hfilter, ITER_T wfilter, ITER_T hstride,
               ITER_T wstride, ITER_T hpadl, ITER_T hpadr, ITER_T wpadl,
               ITER_T wpadr, INT_T* const ret);

/**
 * @brief Computes the convolution of batched and multi-channeled 2D image A with filter B, and stores the result in C, using tmp as a buffer
 * Dimensions:	A, B, C are matrices, dim(A) = [N][H][W][CI], dim(B) = [G][HF][WF][CINF][COUTF], dim(C) = [N][HOUT][WOUT][COUTF*G], dim(tmp) = [HF*WF*CINF]
 * @param[in]       A            pointer to input matrix which corresponds to image
 * @param[in]       B            pointer to input matrix which corresponds to convolution filter
 * @param[out]      C            pointer to output matrix
 * @param[in]       tmp          pointer to temp storage
 * @param[in]       N            dimension of input matrix A, dimension of output matrix C
 * @param[in]       H            dimension of input matrix A
 * @param[in]       W            dimension of input matrix A
 * @param[in]       CI           dimension of input matrix A
 * @param[in]       HF           dimension of input matrix B
 * @param[in]       WF           dimension of input matrix B
 * @param[in]       CINF         dimension of input matrix B
 * @param[in]       COUTF        dimension of input matrix B
 * @param[in]       HOUT         dimension of input matrix C
 * @param[in]       WOUT         dimension of input matrix C
 * @param[in]       HPADL        Thickness of padding on top of the image
 * @param[in]       HPADR        Thickness of padding on bottom of the image
 * @param[in]       WPADL        Thickness of padding on left of the image
 * @param[in]       WPADR        Thickness of padding on right of the image
 * @param[in]       HSTR         Convolution horizontal stride
 * @param[in]       WSTR         Convolution vertical stride
 * @param[in]       HDL          Convolution horizontal dilations
 * @param[in]       WDL          Convolution vertical dilations
 * @param[in]       G            Number of groups
 * @param[in]       shrA         scale factor for input matrix A. dividing input matrices' elements to prevent overflows
 * @param[in]       shrB         scale factor for input matrix B. dividing input matrices' elements to prevent overflows
 * @param[in]       H1           fixed point arithmetic
 * @param[in]       H2           fixed point arithmetic
 * @return          none
 * @example         A            = {
 *                                    11, 220,
 *                                    130, 40,
 *
 *                                    50, 60,
 *                                    66, 76,
 *
 *
 *                                    86, 910,
 *                                    411, 312,
 *
 *                                    513, 514,
 *                                    715, 716
 *                                 }
 * 100, 992, 15, 26, 27, 8, 3, 4, 5, 2, 2, 2, 7, 8, 29, 140
 *
 *                  B            = {
 *                                    100, 992,
 *                                    15, 26,
 *
 *                                    27, 8,
 *                                    3, 4,
 *
 *
 *                                    5, 2,
 *                                    2, 2,
 *
 *                                    7, 8,
 *                                    29, 140
 *                                 }
 *                  shrA         = 8
 *                  shrB         = 8
 *                  C            = {
 *                                    0, 0,
 *                                    0, 0,
 *
 *                                    7, 6,
 *                                    7, 6,
 *
 *                                    0, 0,
 *                                    0, 0,
 *
 *                                    39, 33,
 *                                    39, 33
 *                                 }
 */
void q_convolution(const INT_T* const mat, const INT_T* const filter,
                   INT_T* const treesumBuffer, INT_T N, INT_T H, INT_T W,
                   INT_T CIN, INT_T HF,INT_T WF, INT_T CINF, INT_T COUTF,
                   INT_T HOUT, INT_T WOUT, INT_T HPADL, INT_T HPADR, INT_T WPADL,
                   INT_T WPADR, INT_T HSTR, INT_T WSTR, INT_T HDL, INT_T WDL,
                   INT_T G, INT_T* const ret, INT_T shrA, INT_T shrB, INT_T H1,
                   INT_T H2);

#endif
