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
// Function for subtracting every element of a vector B from a scalar a.
// The resultant vector has elements C_{i} = a - B_{i}.
void v_q_scalar_sub(INT_T scalar, const INT_T* const vec, ITER_T len,
                    INT_T* const ret, SCALE_T scscalar, SCALE_T scvec, SCALE_T scret);
// Function for subtracting a scalar b from every element of a vector A.
// The resultant vector has elements C_{i} = A_{i} - b.
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
 * Note: No saturation is done here, and hence, the output might overflow with a large input.
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
 * Note: No saturation is done here, and hence, the output might overflow with a large input.
 * @param[in]       vec       pointer to vector whose exponential scaling is to be performed
 * @param[in]       len       length of the vector
 * @param[in]       scvec     scaling factor for input vector
 * @param[in]       scret     scaling factor for output vector
 * @param[out]      ret       pointer to the output vector
 * @return          none
 * @example         formula   = exp((float)vec_{i} / scvec) * scret
 *                  vec       = {13, 54, 34, 35, 87}
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
 * @brief Performs the row-order or the column-order reversal of the 2-D input matrix.
 * @param[in]       mat       pointer to the (row / column-major) input matrix on which reversal is to be performed
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
 * col_indices and mat_values combined are a sparse representation; dim(vec) = [ndims].
 * mat_values[i] is the i^th non-zero value of the input matrix, and col_indices[i] encodes the location of mat_values[i].
 * Number of zeroes before col_indices[i] : row of mat_values[i]
 * col_indices[i - l] where l is the number of zeroes before col_indices[i]: column of mat_values[i]
 * @param[in]       col_indices  pointer to input matrix which stores the column indices of non-zero values of matrix A
 * @param[in]       mat_values   pointer to input matrix which stores the non-zero values of matrix A
 * @param[in]       vec          pointer to the input vector
 * @param[in]       ndims        dimension of the multiplication vector
 * @param[out]      ret          pointer to the output matrix
 * @param[in]       scmat        scale factor of the input matrix
 * @param[in]       scvec        scale factor of the input vector
 * @param[in]       scret        scale factor of the output matrix
   @return          none
 * @example         mat          = {{10, 20, 30, 40, 50, 60, 70, 0, 0, 0, 0, 0, 0, 0},
 *                                  {0, 80, 0, 90, 0, 100, 0, 110, 0, 120, 0, 130, 0, 140}}
 *                  col_indices  = {1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6, 8, 10, 12, 14, 0}
 *                  mat_values   = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140}
 *                  vec          = {1, 2}
 *                  ndims        = 2
 *                  scmat        = 1
 *                  scvec        = 2
 *                  scret        = 4
 *                  ret          = {1, 22, 3, 27, 6, 32, 8, 27, 0, 30, 0, 32, 0, 35}
 */
void m_q_sparse_mulvec(const ITER_T* const col_indices, const INT_T* const mat_values,
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
 * @brief Computes the maxpool operation on the input tensor with the given parameters.
 * Note: This implementation assumes no dilation by default.
 * @param[in]       input     pointer to the tensor on which max-pooling is to be performed
 * @param[out]      output    pointer to the output tensor
 * @param[in]       N         number of batches of the input tensor
 * @param[in]       H         number of rows of the input tensor
 * @param[in]       W         number of columns of the input tensor
 * @param[in]       CIn       number of channels of the input tensor
 * @param[in]       HF        number of rows of the pooling filter
 * @param[in]       WF        number of columns of the pooling filter
 * @param[in]       CF        number of channels of the pooling filter
 * @param[in]       COut      number of channels of the output tensor
 * @param[in]       HOut      number of rows of the output tensor
 * @param[in]       WOut      number of columns of the output tensor
 * @param[in]       G         number of groups of pooling filters
 * @param[in]       HPadU     padding over the top row
 * @param[in]       HPadD     padding under the bottom row
 * @param[in]       WPadL     padding before the leftmost column
 * @param[in]       WPadR     padding after the rightmost column
 * @param[in]       HStride   stride of the pooling filter along the rows, used for moving the receptive field horizontally within the larger image
 * @param[in]       WStride   stride of the pooling filter along the columns, used for moving the receptive field vertically within the larger image
 * @param[in]       scinput   scale of the input tensor
 * @param[in]       scoutput  scale of the output tensor
 * @return          none
 * @example         Please refer the test-case: test_quantized_maxpool()
 */
void q_maxpool(const INT_T* const input, INT_T* const output, ITER_T N,
               ITER_T H, ITER_T W, ITER_T CIn, ITER_T HF, ITER_T WF, ITER_T CF,
               ITER_T COut, ITER_T HOut, ITER_T WOut, ITER_T G, S_ITER_T HPadU,
               S_ITER_T HPadD, S_ITER_T WPadL, S_ITER_T WPadR, ITER_T HStride,
               ITER_T WStride, SCALE_T scinput, SCALE_T scoutput);

/**
 * @brief Computes the maxpool operation on the input tensor with the given parameters.
 * Note: This implementation assumes no dilation by default.
 * @param[in]       input          pointer to the tensor on which convolution is to be performed
 * @param[in]       filter         pointer to the convolutional filter tensor
 * @param[out]      output         pointer to the output tensor
 * @param[in]       treesumBuffer  pointer to the buffer for computing TreeSum accumulation
 * @param[in]       N              number of batches of the input tensor
 * @param[in]       H              number of rows of the input tensor
 * @param[in]       W              number of columns of the input tensor
 * @param[in]       CIn            number of channels of the input tensor
 * @param[in]       HF             number of rows of the convolutional filter
 * @param[in]       WF             number of columns of the convolutional filter
 * @param[in]       CF             number of channels of the convolutional filter
 * @param[in]       COut           number of channels of the output tensor
 * @param[in]       HOut           number of rows of the output tensor
 * @param[in]       WOut           number of columns of the output tensor
 * @param[in]       G              number of groups of convolutional filters
 * @param[in]       HPadU          padding over the top row
 * @param[in]       HPadD          padding under the bottom row
 * @param[in]       WPadL          padding before the leftmost column
 * @param[in]       WPadR          padding after the rightmost column
 * @param[in]       HStride        stride of the convolution filter along the rows, used for moving the receptive field horizontally within the larger image
 * @param[in]       WStride        stride of the convolution filter along the columns, used for moving the receptive field vertically within the larger image
 * @param[in]       H1             depth parameter for division by two used in TreeSum
 * @param[in]       H2             depth parameter for direct sum used in TreeSum
 * @param[in]       scinput        scale of the input tensor
 * @param[in]       scoutput       scale of the output tensor
 * @return          none
 * @example         Please refer the test-case: test_quantized_convolution()
 */
void q_convolution(const INT_T* const input, const INT_T* const filter,
                   INT_T* const output, INTM_T* const treesumBuffer, ITER_T N,
                   ITER_T H, ITER_T W, ITER_T CIn, ITER_T HF, ITER_T WF,
                   ITER_T CF, ITER_T COut, ITER_T HOut, ITER_T WOut, ITER_T G,
                   S_ITER_T HPadU, S_ITER_T HPadD, S_ITER_T WPadL,
                   S_ITER_T WPadR, ITER_T HStride, ITER_T WStride, SCALE_T H1,
                   SCALE_T H2, SCALE_T scinput, SCALE_T scoutput);

#endif
