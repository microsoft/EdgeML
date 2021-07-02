// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __UTILS_H__
#define __UTILS_H__

#include <math.h>
#include <float.h>

float min(float a, float b);
float max(float a, float b);

float relu(float x);
float sigmoid(float x);
float tanhyperbolic(float x);
float quantTanh(float x);
float quantSigmoid(float x);

void v_relu(const float* const vec, unsigned len, float* const ret);
void v_sigmoid(const float* const vec, unsigned len, float* const ret);
void v_tanh(const float* const vec, unsigned len, float* const ret);
void v_quantSigmoid(const float* const vec, unsigned len, float* const ret);
void v_quantTanh(const float* const vec, unsigned len, float* const ret);

/* Scaled matrix-vector multiplication:  ret = alpha * ret + beta * mat * vec
   alpha and beta are scalars
   ret is of size nrows, vec is of size ncols
   mat is of size nrows * ncols, stored in row major */
void matVec(const float* const mat, const float* const vec,
  unsigned nrows, unsigned ncols,
  float alpha, float beta,
  float* const ret);

/* 
  Matrix-vector multiplication with a row offset
  This function was developed primarily for the conv1d function. This helps bypass the permutation of the time and channel axis
  ret is of size nrows, vec is of size ncols
  mat is of size nrows * ncols, stored in row major
  depthwise is to change the matVec to depthwise specific convolutions
  row_stride is the offset factor between two adjacent rows
  Note : This matrix-vector multiplication is useful for matrices where a certain number of columns are dropped
  For a normal matVec case, this value will be ncols
  Eg : For a 400 x 400 matrix and a 100 length vector, we can consider the top 400 x 100 elements for the multiplication. 
   Eg : For a 400 x 400 matrix and a 100 length vector, we can consider the top 400 x 100 elements for the multiplication. 
  Eg : For a 400 x 400 matrix and a 100 length vector, we can consider the top 400 x 100 elements for the multiplication. 
      For this eg ncols will be 100 and row_stride will be 400
  vec_stride is the offset fector between 2 elements in a vector i.e. the elements of a vector are placed at "n" intervals
  For a normal matVec case, this value will be 1
  Eg : For matVec with a 400 x 100 matrix a vector of length 100 is needed. 
   Eg : For matVec with a 400 x 100 matrix a vector of length 100 is needed. 
  Eg : For matVec with a 400 x 100 matrix a vector of length 100 is needed. 
      So it's possible to enter a 400 length vector and consider every 4th element. 
        So it's possible to enter a 400 length vector and consider every 4th element. 
      So it's possible to enter a 400 length vector and consider every 4th element. 
      For this ncols will be 100 and vec_stride will be 4 
*/
void offset_matVec_conv1d(const float* mat, const float* vec,
  unsigned nrows, unsigned ncols,
  unsigned row_stride, unsigned vec_stride,
  unsigned depthwise, float* ret);

/* 
  Tiled (cache-blocked) implementation of the Matrix Multiplication
  Note: If only the MatMul output is needed, then please use calloc to initialize the output
  An alternative is to use malloc, followed by memset 0
  There is second way to use this function. This is for adding the result of the MatMul to a pre-existing matrix
  If there is a pre-existing [nrows, ncols] matrix that needs to be added to the MatMul output, then pass that matrix directly
  This MatMul adds the result on the pre-existing values in ret. Hence either a zero initialized or a pre-existing mat is needed
  matA           first matrix; shape = [nrows, ncommon]
  matB           second matrix; shape = [ncommon, ncols]
  nrows          number of rows in the first matrix
  ncommon        number of columns in the first matrix/number of rows in the second matrix
  ncols          number of columns in the second matrix
  total_comm_A   The actual offset factor between 2 rows for matA. Used if we need fewer columns than the actual number stored
  total_cols_B   The actual offset factor between 2 rows for matB. Used if we need fewer columns than the actual number stored. 
   total_cols_B   The actual offset factor between 2 rows for matB. Used if we need fewer columns than the actual number stored. 
  total_cols_B   The actual offset factor between 2 rows for matB. Used if we need fewer columns than the actual number stored. 
  ret            matrix multiplication output. shape = [nrows, ncols]
  block_size     tile/block size for optimal cache performance. A hardware specific parameter
*/
void tiledMatMul_float(const float* const matA, const float* const matB,
  unsigned nrows, unsigned ncommon, unsigned ncols,
  unsigned total_comm_A, unsigned total_cols_B,
  float* const ret, unsigned block_size);

/* 
  Tiled (cache-blocked) implementation of the Matrix Multiplication, but with matB stored in the transposed format
  The result will the same as the regular MatMul but the matrix B provided will be pre-transposed (before the storage or usage)
  Note: If only the MatMul output is needed, then please use calloc to initialize the output
  An alternative is to use malloc, followed by memset 0
  There is second way to use this function. This is for adding the result of the MatMul to a pre-existing matrix
  If there is a pre-existing [nrows, ncols] matrix that needs to be added to the MatMul output, then pass that matrix directly
  This MatMul adds the result on the pre-existing values in ret. Hence either a zero initialized or a pre-existing mat is needed
  matA           first matrix; shape = [nrows, ncommon]
  matB           second matrix; shape = [ncols, ncommon]
  nrows          number of rows in the first matrix
  ncommon        number of columns in the first matrix/number of rows in the second matrix
  ncols          number of columns in the second matrix
  total_comm_A   The actual offset factor between 2 rows for matA. Used if we need fewer columns than the actual number stored
  total_comm_B   The actual offset factor between 2 rows for matB. Used if we need fewer columns than the actual number stored. 
   total_comm_B   The actual offset factor between 2 rows for matB. Used if we need fewer columns than the actual number stored. 
  total_comm_B   The actual offset factor between 2 rows for matB. Used if we need fewer columns than the actual number stored. 
                 Since matB is transposed the columns are now the ncomm axis
  ret            matrix multiplication output. shape = [nrows, ncols]
  block_size     tile/block size for optimal cache performance. A hardware specific parameter
*/
void transposed_tiledMatMul(const float* const matA, const float* const matB,
  unsigned nrows, unsigned ncommon, unsigned ncols,
  unsigned total_comm_A, unsigned total_comm_B,
  float* const ret, unsigned block_size);

// scaled vector addition: ret = scalar1 * vec1 + scalar2 * vector2
void v_add(float scalar1, const float* const vec1,
  float scalar2, const float* const vec2,
  unsigned len, float* const ret);

// point-wise vector multiplication ret = vec2 * vec1 
void v_mult(const float* const vec1, const float* const vec2,
  unsigned len, float* const ret);

// point-wise vector division ret = vec2 / vec1 
void v_div(const float* const vec1, const float* const vec2,
  unsigned len, float* const ret);

// Return squared Euclidean distance between vec1 and vec2
float l2squared(const float* const vec1,
  const float* const vec2, unsigned dim);

// Return index with max value, if tied, return first tied index.
unsigned argmax(const float* const vec, unsigned len);

// ret[i] = exp(input[i]) / \sum_i exp(input[i])
void softmax(const float* const input, unsigned len, float* const ret);

/* Custom non-linear layer for the phoneme detection model. It can be used for other time-series problems if necessary
   output_signal    pointer to the output signal, size = out_time * (in_channels / 2)
   input_signal     pointer to the input signal. size = in_time * in_channels
   in_time          number of input time steps
   in_channels      number of input channels. The output will have the half the number of input channels. 
                    Necessary for in_channels % 2 == 0
 */
void semi_sigmoid_tanh(float* output_signal, const float* const input_signal,
  unsigned in_time, unsigned in_channels);

#endif
