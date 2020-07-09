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

// Functions for calculating quantized operations and activations.
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
                ITER_T H1, ITER_T H2);
/**
 * @brief Figures out the index of largest element in vector
 * @param[in]       vec          pointer to input vector
 * @param[in]       len          length of the vector
 * @param[in]       index        pointer to a variable where the index of largest element found, to be stored
 * @return          none
 * @example         vec          = {12, 24, 54, 1, 2, 10}
 *                  index        = 2
*/
void arg_max(INT_T *vec, INT_T len, INT_T *index);

/**
 * @brief Performs the transpose of matrix
 * @param[in]       mat_in       pointer to input matrix whose transpose is to be performed.
 * @param[in]       mat_out      pointer to output matrix which will hold transpose of input matrix mat_in
 * @param[in]       nrows        number of rows of output matrix
 * @param[in]       ncols        number of columns of output matrix
   @return          none
 * @example         vec_in       = 1, 2
 *                                 4, 5
 *                            
 *                  vec_out      = 1 4
 *                                 2 5
 * 
 * @example         vec_in       = 1, 2, 3
 *                                 4, 5, 6
 *                   
 *                  vec_out      = 1  4
 *                                 2  5
 *                                 3  6
*/
void transpose(INT_T *mat_in, INT_T *mat_out, INT_T nrows, INT_T ncols);

/**
 * @brief Performs the channel-wise addition/subtraction of matrices.
 * Dimensions: 	mat_in, mat_bias, mat_out are matrices, dim(mat_in) = dim(mat_out) = [nbatch][nrows][ncols][nchannel], dim(mat_bias) = [nchannel]
 * For c over all channel (nchannel) dimensions, add/subtract scalar mat_bias[nchannel] to all values of mat_in[:][:][:][nchannel] and store in mat_out.
 * @param[in]       mat_in        pointer to input matrix on which addition/substraction to be performed
 * @param[in]       mat_bias      pointer to bias matrix which is added/subsstracted to input matrix mat_in
 * @param[out]      mat_out       pointer to output matrix where the final operational result is to be stored.
 * @param[in]       nbatch        number of batch
 * @param[in]       nrows         rows of matrix
 * @param[in]       ncols         columns of matrix
 * @param[in]       nchannel      number of channel
 * @param[in]       scl_a         Scaling factor for mat_in. scl_a, scl_b are used to bring matrices A and B to the same scale for addition
 * @param[in]       scl_b         Scaling factor for mat_bias. scl_a, scl_b are used to bring matrices A and B to the same scale for addition
 * @param[in]       scl_out       Scaling factor for mat_out. scl_out adjusts the output matrix if required to prevent overflows
 * @param[in]       add           flag to defining functionality. 1 for addtion 0 for substraction
 * @return          none
 * @example         mat_in        = { { {1324, 5453}, {3454, 3435}},
 *                                    { {8789, 3411}, {5412, 8934}} }, 
 *                                  { { {6895, 1211}, {6790, 5425}}, 
 *                                    { {8976, 4539}, {9348, 9321}} }}
 * 
 *                  mat_bias      = { { {8452, 2341}, {9383, 2353}}, 
*                                     { {4522, 6232}, {2562, 565}} },
 *                                    { {4564, 7756}, {2585, 8735}}, 
 *                                    { {3525, 4341}, {4656, 2313}} }}
 *                  scl_a         = 1
 *                  scl_b         = 2
 *                  scl_out       = 2
 *                  mat_out       = { { {2775, 3311}, {3840, 2302}}, 
 *                                    { {6507, 2290}, {4819, 5052}} },
 *                                  { { {5560, 1190}, {5508, 3297}}, 
 *                                    { {6601, 2854}, {6787, 5245}} }}
*/
void add_or_sub_cir_4D(INT_T *mat_in, const INT_T *mat_bias, INT_T *mat_out, 
                       INT_T nbatch, INT_T nrows, INT_T ncols, INT_T nchannel, 
                       INT_T scl_a, INT_T scl_b, INT_T scl_out, uint8_t add);

/**
 * @brief Performs the channel-wise addition/subtraction of matrices
 * Dimensions: 	mat_in, mat_bias, mat_out are matrices, dim(mat_in) = dim(mat_out) = [nbatch][nrows][ncols][nchannel], dim(mat_bias) = [nchannel]
 * For c over all channel (nchannel) dimensions, add/subtract scalar mat_bias[nchannel] to all values of mat_in[:][:][:][nchannel] and store in mat_out.
 * @param[in]       mat_in       pointer to input matrix on which addition/substraction to be performed
 * @param[in]       mat_bias     pointer to bias matrix which is added/subsstracted to input matrix
 * @param[out]      mat_out      pointer to output matrix where the final operational result is to be stored.
 * @param[in]       nrows        rows of matrix
 * @param[in]       ncols        columns of matrix
 * @param[in]       scl_a        Scaling factor for mat_in matrix
 * @param[in]       scl_b        Scaling factor for mat_bias matrix
 * @param[in]       scl_out      Scaling factor for mat_out matrix
 * @param[in]       add          flag to defining functionality. 1 for addtion 0 for substraction
 * @return          none
 * @example         mat_in       = {1324, 5453, 3454, 3435, 8789, 3411}
 *                  mat_bias     = {8452, 2341, 9383, 2353, 4522, 6232}
 *                  scl_a        = 1
 *                  scl_b        = 2
 *                  scl_out      = 2
 *                  mat_out      = {2775, 3311, 4072, 2305, 6507, 2290}
*/
void add_or_sub_cir_2D(INT_T *mat_in, const INT_T *mat_bias, INT_T *mat_out, 
                       INT_T nrows, INT_T ncols, INT_T scl_a, INT_T scl_b, 
                       INT_T scl_out, uint8_t add);

/**
 * @brief Eliminate any negative value element present in vector by replacing it with 0.
 * @param[in/out]   mat        pointer to matrix from which negative elements to be eliminated by replacing with 0
 * @param[in]       length     length of nD matrix
 * @return          none
 * @example         length     = 4 
                    mat        = {1324, -5453, 3454, -3435, 8789}
 *                  mat        = {1324, 0, 3454, 0, 8789}
*/
void relu(INT_T *mat, INT_T length);

/**
 * @brief Computes exponentiation of all elements in mat_in (interpreted as a floating point value) to the base e and stores the result in mat_out
 * @param[in]       mat_in       pointer to matrix whose exponential scaling is to be performed.
 * @param[in]       length       length of matrix. length = nrows * ncols
 * @param[in]       scl_in       scaling factor for input matrix. Dividing (float division) each element of matrix mat_in by shrA gives the floating point matrix of mat_in
 * @param[in]       scl_out      scaling factor for output matrix. Dividing (float division) each element of matrix mat_out by shrB gives the floating point matrix of mat_out
 * @param[out]      mat_out      pointer to matrix while will store exponential scaling performed on input matrix.
 * @return          none
 * @example         mat_in       = {13, 54, 34, 35, 87}
 *                  scl_in       = 100
 *                  scl_out      = 200
 *                  mat_out      = {227, 343, 280, 283, 477}
*/
void exp_scale(INT_T *mat_in, INT_T length, INT_T scl_in, INT_T scl_out, INT_T *mat_out);

/**
 * @brief Performs down scaling on matrix. Divides all elements of mat by scale and stores the result in mat. 
 * @param[in/out]   mat          pointer to matrix on which down scaling is to be performed.
 * @param[in]       length       length of matrix
 * @param[in]       scale        scaling factor of matrix
 * @return          none
 * @example         mat          = {4232, -9879, -2342,1232}
 *                  length       = 4
 *                  scale        = 37
 *                  mat          = {114, -267, -63, 33}
*/
void adjust_scale_shr(INT_T *mat, INT_T length, INT_T scale);

/**
 * @brief Performs up scaling on matrix. Multiplies all elements of mat by scale and stores the result in mat
 * @param[in/out]   mat          pointer to matrix on which up scaling is to be performed.
 * @param[in]       length       length of matrix
 * @param[in]       scale        scaling factor of matrix
 * @return          none
 * @example         mat          = {423, -987, -2342,1232}
 *                  length       = 4
 *                  scale        = 10
 *                  mat          = {4230, -9870, -23420, 12320}
*/
void adjust_scale_shl(INT_T *mat, INT_T length, INT_T scale);

/**
 * @brief Reverses the matrix A along axis (can be 0 for axis nrows and 1 for axis ncols)
 * @param[in]       mat_in       pointer to matrix on which reversal is to be performed.
 * @param[in]       axis         direction of reversal. axis = 0 for reversing along rows, axis = 1 for reversing along columns.
 * @param[in]       nrows        rows of matrix
 * @param[in]       ncols        columns of matrix
 * @param[out]      mat_out      pointer to matrix which will store the performed reversal operation on input matrix
 * @return          none
 * @example         mat_in       = {4232, -9879, -2342, 1232, -3242, 8432, 9823, 2342}
 *                  ncols        = 4
 *                  axis         = 1
 *                  mat_out      = {1232, -2342, -9879, 4232, 2342, 9823, 8432, -3242}

*/
void Reverse2(INT_T *mat_in, INT_T axis, INT_T nrows, INT_T ncols, INT_T *mat_out);

/**
 * @brief Computes the convolution of batched and multi-channeled 2D image A with filter B, and stores the result in C, using tmp as a buffer
 * Dimensions:	A, B, C are matrices, dim(A) = [N][H][W][CI], dim(B) = [G][HF][WF][CINF][COUTF], dim(C) = [N][HOUT][WOUT][COUTF*G], dim(tmp) = [HF*WF*CINF]
 * @param[in]       A           pointer to input matrix which corresponds to image
 * @param[in]       B           pointer to input matrix which corresponds to convolution filter
 * @param[out]      C           pointer to output matrix
 * @param[in]       tmp         pointer to temp storage
 * @param[in]       N           dimension of input matrix A, dimension of output matrix C
 * @param[in]       H           dimension of input matrix A
 * @param[in]       W           dimension of input matrix A
 * @param[in]       CI          dimension of input matrix A
 * @param[in]       HF          dimension of input matrix B
 * @param[in]       WF          dimension of input matrix B
 * @param[in]       CINF        dimension of input matrix B
 * @param[in]       COUTF       dimension of input matrix B
 * @param[in]       HOUT        dimension of input matrix C
 * @param[in]       WOUT        dimension of input matrix C
 * @param[in]       HPADL       Thickness of padding on top of the image
 * @param[in]       HPADR       Thickness of padding on bottom of the image
 * @param[in]       WPADL       Thickness of padding on left of the image
 * @param[in]       WPADR       Thickness of padding on right of the image
 * @param[in]       HSTR        Convolution horizontal stride
 * @param[in]       WSTR        Convolution vertical stride
 * @param[in]       HDL         Convolution horizontal dilations
 * @param[in]       WDL         Convolution vertical dilations
 * @param[in]       G           Number of groups
 * @param[in]       shrA        scale factor for input matrix A. dividing input matrices' elements to prevent overflows
 * @param[in]       shrB        scale factor for input matrix B. dividing input matrices' elements to prevent overflows
 * @param[in]       H1          fixed point arithmetic
 * @param[in]       H2          fixed point arithmetic
   @return          none
 * @example         A           = {
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
 *                  B           = {
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
 *                  shrA        = 8
 *                  shrB        = 8
 *                  C           = {
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
 *                                }
 *                  
 *
*/
void convolution(INT_T *A, const INT_T *B, INT_T *C, INT_T *tmp, 
                 INT_T N, INT_T H, INT_T W, INT_T CIN, INT_T HF,        
                 INT_T WF, INT_T CINF, INT_T COUTF, INT_T HOUT,         
                 INT_T WOUT, INT_T HPADL, INT_T HPADR, INT_T WPADL,     
                 INT_T WPADR, INT_T HSTR, INT_T WSTR, INT_T HDL,        
                 INT_T WDL, INT_T G, INT_T shrA, INT_T shrB, INT_T H1,  
                 INT_T H2);

/**
 * @brief Computes the sigmoid activation for all elements of mat_in and stores the result in mat_out
 * Dimensions:	mat_in, mat_out are matrices, dim(mat_in) = dim(mat_out) = [nrows][ncols]; div, add, sigmoid_limit, scale_in, scale_out are integers.
 * @param[in]       mat_in              pointer to input matrix
 * @param[in]       nrows               rows of matrices
 * @param[in]       ncols               columns of matrices
 * @param[in]       div                 division value
 * @param[in]       add                 addition value
 * @param[in]       sigmoid_limit       sigmoid limit value
 * @param[in]       scale_in            Dividing (float division) each element of matrix mat_in by scale_in gives the floating point matrix of mat_in
 * @param[in]       scale_out           Dividing (float division) each element of matrix mat_out by scale_out gives the floating point matrix of mat_out
 * @param[out]      mat_out             pointer to output matrix
   @return          none
 * @example         mat_in              = 1, 2, 3, 4, 5, 6, 7, 8
 *                  nrows               = 4
 *                  ncols               = 4
 *                  div                 = 2
 *                  add                 = 1
 *                  sigmoid_limit       = 5
 *                  scale_in            = 5
 *                  scale_out           = 10
 *                  mat_out             = 21, 27, 11, 15, 53, 67, 23, 31
*/

void sigmoid(INT_T *mat_in, INT_T nrows, INT_T ncols, INT_T div, INT_T add, 
             INT_T sigmoid_limit, INT_T scale_in, INT_T scale_out, INT_T *mat_out);

/**
 * @brief  Sparse Matrix Multiplication. Compute A * B and store it in C.
 * Dimensions: 	A, B, C are matrices. dim(A) = [I][J], dim(B) = [J][1], dim(C)  [I][1]
 * Aval, Aidx combined is a sparse representation of A. dim(Aval) = [K], dim(Aidx) = [K+J].
 * Representation:	Aval[i] is the i^th non-zero value of A, and Aidx[i] encodes the location of Aval[i].
 * Number of zeroes before Aidx[i] : row of Aval[i]
 * Aidx[i] + ... + Aidx[l] where l is the largest value less than i such that A[idx] = 0 : column of Aval[i]
 * @param[in]       Aidx     pointer to input matrix which eveluates to matrix A
 * @param[in]       Aval     pointer to input matrix which eveluates to matrix A
 * @param[in]       B        pointer to input matrix which performs sparce matrix multiplication with matrix A
 * @param[out]      C        pointer to output matrix
 * @param[in]       K        dimension of Aval matrix.
 * @param[in]       shrA     scaling factor of input matrix A. shrA, adjusts the input matrix if required to prevent overflows
 * @param[in]       shrB     scaling factor of input matrix B. schrB adjusts the input matrix if required to prevent overflows
 * @param[in]       shrC     scaling factor of input matrix C. shrC adjusts the output matrix if required to prevent overflows
   @return          none
 * @example         Aidx     = 1, 2, 3, 4, 5, 6, 7 ,0
 *                  Aval     = 10, 20, 30, 40, 50, 60, 70, 80
 *                  B        = 1, 2
 *                  shrA     = 1
 *                  shrB     = 2
 *                  shrC     = 3
 *                  C        = 1 ,29 ,5 ,36 ,8 ,43 ,11 ,36
*/
void sp_mat_mul(const INT_T *Aidx, const INT_T *Aval, INT_T **B, INT_T *C, INT_T K,
                INT_T shrA, INT_T shrB, INT_T shrC);

/**
 * @brief Computes the maxpool of mat_in and stores the result in mat_out
 * Dimensions:	mat_in, mat_out are matrices, dim(mat_in) = dim(mat_out) = [N][H][W][C]
 * @param[in]       mat_in      pointer to input matrix
 * @param[out]      mat_out     pointer to output matrix
 * @param[in]       N           dimension of matrices
 * @param[in]       H           dimension of matrices
 * @param[in]       W           dimension of matrices
 * @param[in]       C           dimension of matrices
 * @param[in]       FH          FH, FW : Size of filter amongst which max is taken
 * @param[in]       FW          FH, FW : Size of filter amongst which max is taken
 * @param[in]       strideH     Convolution horizontal stride
 * @param[in]       strideW     Convolution vertical stride
 * @param[in]       HPADL       Thickness of padding on top of the image
 * @param[in]       HPADR       Thickness of padding on bottom of the image
 * @param[in]       WPADL       Thickness of padding on left of the image
 * @param[in]       WPADR       Thickness of padding on right of the image
 * @param[in]       
   @return          none
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
                    strideW      = 1
 *                  HPADL, HPADR = 1, 1
 *                  WPADL, WPADR = 1, 1
 *                  mat_out      = {
 *                                    7, 8, 
 *                                    9, 10, 
 * 
 *                                    11, 12, 
 *                                    13, 14, 
 * 
 *                                    15, 16, 
 *                                    15, 16, 
 *                       
 *                                    15, 16, 
 *                                    15, 16
 *                                  }
*/

void maxpool(INT_T *mat_in, INT_T *mat_out, INT_T N, INT_T H, INT_T W, INT_T C, INT_T FH, 
             INT_T FW, INT_T strideH, INT_T strideW, INT_T HPADL, INT_T HPADR, 
            INT_T WPADL, INT_T WPADR);

#endif
