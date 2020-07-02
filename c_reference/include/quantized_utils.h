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
 * @example         vec          = {12, 24,54, 1, 2, 10}
 *                  index        = 2
*/
void arg_max(INT_T *vec, INT_T len, INT_T *index);

/**
 * @brief Performs the transpose of matrix
 * @param[in]       mat_in       pointer to input matrix whose transpose is to be performed.
 * @param[in]       mat_out      pointer to output matrix which will hold transpose of input matrix
 * @param[in]       nrows        number of rows of input matrix
 * @param[in]       ncols        number of columns of input matrix
   @return          none
 * @example         vec_in       = 1, 2, 3
 *                                 4, 5, 6
 *                            
 *                  vec_out      = 1 4
 *                                 2 5
 *                                 3 6
*/
void transpose(INT_T *mat_in, INT_T *mat_out, INT_T nrows, INT_T ncols);

/**
 * @brief Performs the addition/substraction and then scaling of 4D circle.
 * @param[in]       mat_in        pointer to one input matrix on which addition/substraction to be performed
 * @param[in]       mat_bias      pointer to other input matrix on which addition/substraction to be performed
 * @param[out]      mat_out       pointer to output matrix where the final operational result is to be stored.
 * @param[in]       nbatch        number of batch
 * @param[in]       nrows         rows of matrix
 * @param[in]       ncols         columns of matrix
 * @param[in]       nchannel      number of channel
 * @param[in]       scl_a         Scaling factor for mat_in
 * @param[in]       scl_b         Scaling factor for mat_bias
 * @param[in]       scl_out       Scaling factor for mat_out
 * @param[in]       add           flag to defining functionality. 1 for addtion 0 for substraction
 * @return          none
 * @example         mat_in        = {1324, 5453, 3454, 3435, 8789}
 *                  mat_bias      = {8452, 2341, 9383, 2353, 4522}
 *                  scl_a         = 10
 *                  scl_b         = 20
 *                  scl_out       = 30
 *                  mat_out       = {18,21,25,14,43}
*/
void add_or_sub_cir_4D(INT_T *mat_in, const INT_T *mat_bias, INT_T *mat_out, INT_T nbatch, INT_T nrows, INT_T ncols, INT_T nchannel, INT_T scl_a, INT_T scl_b, INT_T scl_out, uint8_t add);

/**
 * @brief Performs the addition/substraction and then scaling of 2D circle.
 * @param[in]       mat_in       pointer to matrix on which addition/substraction to be performed
 * @param[in]       mat_bias     pointer to other matrix on which addition/substraction to be performed
 * @param[out]      mat_out      pointer to output matrix where the final operational result is to be stored.
 * @param[in]       nrows        rows of matrix
 * @param[in]       ncols        columns of matrix
 * @param[in]       scl_a        Scaling factor for mat_in matrix
 * @param[in]       scl_b        Scaling factor for mat_bias matrix
 * @param[in]       scl_out      Scaling factor for mat_out matrix
 * @param[in]       add          flag to defining functionality. 1 for addtion 0 for substraction
 * @return          none
 * @example         mat_in       = {1324, 5453, 3454, 3435, 8789}
 *                  mat_bias     = {8452, 2341, 9383, 2353, 4522}
 *                  scl_a        = 10
 *                  scl_b        = 20
 *                  scl_out      = 30
 *                  mat_out      = {18,21,25,14,43}
*/
void add_or_sub_cir_2D(INT_T *mat_in, const INT_T *mat_bias, INT_T *mat_out, INT_T nrows, INT_T ncols, INT_T scl_a, INT_T scl_b, INT_T scl_out, uint8_t add);

/**
 * @brief Eliminate any negative value element present in 2D/4D matrix by replacing it with 0.
 * @param[in/out]   mat        pointer to matrix from which negative elements to be eliminated by replacing with 0
 * @param[in]       length     length of nD matrix
 * @return          none
 * @example         length     = 4 
                    mat        = {1324, -5453, 3454, -3435, 8789}
 *                  mat        = {1324, 0, 3454, 0, 8789}
*/
void relu_nD(INT_T *mat, INT_T length);

/**
 * @brief Performs exponential scaling on matrix.
 * @param[in]       mat_in       pointer to matrix whose exponential scaling is to be performed.
 * @param[in]       length       length of matrix. length = nrows * ncols
 * @param[in]       scl_in       scaling factor for input matrix
 * @param[in]       scl_out      scaling factor for output matrix
 * @param[out]      mat_out      pointer to matrix while will store exponential scaling performed on input matrix
 * @return          none
 * @example         mat_in       = {13, 54, 34, 35, 87}
 *                  scl_in       = 100
 *                  scl_out      = 200
 *                  mat_out      = {227,343,280,283,477}
*/
void exp_scale(INT_T *mat_in, INT_T length, INT_T scl_in, INT_T scl_out, INT_T *mat_out);

/**
 * @brief Performs down scaling on matrix.
 * @param[in/out]   mat          pointer to matrix on which down scaling is to be performed.
 * @param[in]       length       length of matrix
 * @param[in]       scale        scaling factor of matrix
 * @return          none
 * @example         mat          = {4232, -9879, -2342,1232}
 *                  length       = 4
 *                  scale        = 37
 *                  mat          = {114 ,-267 ,-63 ,33}
*/
void adjust_scale_shr(INT_T *mat, INT_T length, INT_T scale);

/**
 * @brief Performs up scaling on matrix.
 * @param[in/out]   mat          pointer to matrix on which up scaling is to be performed.
 * @param[in]       length       length of matrix
 * @param[in]       scale        scaling factor of matrix
 * @return          none
 * @example         mat          = {4232, -987, -2342,1232}
 *                  length       = 4
 *                  scale        = 10
 *                  mat          = {42320, -9870, -23420,12320}
*/
void adjust_scale_shl(INT_T *mat, INT_T length, INT_T scale);

/**
 * @brief Performs reversal on matrix by given columns and in given direction
 * @param[in]       mat_in       pointer to matrix on which reversal is to be performed.
 * @param[in]       axis         direction of reversal. 1 for clockwise and 0 for anticlockwise.
 * @param[in]       nrows        rows of matrix
 * @param[in]       ncols        columns of matrix
 * @param[out]      mat_out      pointer to matrix which will store the performed reversal operation on input matrix
 * @return          none
 * @example         mat_in       = {4232,-9879,-2342,1232,-3242,8432,9823,2342}
 *                  ncols        = 4
 *                  mat_out      = {1232,-2342,-9879,4232,2342,9823,8432,-3242}

*/
void Reverse2(INT_T *mat_in, INT_T axis, INT_T nrows, INT_T ncols, INT_T *mat_out);


/**
 * @brief Performs maxpool operation
 * @param[in]       A           pointer to input matrix
 * @param[in]       B           pointer to input matrix
 * @param[out]      C           pointer to output matrix
 * @param[in]       tmp         pointer to temp storage
 * @param[in]       N           dimension of input matrix A
 * @param[in]       H           dimension of input matrix A
 * @param[in]       W           dimension of input matrix A
 * @param[in]       CI          dimension of input matrix A
 * @param[in]       HF          dimension of input matrix B
 * @param[in]       WF          dimension of input matrix B
 * @param[in]       CO          dimension of input matrix B
 * @param[in]       shrA        dividing input matrices elements to prevent overflows
 * @param[in]       shrB        dividing input matrices elements to prevent overflows
 * @param[in]       H1          parameters for Tree Sum
 * @param[in]       H2          parameters for Tree Sum
   @return          none
 * @example         A           = 1, 2, 3, 4, 5, 6, 7 ,8
 *                  B           = 1,2,5,6,7,8,3,4
 *                  C           = 21 ,27 ,11 ,15 ,53 ,67 ,23 ,31
 *                  
 *
*/
void conv(INT_T *A, const INT_T *B, INT_T *C, INT_T *tmp,                   \
          INT_T N, INT_T H, INT_T W, INT_T CI, INT_T HF,                    \
          INT_T WF, INT_T CO, INT_T shrA, INT_T shrB, INT_T H1, INT_T H2);


/**
 * @brief Performs maxpool operation
 * @param[in]       A           pointer to input matrix
 * @param[in]       B           pointer to input matrix
 * @param[out]      C           pointer to output matrix
 * @param[in]       tmp         pointer to temp storage
 * @param[in]       N           dimension of input matrix A
 * @param[in]       H           dimension of input matrix A
 * @param[in]       W           dimension of input matrix A
 * @param[in]       CI          dimension of input matrix A
 * @param[in]       HF          dimension of input matrix B
 * @param[in]       WF          dimension of input matrix B
 * @param[in]       CINF        dimension of input matrix B
 * @param[in]       COUTF       dimension of input matrix B
 * @param[in]       HOUT        dimension of input matrix B
 * @param[in]       WOUT        dimension of input matrix B
 * @param[in]       HPADL       Thickness of padding on top, bottom of the image
 * @param[in]       HPADR       Thickness of padding on top, bottom of the image
 * @param[in]       WPADL       Thickness of padding on left, right of the image
 * @param[in]       WPADR       Thickness of padding on left, right of the image
 * @param[in]       HSTR        Convolution horizontal, vertical stride
 * @param[in]       HDL         Convolution horizontal, vertical dilations
 * @param[in]       WDL         Convolution horizontal, vertical dilations
 * @param[in]       G           Number of groups
 * @param[in]       shrA        scale factor for input matrix A
 * @param[in]       shrB        scale factor for input matrix B
 * @param[in]       H1          fixed point arithmetic
 * @param[in]       H2          fixed point arithmetic
   @return          none
 * @example         A           = 1, 2, 3, 4, 5, 6, 7 ,8
 *                  B           = 1,2,5,6,7,8,3,4
 *                  C           = 21 ,27 ,11 ,15 ,53 ,67 ,23 ,31
 *                  
 *
*/
void convolution(INT_T *A, const INT_T *B, INT_T *C, INT_T *tmp, 
                 INT_T N, INT_T H, INT_T W, INT_T CIN, INT_T HF,        \
                 INT_T WF, INT_T CINF, INT_T COUTF, INT_T HOUT,         \
                 INT_T WOUT, INT_T HPADL, INT_T HPADR, INT_T WPADL,     \
                 INT_T WPADR, INT_T HSTR, INT_T WSTR, INT_T HDL,        \
                 INT_T WDL, INT_T G, INT_T shrA, INT_T shrB, INT_T H1,  \
                 INT_T H2);

/**
 * @brief Performs sihmoid operation
 * @param[in]       mat_in              pointer to input matrix
 * @param[in]       nrows               rows of input matrix
 * @param[in]       ncols               columns of input matrix
 * @param[in]       div                 division value
 * @param[in]       add                 addition value
 * @param[in]       sigmoid_limit       sigmoid limit value
 * @param[in]       scale_in            scaling factor of input matnrix
 * @param[in]       scale_out           scaling factor of output matrix
 * @param[out]    mat_out             pointer to output matrix
   @return          none
 * @example         mat_in              = 1, 2, 3, 4, 5, 6, 7 ,8
 *                  nrows               = 4, 4, 2, 1, 5, 5, 10
 *                  ncols               = 4
 *                  div                 = 2
 *                  add                 = 1
 *                  sigmoid_limit       = 5
 *                  scale_in            = 5
 *                  scale_out           = 10
 *                  mat_out             = 21 ,27 ,11 ,15 ,53 ,67 ,23 ,31
 *                  C                   = 1 ,29 ,5 ,36 ,8 ,43 ,11 ,36
*/

void sigmoid(INT_T *mat_in, INT_T nrows, INT_T ncols, INT_T div, INT_T add, INT_T sigmoid_limit, \
             INT_T scale_in, INT_T scale_out, INT_T *mat_out);

/**
 * @brief Performs sparse Matrix Multiplication
 * @param[in]       Aidx     pointer to input matrix which eveluates to matrix A
 * @param[in]       Aval     pointer to input matrix which eveluates to matrix A
 * @param[in]       B        pointer to input matrix which performs sparce matrix multiplication with matrix A
 * @param[out]      C        pointer to output matrix
 * @param[in]       K        number of rows of matrix B
 * @param[in]       shrA     scaling factor of input matrix A
 * @param[in]       shrB     scaling factor of input matrix B
 * @param[in]       shrC     scaling factor of input matrix C
   @return          none
 * @example         Aidx     = 1, 2, 3, 4, 5, 6, 7 ,0
 *                  Aval     = 10, 20, 30, 40, 50, 60, 70, 80
 *                  C        = 1 ,29 ,5 ,36 ,8 ,43 ,11 ,36
*/
void sp_mat_mul(const INT_T *Aidx, const INT_T *Aval, INT_T **B, INT_T *C, INT_T K, \
                INT_T shrA, INT_T shrB, INT_T shrC);

/**
 * @brief Performs maxpool operation
 * @param[in]       mat_in      pointer to input matrix
 * @param[out]      mat_out     pointer to output matrix
 * @param[in]       N           dimension of input matrix
 * @param[in]       H           dimension of input matrix
 * @param[in]       W           dimension of input matrix
 * @param[in]       C           dimension of input matrix
 * @param[in]       FH          FH, FW : Size of filter amongst which max is taken
 * @param[in]       FW          FH, FW : Size of filter amongst which max is taken
 * @param[in]       strideH     Convolution horizontal, vertical stride
 * @param[in]       strideW     Convolution horizontal, vertical stride
 * @param[in]       HPADL       Thickness of padding on top, bottom of the image
 * @param[in]       HPADR       Thickness of padding on top, bottom of the image
 * @param[in]       WPADL       Thickness of padding on left, right of the image
 * @param[in]       WPADR       Thickness of padding on left, right of the image
 * @param[in]       
   @return          none
 * @example         mat_in       = 1, 2, 3, 4, 5, 6
 *                  N,H,W,C      = 3, 2, 1, 1
 *                  FW, FH       = 3, 3
 *                  strideH      = 1
                    strideW      = 1
 *                  HPADL, HPADR = 1, 1
 *                  WPADL, WPADR = 1, 1
 *                  mat_out      = 13 ,14 ,15 ,16 ,15 ,16
 *
*/

void maxpool(INT_T *mat_in, INT_T *mat_out, INT_T N, INT_T H, INT_T W, INT_T C, INT_T FH, \
             INT_T FW, INT_T strideH, INT_T strideW, INT_T HPADL, INT_T HPADR, \
            INT_T WPADL, INT_T WPADR);

#endif