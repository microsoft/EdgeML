// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __QUANTIZED_UTILS_H__
#define __QUANTIZED_UTILS_H__

#include <math.h>
#include "quantized_datatypes.h"

// Function for saturating the input to the required format.
// This function isn't used actively because of SeeDot generated scales
// ensuring the overflows aren't a possibility.
static inline Q15_T q15_saturate(Q31_T inp) {
    if (inp > Q15_TMAX){
        return (Q15_T)Q15_TMAX;
    } else if (inp < Q15_TMIN) {
        return (Q15_T)Q15_TMIN;
    } else {
        return (Q15_T)inp;
    }
}

// These functions are used to provide a truncation of input to a specific
// range within the ReLU operation.
static inline Q7_T q7_relu(Q7_T inp, Q7_T limit) {
    if (inp > limit){
        return limit;
    } else if (inp < 0) {
        return 0;
    } else {
        return inp;
    }
}

static inline Q15_T q15_relu(Q15_T inp, Q15_T limit) {
    if (inp > limit){
        return limit;
    } else if (inp < 0) {
        return 0;
    } else {
        return inp;
    }
}

static inline Q31_T q31_relu(Q31_T inp, Q31_T limit) {
    if (inp > limit){
        return limit;
    } else if (inp < 0) {
        return 0;
    } else {
        return inp;
    }
}

static const Q15_T exp_table_A[256] = {16384, 15391, 14459, 13583, 12760, 11987, 11261, 10578, 9937, 9335, 8770, 8238, 7739, 7270, 6830, 6416, 6027, 5662, 5319, 4997, 4694, 4410, 4143, 3892, 3656, 3434, 3226, 3031, 2847, 2675, 2513, 2360, 2217, 2083, 1957, 1838, 1727, 1622, 1524, 1432, 1345, 1263, 1187, 1115, 1047, 984, 924, 868, 816, 766, 720, 676, 635, 597, 561, 527, 495, 465, 437, 410, 385, 362, 340, 319, 300, 282, 265, 249, 234, 220, 206, 194, 182, 171, 161, 151, 142, 133, 125, 118, 110, 104, 97, 92, 86, 81, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 36, 34, 32, 30, 28, 26, 25, 23, 22, 20, 19, 18, 17, 16, 15, 14, 13, 12, 12, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static const Q15_T exp_table_B[128] = {16384, 16376, 16368, 16360, 16352, 16344, 16336, 16328, 16320, 16312, 16304, 16296, 16288, 16280, 16272, 16264, 16256, 16249, 16241, 16233, 16225, 16217, 16209, 16201, 16193, 16185, 16177, 16169, 16162, 16154, 16146, 16138, 16130, 16122, 16114, 16106, 16099, 16091, 16083, 16075, 16067, 16059, 16051, 16044, 16036, 16028, 16020, 16012, 16004, 15997, 15989, 15981, 15973, 15965, 15958, 15950, 15942, 15934, 15927, 15919, 15911, 15903, 15895, 15888, 15880, 15872, 15864, 15857, 15849, 15841, 15833, 15826, 15818, 15810, 15803, 15795, 15787, 15779, 15772, 15764, 15756, 15749, 15741, 15733, 15726, 15718, 15710, 15703, 15695, 15687, 15680, 15672, 15664, 15657, 15649, 15641, 15634, 15626, 15618, 15611, 15603, 15596, 15588, 15580, 15573, 15565, 15558, 15550, 15542, 15535, 15527, 15520, 15512, 15504, 15497, 15489, 15482, 15474, 15467, 15459, 15452, 15444, 15437, 15429, 15421, 15414, 15406, 15399};

static inline Q15_T exp_base_16(Q15_T inp, Q15_T scale) {
  Q15_T val = (inp == -32768) ? 32767 : -inp;
  Q15_T val1 = val % 128;
  val >>= 7;
  Q31_T ret = (Q31_T)exp_table_A[val] * (Q31_T)exp_table_B[val1];
  return (Q15_T)((ret / scale) >> 14);
}

/**
 * @brief Compute the element-wise addition between two vectors.
 * @param[in]       vec1      pointer to the first input vector
 * @param[in]       vec2      pointer to the second input vector
 * @param[in]       len       length of the input vectors
 * @param[out]      ret       pointer to the vector storing the output
 * @param[in]       scvec1    scale factor of the first input vector
 * @param[in]       scvec2    scale factor of the second input vector
 * @param[in]       scret     scale factor of the output vector
 * @param[in]       demote    scale factor for output variable demotion
 * @return          none
 * @example         vec1      = {-425, -169, -3534, 524, -2739, 87, 52, 292}
 *                  vec2      = {-18777, -9518, 4055, -7309, 8584, -17257, -5280, -7933}
 *                  len       = 8
 *                  scvec1    = 1
 *                  scvec2    = 8
 *                  scret     = 1
 *                  demote    = 1
 *                  ret       = {-2772, -1358, -3028, -389, -1666, -2070, -608, -699}
 */
void q15_v_add(const Q15_T* vec1, const Q15_T* vec2, ITER_T len, Q15_T* ret,
               SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret, SCALE_T demote);
/**
 * @brief Compute the element-wise subtraction between two vectors.
 * @param[in]       vec1      pointer to the first input vector
 * @param[in]       vec2      pointer to the second input vector
 * @param[in]       len       length of the input vectors
 * @param[out]      ret       pointer to the vector storing the output
 * @param[in]       scvec1    scale factor of the first input vector
 * @param[in]       scvec2    scale factor of the second input vector
 * @param[in]       scret     scale factor of the output vector
 * @return          none
 * @example         vec1      = {-425, -169, -3534, 524, -2739, 87, 52, 292}
 *                  vec2      = {-18777, -9518, 4055, -7309, 8584, -17257, -5280, -7933}
 *                  len       = 8
 *                  scvec1    = 1
 *                  scvec2    = 8
 *                  scret     = 1
 *                  ret       = {1922, 1020, -4040, 1437, -3812, 2244, 712, 1283}
 */
void q7_v_sub(const Q7_T* vec1, const Q7_T* vec2, ITER_T len, Q7_T* ret,
              SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret);
void q15_v_sub(const Q15_T* vec1, const Q15_T* vec2, ITER_T len, Q15_T* ret,
               SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret);
/**
 * @brief Compute the element-wise product (also known as Hadamard product) between two vectors.
 * @param[in]       vec1      pointer to the first input vector
 * @param[in]       vec2      pointer to the second input vector
 * @param[in]       len       length of the input vectors
 * @param[out]      ret       pointer to the vector storing the output
 * @param[in]       scvec1    scale factor of the first input vector
 * @param[in]       scvec2    scale factor of the second input vector
 * @return          none
 * @example         vec1      = {16378, 13638, 16378, 9787, 14861, 16378, 10661, 11018}
 *                  vec2      = {178, 1064, -2048, 1718, -1663, 851, 1244, 1282}
 *                  len       = 8
 *                  scvec1    = 32
 *                  scvec2    = 64
 *                  ret       = {1423, 7085, -16378, 8209, -12067, 6805, 6475, 6897}
 */
void q7_v_hadamard(const Q7_T* vec1, const Q7_T* vec2, ITER_T len, Q7_T* ret,
                   SCALE_T scvec1, SCALE_T scvec2);
void q15_v_hadamard(const Q15_T* vec1, const Q15_T* vec2, ITER_T len, Q15_T* ret,
                    SCALE_T scvec1, SCALE_T scvec2);
/**
 * @brief Compute the element-wise Sigmoid activation on the input vector.
 * @param[in]       vec            pointer to the input vector
 * @param[in]       len            length of the input vector
 * @param[out]      ret            pointer to the vector storing the output
 * @param[in]       div            division factor of the input vector
 * @param[in]       add            addition offset of the input vector
 * @param[in]       sigmoid_limit  saturation limit for the Sigmoid activation
 * @param[in]       scale_in       scale factor of the input vector
 * @param[in]       scale_out      scale factor of the output vector
 * @param[in]       use_tables     flag for using pre-computed (base 16) exp tables for calculating Sigmoid on the input
 * @return          none
 * @example         formula        = saturate(0, (vec_{i} / div) + add, sigmoid_limit) * 2^{scale_out - scale_in} (use_tables set to 0)
 *                  vec            = {-2772, -1358, -3028, -389, -1666, -2070, -608, -699}
 *                  len            = 8
 *                  div            = 2
 *                  add            = 1024
 *                  sigmoid_limit  = 2048
 *                  scale_in       = 11
 *                  scale_out      = 14
 *                  use_tables     = 0
 *                  ret            = {0, 2760, 0, 6640, 1528, 0, 5760, 5400}
 */
void q15_v_sigmoid(const Q15_T* vec, ITER_T len, Q15_T* ret, Q15_T div,
                   Q15_T add, Q15_T sigmoid_limit, SCALE_T scale_in,
                   SCALE_T scale_out, ITER_T use_tables);
/**
 * @brief Compute the element-wise TanHyperbolic activation on the input vector.
 * @param[in]       vec            pointer to the input vector
 * @param[in]       len            length of the input vector
 * @param[out]      ret            pointer to the vector storing the output
 * @param[in]       scale_in       scale factor of the input vector
 * @param[in]       scale_out      scale factor of the output vector
 * @param[in]       use_tables     flag for using pre-computed (base 16) exp tables for calculating TanH on the input
 * @return          none
 * @example         formula        = saturate(-2^{scale_in}, vec_{i}, 2^{scale_in}) * 2^{scale_out - scale_in} (use_tables set to 0)
 *                  vec            = {178, 1064, -4162, 1718, -1663, 851, 1244, 1282}
 *                  len            = 8
 *                  scale_in       = 11
 *                  scale_out      = 11
 *                  use_tables     = 0
 *                  ret            = {178, 1064, -2048, 1718, -1663, 851, 1244, 1282}
 */
void q15_v_tanh(const Q15_T* vec, ITER_T len, Q15_T* ret, SCALE_T scale_in,
                SCALE_T scale_out, ITER_T use_tables);
/**
 * @brief Compute the addition of a scalar to every element of a vector.
 * @param[in]       scalar    the input scalar to be added to a vector
 * @param[in]       vec       pointer to the input vector
 * @param[in]       len       length of the input vector
 * @param[out]      ret       pointer to the vector storing the output
 * @param[in]       scscalar  scale factor of the input scalar
 * @param[in]       scvec     scale factor of the input vector
 * @param[in]       scret     scale factor of the output vector
 * @return          none
 * @example         scalar    = 30111
 *                  vec       = {16261, 13521, 16261, 9670, 14744, 16261, 10544, 10901}
 *                  len       = 8
 *                  scscalar  = 256
 *                  scvec     = 1
 *                  scret     = 1
 *                  ret       = {16378, 13638, 16378, 9787, 14861, 16378, 10661, 11018}
 */
void q15_v_scalar_add(Q15_T scalar, const Q15_T* vec, ITER_T len, Q15_T* ret,
                      SCALE_T scscalar, SCALE_T scvec, SCALE_T scret);
/**
 * @brief Compute the subtraction of every element of a vector (B) from a scalar (a). The resultant vector has elements C_{i} = a - B_{i}.
 * @param[in]       scalar    the input scalar
 * @param[in]       vec       pointer to the input vector to be subtracted
 * @param[in]       len       length of the input vector
 * @param[out]      ret       pointer to the vector storing the output
 * @param[in]       scscalar  scale factor of the input scalar
 * @param[in]       scvec     scale factor of the input vector
 * @param[in]       scret     scale factor of the output vector
 * @return          none
 * @example         scalar    = 16384
 *                  vec       = {0, 2760, 0, 6640, 1528, 0, 5760, 5400}
 *                  len       = 8
 *                  scscalar  = 1
 *                  scvec     = 1
 *                  scret     = 1
 *                  ret       = {16384, 13624, 16384, 9744, 14856, 16384, 10624, 10984}
 */
void q15_v_scalar_sub(Q15_T scalar, const Q15_T* vec, ITER_T len, Q15_T* ret,
                      SCALE_T scscalar, SCALE_T scvec, SCALE_T scret);
/**
 * @brief Compute the multiplication of a scalar to every element of a vector.
 * @param[in]       scalar    the input scalar to be multiplied
 * @param[in]       vec       pointer to the input vector
 * @param[in]       len       length of the input vector
 * @param[out]      ret       pointer to the vector storing the output
 * @param[in]       scscalar  scale factor of the input scalar
 * @param[in]       scvec     scale factor of the input vector
 * @return          none
 * @example         scalar    = 32522
 *                  vec       = {16384, 13624, 16384, 9744, 14856, 16384, 10624, 10984}
 *                  len       = 8
 *                  scscalar  = 128
 *                  scvec     = 256
 *                  ret       = {16261, 13521, 16261, 9670, 14744, 16261, 10544, 10901}
 */
void q15_v_scalar_mul(Q15_T scalar, const Q15_T* vec, ITER_T len, Q15_T* ret,
                      SCALE_T scscalar, SCALE_T scvec);
/**
 * @brief Finds the index of largest element in a vector.
 * @param[in]       vec       pointer to input vector
 * @param[in]       len       length of the vector
 * @param[out]      ret       pointer variable storing the index of the largest element in the vector
 * @return          none
 * @example         vec       = {12, 24, 54, 1, 2, 10}
 *                  len       = 6
 *                  *ret      = 2
 */
void q15_v_argmax(const Q15_T* const vec, ITER_T len, ITER_T* const ret);
/**
 * @brief Performs element-wise up-scaling on a vector.
 * @param[in]       vec       pointer to the vector on which up-scaling is to be performed
 * @param[in]       len       length of the vector
 * @param[out]      ret       pointer to the output vector
 * @param[in]       scvec     scaling factor of the vector
 * @return          none
 * @example         vec       = {423, -987, -2342, 1232}
 *                  len       = 4
 *                  scvec     = 10
 *                  ret       = {4230, -9870, -23420, 12320}
 */
void q15_v_scale_up(const Q15_T* vec, ITER_T len, Q15_T* ret, SCALE_T scvec);
/**
 * @brief Performs element-wise down-scaling on a vector.
 * @param[in]       vec       pointer to the vector on which down-scaling is to be performed
 * @param[in]       len       length of the vector
 * @param[out]      ret       pointer to the output vector
 * @param[in]       scvec     scaling factor of the vector
 * @return          none
 * @example         vec       = {4232, -9879, -2342, 1232}
 *                  len       = 4
 *                  scvec     = 37
 *                  ret       = {114, -267, -63, 33}
 */
void q15_v_scale_down(const Q15_T* vec, ITER_T len, Q15_T* ret, SCALE_T scvec);

/**
 * @brief Performs the matrix multiplication of a matrix and a vector.
 * @param[in]       mat       pointer to input matrix in row-major order
 * @param[in]       vec       pointer to the input vector
 * @param[in]       nrows     number of rows of the input matrix
 * @param[in]       ncols     number of columns of the input matrix
 * @param[out]      ret       pointer to the output vector
 * @param[in]       scmat     scale factor of the input matrix
 * @param[in]       scvec     scale factor of the input vector
 * @param[in]       H1        depth parameter for division-by-two used in TreeSum
 * @param[in]       H2        depth parameter for direct sum used in TreeSum

 * @return          none
 * @example         mat       = { {7069, -10389, 1562, -1992},
 *                                {3262, -37, -1143, -995},
 *                                {5513, -17035, -14615, -6636},
 *                                {4733, -403, 4106, -1104},
 *                                {-2707, -1287, -18128, -1832},
 *                                {-10108, -137, 2064, 1207},
 *                                {5233, 226, 831, -1909},
 *                                {4489, -1099, 2845, -1261} }
 *                  vec       = {1040, 1919, 4254, 4024}
 *                  nrows     = 8
 *                  ncols     = 4
 *                  scmat     = 128
 *                  scvec     = 64
 *                  H1        = 2
 *                  H2        = 0
 *                  ret       = {-425, -169, -3534, 524, -2739, 87, 52, 292}
 */
void q15xq7_q15_m_mulvec(const Q15_T* mat, const Q7_T* const vec, ITER_T nrows,
                         ITER_T ncols, Q15_T* ret, SCALE_T scmat,
                         SCALE_T scvec, SCALE_T H1, SCALE_T H2);
void q15_m_mulvec(const Q15_T* mat, const Q15_T* const vec, ITER_T nrows,
                  ITER_T ncols, Q15_T* ret, SCALE_T scmat, SCALE_T scvec,
                  SCALE_T H1, SCALE_T H2);

/**
 * @brief Performs the element-wise addition of two input tensors.
 * dim(ten1) = dim(ten2) = [nbatches][nrows][ncols][nchannels]
 * @param[in]       ten1      pointer to the first input tensor
 * @param[in]       ten2      pointer to the second input tensor
 * @param[in]       nbatches  number of batches of the input tensor
 * @param[in]       nrows     number of rows of the input tensor
 * @param[in]       ncols     number of columns of the input tensor
 * @param[in]       nchannels number of channels of the input tensor
 * @param[out]      ret       pointer to the output tensor
 * @param[in]       scten1    scaling factor for the first input tensor
 * @param[in]       scten2    scaling factor for the second input tensor
 * @param[in]       scret     scaling factor for the output tensor
 * @return          none
 * @example         ten1      = { { {1324, 5453}, {3454, 3435} },
 *                                { {8789, 3411}, {5412, 8934} } },
 *                              { { {6895, 1211}, {6790, 5425} },
 *                                { {8976, 4539}, {9348, 9321} } }
 *                  ten2      = { { {1324, 5453}, {3454, 3435} },
 *                                { {8789, 3411}, {5412, 8934} } },
 *                              { { {6895, 1211}, {6790, 5425} },
 *                                { {8976, 4539}, {9348, 9321} } }
 *                  nbatches  = 2
 *                  nrows     = 2
 *                  ncols     = 2
 *                  nchannels = 2
 *                  scten1    = 2
 *                  scten2    = 2
 *                  scret     = 1
 *                  ret       = { { {1324, 5452}, {3454, 3434} },
 *                                { {8788, 3410}, {5412, 8934} } },
 *                              { { {6894, 1210}, {6790, 5424} },
 *                                { {8976, 4538}, {9348, 9320} } }
 */
void q7_t_add(const Q7_T* ten1, const Q7_T* ten2, ITER_T nbatches,
              ITER_T nrows, ITER_T ncols, ITER_T nchannels, Q7_T* ret,
              SCALE_T scten1, SCALE_T scten2, SCALE_T scret);
void q15_t_add(const Q15_T* ten1, const Q15_T* ten2, ITER_T nbatches,
               ITER_T nrows, ITER_T ncols, ITER_T nchannels, Q15_T* ret,
               SCALE_T scten1, SCALE_T scten2, SCALE_T scret);
/**
 * @brief Performs the channel-wise addition of a bias term to the input tensor.
 * dim(ten) = dim(ret) = [nbatches][nrows][ncols][nchannels]; dim(vec) = [nchannels].
 * @param[in]       ten       pointer to the input tensor on which addition is to be performed
 * @param[in]       vec       pointer to the bias vector which is to be added
 * @param[in]       nbatches  number of batches of the input tensor
 * @param[in]       nrows     number of rows of the input tensor
 * @param[in]       ncols     number of columns of the input tensor
 * @param[in]       nchannels number of channels of the input tensor
 * @param[out]      ret       pointer to the output tensor
 * @param[in]       scten     scaling factor for the input tensor
 * @param[in]       scvec     scaling factor for the bias vector
 * @param[in]       scret     scaling factor for the output tensor
 * @return          none
 * @example         ten       = { { {1324, 5453}, {3454, 3435} },
 *                                { {8789, 3411}, {5412, 8934} } },
 *                              { { {6895, 1211}, {6790, 5425} },
 *                                { {8976, 4539}, {9348, 9321} } }
 *                  vec       = {8452, 2341}
 *                  nbatches  = 2
 *                  nrows     = 2
 *                  ncols     = 2
 *                  nchannels = 2
 *                  scten     = 1
 *                  scvec     = 2
 *                  scret     = 2
 *                  ret       = { { {2775, 3311}, {4072, 2305} },
 *                                { {6507, 2290}, {5051, 5055} } },
 *                              { { {5560, 1190}, {5740, 3300} },
 *                                { {6601, 2854}, {7019, 5248} } }
 */
void q7xq15_q7_t_add_vec(const Q7_T* ten, const Q15_T* const vec,
                         ITER_T nbatches, ITER_T nrows, ITER_T ncols,
                         ITER_T nchannels, Q7_T* ret, SCALE_T scmat,
                         SCALE_T scvec, SCALE_T scret);
void q15_t_add_vec(const Q15_T* ten, const Q15_T* const vec, ITER_T nbatches,
                   ITER_T nrows, ITER_T ncols, ITER_T nchannels, Q15_T* ret,
                   SCALE_T scmat, SCALE_T scvec, SCALE_T scret);
/**
 * @brief Replace any negative element present in the tensor with zero and clips positive elements to the limit.
 * @param[in]       ten       pointer to tensor on which element-wise ReLU6 operation is to be applied
 * @param[in]       nbatches  number of batches of the input tensor
 * @param[in]       nrows     number of rows of the input tensor
 * @param[in]       ncols     number of columns of the input tensor
 * @param[in]       nchannels number of channels of the input tensor
 * @param[out]      ret       pointer to the output tensor
 * @param[in]       limit     upper threshold of the ReLU operation
 * @param[in]       div       scaling factor for the input tensor
 * @param[in]
 * @return          none
 * @example         ten       = { { {124, 53}, {45, 35} },
 *                                { {87, -11}, {54, 89} } },
 *                              { { {95, -12}, {90, 42} },
 *                                { {76, 39}, {93, 21} } }
 *                  nbatches  = 2
 *                  nrows     = 2
 *                  ncols     = 2
 *                  nchannels = 2
 *                  limit     = 64
 *                  div       = 1
 *                  ret       = { { {64, 53}, {45, 35} },
 *                                { {64, 0}, {54, 64} } },
 *                              { { {64, 0}, {64, 42} },
 *                                { {64, 39}, {64, 21} } }
 */
void q7_t_relu(const Q7_T* ten, ITER_T nbatches, ITER_T nrows, ITER_T ncols,
               ITER_T nchannels, Q7_T* ret, Q7_T limit, Q7_T div);
/**
 * @brief Computes the L2-Norm for each channel of the input tensor, and divides each number in that channel by it.
 * dim(ten) = dim(ret) = [nbatches][nrows][ncols][nchannels].
 * @param[in]       ten       pointer to tensor on which channel-wise L2-Norm operation is to be applied
 * @param[in]       nbatches  number of batches of the input tensor
 * @param[in]       nrows     number of rows of the input tensor
 * @param[in]       ncols     number of columns of the input tensor
 * @param[in]       nchannels number of channels of the input tensor
 * @param[out]      ret       pointer to the output tensor
 * @param[in]       scale_in  scale factor of the input tensor
 * @param[in]       scale_out scale factor of the output tensor
 * @return          none
 * @example         ten       = { { {1324, 5453}, {3454, 3435} },
 *                                { {8789, 3411}, {5412, 8934} } },
 *                              { { {6895, 1211}, {6790, 5425} },
 *                                { {8976, 4539}, {9348, 9321} } }
 *                  nbatches  = 2
 *                  nrows     = 2
 *                  ncols     = 2
 *                  nchannels = 2
 *                  scale_in  = 1
 *                  scale_out = 1
 *                  ret       = { { {662, 2726}, {1727, 1717} },
 *                                { {4394, 1705}, {2706, 4467} } },
 *                              { { {3447, 605}, {3395, 2712} },
 *                                { {4488, 2269}, {4674, 4660} } }
 */
void q15_t_l2_norm(const Q15_T* ten, ITER_T nbatches, ITER_T nrows,
                   ITER_T ncols, ITER_T nchannels, Q15_T* ret,
                   SCALE_T scale_in, SCALE_T scale_out);

/**
 * @brief Computes the maxpool operation on the input tensor with the given parameters.
 * @param[in]       input          pointer to the tensor on which convolution is to be performed
 * @param[in]       filter         pointer to the convolutional filter tensor
 * @param[out]      output         pointer to the output tensor
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
 * @param[in]       HDilation      dilation of the convolution filter along the rows (number of skipped input rows between two consecutive filter rows is HDilation - 1)
 * @param[in]       WDilation      dilation of the convolution filter along the columns (number of skipped input columns between two consecutive filter rows is WDilation - 1)
 * @param[in]       scinput        scale of the input tensor
 * @param[in]       scoutput       scale of the output tensor
 * @param[in]       demote         scale factor for output variable demotion
 * @return          none
 * @example         Please refer the test-case: test_q15_convolution() in file: c_reference/tests/utils/test_quantized_utils.c
 */
void q7xq15_q7_convolution(const Q7_T* const input, const Q15_T* const filter,
  Q7_T* const output, ITER_T N, ITER_T H, ITER_T W, ITER_T CIn, ITER_T HF,
  ITER_T WF, ITER_T CF, ITER_T COut, ITER_T HOut, ITER_T WOut, ITER_T G,
  S_ITER_T HPadU, S_ITER_T HPadD, S_ITER_T WPadL, S_ITER_T WPadR,
  ITER_T HStride, ITER_T WStride, ITER_T HDilation, ITER_T WDilation,
  SCALE_T scinput, SCALE_T scoutput, SCALE_T demote);
void q7xq15_q15_convolution(const Q7_T* const input, const Q15_T* const filter,
  Q15_T* const output, ITER_T N, ITER_T H, ITER_T W, ITER_T CIn, ITER_T HF,
  ITER_T WF, ITER_T CF, ITER_T COut, ITER_T HOut, ITER_T WOut, ITER_T G,
  S_ITER_T HPadU, S_ITER_T HPadD, S_ITER_T WPadL, S_ITER_T WPadR,
  ITER_T HStride, ITER_T WStride, ITER_T HDilation, ITER_T WDilation,
  SCALE_T scinput, SCALE_T scoutput, SCALE_T demote);
void q15_convolution(const Q15_T* const input, const Q15_T* const filter,
  Q15_T* const output, ITER_T N, ITER_T H, ITER_T W, ITER_T CIn, ITER_T HF,
  ITER_T WF, ITER_T CF, ITER_T COut, ITER_T HOut, ITER_T WOut, ITER_T G,
  S_ITER_T HPadU, S_ITER_T HPadD, S_ITER_T WPadL, S_ITER_T WPadR,
  ITER_T HStride, ITER_T WStride, ITER_T HDilation, ITER_T WDilation,
  SCALE_T scinput, SCALE_T scoutput, SCALE_T demote);

#endif
