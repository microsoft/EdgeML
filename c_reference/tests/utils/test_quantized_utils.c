// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stdio.h>
#include "quantized_utils.h"

// All values generated from Seedot on Wider Regression dataset.
// By default, all tests run without using bit-shifting operations.
// Function for matching the predicted and expected quantized outputs.
static int check_output_q7(const Q7_T* const pred, const Q7_T* const expected,
                           unsigned len) {
  for (unsigned i = 0; i < len; i++)
  {
    if (pred[i] != expected[i]) {
      printf("Output: %d, Expected: %d at Index: %d\n", pred[i], expected[i], i);
      return 1;
    }
  }
  return 0;
}

static int check_output_q15(const Q15_T* const pred, const Q15_T* const expected,
                            unsigned len) {
  for (unsigned i = 0; i < len; i++)
  {
    if (pred[i] != expected[i]) {
      printf("Output: %d, Expected: %d at Index: %d\n", pred[i], expected[i], i);
      return 1;
    }
  }
  return 0;
}

// Test q15_v_add() function.
int test_q15_v_add() {
  const Q15_T qvec_A[8] = {-425, -169, -3534, 524, -2739, 87, 52, 292};
  const Q15_T qvec_B[8] = {-18777, -9518, 4055, -7309, 8584, -17257, -5280, -7933};
  Q15_T pred[8];

  #ifdef SHIFT
    const Q15_T expected[8] = {-2773, -1359, -3028, -390, -1666, -2071, -608, -700};
    q15_v_add(&qvec_A[0], &qvec_B[0], 8, &pred[0], 0, 3, 0, 0);
  #else
    const Q15_T expected[8] = {-2772, -1358, -3028, -389, -1666, -2070, -608, -699};
    q15_v_add(&qvec_A[0], &qvec_B[0], 8, &pred[0], 1, 8, 1, 1);
  #endif

  return check_output_q15(pred, expected, 8);
}

// Test q7_v_sub() function.
int test_q7_v_sub() {
  const Q7_T qvec_A[8] = {-42, -16, -35, 54, -29, 87, 52, 22};
  const Q7_T qvec_B[8] = {-17, -98, 40, -73, 85, -125, -80, -79};
  Q7_T pred[8];

  #ifdef SHIFT
    const Q7_T expected[8] = {-39, -3, -40, 64, -39, 103, 62, 32};
    q7_v_sub(&qvec_A[0], &qvec_B[0], 8, &pred[0], 0, 3, 0);
  #else
    const Q7_T expected[8] = {-40, -4, -40, 63, -39, 102, 62, 31};
    q7_v_sub(&qvec_A[0], &qvec_B[0], 8, &pred[0], 1, 8, 1);
  #endif

  return check_output_q7(pred, expected, 8);
}

// Test q15_v_sub() function.
int test_q15_v_sub() {
  const Q15_T qvec_A[8] = {-425, -169, -3534, 524, -2739, 87, 52, 292};
  const Q15_T qvec_B[8] = {-18777, -9518, 4055, -7309, 8584, -17257, -5280, -7933};
  Q15_T pred[8];

  #ifdef SHIFT
    const Q15_T expected[8] = {1923, 1021, -4040, 1438, -3812, 2245, 712, 1284};
    q15_v_sub(&qvec_A[0], &qvec_B[0], 8, &pred[0], 0, 3, 0);
  #else
    const Q15_T expected[8] = {1922, 1020, -4040, 1437, -3812, 2244, 712, 1283};
    q15_v_sub(&qvec_A[0], &qvec_B[0], 8, &pred[0], 1, 8, 1);
  #endif

  return check_output_q15(pred, expected, 8);
}

// Test q7_v_hadamard() function.
int test_q7_v_hadamard() {
  const Q7_T qvec_A[8] = {64, 57, 64, 37, 60, 64, 40, 43};
  const Q7_T qvec_B[8] = {18, 14, -48, 118, -16, 85, 124, 122};
  Q7_T pred[8];

  #ifdef SHIFT
    const Q7_T expected[8] = {18, 12, -48, 68, -15, 85, 77, 81};
    q7_v_hadamard(&qvec_A[0], &qvec_B[0], 8, &pred[0], 3, 3);
  #else
    const Q7_T expected[8] = {18, 12, -48, 68, -15, 85, 77, 81};
    q7_v_hadamard(&qvec_A[0], &qvec_B[0], 8, &pred[0], 8, 8);
  #endif

  return check_output_q7(pred, expected, 8);
}

// Test q15_v_hadamard() function.
int test_q15_v_hadamard() {
  const Q15_T qvec_A[8] = {16378, 13638, 16378, 9787, 14861, 16378, 10661, 11018};
  const Q15_T qvec_B[8] = {178, 1064, -2048, 1718, -1663, 851, 1244, 1282};
  Q15_T pred[8];

  #ifdef SHIFT
    const Q15_T expected[8] = {1423, 7085, -16378, 8209, -12068, 6805, 6475, 6897};
    q15_v_hadamard(&qvec_A[0], &qvec_B[0], 8, &pred[0], 5, 6);
  #else
    const Q15_T expected[8] = {1423, 7085, -16378, 8209, -12067, 6805, 6475, 6897};
    q15_v_hadamard(&qvec_A[0], &qvec_B[0], 8, &pred[0], 32, 64);
  #endif

  return check_output_q15(pred, expected, 8);
}

// Test q15_v_sigmoid() function.
int test_q15_v_sigmoid() {
  const Q15_T qvec_A[8] = {-2772, -1358, -3028, -389, -1666, -2070, -608, -699};
  const Q15_T expected_A[8] = {0, 2760, 0, 6640, 1528, 0, 5760, 5400};
  const Q15_T expected_B[8] = {3363, 5571, 3041, 7416, 5031, 4371, 6984, 6807};
  Q15_T pred_A[8], pred_B[8];

  q15_v_sigmoid(&qvec_A[0], 8, &pred_A[0], 2, 1024, 2048, 11, 14, 0);
  q15_v_sigmoid(&qvec_A[0], 8, &pred_B[0], 2, 1024, 2048, 11, 14, 1);
  return (check_output_q15(pred_A, expected_A, 8) || check_output_q15(pred_B, expected_B, 8));
}

// Test q15_v_tanh() function.
int test_q15_v_tanh() {
  const Q15_T qvec_A[8] = {178, 1064, -4162, 1718, -1663, 853, 1244, 1282};
  const Q15_T expected_A[8] = {178, 1064, -2048, 1718, -1663, 853, 1244, 1282};
  const Q15_T expected_B[8] = {1420, 7822, -15831, 11227, -10989, 6455, 8885, 9098};
  Q15_T pred_A[8], pred_B[8];

  q15_v_tanh(&qvec_A[0], 8, &pred_A[0], 11, 11, 0);
  q15_v_tanh(&qvec_A[0], 8, &pred_B[0], 11, 11, 1);
  return (check_output_q15(pred_A, expected_A, 8) || check_output_q15(pred_B, expected_B, 8));
}

// Test q15_v_scalar_add() function.
int test_q15_v_scalar_add() {
  const Q15_T qscalar_A = 30111;
  const Q15_T qvec_B[8] = {16261, 13521, 16261, 9670, 14744, 16261, 10544, 10901};
  const Q15_T expected[8] = {16378, 13638, 16378, 9787, 14861, 16378, 10661, 11018};
  Q15_T pred[8];

  #ifdef SHIFT
    q15_v_scalar_add(qscalar_A, &qvec_B[0], 8, &pred[0], 8, 0, 0);
  #else
    q15_v_scalar_add(qscalar_A, &qvec_B[0], 8, &pred[0], 256, 1, 1);
  #endif

  return check_output_q15(pred, expected, 8);
}

// Test q15_v_scalar_sub() function.
int test_q15_v_scalar_sub() {
  const Q15_T qscalar_A = 16384;
  const Q15_T qvec_B[8] = {0, 2760, 0, 6640, 1528, 0, 5760, 5400};
  const Q15_T expected[8] = {16384, 13624, 16384, 9744, 14856, 16384, 10624, 10984};
  Q15_T pred[8];

  #ifdef SHIFT
    q15_v_scalar_sub(qscalar_A, &qvec_B[0], 8, &pred[0], 0, 0, 0);
  #else
    q15_v_scalar_sub(qscalar_A, &qvec_B[0], 8, &pred[0], 1, 1, 1);
  #endif

  return check_output_q15(pred, expected, 8);
}

// Test q15_v_scalar_mul() function.
int test_q15_v_scalar_mul() {
  const Q15_T qscalar_A = 32522;
  const Q15_T qvec_B[8] = {16384, 13624, 16384, 9744, 14856, 16384, 10624, 10984};
  const Q15_T expected[8] = {16261, 13521, 16261, 9670, 14744, 16261, 10544, 10901};
  Q15_T pred[8];

  #ifdef SHIFT
    q15_v_scalar_mul(qscalar_A, &qvec_B[0], 8, &pred[0], 7, 8);
  #else
    q15_v_scalar_mul(qscalar_A, &qvec_B[0], 8, &pred[0], 128, 256);
  #endif

  return check_output_q15(pred, expected, 8);
}

// Test q15_v_argmax() function.
int test_q15_v_argmax() {
  const Q15_T qvec_A[8] = {1675, 9870, -9876, -1234, 5674, 28765, 9876, 12654};
  const ITER_T expected[1] = {5};
  ITER_T pred[1];

  q15_v_argmax(&qvec_A[0], 8, &pred[0]);
  return check_output_q15((const Q15_T*)pred, (const Q15_T*)expected, 1);
}

// Test q15_v_scale_up() function.
int test_q15_v_scale_up() {
  Q15_T qvec_A[16] = {423, -987, -2342, 1232, -324, 843, 982, 2342, 343, 654, 987, 654, 567, 2876, 987, 1265};
  Q15_T expected[16] = {846, -1974, -4684, 2464, -648, 1686, 1964, 4684, 686, 1308, 1974, 1308, 1134, 5752, 1974, 2530};

  #ifdef SHIFT
    q15_v_scale_up(&qvec_A[0], 16, &qvec_A[0], 1);
  #else
    q15_v_scale_up(&qvec_A[0], 16, &qvec_A[0], 2);
  #endif

  return check_output_q15(qvec_A, expected, 16);
}

// Test q15_v_scale_down() function.
int test_q15_v_scale_down() {
  Q15_T qvec_A[16] = {4232, -9879, -2342, 1232, -3242, 8432, 9823, 2342, 343, 6543, 9876, 6542, 5674, 28765, 9876, 12654};

  #ifdef SHIFT
    const Q15_T expected[16] = {2116, -4940, -1171, 616, -1621, 4216, 4911, 1171, 171, 3271, 4938, 3271, 2837, 14382, 4938, 6327};
    q15_v_scale_down(&qvec_A[0], 16, &qvec_A[0], 1);
  #else
    const Q15_T expected[16] = {2116, -4939, -1171, 616, -1621, 4216, 4911, 1171, 171, 3271, 4938, 3271, 2837, 14382, 4938, 6327};
    q15_v_scale_down(&qvec_A[0], 16, &qvec_A[0], 2);
  #endif

  return check_output_q15(qvec_A, expected, 16);
}

// Test q15xq7_q15_m_mulvec() function.
int test_q15xq7_q15_m_mulvec() {
  const Q15_T qmat_A[8 * 4] = {7069, -10389, 1562, -1992, 3262, -37, -1143, -995, 5513, -17035, -14615, -6636, 4733, -403, 4106, -1104, -2707, -1287, -18128, -1832, -10108, -137, 2064, 1207, 5233, 226, 831, -1909, 4489, -1099, 2845, -1261};
  const Q7_T qvec_B[4] = {104, 119, 42, 24};
  Q15_T pred[8];

  #ifdef SHIFT
    const Q15_T expected[8] = {-15, 8, -68, 18, -38, -30, 17, 12};
    q15xq7_q15_m_mulvec(&qmat_A[0], &qvec_B[0], 8, 4, &pred[0], 7, 6, 2, 0);
  #else
    const Q15_T expected[8] = {-14, 8, -67, 18, -37, -29, 17, 12};
    q15xq7_q15_m_mulvec(&qmat_A[0], &qvec_B[0], 8, 4, &pred[0], 128, 64, 4, 0);
  #endif

  return check_output_q15(pred, expected, 8);
}

// Test q15_m_mulvec() function.
int test_q15_m_mulvec() {
  const Q15_T qmat_A[8 * 4] = {7069, -10389, 1562, -1992, 3262, -37, -1143, -995, 5513, -17035, -14615, -6636, 4733, -403, 4106, -1104, -2707, -1287, -18128, -1832, -10108, -137, 2064, 1207, 5233, 226, 831, -1909, 4489, -1099, 2845, -1261};
  const Q15_T qvec_B[4] = {1040, 1919, 4254, 4024};
  Q15_T pred[8];

  #ifdef SHIFT
    const Q15_T expected[8] = {-426, -170, -3535, 524, -2740, 87, 52, 292};
    q15_m_mulvec(&qmat_A[0], &qvec_B[0], 8, 4, &pred[0], 7, 6, 2, 0);
  #else
    const Q15_T expected[8] = {-425, -169, -3534, 524, -2739, 87, 52, 292};
    q15_m_mulvec(&qmat_A[0], &qvec_B[0], 8, 4, &pred[0], 128, 64, 4, 0);
  #endif

  return check_output_q15(pred, expected, 8);
}

// Test q7_t_add() function.
int test_q7_t_add() {
  const Q7_T qten_A[2 * 2 * 2 * 2] = {124, 55,
                                      45, 43,

                                      87, 41,
                                      54, 89,


                                      69, 12,
                                      60, 54,

                                      76, 49,
                                      93, 32};
  const Q7_T qten_B[2 * 2 * 2 * 2] = {27, 31,
                                      38, 22,

                                      65, 29,
                                      81, 55,


                                      60, 119,
                                      8, 29,

                                      66, 85,
                                      87, 24};
  const Q7_T expected[2 * 2 * 2 * 2] = {68, 34,
                                        31, 26,

                                        59, 27,
                                        47, 57,


                                        49, 35,
                                        32, 34,

                                        54, 45,
                                        67, 22};
  Q7_T pred[16];

  #ifdef SHIFT
    q7_t_add(&qten_A[0], &qten_B[0], 2, 2, 2, 2, &pred[0], 0, 1, 1);
  #else
    q7_t_add(&qten_A[0], &qten_B[0], 2, 2, 2, 2, &pred[0], 1, 2, 2);
  #endif

  return check_output_q7(pred, expected, 16);
}

// Test q15_t_add() function.
int test_q15_t_add() {
  const Q15_T qten_A[2 * 2 * 2 * 2] = {1324, 5453,
                                       3454, 3435,

                                       8789, 3411,
                                       5412, 8934,


                                       6895, 1211,
                                       6790, 5425,

                                       8976, 4539,
                                       9348, 9321};
  const Q15_T qten_B[2 * 2 * 2 * 2] = {2775, 3311,
                                       3840, 2302,

                                       6507, 2290,
                                       4819, 5052,


                                       5560, 1190,
                                       5508, 3297,

                                       6601, 2854,
                                       6787, 5245};
  const Q15_T expected[2 * 2 * 2 * 2] = {1355, 3553,
                                         2687, 2292,

                                         6020, 2277,
                                         3910, 5730,


                                         4837, 902,
                                         4772, 3536,

                                         6138, 2982,
                                         6370, 5971};
  Q15_T pred[16];

  #ifdef SHIFT
    q15_t_add(&qten_A[0], &qten_B[0], 2, 2, 2, 2, &pred[0], 0, 1, 1);
  #else
    q15_t_add(&qten_A[0], &qten_B[0], 2, 2, 2, 2, &pred[0], 1, 2, 2);
  #endif

  return check_output_q15(pred, expected, 16);
}

// Test q7xq15_q7_t_add_vec() function.
int test_q7xq15_q7_t_add_vec() {
  const Q7_T qten_A[2 * 2 * 2 * 2] = {24, 43,
                                      54, 33,

                                      89, 31,
                                      12, 93,


                                      68, 121,
                                      90, 25,

                                      76, 45,
                                      34, 21};
  const Q15_T qvec_B[2] = {8452, 2341};
  const Q7_T expected[2 * 2 * 2 * 2] = {45, 30,
                                        60, 25,

                                        77, 24,
                                        39, 55,


                                        67, 69,
                                        78, 21,

                                        71, 31,
                                        50, 19};
  Q7_T pred[16];

  #ifdef SHIFT
    q7xq15_q7_t_add_vec(&qten_A[0], &qvec_B[0], 2, 2, 2, 2, &pred[0], 0, 7, 1);
  #else
    q7xq15_q7_t_add_vec(&qten_A[0], &qvec_B[0], 2, 2, 2, 2, &pred[0], 1, 128, 2);
  #endif

  return check_output_q7(pred, expected, 16);
}

// Test q15_t_add_vec() function.
int test_q15_t_add_vec() {
  const Q15_T qten_A[2 * 2 * 2 * 2] = {1324, 5453,
                                       3454, 3435,

                                       8789, 3411,
                                       5412, 8934,


                                       6895, 1211,
                                       6790, 5425,

                                       8976, 4539,
                                       9348, 9321};
  const Q15_T qvec_B[2] = {8452, 2341};
  const Q15_T expected[2 * 2 * 2 * 2] = {2775, 3311,
                                         3840, 2302,

                                         6507, 2290,
                                         4819, 5052,


                                         5560, 1190,
                                         5508, 3297,

                                         6601, 2854,
                                         6787, 5245};
  Q15_T pred[16];

  #ifdef SHIFT
    q15_t_add_vec(&qten_A[0], &qvec_B[0], 2, 2, 2, 2, &pred[0], 0, 1, 1);
  #else
    q15_t_add_vec(&qten_A[0], &qvec_B[0], 2, 2, 2, 2, &pred[0], 1, 2, 2);
  #endif

  return check_output_q15(pred, expected, 16);
}

// Test q7_v_relu() function.
int test_q7_t_relu() {
  const Q7_T qten_A[2 * 2 * 2 * 2] = {124, 53,
                                      45, 35,

                                      87, -11,
                                      54, 89,


                                      95, -12,
                                      90, 42,

                                      76, 39,
                                      93, 21};
  const Q7_T expected[2 * 2 * 2 * 2] = {64, 53,
                                        45, 35,

                                        64, 0,
                                        54, 64,


                                        64, 0,
                                        64, 42,

                                        64, 39,
                                        64, 21};
  Q7_T pred[16];

  q7_t_relu(&qten_A[0], 2, 2, 2, 2, &pred[0], 64, 1);
  return check_output_q7(pred, expected, 16);
}

// Test q15_t_l2_norm() function.
int test_q15_t_l2_norm() {
  const Q15_T qten_A[2 * 2 * 2 * 2] = {1324, 5453,
                                       3454, 3435,

                                       8789, 3411,
                                       5412, 8934,


                                       6895, 1211,
                                       6790, 5425,

                                       8976, 4539,
                                       9348, 9321};
  const Q15_T expected[2 * 2 * 2 * 2] = {662, 2726,
                                         1727, 1717,

                                         4394, 1705,
                                         2706, 4467,


                                         3447, 605,
                                         3395, 2712,

                                         4488, 2269,
                                         4674, 4660};
  Q15_T pred[16];

  q15_t_l2_norm(&qten_A[0], 2, 2, 2, 2, &pred[0], 1, 1);
  return check_output_q15(pred, expected, 16);
}

// Test q7xq15_q7_convolution() function.
int test_q7xq15_q7_convolution() {
  const Q7_T qmat_A[2 * 2 * 2 * 2] = {11, 22,
                                      13, 40,

                                      50, 60,
                                      66, 76,


                                      86, 91,
                                      11, 31,

                                      53, 14,
                                      71, 16};
  //Convolution Filters
  const Q15_T qmat_B[2 * 2 * 2 * 1 * 1] = {0, 1,
                                           1, 0,


                                           0, 1,
                                           1, 0,};
  const Q15_T qmat_C[1 * 2 * 2 * 2 * 1] = {0, 1,
                                           1, 0,

                                           1, 0,
                                           0, 1};
  const Q15_T qmat_D[2 * 3 * 3 * 1 * 1] = {0, 0, 1,
                                           0, 1, 0,
                                           1, 0, 0,


                                           0, 0, 1,
                                           0, 1, 0,
                                           1, 0, 0};

  const Q7_T expected_A[2 * 1 * 1 * 2] = {15, 25,


                                          16, 11};
  const Q7_T expected_B[2 * 1 * 1 * 1] = {20,


                                          21};
  const Q7_T expected_C[2 * 2 * 2 * 2] = {1, 2,
                                          7, 12,

                                          7, 12,
                                          8, 9,


                                          10, 11,
                                          8, 5,

                                          8, 5,
                                          8, 2};
  const Q7_T expected_D[2 * 3 * 3 * 2] = {0, 0,
                                          0, 0,
                                          1, 1,

                                          0, 0,
                                          0, 0,
                                          0, 0,

                                          0, 1,
                                          0, 0,
                                          0, 0,


                                          0, 0,
                                          0, 0,
                                          1, 0,

                                          0, 0,
                                          0, 0,
                                          0, 0,

                                          0, 0,
                                          0, 0,
                                          0, 0};
  Q7_T pred_A[2 * 1 * 1 * 2], pred_B[2 * 1 * 1 * 1], pred_C[2 * 2 * 2 * 2], pred_D[2 * 3 * 3 * 2];

  #ifdef SHIFT
    q7xq15_q7_convolution(qmat_A, qmat_B, pred_A, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 2);
    q7xq15_q7_convolution(qmat_A, qmat_C, pred_B, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 3);
    q7xq15_q7_convolution(qmat_A, qmat_D, pred_C, 2, 2, 2, 2, 3, 3, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3);
    q7xq15_q7_convolution(qmat_A, qmat_B, pred_D, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 2, 2, 2, 1, 1, 3, 3, 1, 1, 3);
  #else
    q7xq15_q7_convolution(qmat_A, qmat_B, pred_A, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 4);
    q7xq15_q7_convolution(qmat_A, qmat_C, pred_B, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 8);
    q7xq15_q7_convolution(qmat_A, qmat_D, pred_C, 2, 2, 2, 2, 3, 3, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8);
    q7xq15_q7_convolution(qmat_A, qmat_B, pred_D, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 8);
  #endif

  return (check_output_q7(pred_A, expected_A, 4) || check_output_q7(pred_B, expected_B, 2) || check_output_q7(pred_C, expected_C, 16) || check_output_q7(pred_D, expected_D, 36));
}

// Test q7xq15_q15_convolution() function.
int test_q7xq15_q15_convolution() {
  const Q7_T qmat_A[2 * 2 * 2 * 2] = {11, 22,
                                      13, 40,

                                      50, 60,
                                      66, 76,


                                      86, 91,
                                      41, 31,

                                      53, 51,
                                      75, 71};
  //Convolution Filters
  const Q15_T qmat_B[2 * 2 * 2 * 1 * 1] = {0, 1,
                                           1, 0,


                                           0, 1,
                                           1, 0,};
  const Q15_T qmat_C[1 * 2 * 2 * 2 * 1] = {0, 1,
                                           1, 0,

                                           1, 0,
                                           0, 1};
  const Q15_T qmat_D[2 * 3 * 3 * 1 * 1] = {0, 0, 1,
                                           0, 1, 0,
                                           1, 0, 0,


                                           0, 0, 1,
                                           0, 1, 0,
                                           1, 0, 0};

  const Q15_T expected_A[2 * 1 * 1 * 2] = {15, 25,


                                           23, 20};
  const Q15_T expected_B[2 * 1 * 1 * 1] = {20,


                                           32};
  const Q15_T expected_C[2 * 2 * 2 * 2] = {1, 2,
                                           7, 12,

                                           7, 12,
                                           8, 9,


                                           10, 11,
                                           11, 10,

                                           11, 10,
                                           9, 8};
  const Q15_T expected_D[2 * 3 * 3 * 2] = {0, 0,
                                           0, 0,
                                           1, 1,

                                           0, 0,
                                           0, 0,
                                           0, 0,

                                           0, 1,
                                           0, 0,
                                           0, 0,


                                           0, 0,
                                           0, 0,
                                           1, 1,

                                           0, 0,
                                           0, 0,
                                           0, 0,

                                           1, 0,
                                           0, 0,
                                           0, 0};
  Q15_T pred_A[2 * 1 * 1 * 2], pred_B[2 * 1 * 1 * 1], pred_C[2 * 2 * 2 * 2], pred_D[2 * 3 * 3 * 2];

  #ifdef SHIFT
    q7xq15_q15_convolution(qmat_A, qmat_B, pred_A, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 2);
    q7xq15_q15_convolution(qmat_A, qmat_C, pred_B, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 3);
    q7xq15_q15_convolution(qmat_A, qmat_D, pred_C, 2, 2, 2, 2, 3, 3, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3);
    q7xq15_q15_convolution(qmat_A, qmat_B, pred_D, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 2, 2, 2, 1, 1, 3, 3, 1, 1, 3);
  #else
    q7xq15_q15_convolution(qmat_A, qmat_B, pred_A, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 4);
    q7xq15_q15_convolution(qmat_A, qmat_C, pred_B, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 8);
    q7xq15_q15_convolution(qmat_A, qmat_D, pred_C, 2, 2, 2, 2, 3, 3, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8);
    q7xq15_q15_convolution(qmat_A, qmat_B, pred_D, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 8);
  #endif

  return (check_output_q15(pred_A, expected_A, 4) || check_output_q15(pred_B, expected_B, 2) || check_output_q15(pred_C, expected_C, 16) || check_output_q15(pred_D, expected_D, 36));
}

// Test q15_convolution() function.
int test_q15_convolution() {
  const Q15_T qmat_A[2 * 2 * 2 * 2] = {11, 220,
                                       130, 40,

                                       50, 60,
                                       66, 76,


                                       86, 910,
                                       411, 312,

                                       513, 514,
                                       715, 716};
  //Convolution Filters
  const Q15_T qmat_B[2 * 2 * 2 * 1 * 1] = {0, 1,
                                           1, 0,


                                           0, 1,
                                           1, 0,};
  const Q15_T qmat_C[1 * 2 * 2 * 2 * 1] = {0, 1,
                                           1, 0,

                                           1, 0,
                                           0, 1};
  const Q15_T qmat_D[2 * 3 * 3 * 1 * 1] = {0, 0, 1,
                                           0, 1, 0,
                                           1, 0, 0,


                                           0, 0, 1,
                                           0, 1, 0,
                                           1, 0, 0};

  const Q15_T expected_A[2 * 1 * 1 * 2] = {45, 25,


                                           231, 206};
  const Q15_T expected_B[2 * 1 * 1 * 1] = {59,


                                           318};
  const Q15_T expected_C[2 * 2 * 2 * 2] = {1, 27,
                                           22, 12,

                                           22, 12,
                                           8, 9,


                                           10, 113,
                                           115, 103,

                                           115, 103,
                                           89, 89};
  const Q15_T expected_D[2 * 3 * 3 * 2] = {0, 0,
                                           0, 0,
                                           1, 1,

                                           0, 0,
                                           0, 0,
                                           0, 0,

                                           4, 1,
                                           0, 0,
                                           0, 0,


                                           0, 0,
                                           0, 0,
                                           16, 16,

                                           0, 0,
                                           0, 0,
                                           0, 0,

                                           12, 9,
                                           0, 0,
                                           0, 0};
  Q15_T pred_A[2 * 1 * 1 * 2], pred_B[2 * 1 * 1 * 1], pred_C[2 * 2 * 2 * 2], pred_D[2 * 3 * 3 * 2];

  #ifdef SHIFT
    q15_convolution(qmat_A, qmat_B, pred_A, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 2);
    q15_convolution(qmat_A, qmat_C, pred_B, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 3);
    q15_convolution(qmat_A, qmat_D, pred_C, 2, 2, 2, 2, 3, 3, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3);
    q15_convolution(qmat_A, qmat_B, pred_D, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 2, 2, 2, 1, 1, 3, 3, 1, 1, 3);
  #else
    q15_convolution(qmat_A, qmat_B, pred_A, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 4);
    q15_convolution(qmat_A, qmat_C, pred_B, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 8);
    q15_convolution(qmat_A, qmat_D, pred_C, 2, 2, 2, 2, 3, 3, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8);
    q15_convolution(qmat_A, qmat_B, pred_D, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 8);
  #endif

  return (check_output_q15(pred_A, expected_A, 4) || check_output_q15(pred_B, expected_B, 2) || check_output_q15(pred_C, expected_C, 16) || check_output_q15(pred_D, expected_D, 36));
}

int main() {
  if (test_q15_v_add()) {
    printf("Test Failure for q15_v_add()!\n");
  } else if (test_q7_v_sub()) {
    printf("Test Failure for q7_v_sub()!\n");
  } else if (test_q15_v_sub()) {
    printf("Test Failure for q15_v_sub()!\n");
  } else if (test_q7_v_hadamard()) {
    printf("Test Failure for q7_v_hadamard()!\n");
  } else if (test_q15_v_hadamard()) {
    printf("Test Failure for q15_v_hadamard()!\n");
  } else if (test_q15_v_sigmoid()) {
    printf("Test Failure for q15_v_sigmoid()!\n");
  } else if (test_q15_v_tanh()) {
    printf("Test Failure for q15_v_tanh()!\n");
  } else if (test_q15_v_scalar_add()) {
    printf("Test Failure for q15_v_scalar_add()!\n");
  } else if (test_q15_v_scalar_sub()) {
    printf("Test Failure for q15_v_scalar_sub()!\n");
  } else if (test_q15_v_scalar_mul()) {
    printf("Test Failure for q15_v_scalar_mul()!\n");
  } else if (test_q15_v_argmax()) {
    printf("Test Failure for q15_v_argmax()!\n");
  } else if (test_q15_v_scale_up()) {
    printf("Test Failure for q15_v_scale_up()!\n");
  } else if (test_q15_v_scale_down()) {
    printf("Test Failure for q15_v_scale_down()!\n");
  } else if (test_q15xq7_q15_m_mulvec()) {
    printf("Test Failure for q15xq7_q15_m_mulvec()!\n");
  }  else if (test_q15_m_mulvec()) {
    printf("Test Failure for q15_m_mulvec()!\n");
  } else if (test_q7_t_add()) {
    printf("Test Failure for q7_t_add()!\n");
  } else if (test_q15_t_add()) {
    printf("Test Failure for q15_t_add()!\n");
  } else if (test_q7xq15_q7_t_add_vec()) {
    printf("Test Failure for q7xq15_q7_t_add_vec()!\n");
  } else if (test_q15_t_add_vec()) {
    printf("Test Failure for q15_t_add_vec()!\n");
  } else if (test_q7_t_relu()) {
    printf("Test Failure for q7_t_relu()!\n");
  } else if (test_q15_t_l2_norm()) {
    printf("Test Failure for q15_t_l2_norm()!\n");
  } else if (test_q7xq15_q7_convolution()) {
    printf("Test Failure for q7xq15_q7_convolution()!\n");
  } else if (test_q7xq15_q15_convolution()) {
    printf("Test Failure for q7xq15_q15_convolution()!\n");
  } else if (test_q15_convolution()) {
    printf("Test Failure for q15_convolution()!\n");
  } else {
    printf("All Tests Passed!\n");
    return 0;
  }

  return -1;
}
