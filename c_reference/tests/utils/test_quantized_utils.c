// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stdio.h>
#include "quantized_utils.h"

// All values generated from Seedot on Wider Regression dataset.
// By default, all tests run without using bit-shifting operations.
// Function for matching the predicted and expected quantized outputs.
int check_output(const INT_T* const pred, const INT_T* const expected,
                 unsigned len) {
  for (unsigned i = 0; i < len; i++)
  {
    if (pred[i] != expected[i]) {
      return 1;
    }
  }
  return 0;
}

// Test v_q_add() function.
int test_v_q_add() {
  const INT_T qvec_A[8] = {-425, -169, -3534, 524, -2739, 87, 52, 292};
  const INT_T qvec_B[8] = {-18777, -9518, 4055, -7309, 8584, -17257, -5280, -7933};
  const INT_T expected[8] = {-2772, -1358, -3028, -389, -1666, -2070, -608, -699};
  INT_T pred[8];

  v_q_add(&qvec_A[0], &qvec_B[0], 8, &pred[0], 1, 8, 1);
  return check_output(pred, expected, 8);
}

// Test v_q_sub() function.
int test_v_q_sub() {
  const INT_T qvec_A[8] = {-425, -169, -3534, 524, -2739, 87, 52, 292};
  const INT_T qvec_B[8] = {-18777, -9518, 4055, -7309, 8584, -17257, -5280, -7933};
  const INT_T expected[8] = {1922, 1020, -4040, 1437, -3812, 2244, 712, 1283};
  INT_T pred[8];

  v_q_sub(&qvec_A[0], &qvec_B[0], 8, &pred[0], 1, 8, 1);
  return check_output(pred, expected, 8);
}

// Test v_q_hadamard() function.
int test_v_q_hadamard() {
  const INT_T qvec_A[8] = {16378, 13638, 16378, 9787, 14861, 16378, 10661, 11018};
  const INT_T qvec_B[8] = {178, 1064, -2048, 1718, -1663, 851, 1244, 1282};
  const INT_T expected[8] = {1423, 7085, -16378, 8209, -12067, 6805, 6475, 6897};
  INT_T pred[8];

  v_q_hadamard(&qvec_A[0], &qvec_B[0], 8, &pred[0], 32, 64);
  return check_output(pred, expected, 8);
}

// Test v_q_sigmoid() function.
int test_v_q_sigmoid() {
  const INT_T qvec_A[8] = {-2772, -1358, -3028, -389, -1666, -2070, -608, -699};
  const INT_T expected[8] = {0, 2760, 0, 6640, 1528, 0, 5760, 5400};
  INT_T pred[8];

  v_q_sigmoid(&qvec_A[0], 8, &pred[0], 2, 1024, 2048, 11, 14);
  return check_output(pred, expected, 8);
}

// Test v_q_tanh() function.
int test_v_q_tanh() {
  const INT_T qvec_A[8] = {178, 1064, -4162, 1718, -1663, 851, 1244, 1282};
  const INT_T expected[8] = {178, 1064, -2048, 1718, -1663, 851, 1244, 1282};
  INT_T pred[8];

  v_q_tanh(&qvec_A[0], 8, &pred[0], 11, 11);
  return check_output(pred, expected, 8);
}

// Test v_q_scalar_add() function.
int test_v_q_scalar_add() {
  const INT_T qscalar_A = 30111;
  const INT_T qvec_B[8] = {16261, 13521, 16261, 9670, 14744, 16261, 10544, 10901};
  const INT_T expected[8] = {16378, 13638, 16378, 9787, 14861, 16378, 10661, 11018};
  INT_T pred[8];

  v_q_scalar_add(qscalar_A, &qvec_B[0], 8, &pred[0], 256, 1, 1);
  return check_output(pred, expected, 8);
}

// Test v_q_scalar_sub() function.
int test_v_q_scalar_sub() {
  const INT_T qscalar_A = 16384;
  const INT_T qvec_B[8] = {0, 2760, 0, 6640, 1528, 0, 5760, 5400};
  const INT_T expected[8] = {16384, 13624, 16384, 9744, 14856, 16384, 10624, 10984};
  INT_T pred[8];

  v_q_scalar_sub(qscalar_A, &qvec_B[0], 8, &pred[0], 1, 1, 1);
  return check_output(pred, expected, 8);
}

// Test v_q_scalar_mul() function.
int test_v_q_scalar_mul() {
  const INT_T qscalar_A = 32522;
  const INT_T qvec_B[8] = {16384, 13624, 16384, 9744, 14856, 16384, 10624, 10984};
  const INT_T expected[8] = {16261, 13521, 16261, 9670, 14744, 16261, 10544, 10901};
  INT_T pred[8];

  v_q_scalar_mul(qscalar_A, &qvec_B[0], 8, &pred[0], 128, 256);
  return check_output(pred, expected, 8);
}

// Test m_q_mulvec() function.
int test_m_q_mulvec() {
  const INT_T qmat_A[8 * 4] = {7069, -10389, 1562, -1992, 3262, -37, -1143, -995, 5513, -17035, -14615, -6636, 4733, -403, 4106, -1104, -2707, -1287, -18128, -1832, -10108, -137, 2064, 1207, 5233, 226, 831, -1909, 4489, -1099, 2845, -1261};
  const INT_T qvec_B[4] = {1040, 1919, 4254, 4024};
  const INT_T expected[8] = {-425, -169, -3534, 524, -2739, 87, 52, 292};
  INT_T pred[8];

  m_q_mulvec(&qmat_A[0], &qvec_B[0], 8, 4, &pred[0], 128, 64, 2, 0);
  return check_output(pred, expected, 8);
}

int main() {
  if (test_v_q_add()) {
    printf("Test Failure for v_q_add()!\n");
  } else if (test_v_q_sub()) {
    printf("Test Failure for v_q_sub()!\n");
  } else if (test_v_q_hadamard()) {
    printf("Test Failure for v_q_hadamard()!\n");
  } else if (test_v_q_sigmoid()) {
    printf("Test Failure for v_q_sigmoid()!\n");
  } else if (test_v_q_tanh()) {
    printf("Test Failure for v_q_tanh()!\n");
  } else if (test_v_q_scalar_add()) {
    printf("Test Failure for v_q_scalar_add()!\n");
  } else if (test_v_q_scalar_sub()) {
    printf("Test Failure for v_q_scalar_sub()!\n");
  } else if (test_v_q_scalar_mul()) {
    printf("Test Failure for v_q_scalar_mul()!\n");
  } else if (test_m_q_mulvec()) {
    printf("Test Failure for m_q_mulvec()!\n");
  } else {
    printf("All Tests Passed!\n");
    return 0;
  }

  return -1;
}
