// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "quantized_utils.h"

MYINT min(MYINT a, MYINT b) {
  return (a < b) ? a : b;
}

MYINT max(MYINT a, MYINT b) {
  return (a > b) ? a : b;
}

MYINT q16_sigmoid(MYINT x) {
  return 8 * max(min((x + 2048) / 2, 2048), 0);
}

MYINT q16_tanh(MYINT x) {
  return 8 * max(min(x, 2048), -2048);
}

void m_reverse(const MYINT* const A, MYINT* const B, MYITE nrows, MYITE ncols) {
  for (MYITE i = 0; i < nrows; i++) {
    for (MYITE j = 0; j < ncols; j++) {
      B[i * ncols + j] = A[(nrows - i - 1) * ncols + j];
    }
  }
}

void m_q_sigmoid(const MYINT* const A, MYITE nrows, MYITE ncols,
                      MYINT* const B) {
  for (MYITE i = 0; i < nrows; i++) {
    for (MYITE j = 0; j < ncols; j++) {
      B[i * ncols + j] = q16_sigmoid(A[i * ncols + j]);
    }
  }
}

// Currently this uses variable pertaining to int16_t quantization.
void m_q_tanh(const MYINT* const A, MYITE nrows, MYITE ncols,
                   MYINT* const B) {
  for (MYITE i = 0; i < nrows; i++) {
    for (MYITE j = 0; j < ncols; j++) {
      B[i * ncols + j] = q16_tanh(A[i * ncols + j]);
    }
  }
}

void m_q_scalar_add(MYINT A, const MYINT* const B, MYINT* const C,
                    MYITE nrows, MYITE ncols, MYSCL scA, MYSCL scB, MYSCL scC) {
  MYSCL shrmin = min(scA, scB);
  #ifdef SHIFT
    MYSCL shra = scA - shrmin;
    MYSCL shrb = scB - shrmin;
    MYSCL shrc = shrmin - scC;
  #else
    MYSCL shra = scA / shrmin;
    MYSCL shrb = scB / shrmin;
    MYSCL shrc = shrmin / scC;
  #endif

  for (MYITE i = 0; i < nrows; i++) {
    for (MYITE j = 0; j < ncols; j++) {
      #ifdef SHIFT
        C[i * ncols + j] = ((A >> (shra + shrc)) +
                            (B[i * ncols + j] >> (shrb + shrc)));
      #else
        C[i * ncols + j] = ((A / (shra * shrc)) +
                            (B[i * ncols + j] / (shrb * shrc)));
      #endif
    }
  }
}

void m_q_scalar_sub(MYINT A, const MYINT* const B, MYINT* const C, MYITE nrows,
                    MYITE ncols, MYSCL scA, MYSCL scB, MYSCL scC) {
  MYSCL shrmin = min(scA, scB);
  #ifdef SHIFT
    MYSCL shra = scA - shrmin;
    MYSCL shrb = scB - shrmin;
    MYSCL shrc = shrmin - scC;
  #else
    MYSCL shra = scA / shrmin;
    MYSCL shrb = scB / shrmin;
    MYSCL shrc = shrmin / scC;
  #endif

  for (MYITE i = 0; i < nrows; i++) {
    for (MYITE j = 0; j < ncols; j++) {
      #ifdef SHIFT
        C[i * ncols + j] = ((A >> (shra + shrc)) -
                            (B[i * ncols + j] >> (shrb + shrc)));
      #else
        C[i * ncols + j] = ((A / (shra * shrc)) -
                            (B[i * ncols + j] / (shrb * shrc)));
      #endif
    }
  }
}

void m_q_scalar_mul(MYINT A, const MYINT* const B, MYINT* const C, MYITE nrows,
                    MYITE ncols, MYSCL scA, MYSCL scB, MYSCL scC) {
  #ifdef SHIFT
    MYSCL shr = (scA + scB) - scC;
  #else
    MYSCL shr = (scA * scB) / scC;
  #endif

  for (MYITE i = 0; i < nrows; i++) {
    for (MYITE j = 0; j < ncols; j++) {
      #ifdef SHIFT
        C[i * ncols + j] = ((MYINM)A * (MYINM)B[i * ncols + j]) >> shr;
      #else
        C[i * ncols + j] = ((MYINM)A * (MYINM)B[i * ncols + j]) / shr;
      #endif
    }
  }
}

void m_q_hadamard(const MYINT* const A, const MYINT* const B, MYINT* const C,
                  MYITE nrows, MYITE ncols, MYSCL scA, MYSCL scB, MYSCL scC) {
  #ifdef SHIFT
    MYSCL shr = (scA + scB) - scC;
  #else
    MYSCL shr = (scA * scB) / scC;
  #endif

  for (MYITE i = 0; i < nrows; i++) {
    for (MYITE j = 0; j < ncols; j++) {
      #ifdef SHIFT
        C[i * ncols + j] = ((MYINM)A[i * ncols + j] *
                            (MYINM)B[i * ncols + j]) >> shr;
      #else
        C[i * ncols + j] = ((MYINM)A[i * ncols + j] *
                            (MYINM)B[i * ncols + j]) / shr;
      #endif
    }
  }
}

void m_q_add(const MYINT* const A, const MYINT* const B, MYINT* const C,
             MYITE nrows, MYITE ncols, MYSCL scA, MYSCL scB, MYSCL scC) {
  MYSCL shrmin = min(scA, scB);
  #ifdef SHIFT
    MYSCL shra = scA - shrmin;
    MYSCL shrb = scB - shrmin;
    MYSCL shrc = shrmin - scC;
  #else
    MYSCL shra = scA / shrmin;
    MYSCL shrb = scB / shrmin;
    MYSCL shrc = shrmin / scC;
  #endif

  for (MYITE i = 0; i < nrows; i++) {
    for (MYITE j = 0; j < ncols; j++) {
      #ifdef SHIFT
        C[i * ncols + j] = ((A[i * ncols + j] >> (shra + shrc)) +
                        (B[i * ncols + j] >> (shrb + shrc)));
      #else
        C[i * ncols + j] = ((A[i * ncols + j] / (shra * shrc)) +
                        (B[i * ncols + j] / (shrb * shrc)));
      #endif
    }
  }
}

void m_q_mul(const MYINT* const A, const MYINT* const B, MYINT* const C,
             MYITE nrows, MYITE ncols, MYITE nmid, MYSCL scA, MYSCL scB,
             MYSCL scC) {
  #ifdef SHIFT
    MYSCL addshrP = 1, addshr = 0;
    while (addshrP < ncols) {
      addshrP <<= 2;
      addshr += 1;
    }
  #else
    MYSCL addshr = 1;
    while (addshr < ncols) {
      addshr <<= 2;
    }
  #endif
  
  #ifdef SHIFT
    MYSCL shr = scA + scB - scC - addshr;
  #else
    MYSCL shr = (scA * scB) / (scC * addshr);
  #endif

  for (MYITE i = 0; i < nrows; i++) {
    for (MYITE k = 0; k < nmid; k++) {
      MYINM s = 0;
      for(MYITE j = 0; j < ncols; j++) {
        #ifdef SHIFT
          s += ((MYINM)A[i * ncols + j] * (MYINM)B[j * nmid + k]) >> addshr;
        #else
          s += ((MYINM)A[i * ncols + j] * (MYINM)B[j * nmid + k]) / addshr;
        #endif
      }
      #ifdef SHIFT
        C[i * nmid + k] = s >> shr;
      #else
        C[i * nmid + k] = s / shr;
      #endif
    }
  }
}
