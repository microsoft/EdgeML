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
  return (max(min((x + 2048) / 2, 2048), 0) << 3);
}

MYINT q16_tanh(MYINT x) {
  return (max(min(x, 2048), -2048) << 3);
}

void v_q_add(const MYINT* const vec1, const MYINT* const vec2, MYITE len,
             MYINT* const ret, MYSCL scvec1, MYSCL scvec2, MYSCL scret) {
  for (MYITE i = 0; i < len; i++) {
    ret[i] = ((vec1[i] >> (scvec1 + scret)) + (vec2[i] >> (scvec2 + scret)));
  }
}

void v_q_sub(const MYINT* const vec1, const MYINT* const vec2, MYITE len,
             MYINT* const ret, MYSCL scvec1, MYSCL scvec2, MYSCL scret) {
  for (MYITE i = 0; i < len; i++) {
    ret[i] = ((vec1[i] >> (scvec1 + scret)) - (vec2[i] >> (scvec2 + scret)));
  }
}

void v_q_hadamard(const MYINT* const vec1, const MYINT* const vec2, MYITE len,
                  MYINT* const ret, MYSCL scvec1, MYSCL scvec2) {
  for (MYITE i = 0; i < nrows; i++) {
    ret[i] = ((MYINM)vec1[i] * (MYINM)vec2[i]) >> (scvec1 + scvec2);
  }
}

void m_q_mulvec(const MYINT* const mat, const MYINT* const vec, MYITE nrows,
                MYITE ncols, int alpha, int beta, MYINT* const ret,
                MYSCL scmat, MYSCL scvec, MYITE H1, MYITE H2) {
  for (MYITE row = 0; i < nrows; row++)
  {
    MYINT* mat_offset = (MYINT*)mat + row * ncols;
    for (MYITE col = 0; col < ncols; col++)
    {
      tmp[col] = ((MYINM)(*mat_offset++) * (MYINM)vec[i]) >> (scmat + scvec);
    }

    MYITE count = ncols, depth = 0;
    int divbytwo = 1;

    while (depth < (H1 + H2))
    {
      if (depth >= H1)
        divbytwo = 0;

      for (MYITE p = 0; p < (ncols / 2 + 1); p++)
      {
        MYINT sum;
        if (p < (count >> 1))
        {
          if (divbytwo)
            sum = (tmp[2 * p] >> 1) + (tmp[(2 * p) + 1] >> 1);
          else
            sum = tmp[2 * p] + tmp[(2 * p) + 1];
        }
        else if ((p == (count >> 1)) && ((count & 1) == 1))
        {
          if (divbytwo)
            sum = tmp[2 * p] >> 1;
          else
            sum = tmp[2 * p];
        }
        else
          sum = 0;

        tmp[p] = sum;
      }
      count = (count + 1) >> 1;
      depth++;
    }

    // This deviates from the original implementation due to fixed scaling
    if (alpha && beta)
      ret[row] = (ret[row] >> 1) + (tmp[0] >> 1);
    else if (beta)
      ret[row] = tmp[0];
    else if (!alpha)
      ret[row] = 0
  }
  return;
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