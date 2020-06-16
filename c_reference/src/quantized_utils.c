// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "quantized_utils.h"

MYINT min(MYINT a, MYINT b) {
  return (a < b) ? a : b;
}

MYINT max(MYINT a, MYINT b) {
  return (a > b) ? a : b;
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
  for (MYITE i = 0; i < len; i++) {
    ret[i] = ((MYINM)vec1[i] * (MYINM)vec2[i]) >> (scvec1 + scvec2);
  }
}

void v_q_sigmoid(const MYINT* const vec, MYITE len, MYINT* const ret, MYINT div,
                 MYINT add, MYINT sigmoid_limit, MYSCL scale_in,
                 MYSCL scale_out) {
  for (MYITE i = 0; i < len; i++) {
    MYINT x = (vec[i] / div) + add;

    if (x >= sigmoid_limit) {
      ret[i] = sigmoid_limit << (scale_out - scale_in);
    } else if (x <= 0) {
      ret[i] = 0;
    } else {
      ret[i] = x << (scale_out - scale_in);
    }
  }
}

void v_q_tanh(const MYINT* const vec, MYITE len, MYINT* const ret,
              MYSCL scale_in, MYSCL scale_out) {
  for (MYITE i = 0; i < len; i++) {
    if (vec[i] >= scale_in) {
      ret[i] = scale_in << (scale_out - scale_in);
    } else if (vec[i] <= -scale_in) {
      ret[i] = -scale_in << (scale_out - scale_in);
    } else {
      ret[i] = vec[i] << (scale_out - scale_in);
    }
  }
}

void v_q_scalar_add(MYINT scalar, const MYINT* const vec, MYITE len,
                    MYINT* const ret, MYSCL scscalar, MYSCL scvec, MYSCL scret) {
  for (MYITE i = 0; i < len; i++) {
    ret[i] = ((scalar >> (scscalar + scret)) + (vec[i] >> (scvec + scret)));
  }
}

void v_q_scalar_sub(MYINT scalar, const MYINT* const vec, MYITE len,
                    MYINT* const ret, MYSCL scscalar, MYSCL scvec, MYSCL scret) {
  for (MYITE i = 0; i < len; i++) {
    ret[i] = ((scalar >> (scscalar + scret)) - (vec[i] >> (scvec + scret)));
  }
}

void v_q_scalar_mul(MYINT scalar, const MYINT* const vec, MYITE len,
                    MYINT* const ret, MYSCL scscalar, MYSCL scvec) {
  for (MYITE i = 0; i < len; i++) {
    ret[i] = ((MYINM)scalar * (MYINM)vec[i]) >> (scscalar + scvec);
  }
}

void m_q_mulvec(const MYINT* const mat, const MYINT* const vec, MYITE nrows,
                MYITE ncols, MYINT* const ret, MYSCL scmat, MYSCL scvec,
                MYITE H1, MYITE H2) {
  MYINT tmp[ncols];
  for (MYITE row = 0; row < nrows; row++)
  {
    MYINT* mat_offset = (MYINT*)mat + row * ncols;

    for (MYITE col = 0; col < ncols; col++) {
      tmp[col] = ((MYINM)(*mat_offset++) * (MYINM)vec[col]) >> (scmat + scvec);
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
    ret[row] = tmp[0];
  }
}
