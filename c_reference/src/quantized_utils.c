// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "quantized_utils.h"

void v_q_add(const MYINT* const vec1, const MYINT* const vec2, MYITE len,
             MYINT* const ret, MYSCL scvec1, MYSCL scvec2, MYSCL scret) {
  for (MYITE i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = ((vec1[i] >> (scvec1 + scret)) + (vec2[i] >> (scvec2 + scret)));
    #else
      ret[i] = ((vec1[i] / scvec1) / scret) + ((vec2[i] / scvec2) / scret);
    #endif
  }
}

void v_q_sub(const MYINT* const vec1, const MYINT* const vec2, MYITE len,
             MYINT* const ret, MYSCL scvec1, MYSCL scvec2, MYSCL scret) {
  for (MYITE i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = ((vec1[i] >> (scvec1 + scret)) - (vec2[i] >> (scvec2 + scret)));
    #else
      ret[i] = ((vec1[i] / scvec1) / scret) - ((vec2[i] / scvec2) / scret);
    #endif
  }
}

void v_q_hadamard(const MYINT* const vec1, const MYINT* const vec2, MYITE len,
                  MYINT* const ret, MYSCL scvec1, MYSCL scvec2) {
  for (MYITE i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = ((MYINM)vec1[i] * (MYINM)vec2[i]) >> (scvec1 + scvec2);
    #else
      ret[i] = ((((MYINM)vec1[i] * (MYINM)vec2[i]) / scvec1) / scvec2);
    #endif
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
  MYINT scale = (1 << scale_in);
  for (MYITE i = 0; i < len; i++) {
    if (vec[i] >= scale) {
      ret[i] = scale;
    } else if (vec[i] <= -scale) {
      ret[i] = (-scale);
    } else {
      ret[i] = vec[i];
    }
    ret[i] <<= (scale_out - scale_in);
  }
}

void v_q_scalar_add(MYINT scalar, const MYINT* const vec, MYITE len,
                    MYINT* const ret, MYSCL scscalar, MYSCL scvec, MYSCL scret) {
  for (MYITE i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = ((scalar >> (scscalar + scret)) + (vec[i] >> (scvec + scret)));
    #else
      ret[i] = ((scalar / scscalar) / scret) + ((vec[i] / scvec) / scret);
    #endif
  }
}

void v_q_scalar_sub(MYINT scalar, const MYINT* const vec, MYITE len,
                    MYINT* const ret, MYSCL scscalar, MYSCL scvec, MYSCL scret) {
  for (MYITE i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = ((scalar >> (scscalar + scret)) - (vec[i] >> (scvec + scret)));
    #else
      ret[i] = ((scalar / scscalar) / scret) - ((vec[i] / scvec) / scret);
    #endif
  }
}

void v_q_scalar_mul(MYINT scalar, const MYINT* const vec, MYITE len,
                    MYINT* const ret, MYSCL scscalar, MYSCL scvec) {
  for (MYITE i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = ((MYINM)scalar * (MYINM)vec[i]) >> (scscalar + scvec);
    #else
      ret[i] = ((((MYINM)scalar * (MYINM)vec[i]) / scscalar) / scvec);
    #endif
  }
}

void m_q_mulvec(const MYINT* const mat, const MYINT* const vec, MYITE nrows,
                MYITE ncols, MYINT* const ret, MYSCL scmat, MYSCL scvec,
                MYITE H1, MYITE H2) {
  MYINM tmp[ncols];
  for (MYITE row = 0; row < nrows; row++) {
    MYINT* mat_offset = (MYINT*)mat + row * ncols;

    for (MYITE col = 0; col < ncols; col++) {
      tmp[col] = ((MYINM)(*mat_offset++) * (MYINM)vec[col]);
    }

    MYITE count = ncols, depth = 0;
    int divbytwo = 1;

    while (depth < (H1 + H2)) {
      if (depth >= H1)
        divbytwo = 0;

      for (MYITE p = 0; p < ((ncols >> 1) + 1); p++) {
        MYINM sum;
        if (p < (count >> 1)) {
          if (divbytwo == 1) {
            #ifdef SHIFT
              sum = (tmp[2 * p] >> 1) + (tmp[(2 * p) + 1] >> 1);
            #else
              sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
            #endif
          } else {
            sum = tmp[2 * p] + tmp[(2 * p) + 1];
          }
        } else if ((p == (count >> 1)) && ((count & 1) == 1)) {
          if (divbytwo == 1) {
            #ifdef SHIFT
              sum = (tmp[2 * p] >> 1);
            #else
              sum = tmp[2 * p] / 2;
            #endif
          } else {
            sum = tmp[2 * p];
          }
        } else {
          sum = 0;
        }

        tmp[p] = sum;
      }
      count = (count + 1) >> 1;
      depth++;
    }
    #ifdef SHIFT
      ret[row] = (tmp[0] >> (scmat + scvec));
    #else
      ret[row] = ((tmp[0] / scmat) / scvec);
    #endif
  }
}
