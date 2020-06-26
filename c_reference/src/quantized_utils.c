// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "quantized_utils.h"

void v_q_treesum(INTM_T* const vec, ITER_T len, SCALE_T H1, SCALE_T H2) {
  ITER_T count = len, depth = 0;
  int divbytwo = 1;

  while (depth < (H1 + H2)) {
    if (depth >= H1) {
      divbytwo = 0;
    }

    for (ITER_T p = 0; p < ((len >> 1) + 1); p++) {
      if (p < (count >> 1)) {
        if (divbytwo == 1) {
          #ifdef SHIFT
            vec[p] = (vec[2 * p] >> 1) + (vec[(2 * p) + 1] >> 1);
          #else
            vec[p] = vec[2 * p] / 2 + vec[(2 * p) + 1] / 2;
          #endif
        } else {
          vec[p] = vec[2 * p] + vec[(2 * p) + 1];
        }
      } else if ((p == (count >> 1)) && ((count & 1) == 1)) {
        if (divbytwo == 1) {
          #ifdef SHIFT
            vec[p] = (vec[2 * p] >> 1);
          #else
            vec[p] = vec[2 * p] / 2;
          #endif
        } else {
          vec[p] = vec[2 * p];
        }
      } else {
        vec[p] = 0;
      }
    }
    count = (count + 1) >> 1;
    depth++;
  }
}

void v_q_add(const INT_T* const vec1, const INT_T* const vec2, ITER_T len,
             INT_T* const ret, SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret) {
  for (ITER_T i = 0; i < len; i++) {
#ifdef SHIFT
      ret[i] = ((vec1[i] >> (scvec1 + scret)) + (vec2[i] >> (scvec2 + scret)));
#else
      ret[i] = ((vec1[i] / scvec1) / scret) + ((vec2[i] / scvec2) / scret);
#endif
  }
}

void v_q_sub(const INT_T* const vec1, const INT_T* const vec2, ITER_T len,
             INT_T* const ret, SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret) {
  for (ITER_T i = 0; i < len; i++) {
#ifdef SHIFT
      ret[i] = ((vec1[i] >> (scvec1 + scret)) - (vec2[i] >> (scvec2 + scret)));
#else
      ret[i] = ((vec1[i] / scvec1) / scret) - ((vec2[i] / scvec2) / scret);
#endif
  }
}
void v_q_scalar_add(INT_T scalar, const INT_T* const vec, ITER_T len,
                    INT_T* const ret, SCALE_T scscalar, SCALE_T scvec, SCALE_T scret) {
  for (ITER_T i = 0; i < len; i++) {
#ifdef SHIFT
      ret[i] = ((scalar >> (scscalar + scret)) + (vec[i] >> (scvec + scret)));
#else
      ret[i] = ((scalar / scscalar) / scret) + ((vec[i] / scvec) / scret);
#endif
  }
}

void v_q_scalar_sub(INT_T scalar, const INT_T* const vec, ITER_T len,
                    INT_T* const ret, SCALE_T scscalar, SCALE_T scvec, SCALE_T scret) {
  for (ITER_T i = 0; i < len; i++) {
#ifdef SHIFT
      ret[i] = ((scalar >> (scscalar + scret)) - (vec[i] >> (scvec + scret)));
#else
      ret[i] = ((scalar / scscalar) / scret) - ((vec[i] / scvec) / scret);
#endif
  }
}

void v_q_scalar_mul(INT_T scalar, const INT_T* const vec, ITER_T len,
                    INT_T* const ret, SCALE_T scscalar, SCALE_T scvec) {
  for (ITER_T i = 0; i < len; i++) {
#ifdef SHIFT
      ret[i] = ((INTM_T)scalar * (INTM_T)vec[i]) >> (scscalar + scvec);
#else
      ret[i] = ((((INTM_T)scalar * (INTM_T)vec[i]) / scscalar) / scvec);
#endif
  }
}


void v_q_hadamard(const INT_T* const vec1, const INT_T* const vec2, ITER_T len,
                  INT_T* const ret, SCALE_T scvec1, SCALE_T scvec2) {
  for (ITER_T i = 0; i < len; i++) {
#ifdef SHIFT
      ret[i] = ((INTM_T)vec1[i] * (INTM_T)vec2[i]) >> (scvec1 + scvec2);
#else
      ret[i] = ((((INTM_T)vec1[i] * (INTM_T)vec2[i]) / scvec1) / scvec2);
#endif
  }
}

void v_q_sigmoid(const INT_T* const vec, ITER_T len, INT_T* const ret, INT_T div,
                 INT_T add, INT_T sigmoid_limit, SCALE_T scale_in,
                 SCALE_T scale_out) {
  for (ITER_T i = 0; i < len; i++) {
    INT_T x = (vec[i] / div) + add;

    if (x >= sigmoid_limit) {
      ret[i] = sigmoid_limit << (scale_out - scale_in);
    } else if (x <= 0) {
      ret[i] = 0;
    } else {
      ret[i] = x << (scale_out - scale_in);
    }
  }
}

void v_q_tanh(const INT_T* const vec, ITER_T len, INT_T* const ret,
              SCALE_T scale_in, SCALE_T scale_out) {
  INT_T scale = (1 << scale_in);
  for (ITER_T i = 0; i < len; i++) {
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

void m_q_mulvec(const INT_T* const mat, const INT_T* const vec, ITER_T nrows,
                ITER_T ncols, INT_T* const ret, SCALE_T scmat, SCALE_T scvec,
                SCALE_T H1, SCALE_T H2) {
  INTM_T tmp[ncols];
  for (ITER_T row = 0; row < nrows; row++) {
    INT_T* mat_offset = (INT_T*)mat + row * ncols;

    for (ITER_T col = 0; col < ncols; col++) {
      tmp[col] = ((INTM_T)(*mat_offset++) * (INTM_T)vec[col]);
    }

    v_q_treesum(&tmp[0], ncols, H1, H2);
    #ifdef SHIFT
      ret[row] = (tmp[0] >> (scmat + scvec));
    #else
      ret[row] = ((tmp[0] / scmat) / scvec);
    #endif
  }
}
void sp_mat_mul(const INT_T *Aidx, const INT_T *Aval, INT_T **B, INT_T *C, INT_T K, \
                INT_T shrA, INT_T shrB, INT_T shrC) {

  INT_T   ite_idx = 0; 
  INT_T   ite_val = 0;
  INT_T   k       = 0;
  INT_T   idx     = 0;
  INT_T   b       = 0;
  INT_T   a       = 0;
  INT_T   c       = 0;
  
  if(Aidx && Aval && B && C) {

    for (k = 0; k < K; k++) {

      b = B[k * 1][0];
#ifdef SHIFT
      b = b >> shrB;
#else
      b = b / shrB;
#endif
    
      idx = Aidx[ite_idx];
      while (idx != 0) {
        a = Aval[ite_val];
#ifdef SHIFT
        a = a >> shrA
#else
        a = a / shrA;
#endif
        c = a * b;

#ifdef SHIFT
        c = c >> shrC
#else
        c = c / shrC;
#endif
      
        C[idx - 1] += c;

        ite_idx++;
        ite_val++;

        idx = Aidx[ite_idx];
      }
      ite_idx++;
    }
  }
  return;
}

void arg_max(INT_T *A, INT_T len, INT_T *index) {

  INT_T max       = 0;
  INT_T maxIndex  = 0;
  INT_T counter   = 0;
  INT_T i         = 0;
  INT_T x         = 0;

  if(A && index) {
    max = A[0];

    for (i = 0; i < len; i++) {
      x = A[i];

      if (max < x) {
        maxIndex = counter;
        max      = x;
      }

      counter++;
    }

    *index = maxIndex;
  }
  return;
}

void transpose(INT_T *A, INT_T *B, INT_T I, INT_T J) { 
  INT_T i = 0;
  INT_T j = 0;
  
  if( A && B )
  {
    for (i = 0; i < I*J; i++)
    {
      B[j] = A[i];
      j += I;

      if( (i + 1 ) % J == 0 )
      {
        j = ( (i + 1 ) / J );
      }
    }
  }
  return;
}

void add_or_sub_cir_4D(INT_T *A, const INT_T *B, INT_T *X, INT_T N, INT_T H,     \
                  INT_T W, INT_T C, INT_T shrA, INT_T shrB, INT_T shrC, uint8_t add) {
  INT_T n     = 0;
  INT_T c     = 0; 
  INT_T a     = 0;
  INT_T b     = 0;
  INT_T res   = 0;

  if(A && B && X) {
    for (n = 0; n < N * H * W * C; n++) {
          a = A[n];
#ifdef SHIFT
          a >>= shrA;
#else
          a = a / shrA;
#endif /* SHIFT */
          b = B[c++];
          if(c >= C)
              c = 0;
#ifdef SHIFT
          b >>= shrB;
#else
          b = b / shrB;
#endif /* SHIFT */

          if (add)
#ifdef SHIFT
            res = ((a+b) >> shrC);
#else
            res = ( (a + b) / shrC);
#endif /* SHIFT */
          else
#ifdef SHIFT
            res = ((a-b) >> shrC);
#else
            res = ( (a- b) / shrC);
#endif /* SHIFT */
          X[n] = res;
    }
  }

  return;
}


void add_or_sub_cir_2D(INT_T *A, const INT_T *B, INT_T *X, INT_T H, INT_T W,  \
                   INT_T shrA, INT_T shrB, INT_T shrC, uint8_t add) { 

  INT_T h   = 0;
  INT_T w   = 0;
  INT_T a   = 0;
  INT_T b   = 0; 
  INT_T res = 0;

  if(A && B && X) {
    
    for (h = 0; h < H * W; h++) {
        a = A[h];
#ifdef SHIFT
        a >>= shrA;
#else
        a = a / shrA;
#endif /* SHIFT */

        b = B[w++];
        if(w >= W)
          w = 0;
#ifdef SHIFT
        b >>= shrB;
#else
        b = b / shrB;
#endif /* SHIFT */

        if (add) {
#ifdef SHIFT
            res = ((a+b) >> shrC);
#else
            res = ( (a + b) / shrC);
#endif /* SHIFT */
        }
        else {
#ifdef SHIFT
            res = ((a-b) >> shrC);
#else
            res = ( (a- b) / shrC);
#endif /* SHIFT */
        }

        X[h] = res;
    }
  }

  return;
}

/*
   For 2D array, give W = 0, C = 0
*/
void relu_4D(INT_T *A, INT_T N, INT_T H, INT_T W, INT_T C) { 

  INT_T n = 0;
  
  if(A) {

    if(W == 0 && C == 0) {
      
      W = 1;
      C = 1;
    }
    for (n = 0; n < N * H * W * C; n++) {
      if (A[n] < 0)
        A[n] = 0;
    }
  }
  return;
}

void exp_scale(INT_T *A, INT_T I, INT_T J, INT_T shrA, INT_T shrB, INT_T *B) {

  INT_T i = 0;
  
  if(A && B) {
    for (i = 0; i < I*J; i++) {
#ifdef SHIFT
    B[i] = ((INT_T)(exp(((float)A[i]) >> shrA) * shrB));
#else
    B[i] = ((INT_T)(exp(((float)A[i]) / shrA) * shrB));
#endif /* SHIFT */
    }
  }

  return;
}

void sigmoid(INT_T *A, INT_T I, INT_T J, INT_T div, INT_T add, INT_T sigmoid_limit, \
             INT_T scale_in, INT_T scale_out, INT_T *B) {

  INT_T i           = 0;
  INT_T x           = 0;
  INT_T y           = 0;
  INT_T scale_diff  = 0;

#ifdef FLOATEXP
  INT_T z           = 0;
#endif

  if(A && B) {
#ifdef SHIFT
      scale_diff = scale_out >> scale_in;
#else
      scale_diff = scale_out / scale_in;
#endif /* SHIFT */

    for (i = 0; i < I*J; i++) {
#ifdef FLOATEXP
#ifdef SHIFT
      float x = float(A[i]) >> scale_in;

      float y = 1 >> (1 + exp(-x));
#else
      float x = float(A[i]) / scale_in;

      float y = 1 / (1 + exp(-x));
#endif /* SHIFT */

      z = INT_T(y * scale_out);

      B[i] = z;
#else
      x = A[i];
#ifdef SHIFT
      x = (x >> div) + add;
#else
      x = (x / div) + add;
#endif /* SHIFT */

      if (x >= sigmoid_limit)
        y = sigmoid_limit;
      else if (x <= 0)
        y = 0;
      else
        y = x;

      y = y * scale_diff;

      B[i] = y;
#endif
    }
  }

  return;
}

/*
   For 2D array, give K = 0, L = 0
*/
void adjust_scale_shr(INT_T *A, INT_T I, INT_T J, INT_T K, INT_T L, INT_T scale) {

  INT_T i = 0;

  if(A) {

    if(K == 0 && L == 0 ) {

      K = 1;
      L = 1;
    }

    while(i < I * J * K * L ) {
#ifdef SHIFT
      A[i++] >>= scale;
#else
      A[i++] /= scale;
#endif /* SHIFT */
    }
  }

  return;
}

/**
 * Following function does scaling on multidimensional array(4-D array)
   For 2D array, give K = 0, L = 0
 */ 
void adjust_scale_shl(INT_T *A, INT_T I, INT_T J, INT_T K, INT_T L, INT_T scale) {

  INT_T i = 0;

  if(A) {

  if(K == 0 && L == 0 ) {

      K = 1;
      L = 1;
    }    
    while(i < I * J * K * L) {
      A[i++] *= scale;
    }
  }

  return;
}
void Reverse2(INT_T *A, INT_T axis, INT_T I, INT_T J, INT_T *B) {

  INT_T i   = 0;
  INT_T j   = 0;
  INT_T ref = 0;

  if(A && B) {

    j = J - 1;

    if(axis) {

      for ( i = 0; i < I * J; i++) {

          B[i] = A[j--];
          if(j < ref)
          {
              j   = i + J;
              ref = i + 1;
          }
      }
    }
    else
    {
        j   = I*J - J;
        ref = j;

        for ( i = 0; i < I * J; i++) {

          B[i] = A[j++];

          if(j >= (ref + J) ) {
              j = ref - J;
              ref = j;
          }
      }
    }
  }

  return;
}