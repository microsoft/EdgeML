// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "quantized_utils.h"


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
                ITER_T H1, ITER_T H2) {
  INTM_T tmp[ncols];
  for (ITER_T row = 0; row < nrows; row++) {
    INT_T* mat_offset = (INT_T*)mat + row * ncols;

    for (ITER_T col = 0; col < ncols; col++) {
      tmp[col] = ((INTM_T)(*mat_offset++) * (INTM_T)vec[col]);
    }

    ITER_T count = ncols, depth = 0;
    int divbytwo = 1;

    while (depth < (H1 + H2)) {
      if (depth >= H1)
        divbytwo = 0;

      for (ITER_T p = 0; p < ((ncols >> 1) + 1); p++) {
        INTM_T sum;
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

void Transpose(INT_T *A, INT_T *B, INT_T I, INT_T J) { 
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

void AddOrSubCir4D(INT_T *A, const INT_T *B, INT_T *X, INT_T N, INT_T H,     \
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


void AddOrSubCir2D(INT_T *A, const INT_T *B, INT_T *X, INT_T H, INT_T W,  \
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

void Relu4D(INT_T *A, INT_T N, INT_T H, INT_T W, INT_T C) { 

  INT_T n = 0;

  if(A) {
    for (n = 0; n < N * H *W * C; n++) {
      if (A[n] < 0)
        A[n] = 0;
    }
  }
  return;
}

void Relu2D(INT_T *A, INT_T H, INT_T W) {
  INT_T n = 0;

  if(A) {
    for (n = 0; n < H *W ; n++) {
      if (A[n] < 0)
        A[n] = 0;
    }
  }
  return;
}
void Exp(INT_T *A, INT_T I, INT_T J, INT_T shrA, INT_T shrB, INT_T *B) {

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
