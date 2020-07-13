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

void v_q_argmax(const INT_T* const vec, ITER_T len, ITER_T* const ret) {
  INT_T max_value = vec[0];
  ITER_T max_index = 0;

  for (ITER_T i = 1; i < len; i++) {
    if (max_value < vec[i]) {
      max_index = i;
      max_value = vec[i];
    }
  }

  *ret = max_index;
}

void v_q_relu(INT_T* const vec, ITER_T len) {
  for (ITER_T i = 0; i < len; i++) {
    if (vec[i] < 0) {
      vec[i] = 0;
    }
  }
}

void v_q_exp(const INT_T* const vec, ITER_T len, INT_T* const ret,
             SCALE_T scvec, SCALE_T scret) {
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = (((INT_T)exp((float)(vec[i] >> scvec))) << scret);
    #else
      ret[i] = ((INT_T)(exp(((float)vec[i]) / scvec) * scret));
    #endif
  }
}

void v_q_scale_up(INT_T* const vec, ITER_T len, SCALE_T scvec) {
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      mat[i] <<= scvec;
    #else
      mat[i] *= scvec;
    #endif
  }
}

void v_q_scale_down(INT_T* const vec, ITER_T len, SCALE_T scvec) {
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      vec[i] >>= scvec;
    #else
      vec[i] /= scvec;
    #endif
  }
}

void m_q_transpose(const INT_T* const mat, ITER_T nrows, ITER_T ncols,
                   INT_T* const ret) {
  for (ITER_T i = 0; i < nrows; i++) {
    for (ITER_T j = 0; j < ncols; j++) {
      ret[i * ncols + j] = mat[j * nrows + i];
    }
  }
}

void m_q_reverse(const INT_T* const mat, ITER_T nrows, ITER_T ncols, ITER_T axis,
                 INT_T* const ret) {
  for (ITER_T i = 0; i < nrows; i++) {
    for (ITER_T j = 0; j < ncols; j++) {
      if (axis == 0) {
        ret[i * ncols + j] = mat[(nrows - 1 - i) * ncols + j];
      } else {
        ret[i * ncols + j] = mat[i * ncols + (ncols - 1 - j)];
      }
    }
  }
}

void m_q_add_sub_2D(const INT_T* const mat, const INT_T* const vec,
                    ITER_T nrows, ITER_T ncols, INT_T* const ret,
                    SCALE_T scmat, SCALE_T scvec, SCALE_T scret, uint8_t add) {
  for (ITER_T i = 0, w = 0; i < nrows * ncols; i++, w++) {
    if (w >= ncols) {
      w = 0;
    }

    if (add) {
      #ifdef SHIFT
        ret[i] = (mat[i] >> (scmat + scret)) + (vec[w] >> (scvec + scret));
      #else
        ret[i] = ((mat[i] / scmat) / scret) + ((vec[w] / scvec) / scret);
      #endif
    } else {
      #ifdef SHIFT
        ret[i] = (mat[i] >> (scmat + scret)) - (vec[w] >> (scvec + scret));
      #else
        ret[i] = ((mat[i] / scmat) / scret) - ((vec[w] / scvec) / scret);
      #endif
    }
  }
}

void m_q_add_sub_4D(const INT_T* const mat, const INT_T* const vec,
                    ITER_T nbatches, ITER_T nrows, ITER_T ncols,
                    ITER_T nchannels, INT_T* const ret, SCALE_T scmat,
                    SCALE_T scvec, SCALE_T scret, uint8_t add) {
  for (ITER_T i = 0, c = 0; i < nbatches * nrows * ncols * nchannels; i++, c++) {
    if (c >= nchannels) {
      c = 0;
    }

    if (add) {
      #ifdef SHIFT
        ret[i] = (mat[i] >> (scmat + scret)) + (vec[c] >> (scvec + scret));
      #else
        ret[i] = ((mat[i] / scmat) / scret) + ((vec[c] / scvec) / scret);
      #endif
    } else {
      #ifdef SHIFT
        ret[i] = (mat[i] >> (scmat + scret)) - (vec[c] >> (scvec + scret));
      #else
        ret[i] = ((mat[i] / scmat) / scret) - ((vec[c] / scvec) / scret);
      #endif
    }
  }
}

void sp_mat_mul(const INT_T *Aidx, const INT_T *Aval, INT_T **B, INT_T *C, INT_T K,
                INT_T shrA, INT_T shrB, INT_T shrC) {
  INT_T k       = 0;
  INT_T b       = 0;
  INT_T idx     = 0;
  INT_T a       = 0;
  INT_T c       = 0;
  INT_T ite_idx = 0;
  INT_T ite_val = 0;

  for (k = 0; k < K; k++) {
    b = B[k * 1][0];
    #ifdef FASTAPPROX
      #ifdef SHIFT
        b = b >> shrB;
      #else
        b = b / shrB;
      #endif
    #endif

    idx = Aidx[ite_idx];
    while (idx != 0) {
      a = Aval[ite_val];
      #ifdef FASTAPPROX
        #ifdef SHIFT
          a = a >> shrA;
          c = a * b;
          c = c >> shrC;
        #else
          a = a / shrA;
          c = a * b;
          c = c / shrC;
        #endif
      #else
        #ifdef SHIFT
          c = (((INT_T)a * (INT_T)b) >> (shrC + shrA + shrB));
        #else
          c = (((INT_T)a * (INT_T)b) / ((INT_T)shrC * (INT_T)shrA * (INT_T)shrB));
        #endif
      #endif

      C[idx - 1] += c;
      ite_idx++;
      ite_val++;
      idx = Aidx[ite_idx];
    }
    ite_idx++;
  }

  return;
}

void convolution(INT_T *A, const INT_T *B, INT_T *C, INT_T *tmp,INT_T N,
                 INT_T H, INT_T W, INT_T CIN, INT_T HF,INT_T WF,
                 INT_T CINF, INT_T COUTF, INT_T HOUT,INT_T WOUT,
                 INT_T HPADL, INT_T HPADR, INT_T WPADL,INT_T WPADR,
                 INT_T HSTR, INT_T WSTR, INT_T HDL,INT_T WDL, INT_T G,
                 INT_T shrA, INT_T shrB, INT_T H1,INT_T H2) {
  INT_T n         = 0;
  INT_T h         = 0;
  INT_T w         = 0;
  INT_T g         = 0;
  INT_T co        = 0;
  INT_T counter   = 0;
  INT_T hf        = 0;
  INT_T wf        = 0;
  INT_T ci        = 0;
  INT_T a         = 0;
  INT_T b         = 0;
  int64_t temp    = 0;
  INT_T totalEle  = 0;
  INT_T count     = 0;
  uint8_t shr     = 0;
  INT_T sum       = 0;
  INT_T p         = 0;
  INT_T hout      = 0;
  INT_T wout      = 0;
  INT_T depth     = 0;

  INT_T HOffsetL  = HDL * (HF / 2) - HPADL;
  INT_T WOffsetL  = WDL * (WF / 2) - WPADL;
  INT_T HOffsetR  = HDL * (HF / 2) - HPADR;
  INT_T WOffsetR  = WDL * (WF / 2) - WPADR;

  if(A && B && C && tmp) {
    for(n = 0; n < N; n++) {
      for( h = HOffsetL, hout = 0; h < H - HOffsetR; h += HSTR, hout++) {
        for( w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
          for( g = 0; g < G; g++) {
            for( co = 0; co < COUTF; co ++) {
              counter = 0;
              for( hf = -(HF/2); hf <= HF/2; hf++) {
                for( wf = -(WF/2); wf <= WF/2; wf++) {
                  for( ci = 0; ci < CINF; ci++) {
                    a = (((h + HDL * hf) < 0) || ((h + HDL * hf) >= H) || ((w + WDL * wf) < 0) || ((w + WDL * wf) >= W)) ? 0 : A[n * H * W * CIN + (h + HDL * hf) * W * CIN + (w + WDL * wf) * CIN + (ci + g * CINF)];
                    b = B[g * HF * WF * CINF * COUTF + (hf + HF/2) * WF * CINF * COUTF + (wf + WF/2) * CINF * COUTF + ci * COUTF + co];
                    #ifdef FASTAPPROX
                      a = a / shrA;
                      b = b / shrB;
                      tmp[counter] = a * b;
                    #else
                      temp = (((int64_t) a) * ((int64_t)b)) / (((int64_t)shrA) * ((int64_t)shrB));
                      tmp[counter] = (temp);
                    #endif
                    counter++;
                  }
                }
              }

              totalEle  = HF * WF * CINF;
              count     = HF * WF * CINF;
              depth     = 0;
              shr       = 1;

              while (depth < (H1 + H2)) {
                if (depth >= H1)
                  shr = 0;
                #ifdef SHIFT
                  for (p = 0; p < ((totalEle >> 1) + 1); p++) {
                #else
                  for (p = 0; p < (totalEle / 2 + 1); p++) {
                #endif
                    if (p < (count >> 1)) {
                      if (shr)
                        #ifdef SHIFT
                          sum = ( tmp[2 * p] >> 1 ) + ( tmp[(2 * p) + 1] / 2 );
                        #else
                          sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
                        #endif
                      else
                        sum = tmp[2 * p] + tmp[(2 * p) + 1];
                    }
                    else if ((p == (count >> 1)) && ((count & 1) == 1)) {
                      if (shr)
                        #ifdef SHIFT
                          sum = tmp[2 * p] >> 1;
                        #else
                          sum = tmp[2 * p] / 2;
                        #endif
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
              C[n * HOUT * WOUT * (COUTF * G) + hout * WOUT * (COUTF * G) + wout * (COUTF * G) + (co + g * COUTF)] = tmp[0];
            }
          }
        }
      }
    }
  }
}

void maxpool(INT_T *A, INT_T *B, INT_T N, INT_T H, INT_T W, INT_T C, INT_T FH,
             INT_T FW, INT_T strideH, INT_T strideW, INT_T HPADL, INT_T HPADR,
             INT_T WPADL, INT_T WPADR) {
  INT_T n   = 0;
  INT_T ho  = 0;
  INT_T wo  = 0;
  INT_T c   = 0;
  INT_T max = 0;
  INT_T a   = 0;
  INT_T hs  = 0;
  INT_T ws  = 0;
  INT_T HO  = 0;
  INT_T WO  = 0;

if (A && B) {
  #ifdef SHIFT
    HO = H >> strideH;
    WO = W >> strideW;
  #else
    HO = H / strideH;
    WO = W / strideW;
  #endif

  if (A && B) {
    for (n = 0; n < N; n++) {
      for (ho = 0; ho < HO; ho++) {
        for (wo = 0; wo < WO; wo++) {
          for (c = 0; c < C; c++) {
            max = A[n * H * W * C + (strideH * ho) * W * C + (strideW * wo) * C + c];
            for (hs = 0; hs < FH; hs++) {
              for (ws = 0; ws < FW; ws++) {
                a = A[n * H * W * C + ((strideH * ho) + hs) * W * C + ((strideW * wo) + ws) * C + c];
                if (a > max)
                  max = a;
                }
              }
              B[n * HO * WO * C + ho * WO * C + wo * C + c] = max;
            }
          }
        }
      }
    }
  }
  return;
}
