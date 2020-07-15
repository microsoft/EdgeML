// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "quantized_utils.h"
#include <stdio.h>

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

void v_q_sub_scalar(const INT_T* const vec, INT_T scalar, ITER_T len,
                    INT_T* const ret, SCALE_T scvec, SCALE_T scscalar, SCALE_T scret) {
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = ((vec[i] >> (scvec + scret)) - (scalar >> (scscalar + scret)));
    #else
      ret[i] = ((vec[i] / scvec) / scret) - ((scalar / scscalar) / scret);
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
    ret[i] = ((INT_T)(exp(((float)vec[i]) / scvec) * scret));
  }
}

void v_q_scale_up(INT_T* const vec, ITER_T len, SCALE_T scvec) {
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      vec[i] <<= scvec;
    #else
      vec[i] *= scvec;
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

void m_q_add_vec(const INT_T* const mat, const INT_T* const vec,
                 ITER_T nrows, ITER_T ncols, INT_T* const ret,
                 SCALE_T scmat, SCALE_T scvec, SCALE_T scret) {
  ITER_T len = nrows * ncols;
  for (ITER_T i = 0, w = 0; i < len; i++, w++) {
    if (w >= ncols) {
      w = 0;
    }

    #ifdef SHIFT
      ret[i] = ((mat[i] >> (scmat + scret)) + (vec[w] >> (scvec + scret)));
    #else
      ret[i] = ((mat[i] / scmat) / scret) + ((vec[w] / scvec) / scret);
    #endif
  }
}

void m_q_sub_vec(const INT_T* const mat, const INT_T* const vec,
                 ITER_T nrows, ITER_T ncols, INT_T* const ret,
                 SCALE_T scmat, SCALE_T scvec, SCALE_T scret) {
  ITER_T len = nrows * ncols;
  for (ITER_T i = 0, w = 0; i < len; i++, w++) {
    if (w >= ncols) {
      w = 0;
    }

    #ifdef SHIFT
      ret[i] = ((mat[i] >> (scmat + scret)) - (vec[w] >> (scvec + scret)));
    #else
      ret[i] = ((mat[i] / scmat) / scret) - ((vec[w] / scvec) / scret);
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

void m_q_sparse_mulvec(const INT_T* const mat_indices, const INT_T* const mat_values,
                       const INT_T* const vec, ITER_T ndims, INT_T* const ret,
                       SCALE_T scmat, SCALE_T scvec, SCALE_T scret) {
  ITER_T iter_index = 0, iter_value = 0;
  for (ITER_T k = 0; k < ndims; k++) {
    ITER_T index = mat_indices[iter_index];

    while (index != 0) {
      #ifdef SHIFT
        ret[index - 1] += (((INTM_T)mat_values[iter_value] * (INTM_T)vec[k]) >> (scmat + scvec + scret));
      #else
        ret[index - 1] += (((INTM_T)mat_values[iter_value] * (INTM_T)vec[k]) / ((INTM_T)scmat * (INTM_T)scvec * (INTM_T)scret));
      #endif
      iter_index++;
      iter_value++;
      index = mat_indices[iter_index];
    }

    iter_index++;
  }
}

void t_q_add_vec(const INT_T* const mat, const INT_T* const vec,
                 ITER_T nbatches, ITER_T nrows, ITER_T ncols,
                 ITER_T nchannels, INT_T* const ret, SCALE_T scmat,
                 SCALE_T scvec, SCALE_T scret) {
  ITER_T len = nbatches * nrows * ncols * nchannels;
  for (ITER_T i = 0, c = 0; i < len; i++, c++) {
    if (c >= nchannels) {
      c = 0;
    }

    #ifdef SHIFT
      ret[i] = ((mat[i] >> (scmat + scret)) + (vec[c] >> (scvec + scret)));
    #else
      ret[i] = ((mat[i] / scmat) / scret) + ((vec[c] / scvec) / scret);
    #endif
  }
}

void t_q_sub_vec(const INT_T* const mat, const INT_T* const vec,
                 ITER_T nbatches, ITER_T nrows, ITER_T ncols,
                 ITER_T nchannels, INT_T* const ret, SCALE_T scmat,
                 SCALE_T scvec, SCALE_T scret) {
  ITER_T len = nbatches * nrows * ncols * nchannels;
  for (ITER_T i = 0, c = 0; i < len; i++, c++) {
    if (c >= nchannels) {
      c = 0;
    }

    #ifdef SHIFT
      ret[i] = ((mat[i] >> (scmat + scret)) - (vec[c] >> (scvec + scret)));
    #else
      ret[i] = ((mat[i] / scmat) / scret) - ((vec[c] / scvec) / scret);
    #endif
  }
}

void q_maxpool(const INT_T* const mat, ITER_T nbatches, ITER_T nrows, ITER_T ncols,
               ITER_T nchannels, ITER_T hfilter, ITER_T wfilter, ITER_T hstride,
               ITER_T wstride, ITER_T hpadl, ITER_T hpadr, ITER_T wpadl,
               ITER_T wpadr, INT_T* const ret) {
  //CHECK
  ITER_T hoffset = nrows / hstride;
  ITER_T woffset = ncols / wstride;

  for (ITER_T n = 0; n < nbatches; n++) {
    for (ITER_T h = 0; h < hoffset; h++) {
      for (ITER_T w = 0; w < woffset; w++) {
        for (ITER_T c = 0; c < nchannels; c++) {
          INT_T max = mat[n * nrows * ncols * nchannels + (hstride * h) * ncols * nchannels + (wstride * w) * nchannels + c];

          for (ITER_T hs = 0; hs < hfilter; hs++) {
            for (ITER_T ws = 0; ws < wfilter; ws++) {
              INT_T a = mat[n * nrows * ncols * nchannels + ((hstride * h) + hs) * ncols * nchannels + ((wstride * w) + ws) * nchannels + c];
              if (a > max) {
                max = a;
              }
            }
          }

          ret[n * hoffset * woffset * nchannels + h * woffset * nchannels + w * nchannels + c] = max;
        }
      }
    }
  }
}

void q_convolution(const INT_T* const mat, const INT_T* const filter,
                   INT_T* const treesumBuffer, INT_T N, INT_T H, INT_T W,
                   INT_T CIN, INT_T HF,INT_T WF, INT_T CINF, INT_T COUTF,
                   INT_T HOUT, INT_T WOUT, INT_T HPADL, INT_T HPADR, INT_T WPADL,
                   INT_T WPADR, INT_T HSTR, INT_T WSTR, INT_T HDL, INT_T WDL,
                   INT_T G, INT_T* const ret, INT_T shrA, INT_T shrB, INT_T H1,
                   INT_T H2) {
  //Check
  /*ITER_T hoffsetl = HDL * (HF >> 1) - HPADL;
  ITER_T woffsetl = WDL * (WF >> 1) - WPADL;
  ITER_T hoffsetr = HDL * (HF >> 1) - HPADR;
  ITER_T woffsetr = WDL * (WF >> 1) - WPADR;

  for (ITER_T n = 0; n < N; n++) {
    for (ITER_T h = HOffsetL, hout = 0; h < H - HOffsetR; h += HSTR, hout++) {
      for (ITER_T w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
        for (ITER_T g = 0; g < G; g++) {
          for (ITER_T co = 0; co < COUTF; co ++) {

            ITER_T counter = 0;
            for (ITER_T hf = -(HF >> 1); hf <= HF >> 1; hf++) {
              for (ITER_T wf = -(WF >> 1); wf <= WF >> 1; wf++) {
                for (ITER_T ci = 0; ci < CINF; ci++) {
                  MYINT a = (((h + HDL * hf) < 0) || ((h + HDL * hf) >= H) || ((w + WDL * wf) < 0) || ((w + WDL * wf) >= W)) ? 0 : mat[n * H * W * CIN + (h + HDL * hf) * W * CIN + (w + WDL * wf) * CIN + (ci + g * CINF)];
                  MYINT b = filter[(hf + (HF >> 1)) * WF * CINF * COUTF + (wf + (WF >> 1)) * CINF * COUTF + ci * COUTF + co];
                  treesumBuffer[counter] = (((INTM_T) a) * ((INTM_T)b)) / (((INTM_T)shrA) * ((INTM_T)shrB));
                  counter++;
                }
              }
            }

            v_q_treesum(&treeumBuffer[0], HF * WF * CINF, H1, H2);
            ret[n * HOUT * WOUT * (COUTF * G) + hout * WOUT * (COUTF * G) + wout * (COUTF * G) + (co + g * COUTF)] = treesumBuffer[0];
          }
        }
      }
    }
  }*/
}
