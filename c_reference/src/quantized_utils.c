// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "quantized_utils.h"
#ifdef CMSISDSP
  #include <stdlib.h>
  #include "arm_math.h"
#endif

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

void v_q_add(const INT_T* vec1, const INT_T* vec2, ITER_T len,
             INT_T* ret, SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret) {
  #ifdef CMSISDSP
    INT_T *ret2 = malloc(len * sizeof(INT_T));
    arm_shift_q15(vec1, -(scvec1 + scret), ret, len);
    arm_shift_q15(vec2, -(scvec2 + scret), ret2, len);
    arm_add_q15(ret, ret2, ret, len);
    free(ret2);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((vec1[i] >> (scvec1 + scret)) + (vec2[i] >> (scvec2 + scret)));
      #else
        ret[i] = ((vec1[i] / scvec1) / scret) + ((vec2[i] / scvec2) / scret);
      #endif
    }
  #endif
}

void v_q_sub(const INT_T* vec1, const INT_T* vec2, ITER_T len,
             INT_T* ret, SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret) {
  #ifdef CMSISDSP
    INT_T *ret2 = malloc(len * sizeof(INT_T));
    arm_shift_q15(vec1, -(scvec1 + scret), ret, len);
    arm_shift_q15(vec2, -(scvec2 + scret), ret2, len);
    arm_sub_q15(ret, ret2, ret, len);
    free(ret2);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((vec1[i] >> (scvec1 + scret)) - (vec2[i] >> (scvec2 + scret)));
      #else
        ret[i] = ((vec1[i] / scvec1) / scret) - ((vec2[i] / scvec2) / scret);
      #endif
    }
  #endif
}

void v_q_hadamard(const INT_T* vec1, const INT_T* vec2, ITER_T len,
                  INT_T* ret, SCALE_T scvec1, SCALE_T scvec2) {
  #ifdef CMSISDSP
    // Need to test the variation with scaling first as well.
    arm_mult_q15(vec1, vec2, ret, len);
    arm_shift_q15(ret, -(scvec1 + scvec2), ret, len);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((INTM_T)vec1[i] * (INTM_T)vec2[i]) >> (scvec1 + scvec2);
      #else
        ret[i] = ((((INTM_T)vec1[i] * (INTM_T)vec2[i]) / scvec1) / scvec2);
      #endif
    }
  #endif
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

void v_q_scalar_add(INT_T scalar, const INT_T* vec, ITER_T len,
                    INT_T* ret, SCALE_T scscalar, SCALE_T scvec, SCALE_T scret) {
  #ifdef CMSISDSP
    arm_shift_q15(vec, -(scvec + scret), ret, len);
    arm_offset_q15(ret, (scalar >> (scscalar + scret)), ret, len);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((scalar >> (scscalar + scret)) + (vec[i] >> (scvec + scret)));
      #else
        ret[i] = ((scalar / scscalar) / scret) + ((vec[i] / scvec) / scret);
      #endif
    }
  #endif
}

void v_q_scalar_sub(INT_T scalar, const INT_T* vec, ITER_T len,
                    INT_T* ret, SCALE_T scscalar, SCALE_T scvec, SCALE_T scret) {
  #ifdef CMSISDSP
    arm_shift_q15(vec, -(scvec + scret), ret, len);
    arm_negate_q15(ret, ret, len);
    arm_offset_q15(ret, (scalar >> (scscalar + scret)), ret, len);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((scalar >> (scscalar + scret)) - (vec[i] >> (scvec + scret)));
      #else
        ret[i] = ((scalar / scscalar) / scret) - ((vec[i] / scvec) / scret);
      #endif
    }
  #endif
}

void v_q_sub_scalar(const INT_T* vec, INT_T scalar, ITER_T len,
                    INT_T* ret, SCALE_T scvec, SCALE_T scscalar, SCALE_T scret) {
  #ifdef CMSISDSP
    arm_shift_q15(vec, -(scvec + scret), ret, len);
    arm_offset_q15(ret, -(scalar >> (scscalar + scret)), ret, len);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((vec[i] >> (scvec + scret)) - (scalar >> (scscalar + scret)));
      #else
        ret[i] = ((vec[i] / scvec) / scret) - ((scalar / scscalar) / scret);
      #endif
    }
  #endif
}

void v_q_scalar_mul(INT_T scalar, const INT_T* vec, ITER_T len,
                    INT_T* ret, SCALE_T scscalar, SCALE_T scvec) {
  #ifdef CMSISDSP
    arm_scale_q15(vec, scalar, -(scscalar + scvec), ret, len);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((INTM_T)scalar * (INTM_T)vec[i]) >> (scscalar + scvec);
      #else
        ret[i] = ((((INTM_T)scalar * (INTM_T)vec[i]) / scscalar) / scvec);
      #endif
    }
  #endif
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
  ITER_T len = nrows * ncols, counter = 0;
  for (ITER_T i = 0; i < len; i++) {
    if (counter >= len) {
      counter -= len - 1;
    }

    ret[i] = mat[counter];
    counter += nrows;
  }
}

void m_q_reverse(const INT_T* const mat, ITER_T nrows, ITER_T ncols, ITER_T axis,
                 INT_T* const ret) {
  ITER_T len = nrows * ncols;

  if (axis == 0) {
    ITER_T col_counter = 0, row_index = len - ncols;

    for (ITER_T i = 0; i < len; i++) {
      if (col_counter >= ncols) {
        col_counter = 0;
        row_index -= ncols;
      }

      ret[i] = mat[row_index + col_counter];
      col_counter++;
    }
  } else {
    S_ITER_T row_counter = ncols - 1;
    ITER_T col_index = 0;

    for (ITER_T i = 0; i < len; i++) {
      if (row_counter < 0) {
        row_counter = ncols - 1;
        col_index += ncols;
      }

      ret[i] = mat[col_index + (ITER_T)row_counter];
      row_counter--;
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

void m_q_mulvec(INT_T* mat, INT_T* vec, ITER_T nrows,
                ITER_T ncols, INT_T* ret, SCALE_T scmat, SCALE_T scvec,
                SCALE_T H1, SCALE_T H2) {
  #ifdef CMSISDSP
    INT_T tmp[ncols];
    arm_matrix_instance_q15 A, B, C;
    arm_mat_init_q15(&A, nrows, ncols, mat);
    arm_mat_init_q15(&B, ncols, 1, vec);
    arm_mat_init_q15(&C, nrows, 1, ret);
    arm_mat_mult_fast_q15(&A, &B, &C, &tmp);
  #else
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
  #endif
}

void m_q_sparse_mulvec(const ITER_T* const col_indices, const INT_T* const mat_values,
                       const INT_T* const vec, ITER_T ndims, INT_T* const ret,
                       SCALE_T scmat, SCALE_T scvec, SCALE_T scret) {
  ITER_T iter_index = 0, iter_value = 0;
  for (ITER_T k = 0; k < ndims; k++) {
    ITER_T index = col_indices[iter_index];

    while (index != 0) {
      #ifdef SHIFT
        ret[index - 1] += (((INTM_T)mat_values[iter_value] * (INTM_T)vec[k]) >> (scmat + scvec + scret));
      #else
        ret[index - 1] += (((INTM_T)mat_values[iter_value] * (INTM_T)vec[k]) / ((INTM_T)scmat * (INTM_T)scvec * (INTM_T)scret));
      #endif
      iter_index++;
      iter_value++;
      index = col_indices[iter_index];
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

void q_maxpool(const INT_T* const input, INT_T* const output, ITER_T N,
               ITER_T H, ITER_T W, ITER_T CIn, ITER_T HF, ITER_T WF, ITER_T CF,
               ITER_T COut, ITER_T HOut, ITER_T WOut, ITER_T G, S_ITER_T HPadU,
               S_ITER_T HPadD, S_ITER_T WPadL, S_ITER_T WPadR, ITER_T HStride,
               ITER_T WStride, ITER_T HDilation, ITER_T WDilation,
               SCALE_T scinput, SCALE_T scoutput) {
  S_ITER_T HOffsetL = ((S_ITER_T)HDilation * (S_ITER_T)((HF - 1) >> 1)) - HPadU;
  S_ITER_T WOffsetL = ((S_ITER_T)WDilation * (S_ITER_T)((WF - 1) >> 1)) - WPadL;
  S_ITER_T HOffsetR = ((S_ITER_T)HDilation * (S_ITER_T)(HF >> 1)) - HPadD;
  S_ITER_T WOffsetR = ((S_ITER_T)WDilation * (S_ITER_T)(WF >> 1)) - WPadR;

  ITER_T HOffsetIn = W * CIn;
  ITER_T NOffsetIn = H * HOffsetIn;
  ITER_T WOffsetOut = (COut * G);
  ITER_T HOffsetOut = WOut * WOffsetOut;
  ITER_T NOffsetOut = HOut * HOffsetOut;
  for (ITER_T n = 0; n < N; n++) {
    ITER_T hout = 0;
    ITER_T NIndexIn = n * NOffsetIn;
    ITER_T NIndexOut = n * NOffsetOut;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; h += (S_ITER_T)HStride, hout++) {
      ITER_T wout = 0;
      ITER_T HIndexOut = hout * HOffsetOut;
      for (S_ITER_T w = WOffsetL; w < (S_ITER_T)W - WOffsetR; w += (S_ITER_T)WStride, wout++) {
        ITER_T WIndexOut = wout * WOffsetOut;
        for (ITER_T g = 0; g < G; g++) {
          ITER_T CIndexIn = g * CF;
          ITER_T CIndexOut = g * COut;
          for (ITER_T c = 0; c < COut; c++) {

            INT_T max = INT_TMIN;
            for (S_ITER_T hf = -((HF - 1) >> 1); hf <= (HF >> 1); hf++) {
              S_ITER_T hoffset = h + ((S_ITER_T)HDilation * hf);
              ITER_T HIndexIn = ((ITER_T)hoffset) * HOffsetIn;
              for (S_ITER_T wf = -((WF - 1) >> 1); wf <= (WF >> 1); wf++) {
                S_ITER_T woffset = w + ((S_ITER_T)WDilation * wf);
                ITER_T WIndexIn = ((ITER_T)woffset) * CIn;
                for (ITER_T cf = 0; cf < CF; cf++) {
                  if ((hoffset < 0) || (hoffset >= (S_ITER_T)H) || (woffset < 0) || (woffset >= (S_ITER_T)W)) {
                    if (max < 0) {
                      max = 0;
                    }
                  } else {
                    INT_T a = input[NIndexIn + HIndexIn + WIndexIn + (cf + CIndexIn)];
                    if (max < a) {
                      max = a;
                    }
                  }
                }
              }
            }

            #ifdef SHIFT
              output[NIndexOut + HIndexOut + WIndexOut + (c + CIndexOut)] = (max >> (scinput + scoutput));
            #else
              output[NIndexOut + HIndexOut + WIndexOut + (c + CIndexOut)] = ((max / scinput) / scoutput);
            #endif
          }
        }
      }
    }
  }
}

void q_convolution(const INT_T* const input, const INT_T* const filter,
                   INT_T* const output, INTM_T* const treesumBuffer, ITER_T N,
                   ITER_T H, ITER_T W, ITER_T CIn, ITER_T HF, ITER_T WF,
                   ITER_T CF, ITER_T COut, ITER_T HOut, ITER_T WOut, ITER_T G,
                   S_ITER_T HPadU, S_ITER_T HPadD, S_ITER_T WPadL,
                   S_ITER_T WPadR, ITER_T HStride, ITER_T WStride,
                   ITER_T HDilation, ITER_T WDilation, SCALE_T H1, SCALE_T H2,
                   SCALE_T scinput, SCALE_T scoutput) {
  S_ITER_T HOffsetL = ((S_ITER_T)HDilation * (S_ITER_T)((HF - 1) >> 1)) - HPadU;
  S_ITER_T WOffsetL = ((S_ITER_T)WDilation * (S_ITER_T)((WF - 1) >> 1)) - WPadL;
  S_ITER_T HOffsetR = ((S_ITER_T)HDilation * (S_ITER_T)(HF >> 1)) - HPadD;
  S_ITER_T WOffsetR = ((S_ITER_T)WDilation * (S_ITER_T)(WF >> 1)) - WPadR;

  ITER_T HOffsetIn = W * CIn;
  ITER_T NOffsetIn = H * HOffsetIn;
  ITER_T WOffsetF = CF * COut;
  ITER_T HOffsetF = WF * WOffsetF;
  ITER_T WOffsetOut = (COut * G);
  ITER_T HOffsetOut = WOut * WOffsetOut;
  ITER_T NOffsetOut = HOut * HOffsetOut;
  for (ITER_T n = 0; n < N; n++) {
    ITER_T hout = 0;
    ITER_T NIndexIn = n * NOffsetIn;
    ITER_T NIndexOut = n * NOffsetOut;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; h += (S_ITER_T)HStride, hout++) {
      ITER_T wout = 0;
      ITER_T HIndexOut = hout * HOffsetOut;
      for (S_ITER_T w = WOffsetL; w < (S_ITER_T)W - WOffsetR; w += (S_ITER_T)WStride, wout++) {
        ITER_T WIndexOut = wout * WOffsetOut;
        for (ITER_T g = 0; g < G; g++) {
          ITER_T CIndexIn = g * CF;
          ITER_T CIndexOut = g * COut;
          for (ITER_T c = 0; c < COut; c++) {

            ITER_T counter = 0;
            for (S_ITER_T hf = -((HF - 1) >> 1); hf <= (HF >> 1); hf++) {
              S_ITER_T hoffset = h + ((S_ITER_T)HDilation * hf);
              ITER_T HIndexIn = ((ITER_T)hoffset) * HOffsetIn;
              ITER_T HIndexF = ((ITER_T)(hf + ((HF - 1) >> 1))) * HOffsetF;
              for (S_ITER_T wf = -((WF - 1) >> 1); wf <= (WF >> 1); wf++) {
                S_ITER_T woffset = w + ((S_ITER_T)WDilation * wf);
                ITER_T WIndexIn = ((ITER_T)woffset) * CIn;
                ITER_T WIndexF = ((ITER_T)(wf + ((WF - 1) >> 1))) * WOffsetF;
                for (ITER_T cf = 0; cf < CF; cf++) {
                  if ((hoffset < 0) || (hoffset >= (S_ITER_T)H) || (woffset < 0) || (woffset >= (S_ITER_T)W)) {
                    treesumBuffer[counter] = 0;
                  } else {
                    treesumBuffer[counter] = ((INTM_T)input[NIndexIn + HIndexIn + WIndexIn + (cf + CIndexIn)]) *
                      ((INTM_T)filter[HIndexF + WIndexF + (c + cf * COut)]);
                  }
                  counter++;
                }
              }
            }

            v_q_treesum(&treesumBuffer[0], HF * WF * CF, H1, H2);
            #ifdef SHIFT
              output[NIndexOut + HIndexOut + WIndexOut + (c + CIndexOut)] = (treesumBuffer[0] >> (scinput + scoutput));
            #else
              output[NIndexOut + HIndexOut + WIndexOut + (c + CIndexOut)] = ((treesumBuffer[0] / scinput) / scoutput);
            #endif
          }
        }
      }
    }
  }
}
