// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "quantized_mbconv.h"

void q7_mbconv_block(const Q7_T* const input, const Q7_T* const filter1,
  const Q7_T* const BN1W, const Q7_T* const BN1B, const Q7_T* const filter2,
  const Q7_T* const BN2W, const Q7_T* const BN2B, const Q7_T* const filter3,
  const Q7_T* const BN3W, const Q7_T* const BN3B, Q7_T* const output,
  Q7_T* const convBuffer1, Q7_T* const convBuffer2, ITER_T N, ITER_T H,
  ITER_T W, ITER_T CIn, ITER_T CTemp, ITER_T HF, ITER_T WF, ITER_T COut,
  ITER_T HOut, ITER_T WOut, S_ITER_T HPadU, S_ITER_T HPadD, S_ITER_T WPadL,
  S_ITER_T WPadR, ITER_T HStride, ITER_T WStride, Q15_T limit1, Q15_T limit2,
  SCALE_T shrU1, SCALE_T shrX1, SCALE_T shrU2, SCALE_T shrX2, SCALE_T shrU3,
  SCALE_T shrW3, SCALE_T shlU1, SCALE_T shlX1, SCALE_T shlU2, SCALE_T shlX2,
  SCALE_T shlU3, SCALE_T shlW3) {
  S_ITER_T HOffsetFL = (HF - 1) >> 1;
  S_ITER_T WOffsetFL = (WF - 1) >> 1;
  S_ITER_T HOffsetFR = HF >> 1;
  S_ITER_T WOffsetFR = WF >> 1;

  S_ITER_T HOffsetL = HOffsetFL - HPadU;
  S_ITER_T WOffsetL = WOffsetFL - WPadL;
  S_ITER_T HOffsetR = HOffsetFR - HPadD;
  S_ITER_T WOffsetR = WOffsetFR - WPadR;

  ITER_T HOffsetIn = W * CIn;
  ITER_T NOffsetIn = H * HOffsetIn;
  ITER_T HOffsetC1 = W * CTemp;
  ITER_T GOffsetF = HF * WF;
  ITER_T HOffsetOut = WOut * COut;
  ITER_T NOffsetOut = HOut * HOffsetOut;

  Q31_T sum;
  for (ITER_T n = 0; n < N; n++) {
    ITER_T NIndexIn = n * NOffsetIn;
    ITER_T NIndexOut = n * NOffsetOut;
    ITER_T margin = 0;
    if ((S_ITER_T)HF - HPadU - (S_ITER_T)HStride > 0) {
      margin = (ITER_T)((S_ITER_T)HF - HPadU - (S_ITER_T)HStride);
    }

    for (ITER_T i = 0; i < margin; i++) {
      ITER_T HIndexIn = i * HOffsetIn + NIndexIn;
      ITER_T HIndexC1 = i * HOffsetC1;
      for (ITER_T j = 0; j < W; j++) {
        ITER_T WIndexIn = j * CIn + HIndexIn;
        Q7_T* convBuffer1_offset = ((Q7_T*)convBuffer1) + j * CTemp + HIndexC1;
        for (ITER_T k = 0; k < CTemp; k++) {
          sum = 0;
          Q7_T* input_offset = (Q7_T*)input + WIndexIn;
          Q7_T* filter1_offset = (Q7_T*)filter1 + k;
          ITER_T in_channels = CIn;

          #ifdef LOOP_UNROLL
            ITER_T len_unroll = in_channels >> 2;
            in_channels = in_channels % 4;
            while (len_unroll--) {
              sum += ((Q15_T)*input_offset++) * ((Q15_T)*filter1_offset);
              filter1_offset += CTemp;
              sum += ((Q15_T)*input_offset++) * ((Q15_T)*filter1_offset);
              filter1_offset += CTemp;
              sum += ((Q15_T)*input_offset++) * ((Q15_T)*filter1_offset);
              filter1_offset += CTemp;
              sum += ((Q15_T)*input_offset++) * ((Q15_T)*filter1_offset);
              filter1_offset += CTemp;
            }
          #endif

          while (in_channels--) {
            sum += ((Q15_T)*input_offset++) * ((Q15_T)*filter1_offset);
            filter1_offset += CTemp;
          }

          #ifdef SHIFT
            Q15_T x = (((Q15_T)(((sum << shlU1) >> shrU1) + BN1B[k])) *
                       ((Q15_T)BN1W[k]));
          #else
            Q15_T x = (((Q15_T)((sum * shlU1) / shrU1 + BN1B[k])) *
                       ((Q15_T)BN1W[k]));
          #endif
          x = q15_relu(x, limit1);
          #ifdef SHIFT
            *convBuffer1_offset++ = ((x << shlX1) >> shrX1);
          #else
            *convBuffer1_offset++ = (x * shlX1) / shrX1;
          #endif
        }
      }
    }

    ITER_T hout = 0;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; hout++, h += (S_ITER_T)HStride) {
      ITER_T HIndexOut = hout * HOffsetOut + NIndexOut;
      for (ITER_T i = 0; i < HStride; i++) {
        ITER_T iRed = (i + margin + hout * HStride) % HF;
        ITER_T iFull = i + margin + hout * HStride;
        ITER_T HIndexC1 = iRed * HOffsetC1;
        ITER_T HIndexIn = iFull * HOffsetIn + NIndexIn;
        if (iFull < H){
          for (ITER_T j = 0; j < W; j++) {
            ITER_T WIndexIn = j * CIn + HIndexIn;
            Q7_T* convBuffer1_offset = ((Q7_T*)convBuffer1) + j * CTemp + HIndexC1;
            for (ITER_T k = 0; k < CTemp; k++) {
              sum = 0;
              Q7_T* input_offset = (Q7_T*)input + WIndexIn;
              Q7_T* filter1_offset = (Q7_T*)filter1 + k;
              ITER_T in_channels = CIn;

              #ifdef LOOP_UNROLL
                ITER_T len_unroll = in_channels >> 2;
                in_channels = in_channels % 4;
                while (len_unroll--) {
                  sum += ((Q15_T)*input_offset++) * ((Q15_T)*filter1_offset);
                  filter1_offset += CTemp;
                  sum += ((Q15_T)*input_offset++) * ((Q15_T)*filter1_offset);
                  filter1_offset += CTemp;
                  sum += ((Q15_T)*input_offset++) * ((Q15_T)*filter1_offset);
                  filter1_offset += CTemp;
                  sum += ((Q15_T)*input_offset++) * ((Q15_T)*filter1_offset);
                  filter1_offset += CTemp;
                }
              #endif

              while (in_channels--) {
                sum += ((Q15_T)*input_offset++) * ((Q15_T)*filter1_offset);
                filter1_offset += CTemp;
              }

              #ifdef SHIFT
                Q15_T x = (((Q15_T)(((sum << shlU1) >> shrU1) + BN1B[k])) *
                           ((Q15_T)BN1W[k]));
              #else
                Q15_T x = (((Q15_T)((sum * shlU1) / shrU1 + BN1B[k])) *
                           ((Q15_T)BN1W[k]));
              #endif
              x = q15_relu(x, limit1);
              #ifdef SHIFT
                *convBuffer1_offset++ = ((x << shlX1) >> shrX1);
              #else
                *convBuffer1_offset++ = (x * shlX1) / shrX1;
              #endif
            }
          }
        } else {
          Q7_T* convBuffer1_offset = (Q7_T*)convBuffer1 + HIndexC1;
          for (ITER_T j = 0; j < W; j++) {
            Q7_T* BN1B_offset = (Q7_T*)BN1B;
            Q7_T* BN1W_offset = (Q7_T*)BN1W;
            ITER_T temp_channels = CTemp;

            #ifdef LOOP_UNROLL
              ITER_T len_unroll = temp_channels >> 2;
              temp_channels = temp_channels % 4;
              while (len_unroll--) {
                Q15_T w = q15_relu(((Q15_T)*BN1B_offset++) * ((Q15_T)*BN1W_offset++), limit1);
                Q15_T x = q15_relu(((Q15_T)*BN1B_offset++) * ((Q15_T)*BN1W_offset++), limit1);
                Q15_T y = q15_relu(((Q15_T)*BN1B_offset++) * ((Q15_T)*BN1W_offset++), limit1);
                Q15_T z = q15_relu(((Q15_T)*BN1B_offset++) * ((Q15_T)*BN1W_offset++), limit1);
                #ifdef SHIFT
                  *convBuffer1_offset++ = ((w << shlX1) >> shrX1);
                  *convBuffer1_offset++ = ((x << shlX1) >> shrX1);
                  *convBuffer1_offset++ = ((y << shlX1) >> shrX1);
                  *convBuffer1_offset++ = ((z << shlX1) >> shrX1);
                #else
                  *convBuffer1_offset++ = ((w * shlX1) / shrX1);
                  *convBuffer1_offset++ = ((x * shlX1) / shrX1);
                  *convBuffer1_offset++ = ((y * shlX1) / shrX1);
                  *convBuffer1_offset++ = ((z * shlX1) / shrX1);
                #endif
              }
            #endif

            while (temp_channels--) {
              Q15_T w = q15_relu(((Q15_T)*BN1B_offset++) * ((Q15_T)*BN1W_offset++), limit1);
              #ifdef SHIFT
                *convBuffer1_offset++ = ((w << shlX1) >> shrX1);
              #else
                *convBuffer1_offset++ = ((w * shlX1) / shrX1);
              #endif
            }
          }
        }
      }

      ITER_T wout = 0;
      for (S_ITER_T w = WOffsetL; w < ((S_ITER_T)W) - WOffsetR; wout++, w += ((S_ITER_T)WStride)) {
        Q7_T* output_offset = ((Q7_T*)output) + wout * COut + HIndexOut;
        for (ITER_T g = 0; g < CTemp; g++) {
          sum = 0;
          ITER_T GIndexF = g * GOffsetF;
          for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
            S_ITER_T hindex = h + hf;
            if ((hindex < 0) || (hindex >= (S_ITER_T)H)){
              continue;
            }
            ITER_T HIndexC1 = (((ITER_T)hindex) % HF) * HOffsetC1 + g;
            ITER_T HIndexF = ((ITER_T)(hf + HOffsetFL)) * WF + GIndexF;
            for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
              S_ITER_T windex = w + wf;
              if ((windex < 0) || (windex >= (S_ITER_T)W)) {
                continue;
              } else {
                sum += ((Q15_T)convBuffer1[HIndexC1 + ((ITER_T)windex) * CTemp]) *
                       ((Q15_T)filter2[HIndexF + ((ITER_T)(wf + WOffsetFL))]);
              }
            }
          }

          #ifdef SHIFT
            Q15_T x = (((Q15_T)(((sum << shlU2) >> shrU2) + BN2B[g])) *
                       ((Q15_T)BN2W[g]));
          #else
            Q15_T x = (((Q15_T)((sum * shlU2) / shrU2 + BN2B[g])) *
                       ((Q15_T)BN2W[g]));
          #endif
          x = q15_relu(x, limit2);
          #ifdef SHIFT
            convBuffer2[g] = ((x << shlX2) >> shrX2);
          #else
            convBuffer2[g] = (x * shlX2) / shrX2;
          #endif
        }

        for (ITER_T i = 0; i < COut; i++) {
          sum = 0;
          Q7_T* convBuffer2_offset = (Q7_T*)convBuffer2;
          Q7_T* filter3_offset = (Q7_T*)filter3 + i;
          ITER_T temp_channels = CTemp;

          #ifdef LOOP_UNROLL
            ITER_T len_unroll = temp_channels >> 2;
            temp_channels = temp_channels % 4;
            while (len_unroll--) {
              sum += ((Q15_T)*convBuffer2_offset++) * ((Q15_T)*filter3_offset);
              filter3_offset += COut;
              sum += ((Q15_T)*convBuffer2_offset++) * ((Q15_T)*filter3_offset);
              filter3_offset += COut;
              sum += ((Q15_T)*convBuffer2_offset++) * ((Q15_T)*filter3_offset);
              filter3_offset += COut;
              sum += ((Q15_T)*convBuffer2_offset++) * ((Q15_T)*filter3_offset);
              filter3_offset += COut;
            }
          #endif

          while (temp_channels--) {
            sum += ((Q15_T)*convBuffer2_offset++) * ((Q15_T)*filter3_offset);
            filter3_offset += COut;
          }

          #ifdef SHIFT
            *output_offset++ = (((((Q15_T)(((sum << shlU3) >> shrU3) + BN3B[i])) *
                                  ((Q15_T) BN3W[i])) << shlW3) >> shrW3);
          #else
            *output_offset++ = ((((Q15_T)((sum * shlU3) / shrU3 + BN3B[i])) *
                                 ((Q15_T) BN3W[i])) * shlW3) / shrW3;
          #endif
        }
      }
    }
  }
}

void q7xq15_q15_mbconv_block(const Q7_T* const input,
  const Q15_T* const filter1, const Q15_T* const BN1W, const Q15_T* const BN1B,
  const Q15_T* const filter2, const Q15_T* const BN2W, const Q15_T* const BN2B,
  const Q15_T* const filter3, const Q15_T* const BN3W, const Q15_T* const BN3B,
  Q15_T* const output, Q15_T* const convBuffer1, Q15_T* const convBuffer2,
  ITER_T N, ITER_T H, ITER_T W, ITER_T CIn, ITER_T CTemp, ITER_T HF, ITER_T WF,
  ITER_T COut, ITER_T HOut, ITER_T WOut, S_ITER_T HPadU, S_ITER_T HPadD,
  S_ITER_T WPadL, S_ITER_T WPadR, ITER_T HStride, ITER_T WStride, Q31_T limit1,
  Q31_T limit2, SCALE_T shrU1, SCALE_T shrX1, SCALE_T shrU2, SCALE_T shrX2,
  SCALE_T shrU3, SCALE_T shrW3, SCALE_T shlU1, SCALE_T shlX1, SCALE_T shlU2,
  SCALE_T shlX2, SCALE_T shlU3, SCALE_T shlW3) {
  S_ITER_T HOffsetFL = (HF - 1) >> 1;
  S_ITER_T WOffsetFL = (WF - 1) >> 1;
  S_ITER_T HOffsetFR = HF >> 1;
  S_ITER_T WOffsetFR = WF >> 1;

  S_ITER_T HOffsetL = HOffsetFL - HPadU;
  S_ITER_T WOffsetL = WOffsetFL - WPadL;
  S_ITER_T HOffsetR = HOffsetFR - HPadD;
  S_ITER_T WOffsetR = WOffsetFR - WPadR;

  ITER_T HOffsetIn = W * CIn;
  ITER_T NOffsetIn = H * HOffsetIn;
  ITER_T HOffsetC1 = W * CTemp;
  ITER_T GOffsetF = HF * WF;
  ITER_T HOffsetOut = WOut * COut;
  ITER_T NOffsetOut = HOut * HOffsetOut;

  Q31_T sum;
  for (ITER_T n = 0; n < N; n++) {
    ITER_T NIndexIn = n * NOffsetIn;
    ITER_T NIndexOut = n * NOffsetOut;
    ITER_T margin = 0;
    if ((S_ITER_T)HF - HPadU - (S_ITER_T)HStride > 0) {
      margin = (ITER_T)((S_ITER_T)HF - HPadU - (S_ITER_T)HStride);
    }

    for (ITER_T i = 0; i < margin; i++) {
      ITER_T HIndexIn = i * HOffsetIn + NIndexIn;
      ITER_T HIndexC1 = i * HOffsetC1;
      for (ITER_T j = 0; j < W; j++) {
        ITER_T WIndexIn = j * CIn + HIndexIn;
        Q15_T* convBuffer1_offset = ((Q15_T*)convBuffer1) + j * CTemp + HIndexC1;
        for (ITER_T k = 0; k < CTemp; k++) {
          sum = 0;
          Q7_T* input_offset = (Q7_T*)input + WIndexIn;
          Q15_T* filter1_offset = (Q15_T*)filter1 + k;
          ITER_T in_channels = CIn;

          #ifdef LOOP_UNROLL
            ITER_T len_unroll = in_channels >> 2;
            in_channels = in_channels % 4;
            while (len_unroll--) {
              sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
              filter1_offset += CTemp;
              sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
              filter1_offset += CTemp;
              sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
              filter1_offset += CTemp;
              sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
              filter1_offset += CTemp;
            }
          #endif

          while (in_channels--) {
            sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
            filter1_offset += CTemp;
          }

          #ifdef SHIFT
            Q31_T x = (((Q31_T)(((sum << shlU1) >> shrU1) + BN1B[k])) *
                       ((Q31_T)BN1W[k]));
          #else
            Q31_T x = (((Q31_T)((sum * shlU1) / shrU1 + BN1B[k])) *
                       ((Q31_T)BN1W[k]));
          #endif
          x = q31_relu(x, limit1);
          #ifdef SHIFT
            *convBuffer1_offset++ = ((x << shlX1) >> shrX1);
          #else
            *convBuffer1_offset++ = (x * shlX1) / shrX1;
          #endif
        }
      }
    }

    ITER_T hout = 0;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; hout++, h += (S_ITER_T)HStride) {
      ITER_T HIndexOut = hout * HOffsetOut + NIndexOut;
      for (ITER_T i = 0; i < HStride; i++) {
        ITER_T iRed = (i + margin + hout * HStride) % HF;
        ITER_T iFull = i + margin + hout * HStride;
        ITER_T HIndexC1 = iRed * HOffsetC1;
        ITER_T HIndexIn = iFull * HOffsetIn + NIndexIn;
        if (iFull < H){
          for (ITER_T j = 0; j < W; j++) {
            ITER_T WIndexIn = j * CIn + HIndexIn;
            Q15_T* convBuffer1_offset = ((Q15_T*)convBuffer1) + j * CTemp + HIndexC1;
            for (ITER_T k = 0; k < CTemp; k++) {
              sum = 0;
              Q7_T* input_offset = (Q7_T*)input + WIndexIn;
              Q15_T* filter1_offset = (Q15_T*)filter1 + k;
              ITER_T in_channels = CIn;

              #ifdef LOOP_UNROLL
                ITER_T len_unroll = in_channels >> 2;
                in_channels = in_channels % 4;
                while (len_unroll--) {
                  sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                  filter1_offset += CTemp;
                  sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                  filter1_offset += CTemp;
                  sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                  filter1_offset += CTemp;
                  sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                  filter1_offset += CTemp;
                }
              #endif

              while (in_channels--) {
                sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                filter1_offset += CTemp;
              }

              #ifdef SHIFT
                Q31_T x = (((Q31_T)(((sum << shlU1) >> shrU1) + BN1B[k])) *
                           ((Q31_T)BN1W[k]));
              #else
                Q31_T x = (((Q31_T)((sum * shlU1) / shrU1 + BN1B[k])) *
                           ((Q31_T)BN1W[k]));
              #endif
              x = q31_relu(x, limit1);
              #ifdef SHIFT
                *convBuffer1_offset++ = ((x << shlX1) >> shrX1);
              #else
                *convBuffer1_offset++ = (x * shlX1) / shrX1;
              #endif
            }
          }
        } else {
          Q15_T* convBuffer1_offset = (Q15_T*)convBuffer1 + HIndexC1;
          for (ITER_T j = 0; j < W; j++) {
            Q15_T* BN1B_offset = (Q15_T*)BN1B;
            Q15_T* BN1W_offset = (Q15_T*)BN1W;
            ITER_T temp_channels = CTemp;

            #ifdef LOOP_UNROLL
              ITER_T len_unroll = temp_channels >> 2;
              temp_channels = temp_channels % 4;
              while (len_unroll--) {
                Q31_T w = q31_relu(((Q31_T)*BN1B_offset++) * ((Q31_T)*BN1W_offset++), limit1);
                Q31_T x = q31_relu(((Q31_T)*BN1B_offset++) * ((Q31_T)*BN1W_offset++), limit1);
                Q31_T y = q31_relu(((Q31_T)*BN1B_offset++) * ((Q31_T)*BN1W_offset++), limit1);
                Q31_T z = q31_relu(((Q31_T)*BN1B_offset++) * ((Q31_T)*BN1W_offset++), limit1);
                #ifdef SHIFT
                  *convBuffer1_offset++ = ((w << shlX1) >> shrX1);
                  *convBuffer1_offset++ = ((x << shlX1) >> shrX1);
                  *convBuffer1_offset++ = ((y << shlX1) >> shrX1);
                  *convBuffer1_offset++ = ((z << shlX1) >> shrX1);
                #else
                  *convBuffer1_offset++ = ((w * shlX1) / shrX1);
                  *convBuffer1_offset++ = ((x * shlX1) / shrX1);
                  *convBuffer1_offset++ = ((y * shlX1) / shrX1);
                  *convBuffer1_offset++ = ((z * shlX1) / shrX1);
                #endif
              }
            #endif

            while (temp_channels--) {
              Q15_T w = q15_relu(((Q15_T)*BN1B_offset++) * ((Q15_T)*BN1W_offset++), limit1);
              #ifdef SHIFT
                *convBuffer1_offset++ = ((w << shlX1) >> shrX1);
              #else
                *convBuffer1_offset++ = ((w * shlX1) / shrX1);
              #endif
            }
          }
        }
      }

      ITER_T wout = 0;
      for (S_ITER_T w = WOffsetL; w < ((S_ITER_T)W) - WOffsetR; wout++, w += ((S_ITER_T)WStride)) {
        Q15_T* output_offset = ((Q15_T*)output) + wout * COut + HIndexOut;
        for (ITER_T g = 0; g < CTemp; g++) {
          sum = 0;
          ITER_T GIndexF = g * GOffsetF;
          for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
            S_ITER_T hindex = h + hf;
            if ((hindex < 0) || (hindex >= (S_ITER_T)H)){
              continue;
            }
            ITER_T HIndexC1 = (((ITER_T)hindex) % HF) * HOffsetC1 + g;
            ITER_T HIndexF = ((ITER_T)(hf + HOffsetFL)) * WF + GIndexF;
            for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
              S_ITER_T windex = w + wf;
              if ((windex < 0) || (windex >= (S_ITER_T)W)) {
                continue;
              } else {
                sum += ((Q31_T)convBuffer1[HIndexC1 + ((ITER_T)windex) * CTemp]) *
                       ((Q31_T)filter2[HIndexF + ((ITER_T)(wf + WOffsetFL))]);
              }
            }
          }

          #ifdef SHIFT
            Q31_T x = (((Q31_T)(((sum << shlU2) >> shrU2) + BN2B[g])) *
                       ((Q31_T)BN2W[g]));
          #else
            Q31_T x = (((Q31_T)((sum * shlU2) / shrU2 + BN2B[g])) *
                       ((Q31_T)BN2W[g]));
          #endif
          x = q31_relu(x, limit2);
          #ifdef SHIFT
            convBuffer2[g] = ((x << shlX2) >> shrX2);
          #else
            convBuffer2[g] = (x * shlX2) / shrX2;
          #endif
        }

        for (ITER_T i = 0; i < COut; i++) {
          sum = 0;
          Q15_T* convBuffer2_offset = (Q15_T*)convBuffer2;
          Q15_T* filter3_offset = (Q15_T*)filter3 + i;
          ITER_T temp_channels = CTemp;

          #ifdef LOOP_UNROLL
            ITER_T len_unroll = temp_channels >> 2;
            temp_channels = temp_channels % 4;
            while (len_unroll--) {
              sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
              filter3_offset += COut;
              sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
              filter3_offset += COut;
              sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
              filter3_offset += COut;
              sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
              filter3_offset += COut;
            }
          #endif

          while (temp_channels--) {
            sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
            filter3_offset += COut;
          }

          #ifdef SHIFT
            *output_offset++ = (((((Q31_T)(((sum << shlU3) >> shrU3) + BN3B[i])) *
                                  ((Q31_T) BN3W[i])) << shlW3) >> shrW3);
          #else
            *output_offset++ = ((((Q31_T)((sum * shlU3) / shrU3 + BN3B[i])) *
                                 ((Q31_T) BN3W[i])) * shlW3) / shrW3;
          #endif
        }
      }
    }
  }
}

void q15xq7_q7_mbconv_block(const Q15_T* const input,
  const Q7_T* const filter1, const Q7_T* const BN1W, const Q15_T* const BN1B,
  const Q7_T* const filter2, const Q7_T* const BN2W, const Q15_T* const BN2B,
  const Q7_T* const filter3, const Q7_T* const BN3W, const Q15_T* const BN3B,
  Q7_T* const output, Q15_T* const convBuffer1, Q15_T* const convBuffer2,
  ITER_T N, ITER_T H, ITER_T W, ITER_T CIn, ITER_T CTemp, ITER_T HF, ITER_T WF,
  ITER_T COut, ITER_T HOut, ITER_T WOut, S_ITER_T HPadU, S_ITER_T HPadD,
  S_ITER_T WPadL, S_ITER_T WPadR, ITER_T HStride, ITER_T WStride, Q31_T limit1,
  Q31_T limit2, SCALE_T shrU1, SCALE_T shrX1, SCALE_T shrU2, SCALE_T shrX2,
  SCALE_T shrU3, SCALE_T shrW3, SCALE_T shlU1, SCALE_T shlX1, SCALE_T shlU2,
  SCALE_T shlX2, SCALE_T shlU3, SCALE_T shlW3) {
  S_ITER_T HOffsetFL = (HF - 1) >> 1;
  S_ITER_T WOffsetFL = (WF - 1) >> 1;
  S_ITER_T HOffsetFR = HF >> 1;
  S_ITER_T WOffsetFR = WF >> 1;

  S_ITER_T HOffsetL = HOffsetFL - HPadU;
  S_ITER_T WOffsetL = WOffsetFL - WPadL;
  S_ITER_T HOffsetR = HOffsetFR - HPadD;
  S_ITER_T WOffsetR = WOffsetFR - WPadR;

  ITER_T HOffsetIn = W * CIn;
  ITER_T NOffsetIn = H * HOffsetIn;
  ITER_T HOffsetC1 = W * CTemp;
  ITER_T GOffsetF = HF * WF;
  ITER_T HOffsetOut = WOut * COut;
  ITER_T NOffsetOut = HOut * HOffsetOut;

  Q31_T sum;
  for (ITER_T n = 0; n < N; n++) {
    ITER_T NIndexIn = n * NOffsetIn;
    ITER_T NIndexOut = n * NOffsetOut;
    ITER_T margin = 0;
    if ((S_ITER_T)HF - HPadU - (S_ITER_T)HStride > 0) {
      margin = (ITER_T)((S_ITER_T)HF - HPadU - (S_ITER_T)HStride);
    }

    for (ITER_T i = 0; i < margin; i++) {
      ITER_T HIndexIn = i * HOffsetIn + NIndexIn;
      ITER_T HIndexC1 = i * HOffsetC1;
      for (ITER_T j = 0; j < W; j++) {
        ITER_T WIndexIn = j * CIn + HIndexIn;
        Q15_T* convBuffer1_offset = ((Q15_T*)convBuffer1) + j * CTemp + HIndexC1;
        for (ITER_T k = 0; k < CTemp; k++) {
          sum = 0;
          Q15_T* input_offset = (Q15_T*)input + WIndexIn;
          Q7_T* filter1_offset = (Q7_T*)filter1 + k;
          ITER_T in_channels = CIn;

          #ifdef LOOP_UNROLL
            ITER_T len_unroll = in_channels >> 2;
            in_channels = in_channels % 4;
            while (len_unroll--) {
              sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
              filter1_offset += CTemp;
              sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
              filter1_offset += CTemp;
              sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
              filter1_offset += CTemp;
              sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
              filter1_offset += CTemp;
            }
          #endif

          while (in_channels--) {
            sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
            filter1_offset += CTemp;
          }

          #ifdef SHIFT
            Q31_T x = (((Q31_T)(((sum << shlU1) >> shrU1) + BN1B[k])) *
                       ((Q31_T)BN1W[k]));
          #else
            Q31_T x = (((Q31_T)((sum * shlU1) / shrU1 + BN1B[k])) *
                       ((Q31_T)BN1W[k]));
          #endif
          x = q31_relu(x, limit1);
          #ifdef SHIFT
            *convBuffer1_offset++ = ((x << shlX1) >> shrX1);
          #else
            *convBuffer1_offset++ = (x * shlX1) / shrX1;
          #endif
        }
      }
    }

    ITER_T hout = 0;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; hout++, h += (S_ITER_T)HStride) {
      ITER_T HIndexOut = hout * HOffsetOut + NIndexOut;
      for (ITER_T i = 0; i < HStride; i++) {
        ITER_T iRed = (i + margin + hout * HStride) % HF;
        ITER_T iFull = i + margin + hout * HStride;
        ITER_T HIndexC1 = iRed * HOffsetC1;
        ITER_T HIndexIn = iFull * HOffsetIn + NIndexIn;
        if (iFull < H){
          for (ITER_T j = 0; j < W; j++) {
            ITER_T WIndexIn = j * CIn + HIndexIn;
            Q15_T* convBuffer1_offset = ((Q15_T*)convBuffer1) + j * CTemp + HIndexC1;
            for (ITER_T k = 0; k < CTemp; k++) {
              sum = 0;
              Q15_T* input_offset = (Q15_T*)input + WIndexIn;
              Q7_T* filter1_offset = (Q7_T*)filter1 + k;
              ITER_T in_channels = CIn;

              #ifdef LOOP_UNROLL
                ITER_T len_unroll = in_channels >> 2;
                in_channels = in_channels % 4;
                while (len_unroll--) {
                  sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                  filter1_offset += CTemp;
                  sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                  filter1_offset += CTemp;
                  sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                  filter1_offset += CTemp;
                  sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                  filter1_offset += CTemp;
                }
              #endif

              while (in_channels--) {
                sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                filter1_offset += CTemp;
              }

              #ifdef SHIFT
                Q31_T x = (((Q31_T)(((sum << shlU1) >> shrU1) + BN1B[k])) *
                           ((Q31_T)BN1W[k]));
              #else
                Q31_T x = (((Q31_T)((sum * shlU1) / shrU1 + BN1B[k])) *
                           ((Q31_T)BN1W[k]));
              #endif
              x = q31_relu(x, limit1);
              #ifdef SHIFT
                *convBuffer1_offset++ = ((x << shlX1) >> shrX1);
              #else
                *convBuffer1_offset++ = (x * shlX1) / shrX1;
              #endif
            }
          }
        } else {
          Q15_T* convBuffer1_offset = (Q15_T*)convBuffer1 + HIndexC1;
          for (ITER_T j = 0; j < W; j++) {
            Q15_T* BN1B_offset = (Q15_T*)BN1B;
            Q7_T* BN1W_offset = (Q7_T*)BN1W;
            ITER_T temp_channels = CTemp;

            #ifdef LOOP_UNROLL
              ITER_T len_unroll = temp_channels >> 2;
              temp_channels = temp_channels % 4;
              while (len_unroll--) {
                Q31_T w = q31_relu(((Q31_T)*BN1B_offset++) * ((Q31_T)*BN1W_offset++), limit1);
                Q31_T x = q31_relu(((Q31_T)*BN1B_offset++) * ((Q31_T)*BN1W_offset++), limit1);
                Q31_T y = q31_relu(((Q31_T)*BN1B_offset++) * ((Q31_T)*BN1W_offset++), limit1);
                Q31_T z = q31_relu(((Q31_T)*BN1B_offset++) * ((Q31_T)*BN1W_offset++), limit1);
                #ifdef SHIFT
                  *convBuffer1_offset++ = ((w << shlX1) >> shrX1);
                  *convBuffer1_offset++ = ((x << shlX1) >> shrX1);
                  *convBuffer1_offset++ = ((y << shlX1) >> shrX1);
                  *convBuffer1_offset++ = ((z << shlX1) >> shrX1);
                #else
                  *convBuffer1_offset++ = ((w * shlX1) / shrX1);
                  *convBuffer1_offset++ = ((x * shlX1) / shrX1);
                  *convBuffer1_offset++ = ((y * shlX1) / shrX1);
                  *convBuffer1_offset++ = ((z * shlX1) / shrX1);
                #endif
              }
            #endif

            while (temp_channels--) {
              Q31_T w = q31_relu(((Q31_T)*BN1B_offset++) * ((Q31_T)*BN1W_offset++), limit1);
              #ifdef SHIFT
                *convBuffer1_offset++ = ((w << shlX1) >> shrX1);
              #else
                *convBuffer1_offset++ = ((w * shlX1) / shrX1);
              #endif
            }
          }
        }
      }

      ITER_T wout = 0;
      for (S_ITER_T w = WOffsetL; w < ((S_ITER_T)W) - WOffsetR; wout++, w += ((S_ITER_T)WStride)) {
        Q7_T* output_offset = ((Q7_T*)output) + wout * COut + HIndexOut;
        for (ITER_T g = 0; g < CTemp; g++) {
          sum = 0;
          ITER_T GIndexF = g * GOffsetF;
          for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
            S_ITER_T hindex = h + hf;
            if ((hindex < 0) || (hindex >= (S_ITER_T)H)){
              continue;
            }
            ITER_T HIndexC1 = (((ITER_T)hindex) % HF) * HOffsetC1 + g;
            ITER_T HIndexF = ((ITER_T)(hf + HOffsetFL)) * WF + GIndexF;
            for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
              S_ITER_T windex = w + wf;
              if ((windex < 0) || (windex >= (S_ITER_T)W)) {
                continue;
              } else {
                sum += ((Q31_T)convBuffer1[HIndexC1 + ((ITER_T)windex) * CTemp]) *
                       ((Q31_T)filter2[HIndexF + ((ITER_T)(wf + WOffsetFL))]);
              }
            }
          }

          #ifdef SHIFT
            Q31_T x = (((Q31_T)(((sum << shlU2) >> shrU2) + BN2B[g])) *
                       ((Q31_T)BN2W[g]));
          #else
            Q31_T x = (((Q31_T)((sum * shlU2) / shrU2 + BN2B[g])) *
                       ((Q31_T)BN2W[g]));
          #endif
          x = q31_relu(x, limit2);
          #ifdef SHIFT
            convBuffer2[g] = ((x << shlX2) >> shrX2);
          #else
            convBuffer2[g] = (x * shlX2) / shrX2;
          #endif
        }

        for (ITER_T i = 0; i < COut; i++) {
          sum = 0;
          Q15_T* convBuffer2_offset = (Q15_T*)convBuffer2;
          Q7_T* filter3_offset = (Q7_T*)filter3 + i;
          ITER_T temp_channels = CTemp;

          #ifdef LOOP_UNROLL
            ITER_T len_unroll = temp_channels >> 2;
            temp_channels = temp_channels % 4;
            while (len_unroll--) {
              sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
              filter3_offset += COut;
              sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
              filter3_offset += COut;
              sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
              filter3_offset += COut;
              sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
              filter3_offset += COut;
            }
          #endif

          while (temp_channels--) {
            sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
            filter3_offset += COut;
          }

          #ifdef SHIFT
            *output_offset++ = (((((Q31_T)(((sum << shlU3) >> shrU3) + BN3B[i])) *
                                  ((Q31_T) BN3W[i])) << shlW3) >> shrW3);
          #else
            *output_offset++ = ((((Q31_T)((sum * shlU3) / shrU3 + BN3B[i])) *
                                 ((Q31_T) BN3W[i])) * shlW3) / shrW3;
          #endif
        }
      }
    }
  }
}

void q15xq7_q15_mbconv_block(const Q15_T* const input,
  const Q7_T* const filter1, const Q7_T* const BN1W, const Q15_T* const BN1B,
  const Q7_T* const filter2, const Q7_T* const BN2W, const Q15_T* const BN2B,
  const Q7_T* const filter3, const Q7_T* const BN3W, const Q15_T* const BN3B,
  Q15_T* const output, Q15_T* const convBuffer1, Q15_T* const convBuffer2,
  ITER_T N, ITER_T H, ITER_T W, ITER_T CIn, ITER_T CTemp, ITER_T HF, ITER_T WF,
  ITER_T COut, ITER_T HOut, ITER_T WOut, S_ITER_T HPadU, S_ITER_T HPadD,
  S_ITER_T WPadL, S_ITER_T WPadR, ITER_T HStride, ITER_T WStride, Q31_T limit1,
  Q31_T limit2, SCALE_T shrU1, SCALE_T shrX1, SCALE_T shrU2, SCALE_T shrX2,
  SCALE_T shrU3, SCALE_T shrW3, SCALE_T shlU1, SCALE_T shlX1, SCALE_T shlU2,
  SCALE_T shlX2, SCALE_T shlU3, SCALE_T shlW3) {
  S_ITER_T HOffsetFL = (HF - 1) >> 1;
  S_ITER_T WOffsetFL = (WF - 1) >> 1;
  S_ITER_T HOffsetFR = HF >> 1;
  S_ITER_T WOffsetFR = WF >> 1;

  S_ITER_T HOffsetL = HOffsetFL - HPadU;
  S_ITER_T WOffsetL = WOffsetFL - WPadL;
  S_ITER_T HOffsetR = HOffsetFR - HPadD;
  S_ITER_T WOffsetR = WOffsetFR - WPadR;

  ITER_T HOffsetIn = W * CIn;
  ITER_T NOffsetIn = H * HOffsetIn;
  ITER_T HOffsetC1 = W * CTemp;
  ITER_T GOffsetF = HF * WF;
  ITER_T HOffsetOut = WOut * COut;
  ITER_T NOffsetOut = HOut * HOffsetOut;

  Q31_T sum;
  for (ITER_T n = 0; n < N; n++) {
    ITER_T NIndexIn = n * NOffsetIn;
    ITER_T NIndexOut = n * NOffsetOut;
    ITER_T margin = 0;
    if ((S_ITER_T)HF - HPadU - (S_ITER_T)HStride > 0) {
      margin = (ITER_T)((S_ITER_T)HF - HPadU - (S_ITER_T)HStride);
    }

    for (ITER_T i = 0; i < margin; i++) {
      ITER_T HIndexIn = i * HOffsetIn + NIndexIn;
      ITER_T HIndexC1 = i * HOffsetC1;
      for (ITER_T j = 0; j < W; j++) {
        ITER_T WIndexIn = j * CIn + HIndexIn;
        Q15_T* convBuffer1_offset = ((Q15_T*)convBuffer1) + j * CTemp + HIndexC1;
        for (ITER_T k = 0; k < CTemp; k++) {
          sum = 0;
          Q15_T* input_offset = (Q15_T*)input + WIndexIn;
          Q7_T* filter1_offset = (Q7_T*)filter1 + k;
          ITER_T in_channels = CIn;

          #ifdef LOOP_UNROLL
            ITER_T len_unroll = in_channels >> 2;
            in_channels = in_channels % 4;
            while (len_unroll--) {
              sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
              filter1_offset += CTemp;
              sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
              filter1_offset += CTemp;
              sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
              filter1_offset += CTemp;
              sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
              filter1_offset += CTemp;
            }
          #endif

          while (in_channels--) {
            sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
            filter1_offset += CTemp;
          }

          #ifdef SHIFT
            Q31_T x = (((Q31_T)(((sum << shlU1) >> shrU1) + BN1B[k])) *
                       ((Q31_T)BN1W[k]));
          #else
            Q31_T x = (((Q31_T)((sum * shlU1) / shrU1 + BN1B[k])) *
                       ((Q31_T)BN1W[k]));
          #endif
          x = q31_relu(x, limit1);
          #ifdef SHIFT
            *convBuffer1_offset++ = ((x << shlX1) >> shrX1);
          #else
            *convBuffer1_offset++ = (x * shlX1) / shrX1;
          #endif
        }
      }
    }

    ITER_T hout = 0;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; hout++, h += (S_ITER_T)HStride) {
      ITER_T HIndexOut = hout * HOffsetOut + NIndexOut;
      for (ITER_T i = 0; i < HStride; i++) {
        ITER_T iRed = (i + margin + hout * HStride) % HF;
        ITER_T iFull = i + margin + hout * HStride;
        ITER_T HIndexC1 = iRed * HOffsetC1;
        ITER_T HIndexIn = iFull * HOffsetIn + NIndexIn;
        if (iFull < H){
          for (ITER_T j = 0; j < W; j++) {
            ITER_T WIndexIn = j * CIn + HIndexIn;
            Q15_T* convBuffer1_offset = ((Q15_T*)convBuffer1) + j * CTemp + HIndexC1;
            for (ITER_T k = 0; k < CTemp; k++) {
              sum = 0;
              Q15_T* input_offset = (Q15_T*)input + WIndexIn;
              Q7_T* filter1_offset = (Q7_T*)filter1 + k;
              ITER_T in_channels = CIn;

              #ifdef LOOP_UNROLL
                ITER_T len_unroll = in_channels >> 2;
                in_channels = in_channels % 4;
                while (len_unroll--) {
                  sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                  filter1_offset += CTemp;
                  sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                  filter1_offset += CTemp;
                  sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                  filter1_offset += CTemp;
                  sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                  filter1_offset += CTemp;
                }
              #endif

              while (in_channels--) {
                sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                filter1_offset += CTemp;
              }

              #ifdef SHIFT
                Q31_T x = (((Q31_T)(((sum << shlU1) >> shrU1) + BN1B[k])) *
                           ((Q31_T)BN1W[k]));
              #else
                Q31_T x = (((Q31_T)((sum * shlU1) / shrU1 + BN1B[k])) *
                           ((Q31_T)BN1W[k]));
              #endif
              x = q31_relu(x, limit1);
              #ifdef SHIFT
                *convBuffer1_offset++ = ((x << shlX1) >> shrX1);
              #else
                *convBuffer1_offset++ = (x * shlX1) / shrX1;
              #endif
            }
          }
        } else {
          Q15_T* convBuffer1_offset = (Q15_T*)convBuffer1 + HIndexC1;
          for (ITER_T j = 0; j < W; j++) {
            Q15_T* BN1B_offset = (Q15_T*)BN1B;
            Q7_T* BN1W_offset = (Q7_T*)BN1W;
            ITER_T temp_channels = CTemp;

            #ifdef LOOP_UNROLL
              ITER_T len_unroll = temp_channels >> 2;
              temp_channels = temp_channels % 4;
              while (len_unroll--) {
                Q31_T w = q31_relu(((Q31_T)*BN1B_offset++) * ((Q31_T)*BN1W_offset++), limit1);
                Q31_T x = q31_relu(((Q31_T)*BN1B_offset++) * ((Q31_T)*BN1W_offset++), limit1);
                Q31_T y = q31_relu(((Q31_T)*BN1B_offset++) * ((Q31_T)*BN1W_offset++), limit1);
                Q31_T z = q31_relu(((Q31_T)*BN1B_offset++) * ((Q31_T)*BN1W_offset++), limit1);
                #ifdef SHIFT
                  *convBuffer1_offset++ = ((w << shlX1) >> shrX1);
                  *convBuffer1_offset++ = ((x << shlX1) >> shrX1);
                  *convBuffer1_offset++ = ((y << shlX1) >> shrX1);
                  *convBuffer1_offset++ = ((z << shlX1) >> shrX1);
                #else
                  *convBuffer1_offset++ = ((w * shlX1) / shrX1);
                  *convBuffer1_offset++ = ((x * shlX1) / shrX1);
                  *convBuffer1_offset++ = ((y * shlX1) / shrX1);
                  *convBuffer1_offset++ = ((z * shlX1) / shrX1);
                #endif
              }
            #endif

            while (temp_channels--) {
              Q31_T w = q31_relu(((Q31_T)*BN1B_offset++) * ((Q31_T)*BN1W_offset++), limit1);
              #ifdef SHIFT
                *convBuffer1_offset++ = ((w << shlX1) >> shrX1);
              #else
                *convBuffer1_offset++ = ((w * shlX1) / shrX1);
              #endif
            }
          }
        }
      }

      ITER_T wout = 0;
      for (S_ITER_T w = WOffsetL; w < ((S_ITER_T)W) - WOffsetR; wout++, w += ((S_ITER_T)WStride)) {
        Q15_T* output_offset = ((Q15_T*)output) + wout * COut + HIndexOut;
        for (ITER_T g = 0; g < CTemp; g++) {
          sum = 0;
          ITER_T GIndexF = g * GOffsetF;
          for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
            S_ITER_T hindex = h + hf;
            if ((hindex < 0) || (hindex >= (S_ITER_T)H)){
              continue;
            }
            ITER_T HIndexC1 = (((ITER_T)hindex) % HF) * HOffsetC1 + g;
            ITER_T HIndexF = ((ITER_T)(hf + HOffsetFL)) * WF + GIndexF;
            for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
              S_ITER_T windex = w + wf;
              if ((windex < 0) || (windex >= (S_ITER_T)W)) {
                continue;
              } else {
                sum += ((Q31_T)convBuffer1[HIndexC1 + ((ITER_T)windex) * CTemp]) *
                       ((Q31_T)filter2[HIndexF + ((ITER_T)(wf + WOffsetFL))]);
              }
            }
          }

          #ifdef SHIFT
            Q31_T x = (((Q31_T)(((sum << shlU2) >> shrU2) + BN2B[g])) *
                       ((Q31_T)BN2W[g]));
          #else
            Q31_T x = (((Q31_T)((sum * shlU2) / shrU2 + BN2B[g])) *
                       ((Q31_T)BN2W[g]));
          #endif
          x = q31_relu(x, limit2);
          #ifdef SHIFT
            convBuffer2[g] = ((x << shlX2) >> shrX2);
          #else
            convBuffer2[g] = (x * shlX2) / shrX2;
          #endif
        }

        for (ITER_T i = 0; i < COut; i++) {
          sum = 0;
          Q15_T* convBuffer2_offset = (Q15_T*)convBuffer2;
          Q7_T* filter3_offset = (Q7_T*)filter3 + i;
          ITER_T temp_channels = CTemp;

          #ifdef LOOP_UNROLL
            ITER_T len_unroll = temp_channels >> 2;
            temp_channels = temp_channels % 4;
            while (len_unroll--) {
              sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
              filter3_offset += COut;
              sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
              filter3_offset += COut;
              sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
              filter3_offset += COut;
              sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
              filter3_offset += COut;
            }
          #endif

          while (temp_channels--) {
            sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
            filter3_offset += COut;
          }

          #ifdef SHIFT
            *output_offset++ = (((((Q31_T)(((sum << shlU3) >> shrU3) + BN3B[i])) *
                                  ((Q31_T) BN3W[i])) << shlW3) >> shrW3);
          #else
            *output_offset++ = ((((Q31_T)((sum * shlU3) / shrU3 + BN3B[i])) *
                                 ((Q31_T) BN3W[i])) * shlW3) / shrW3;
          #endif
        }
      }
    }
  }
}

void q15_mbconv_block(const Q15_T* const input, const Q15_T* const filter1,
  const Q15_T* const BN1W, const Q15_T* const BN1B, const Q15_T* const filter2,
  const Q15_T* const BN2W, const Q15_T* const BN2B, const Q15_T* const filter3,
  const Q15_T* const BN3W, const Q15_T* const BN3B, Q15_T* const output,
  Q15_T* const convBuffer1, Q15_T* const convBuffer2, ITER_T N, ITER_T H,
  ITER_T W, ITER_T CIn, ITER_T CTemp, ITER_T HF, ITER_T WF, ITER_T COut,
  ITER_T HOut, ITER_T WOut, S_ITER_T HPadU, S_ITER_T HPadD, S_ITER_T WPadL,
  S_ITER_T WPadR, ITER_T HStride, ITER_T WStride, Q31_T limit1, Q31_T limit2,
  SCALE_T shrU1, SCALE_T shrX1, SCALE_T shrU2, SCALE_T shrX2, SCALE_T shrU3,
  SCALE_T shrW3, SCALE_T shlU1, SCALE_T shlX1, SCALE_T shlU2, SCALE_T shlX2,
  SCALE_T shlU3, SCALE_T shlW3) {
  S_ITER_T HOffsetFL = (HF - 1) >> 1;
  S_ITER_T WOffsetFL = (WF - 1) >> 1;
  S_ITER_T HOffsetFR = HF >> 1;
  S_ITER_T WOffsetFR = WF >> 1;

  S_ITER_T HOffsetL = HOffsetFL - HPadU;
  S_ITER_T WOffsetL = WOffsetFL - WPadL;
  S_ITER_T HOffsetR = HOffsetFR - HPadD;
  S_ITER_T WOffsetR = WOffsetFR - WPadR;

  ITER_T HOffsetIn = W * CIn;
  ITER_T NOffsetIn = H * HOffsetIn;
  ITER_T HOffsetC1 = W * CTemp;
  ITER_T GOffsetF = HF * WF;
  ITER_T HOffsetOut = WOut * COut;
  ITER_T NOffsetOut = HOut * HOffsetOut;

  Q63_T sum;
  for (ITER_T n = 0; n < N; n++) {
    ITER_T NIndexIn = n * NOffsetIn;
    ITER_T NIndexOut = n * NOffsetOut;
    ITER_T margin = 0;
    if ((S_ITER_T)HF - HPadU - (S_ITER_T)HStride > 0) {
      margin = (ITER_T)((S_ITER_T)HF - HPadU - (S_ITER_T)HStride);
    }

    for (ITER_T i = 0; i < margin; i++) {
      ITER_T HIndexIn = i * HOffsetIn + NIndexIn;
      ITER_T HIndexC1 = i * HOffsetC1;
      for (ITER_T j = 0; j < W; j++) {
        ITER_T WIndexIn = j * CIn + HIndexIn;
        Q15_T* convBuffer1_offset = ((Q15_T*)convBuffer1) + j * CTemp + HIndexC1;
        for (ITER_T k = 0; k < CTemp; k++) {
          sum = 0;
          Q15_T* input_offset = (Q15_T*)input + WIndexIn;
          Q15_T* filter1_offset = (Q15_T*)filter1 + k;
          ITER_T in_channels = CIn;

          #ifdef LOOP_UNROLL
            ITER_T len_unroll = in_channels >> 2;
            in_channels = in_channels % 4;
            while (len_unroll--) {
              sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
              filter1_offset += CTemp;
              sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
              filter1_offset += CTemp;
              sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
              filter1_offset += CTemp;
              sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
              filter1_offset += CTemp;
            }
          #endif

          while (in_channels--) {
            sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
            filter1_offset += CTemp;
          }

          #ifdef SHIFT
            Q31_T x = (((Q31_T)(((sum << shlU1) >> shrU1) + BN1B[k])) *
                       ((Q31_T)BN1W[k]));
          #else
            Q31_T x = (((Q31_T)((sum * shlU1) / shrU1 + BN1B[k])) *
                       ((Q31_T)BN1W[k]));
          #endif
          x = q31_relu(x, limit1);
          #ifdef SHIFT
            *convBuffer1_offset++ = ((x << shlX1) >> shrX1);
          #else
            *convBuffer1_offset++ = (x * shlX1) / shrX1;
          #endif
        }
      }
    }

    ITER_T hout = 0;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; hout++, h += (S_ITER_T)HStride) {
      ITER_T HIndexOut = hout * HOffsetOut + NIndexOut;
      for (ITER_T i = 0; i < HStride; i++) {
        ITER_T iRed = (i + margin + hout * HStride) % HF;
        ITER_T iFull = i + margin + hout * HStride;
        ITER_T HIndexC1 = iRed * HOffsetC1;
        ITER_T HIndexIn = iFull * HOffsetIn + NIndexIn;
        if (iFull < H){
          for (ITER_T j = 0; j < W; j++) {
            ITER_T WIndexIn = j * CIn + HIndexIn;
            Q15_T* convBuffer1_offset = ((Q15_T*)convBuffer1) + j * CTemp + HIndexC1;
            for (ITER_T k = 0; k < CTemp; k++) {
              sum = 0;
              Q15_T* input_offset = (Q15_T*)input + WIndexIn;
              Q15_T* filter1_offset = (Q15_T*)filter1 + k;
              ITER_T in_channels = CIn;

              #ifdef LOOP_UNROLL
                ITER_T len_unroll = in_channels >> 2;
                in_channels = in_channels % 4;
                while (len_unroll--) {
                  sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                  filter1_offset += CTemp;
                  sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                  filter1_offset += CTemp;
                  sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                  filter1_offset += CTemp;
                  sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                  filter1_offset += CTemp;
                }
              #endif

              while (in_channels--) {
                sum += ((Q31_T)*input_offset++) * ((Q31_T)*filter1_offset);
                filter1_offset += CTemp;
              }

              #ifdef SHIFT
                Q31_T x = (((Q31_T)(((sum << shlU1) >> shrU1) + BN1B[k])) *
                           ((Q31_T)BN1W[k]));
              #else
                Q31_T x = (((Q31_T)((sum * shlU1) / shrU1 + BN1B[k])) *
                           ((Q31_T)BN1W[k]));
              #endif
              x = q31_relu(x, limit1);
              #ifdef SHIFT
                *convBuffer1_offset++ = ((x << shlX1) >> shrX1);
              #else
                *convBuffer1_offset++ = (x * shlX1) / shrX1;
              #endif
            }
          }
        } else {
          Q15_T* convBuffer1_offset = (Q15_T*)convBuffer1 + HIndexC1;
          for (ITER_T j = 0; j < W; j++) {
            Q15_T* BN1B_offset = (Q15_T*)BN1B;
            Q15_T* BN1W_offset = (Q15_T*)BN1W;
            ITER_T temp_channels = CTemp;

            #ifdef LOOP_UNROLL
              ITER_T len_unroll = temp_channels >> 2;
              temp_channels = temp_channels % 4;
              while (len_unroll--) {
                Q31_T w = q31_relu(((Q31_T)*BN1B_offset++) * ((Q31_T)*BN1W_offset++), limit1);
                Q31_T x = q31_relu(((Q31_T)*BN1B_offset++) * ((Q31_T)*BN1W_offset++), limit1);
                Q31_T y = q31_relu(((Q31_T)*BN1B_offset++) * ((Q31_T)*BN1W_offset++), limit1);
                Q31_T z = q31_relu(((Q31_T)*BN1B_offset++) * ((Q31_T)*BN1W_offset++), limit1);
                #ifdef SHIFT
                  *convBuffer1_offset++ = ((w << shlX1) >> shrX1);
                  *convBuffer1_offset++ = ((x << shlX1) >> shrX1);
                  *convBuffer1_offset++ = ((y << shlX1) >> shrX1);
                  *convBuffer1_offset++ = ((z << shlX1) >> shrX1);
                #else
                  *convBuffer1_offset++ = ((w * shlX1) / shrX1);
                  *convBuffer1_offset++ = ((x * shlX1) / shrX1);
                  *convBuffer1_offset++ = ((y * shlX1) / shrX1);
                  *convBuffer1_offset++ = ((z * shlX1) / shrX1);
                #endif
              }
            #endif

            while (temp_channels--) {
              Q31_T w = q31_relu(((Q31_T)*BN1B_offset++) * ((Q31_T)*BN1W_offset++), limit1);
              #ifdef SHIFT
                *convBuffer1_offset++ = ((w << shlX1) >> shrX1);
              #else
                *convBuffer1_offset++ = ((w * shlX1) / shrX1);
              #endif
            }
          }
        }
      }

      ITER_T wout = 0;
      for (S_ITER_T w = WOffsetL; w < ((S_ITER_T)W) - WOffsetR; wout++, w += ((S_ITER_T)WStride)) {
        Q15_T* output_offset = ((Q15_T*)output) + wout * COut + HIndexOut;
        for (ITER_T g = 0; g < CTemp; g++) {
          sum = 0;
          ITER_T GIndexF = g * GOffsetF;
          for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
            S_ITER_T hindex = h + hf;
            if ((hindex < 0) || (hindex >= (S_ITER_T)H)){
              continue;
            }
            ITER_T HIndexC1 = (((ITER_T)hindex) % HF) * HOffsetC1 + g;
            ITER_T HIndexF = ((ITER_T)(hf + HOffsetFL)) * WF + GIndexF;
            for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
              S_ITER_T windex = w + wf;
              if ((windex < 0) || (windex >= (S_ITER_T)W)) {
                continue;
              } else {
                sum += ((Q31_T)convBuffer1[HIndexC1 + ((ITER_T)windex) * CTemp]) *
                       ((Q31_T)filter2[HIndexF + ((ITER_T)(wf + WOffsetFL))]);
              }
            }
          }

          #ifdef SHIFT
            Q31_T x = (((Q31_T)(((sum << shlU2) >> shrU2) + BN2B[g])) *
                       ((Q31_T)BN2W[g]));
          #else
            Q31_T x = (((Q31_T)((sum * shlU2) / shrU2 + BN2B[g])) *
                       ((Q31_T)BN2W[g]));
          #endif
          x = q31_relu(x, limit2);
          #ifdef SHIFT
            convBuffer2[g] = ((x << shlX2) >> shrX2);
          #else
            convBuffer2[g] = (x * shlX2) / shrX2;
          #endif
        }

        for (ITER_T i = 0; i < COut; i++) {
          sum = 0;
          Q15_T* convBuffer2_offset = (Q15_T*)convBuffer2;
          Q15_T* filter3_offset = (Q15_T*)filter3 + i;
          ITER_T temp_channels = CTemp;

          #ifdef LOOP_UNROLL
            ITER_T len_unroll = temp_channels >> 2;
            temp_channels = temp_channels % 4;
            while (len_unroll--) {
              sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
              filter3_offset += COut;
              sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
              filter3_offset += COut;
              sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
              filter3_offset += COut;
              sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
              filter3_offset += COut;
            }
          #endif

          while (temp_channels--) {
            sum += ((Q31_T)*convBuffer2_offset++) * ((Q31_T)*filter3_offset);
            filter3_offset += COut;
          }

          #ifdef SHIFT
            *output_offset++ = (((((Q31_T)(((sum << shlU3) >> shrU3) + BN3B[i])) *
                                  ((Q31_T) BN3W[i])) << shlW3) >> shrW3);
          #else
            *output_offset++ = ((((Q31_T)((sum * shlU3) / shrU3 + BN3B[i])) *
                                 ((Q31_T) BN3W[i])) * shlW3) / shrW3;
          #endif
        }
      }
    }
  }
}
