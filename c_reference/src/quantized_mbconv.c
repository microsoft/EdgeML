// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "quantized_mbconv.h"

void q7_mbconv_block(const Q7_T* const input, const Q7_T* const filter1,
  const Q7_T* const BN1W, const Q7_T* const BN1B, const Q7_T* const filter2,
  const Q7_T* const BN2W, const Q7_T* const BN2B, const Q7_T* const filter3,
  const Q7_T* const BN3W, const Q7_T* const BN3B, Q7_T* const output,
  Q7_T* const convBuffer1, Q7_T* const convBuffer2, Q15_T* const treesumBuffer,
  ITER_T N, ITER_T H, ITER_T W, ITER_T CIn, ITER_T CTemp, ITER_T HF, ITER_T WF,
  ITER_T COut, ITER_T HOut, ITER_T WOut, S_ITER_T HPadU, S_ITER_T HPadD,
  S_ITER_T WPadL, S_ITER_T WPadR, ITER_T HStride, ITER_T WStride, SCALE_T depth1,
  SCALE_T depth2, SCALE_T depth3, Q15_T limit1, Q15_T limit2, SCALE_T shrU1,
  SCALE_T shrB1, SCALE_T shrX1, SCALE_T shrU2, SCALE_T shrB2, SCALE_T shrX2,
  SCALE_T shrU3, SCALE_T shrB3, SCALE_T shrW3, SCALE_T shlU1, SCALE_T shlB1,
  SCALE_T shlX1, SCALE_T shlU2, SCALE_T shlB2, SCALE_T shlX2, SCALE_T shlU3,
  SCALE_T shlB3, SCALE_T shlW3) {

  S_ITER_T HOffsetFL = (HF - 1) >> 1;
  S_ITER_T WOffsetFL = (WF - 1) >> 1;
  S_ITER_T HOffsetFR = HF >> 1;
  S_ITER_T WOffsetFR = WF >> 1;

  S_ITER_T HOffsetL = HOffsetFL - HPadU;
  S_ITER_T WOffsetL = WOffsetFL - WPadL;
  S_ITER_T HOffsetR = HOffsetFR - HPadD;
  S_ITER_T WOffsetR = WOffsetFR - WPadR;

  for (ITER_T n = 0; n < N; n++) {
    ITER_T margin = 0, nstart = 0;
    if ((S_ITER_T)HF - HPadU - (S_ITER_T)HStride > 0) {
      margin = (ITER_T)((S_ITER_T)HF - HPadU - (S_ITER_T)HStride);
    }
    if (HPadU < 0) {
      // nstart will always be zero unless HPadU is negative.
      nstart = (ITER_T)(-HPadU);
    }

    for (ITER_T i = nstart; i < margin; i++) {
      for (ITER_T j = 0; j < W; j++) {
        for (ITER_T k = 0; k < CTemp; k++) {
          for (ITER_T l = 0; l < CIn; l++) {
            treesumBuffer[l] = ((Q15_T)input[n * H * W * CIn + i * W * CIn + j * CIn + l]) *
                               ((Q15_T)filter1[l * CTemp + k]);
          }

          q15_v_treesum(treesumBuffer, CIn, depth1, 0);
          #ifdef SHIFT
            Q15_T x = (((Q15_T)(((treesumBuffer[0] << shlU1) >> shrU1) + ((BN1B[k] << shlB1) >> shrB1))) *
                        ((Q15_T)BN1W[k]));
          #else
            Q15_T x = (((Q15_T)((treesumBuffer[0] * shlU1) / shrU1 + (BN1B[k] * shlB1) / shrB1)) *
                        ((Q15_T)BN1W[k]));
          #endif
          x = q15_relu(x, limit1);
          #ifdef SHIFT
            convBuffer1[i * W * CTemp + j * CTemp + k] = ((x << shlX1) >> shrX1);
          #else
            convBuffer1[i * W * CTemp + j * CTemp + k] = (x * shlX1) / shrX1;
          #endif
        }
      }
    }

    ITER_T hout = 0;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; hout++, h += (S_ITER_T)HStride) {
      for (ITER_T i = 0; i < HStride; i++) {
        for (ITER_T j = 0; j < W; j++) {
          for (ITER_T k = 0; k < CTemp; k++) {
            ITER_T iRed = (i + margin + hout * HStride) % HF;
            ITER_T iFull = i + margin + hout * HStride;
            for (ITER_T l = 0; l < CIn; l++) {
              if (iFull < H) {
                treesumBuffer[l] = ((Q15_T)input[n * H * W * CIn + iFull * W * CIn + j * CIn + l]) *
                                   ((Q15_T)filter1[l * CTemp + k]);
              } else {
                treesumBuffer[l] = 0;
              }
            }

            q15_v_treesum(treesumBuffer, CIn, depth1, 0);
            #ifdef SHIFT
              Q15_T x = (((Q15_T)(((treesumBuffer[0] << shlU1) >> shrU1) + ((BN1B[k] << shlB1) >> shrB1))) *
                          ((Q15_T)BN1W[k]));
            #else
              Q15_T x = (((Q15_T)((treesumBuffer[0] * shlU1) / shrU1 + (BN1B[k] * shlB1) / shrB1)) *
                          ((Q15_T)BN1W[k]));
            #endif
            x = q15_relu(x, limit1);
            #ifdef SHIFT
              convBuffer1[iRed * W * CTemp + j * CTemp + k] = ((x << shlX1) >> shrX1);
            #else
              convBuffer1[iRed * W * CTemp + j * CTemp + k] = (x * shlX1) / shrX1;
            #endif
          }
        }
      }

      ITER_T wout = 0;
      for (S_ITER_T w = WOffsetL; w < ((S_ITER_T)W) - WOffsetR; wout++, w += ((S_ITER_T)WStride)) {
        for (ITER_T g = 0; g < CTemp; g++) {
          ITER_T counter = 0;
          for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
            for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
              if (((h + hf) < 0) || ((h + hf) >= (S_ITER_T)H) || ((w + wf) < 0) || ((w + wf) >= (S_ITER_T)W)) {
                treesumBuffer[counter] = 0;
              } else {
                treesumBuffer[counter] = ((Q15_T)convBuffer1[(((ITER_T)(h + hf)) % HF) * W * CTemp + ((ITER_T)(w + wf)) * CTemp + g]) *
                                         ((Q15_T)filter2[g * HF * WF + ((ITER_T)(hf + HOffsetFL)) * WF + ((ITER_T)(wf + WOffsetFL))]);
              }
              counter++;
            }
          }

          q15_v_treesum(treesumBuffer, HF * WF, depth2, 0);
          #ifdef SHIFT
            Q15_T x = (((Q15_T)(((treesumBuffer[0] << shlU2) >> shrU2) + ((BN2B[g] << shlB2) >> shrB2))) *
                        ((Q15_T)BN2W[g]));
          #else
            Q15_T x = (((Q15_T)((treesumBuffer[0] * shlU2) / shrU2 + (BN2B[g] * shlB2) / shrB2)) *
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
          for (ITER_T g = 0; g < CTemp; g++) {
            treesumBuffer[g] = ((Q15_T)convBuffer2[g]) * ((Q15_T)filter3[g * COut + i]);
          }

          q15_v_treesum(treesumBuffer, CTemp, depth3, 0);
          #ifdef SHIFT
            output[n * HOut * WOut * COut + hout * WOut * COut + wout * COut + i] =
              (((((Q15_T)(((treesumBuffer[0] << shlU3) >> shrU3) + ((BN3B[i] << shlB3) >> shrB3))) *
                ((Q15_T) BN3W[i])) << shlW3) >> shrW3);
          #else
            output[n * HOut * WOut * COut + hout * WOut * COut + wout * COut + i] =
              ((((Q15_T)((treesumBuffer[0] * shlU3) / shrU3 + (BN3B[i] * shlB3) / shrB3)) *
                ((Q15_T) BN3W[i])) * shlW3) / shrW3;
          #endif
        }
      }
    }
  }
}

void q7xq15_q15_mbconv_block(const Q7_T* const input, const Q15_T* const filter1,
  const Q15_T* const BN1W, const Q15_T* const BN1B, const Q15_T* const filter2,
  const Q15_T* const BN2W, const Q15_T* const BN2B, const Q15_T* const filter3,
  const Q15_T* const BN3W, const Q15_T* const BN3B, Q15_T* const output,
  Q15_T* const convBuffer1, Q15_T* const convBuffer2, Q31_T* const treesumBuffer,
  ITER_T N, ITER_T H, ITER_T W, ITER_T CIn, ITER_T CTemp, ITER_T HF, ITER_T WF,
  ITER_T COut, ITER_T HOut, ITER_T WOut, S_ITER_T HPadU, S_ITER_T HPadD,
  S_ITER_T WPadL, S_ITER_T WPadR, ITER_T HStride, ITER_T WStride, SCALE_T depth1,
  SCALE_T depth2, SCALE_T depth3, Q31_T limit1, Q31_T limit2, L_SCALE_T shrU1,
  L_SCALE_T shrB1, L_SCALE_T shrX1, L_SCALE_T shrU2, L_SCALE_T shrB2,
  L_SCALE_T shrX2, L_SCALE_T shrU3, L_SCALE_T shrB3, L_SCALE_T shrW3,
  L_SCALE_T shlU1, L_SCALE_T shlB1, L_SCALE_T shlX1, L_SCALE_T shlU2,
  L_SCALE_T shlB2, L_SCALE_T shlX2, L_SCALE_T shlU3, L_SCALE_T shlB3,
  L_SCALE_T shlW3) {

  S_ITER_T HOffsetFL = (HF - 1) >> 1;
  S_ITER_T WOffsetFL = (WF - 1) >> 1;
  S_ITER_T HOffsetFR = HF >> 1;
  S_ITER_T WOffsetFR = WF >> 1;

  S_ITER_T HOffsetL = HOffsetFL - HPadU;
  S_ITER_T WOffsetL = WOffsetFL - WPadL;
  S_ITER_T HOffsetR = HOffsetFR - HPadD;
  S_ITER_T WOffsetR = WOffsetFR - WPadR;

  for (ITER_T n = 0; n < N; n++) {
    ITER_T margin = 0, nstart = 0;
    if ((S_ITER_T)HF - HPadU - (S_ITER_T)HStride > 0) {
      margin = (ITER_T)((S_ITER_T)HF - HPadU - (S_ITER_T)HStride);
    }
    if (HPadU < 0) {
      // nstart will always be zero unless HPadU is negative.
      nstart = (ITER_T)(-HPadU);
    }

    for (ITER_T i = nstart; i < margin; i++) {
      for (ITER_T j = 0; j < W; j++) {
        for (ITER_T k = 0; k < CTemp; k++) {
          for (ITER_T l = 0; l < CIn; l++) {
            treesumBuffer[l] = ((Q31_T)input[n * H * W * CIn + i * W * CIn + j * CIn + l]) *
                               ((Q31_T)filter1[l * CTemp + k]);
          }

          q31_v_treesum(treesumBuffer, CIn, depth1, 0);
          #ifdef SHIFT
            Q31_T x = (((Q31_T)(((treesumBuffer[0] << shlU1) >> shrU1) + ((BN1B[k] << shlB1) >> shrB1))) *
                        ((Q31_T)BN1W[k]));
          #else
            Q31_T x = (((Q31_T)((treesumBuffer[0] * shlU1) / shrU1 + (BN1B[k] * shlB1) / shrB1)) *
                        ((Q31_T)BN1W[k]));
          #endif
          x = q31_relu(x, limit1);
          #ifdef SHIFT
            convBuffer1[i * W * CTemp + j * CTemp + k] = ((x << shlX1) >> shrX1);
          #else
            convBuffer1[i * W * CTemp + j * CTemp + k] = (x * shlX1) / shrX1;
          #endif
        }
      }
    }

    ITER_T hout = 0;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; hout++, h += (S_ITER_T)HStride) {
      for (ITER_T i = 0; i < HStride; i++) {
        for (ITER_T j = 0; j < W; j++) {
          for (ITER_T k = 0; k < CTemp; k++) {
            ITER_T iRed = (i + margin + hout * HStride) % HF;
            ITER_T iFull = i + margin + hout * HStride;
            for (ITER_T l = 0; l < CIn; l++) {
              if (iFull < H) {
                treesumBuffer[l] = ((Q31_T)input[n * H * W * CIn + iFull * W * CIn + j * CIn + l]) *
                                   ((Q31_T)filter1[l * CTemp + k]);
              } else {
                treesumBuffer[l] = 0;
              }
            }

            q31_v_treesum(treesumBuffer, CIn, depth1, 0);
            #ifdef SHIFT
              Q31_T x = (((Q31_T)(((treesumBuffer[0] << shlU1) >> shrU1) + ((BN1B[k] << shlB1) >> shrB1))) *
                          ((Q31_T)BN1W[k]));
            #else
              Q31_T x = (((Q31_T)((treesumBuffer[0] * shlU1) / shrU1 + (BN1B[k] * shlB1) / shrB1)) *
                          ((Q31_T)BN1W[k]));
            #endif
            x = q31_relu(x, limit1);
            #ifdef SHIFT
              convBuffer1[iRed * W * CTemp + j * CTemp + k] = ((x << shlX1) >> shrX1);
            #else
              convBuffer1[iRed * W * CTemp + j * CTemp + k] = (x * shlX1) / shrX1;
            #endif
          }
        }
      }

      ITER_T wout = 0;
      for (S_ITER_T w = WOffsetL; w < ((S_ITER_T)W) - WOffsetR; wout++, w += ((S_ITER_T)WStride)) {
        for (ITER_T g = 0; g < CTemp; g++) {
          ITER_T counter = 0;
          for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
            for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
              if (((h + hf) < 0) || ((h + hf) >= (S_ITER_T)H) || ((w + wf) < 0) || ((w + wf) >= (S_ITER_T)W)) {
                treesumBuffer[counter] = 0;
              } else {
                treesumBuffer[counter] = ((Q31_T)convBuffer1[(((ITER_T)(h + hf)) % HF) * W * CTemp + ((ITER_T)(w + wf)) * CTemp + g]) *
                                         ((Q31_T)filter2[g * HF * WF + ((ITER_T)(hf + HOffsetFL)) * WF + ((ITER_T)(wf + WOffsetFL))]);
              }
              counter++;
            }
          }

          q31_v_treesum(treesumBuffer, HF * WF, depth2, 0);
          #ifdef SHIFT
            Q31_T x = (((Q31_T)(((treesumBuffer[0] << shlU2) >> shrU2) + ((BN2B[g] << shlB2) >> shrB2))) *
                        ((Q31_T)BN2W[g]));
          #else
            Q31_T x = (((Q31_T)((treesumBuffer[0] * shlU2) / shrU2 + (BN2B[g] * shlB2) / shrB2)) *
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
          for (ITER_T g = 0; g < CTemp; g++) {
            treesumBuffer[g] = ((Q31_T)convBuffer2[g]) * ((Q31_T)filter3[g * COut + i]);
          }

          q31_v_treesum(treesumBuffer, CTemp, depth3, 0);
          #ifdef SHIFT
            output[n * HOut * WOut * COut + hout * WOut * COut + wout * COut + i] =
              (((((Q31_T)(((treesumBuffer[0] << shlU3) >> shrU3) + ((BN3B[i] << shlB3) >> shrB3))) *
                ((Q31_T) BN3W[i])) << shlW3) >> shrW3);
          #else
            output[n * HOut * WOut * COut + hout * WOut * COut + wout * COut + i] =
              ((((Q31_T)((treesumBuffer[0] * shlU3) / shrU3 + (BN3B[i] * shlB3) / shrB3)) *
                ((Q31_T) BN3W[i])) * shlW3) / shrW3;
          #endif
        }
      }
    }
  }
}

void q15xq7_q7_mbconv_block(const Q15_T* const input, const Q7_T* const filter1,
  const Q7_T* const BN1W, const Q7_T* const BN1B, const Q7_T* const filter2,
  const Q7_T* const BN2W, const Q7_T* const BN2B, const Q7_T* const filter3,
  const Q7_T* const BN3W, const Q7_T* const BN3B, Q7_T* const output,
  Q15_T* const convBuffer1, Q15_T* const convBuffer2, Q31_T* const treesumBuffer,
  ITER_T N, ITER_T H, ITER_T W, ITER_T CIn, ITER_T CTemp, ITER_T HF, ITER_T WF,
  ITER_T COut, ITER_T HOut, ITER_T WOut, S_ITER_T HPadU, S_ITER_T HPadD,
  S_ITER_T WPadL, S_ITER_T WPadR, ITER_T HStride, ITER_T WStride, SCALE_T depth1,
  SCALE_T depth2, SCALE_T depth3, Q31_T limit1, Q31_T limit2, L_SCALE_T shrU1,
  L_SCALE_T shrB1, L_SCALE_T shrX1, L_SCALE_T shrU2, L_SCALE_T shrB2,
  L_SCALE_T shrX2, L_SCALE_T shrU3, L_SCALE_T shrB3, L_SCALE_T shrW3,
  L_SCALE_T shlU1, L_SCALE_T shlB1, L_SCALE_T shlX1, L_SCALE_T shlU2,
  L_SCALE_T shlB2, L_SCALE_T shlX2, L_SCALE_T shlU3, L_SCALE_T shlB3,
  L_SCALE_T shlW3) {

  S_ITER_T HOffsetFL = (HF - 1) >> 1;
  S_ITER_T WOffsetFL = (WF - 1) >> 1;
  S_ITER_T HOffsetFR = HF >> 1;
  S_ITER_T WOffsetFR = WF >> 1;

  S_ITER_T HOffsetL = HOffsetFL - HPadU;
  S_ITER_T WOffsetL = WOffsetFL - WPadL;
  S_ITER_T HOffsetR = HOffsetFR - HPadD;
  S_ITER_T WOffsetR = WOffsetFR - WPadR;

  for (ITER_T n = 0; n < N; n++) {
    ITER_T margin = 0, nstart = 0;
    if ((S_ITER_T)HF - HPadU - (S_ITER_T)HStride > 0) {
      margin = (ITER_T)((S_ITER_T)HF - HPadU - (S_ITER_T)HStride);
    }
    if (HPadU < 0) {
      // nstart will always be zero unless HPadU is negative.
      nstart = (ITER_T)(-HPadU);
    }

    for (ITER_T i = nstart; i < margin; i++) {
      for (ITER_T j = 0; j < W; j++) {
        for (ITER_T k = 0; k < CTemp; k++) {
          for (ITER_T l = 0; l < CIn; l++) {
            treesumBuffer[l] = ((Q31_T)input[n * H * W * CIn + i * W * CIn + j * CIn + l]) *
                               ((Q31_T)filter1[l * CTemp + k]);
          }

          q31_v_treesum(treesumBuffer, CIn, depth1, 0);
          #ifdef SHIFT
            Q31_T x = (((Q31_T)(((treesumBuffer[0] << shlU1) >> shrU1) + ((BN1B[k] << shlB1) >> shrB1))) *
                        ((Q31_T)BN1W[k]));
          #else
            Q31_T x = (((Q31_T)((treesumBuffer[0] * shlU1) / shrU1 + (BN1B[k] * shlB1) / shrB1)) *
                        ((Q31_T)BN1W[k]));
          #endif
          x = q31_relu(x, limit1);
          #ifdef SHIFT
            convBuffer1[i * W * CTemp + j * CTemp + k] = ((x << shlX1) >> shrX1);
          #else
            convBuffer1[i * W * CTemp + j * CTemp + k] = (x * shlX1) / shrX1;
          #endif
        }
      }
    }

    ITER_T hout = 0;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; hout++, h += (S_ITER_T)HStride) {
      for (ITER_T i = 0; i < HStride; i++) {
        for (ITER_T j = 0; j < W; j++) {
          for (ITER_T k = 0; k < CTemp; k++) {
            ITER_T iRed = (i + margin + hout * HStride) % HF;
            ITER_T iFull = i + margin + hout * HStride;
            for (ITER_T l = 0; l < CIn; l++) {
              if (iFull < H) {
                treesumBuffer[l] = ((Q31_T)input[n * H * W * CIn + iFull * W * CIn + j * CIn + l]) *
                                   ((Q31_T)filter1[l * CTemp + k]);
              } else {
                treesumBuffer[l] = 0;
              }
            }

            q31_v_treesum(treesumBuffer, CIn, depth1, 0);
            #ifdef SHIFT
              Q31_T x = (((Q31_T)(((treesumBuffer[0] << shlU1) >> shrU1) + ((BN1B[k] << shlB1) >> shrB1))) *
                          ((Q31_T)BN1W[k]));
            #else
              Q31_T x = (((Q31_T)((treesumBuffer[0] * shlU1) / shrU1 + (BN1B[k] * shlB1) / shrB1)) *
                          ((Q31_T)BN1W[k]));
            #endif
            x = q31_relu(x, limit1);
            #ifdef SHIFT
              convBuffer1[iRed * W * CTemp + j * CTemp + k] = ((x << shlX1) >> shrX1);
            #else
              convBuffer1[iRed * W * CTemp + j * CTemp + k] = (x * shlX1) / shrX1;
            #endif
          }
        }
      }

      ITER_T wout = 0;
      for (S_ITER_T w = WOffsetL; w < ((S_ITER_T)W) - WOffsetR; wout++, w += ((S_ITER_T)WStride)) {
        for (ITER_T g = 0; g < CTemp; g++) {
          ITER_T counter = 0;
          for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
            for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
              if (((h + hf) < 0) || ((h + hf) >= (S_ITER_T)H) || ((w + wf) < 0) || ((w + wf) >= (S_ITER_T)W)) {
                treesumBuffer[counter] = 0;
              } else {
                treesumBuffer[counter] = ((Q31_T)convBuffer1[(((ITER_T)(h + hf)) % HF) * W * CTemp + ((ITER_T)(w + wf)) * CTemp + g]) *
                                         ((Q31_T)filter2[g * HF * WF + ((ITER_T)(hf + HOffsetFL)) * WF + ((ITER_T)(wf + WOffsetFL))]);
              }
              counter++;
            }
          }

          q31_v_treesum(treesumBuffer, HF * WF, depth2, 0);
          #ifdef SHIFT
            Q31_T x = (((Q31_T)(((treesumBuffer[0] << shlU2) >> shrU2) + ((BN2B[g] << shlB2) >> shrB2))) *
                        ((Q31_T)BN2W[g]));
          #else
            Q31_T x = (((Q31_T)((treesumBuffer[0] * shlU2) / shrU2 + (BN2B[g] * shlB2) / shrB2)) *
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
          for (ITER_T g = 0; g < CTemp; g++) {
            treesumBuffer[g] = ((Q31_T)convBuffer2[g]) * ((Q31_T)filter3[g * COut + i]);
          }

          q31_v_treesum(treesumBuffer, CTemp, depth3, 0);
          #ifdef SHIFT
            output[n * HOut * WOut * COut + hout * WOut * COut + wout * COut + i] =
              (((((Q31_T)(((treesumBuffer[0] << shlU3) >> shrU3) + ((BN3B[i] << shlB3) >> shrB3))) *
                ((Q31_T) BN3W[i])) << shlW3) >> shrW3);
          #else
            output[n * HOut * WOut * COut + hout * WOut * COut + wout * COut + i] =
              ((((Q31_T)((treesumBuffer[0] * shlU3) / shrU3 + (BN3B[i] * shlB3) / shrB3)) *
                ((Q31_T) BN3W[i])) * shlW3) / shrW3;
          #endif
        }
      }
    }
  }
}

void q15xq7_q15_mbconv_block(const Q15_T* const input, const Q7_T* const filter1,
  const Q7_T* const BN1W, const Q7_T* const BN1B, const Q7_T* const filter2,
  const Q7_T* const BN2W, const Q7_T* const BN2B, const Q7_T* const filter3,
  const Q7_T* const BN3W, const Q7_T* const BN3B, Q15_T* const output,
  Q15_T* const convBuffer1, Q15_T* const convBuffer2, Q31_T* const treesumBuffer,
  ITER_T N, ITER_T H, ITER_T W, ITER_T CIn, ITER_T CTemp, ITER_T HF, ITER_T WF,
  ITER_T COut, ITER_T HOut, ITER_T WOut, S_ITER_T HPadU, S_ITER_T HPadD,
  S_ITER_T WPadL, S_ITER_T WPadR, ITER_T HStride, ITER_T WStride, SCALE_T depth1,
  SCALE_T depth2, SCALE_T depth3, Q31_T limit1, Q31_T limit2, L_SCALE_T shrU1,
  L_SCALE_T shrB1, L_SCALE_T shrX1, L_SCALE_T shrU2, L_SCALE_T shrB2,
  L_SCALE_T shrX2, L_SCALE_T shrU3, L_SCALE_T shrB3, L_SCALE_T shrW3,
  L_SCALE_T shlU1, L_SCALE_T shlB1, L_SCALE_T shlX1, L_SCALE_T shlU2,
  L_SCALE_T shlB2, L_SCALE_T shlX2, L_SCALE_T shlU3, L_SCALE_T shlB3,
  L_SCALE_T shlW3) {

  S_ITER_T HOffsetFL = (HF - 1) >> 1;
  S_ITER_T WOffsetFL = (WF - 1) >> 1;
  S_ITER_T HOffsetFR = HF >> 1;
  S_ITER_T WOffsetFR = WF >> 1;

  S_ITER_T HOffsetL = HOffsetFL - HPadU;
  S_ITER_T WOffsetL = WOffsetFL - WPadL;
  S_ITER_T HOffsetR = HOffsetFR - HPadD;
  S_ITER_T WOffsetR = WOffsetFR - WPadR;

  for (ITER_T n = 0; n < N; n++) {
    ITER_T margin = 0, nstart = 0;
    if ((S_ITER_T)HF - HPadU - (S_ITER_T)HStride > 0) {
      margin = (ITER_T)((S_ITER_T)HF - HPadU - (S_ITER_T)HStride);
    }
    if (HPadU < 0) {
      // nstart will always be zero unless HPadU is negative.
      nstart = (ITER_T)(-HPadU);
    }

    for (ITER_T i = nstart; i < margin; i++) {
      for (ITER_T j = 0; j < W; j++) {
        for (ITER_T k = 0; k < CTemp; k++) {
          for (ITER_T l = 0; l < CIn; l++) {
            treesumBuffer[l] = ((Q31_T)input[n * H * W * CIn + i * W * CIn + j * CIn + l]) *
                               ((Q31_T)filter1[l * CTemp + k]);
          }

          q31_v_treesum(treesumBuffer, CIn, depth1, 0);
          #ifdef SHIFT
            Q31_T x = (((Q31_T)(((treesumBuffer[0] << shlU1) >> shrU1) + ((BN1B[k] << shlB1) >> shrB1))) *
                        ((Q31_T)BN1W[k]));
          #else
            Q31_T x = (((Q31_T)((treesumBuffer[0] * shlU1) / shrU1 + (BN1B[k] * shlB1) / shrB1)) *
                        ((Q31_T)BN1W[k]));
          #endif
          x = q31_relu(x, limit1);
          #ifdef SHIFT
            convBuffer1[i * W * CTemp + j * CTemp + k] = ((x << shlX1) >> shrX1);
          #else
            convBuffer1[i * W * CTemp + j * CTemp + k] = (x * shlX1) / shrX1;
          #endif
        }
      }
    }

    ITER_T hout = 0;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; hout++, h += (S_ITER_T)HStride) {
      for (ITER_T i = 0; i < HStride; i++) {
        for (ITER_T j = 0; j < W; j++) {
          for (ITER_T k = 0; k < CTemp; k++) {
            ITER_T iRed = (i + margin + hout * HStride) % HF;
            ITER_T iFull = i + margin + hout * HStride;
            for (ITER_T l = 0; l < CIn; l++) {
              if (iFull < H) {
                treesumBuffer[l] = ((Q31_T)input[n * H * W * CIn + iFull * W * CIn + j * CIn + l]) *
                                   ((Q31_T)filter1[l * CTemp + k]);
              } else {
                treesumBuffer[l] = 0;
              }
            }

            q31_v_treesum(treesumBuffer, CIn, depth1, 0);
            #ifdef SHIFT
              Q31_T x = (((Q31_T)(((treesumBuffer[0] << shlU1) >> shrU1) + ((BN1B[k] << shlB1) >> shrB1))) *
                          ((Q31_T)BN1W[k]));
            #else
              Q31_T x = (((Q31_T)((treesumBuffer[0] * shlU1) / shrU1 + (BN1B[k] * shlB1) / shrB1)) *
                          ((Q31_T)BN1W[k]));
            #endif
            x = q31_relu(x, limit1);
            #ifdef SHIFT
              convBuffer1[iRed * W * CTemp + j * CTemp + k] = ((x << shlX1) >> shrX1);
            #else
              convBuffer1[iRed * W * CTemp + j * CTemp + k] = (x * shlX1) / shrX1;
            #endif
          }
        }
      }

      ITER_T wout = 0;
      for (S_ITER_T w = WOffsetL; w < ((S_ITER_T)W) - WOffsetR; wout++, w += ((S_ITER_T)WStride)) {
        for (ITER_T g = 0; g < CTemp; g++) {
          ITER_T counter = 0;
          for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
            for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
              if (((h + hf) < 0) || ((h + hf) >= (S_ITER_T)H) || ((w + wf) < 0) || ((w + wf) >= (S_ITER_T)W)) {
                treesumBuffer[counter] = 0;
              } else {
                treesumBuffer[counter] = ((Q31_T)convBuffer1[(((ITER_T)(h + hf)) % HF) * W * CTemp + ((ITER_T)(w + wf)) * CTemp + g]) *
                                         ((Q31_T)filter2[g * HF * WF + ((ITER_T)(hf + HOffsetFL)) * WF + ((ITER_T)(wf + WOffsetFL))]);
              }
              counter++;
            }
          }

          q31_v_treesum(treesumBuffer, HF * WF, depth2, 0);
          #ifdef SHIFT
            Q31_T x = (((Q31_T)(((treesumBuffer[0] << shlU2) >> shrU2) + ((BN2B[g] << shlB2) >> shrB2))) *
                        ((Q31_T)BN2W[g]));
          #else
            Q31_T x = (((Q31_T)((treesumBuffer[0] * shlU2) / shrU2 + (BN2B[g] * shlB2) / shrB2)) *
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
          for (ITER_T g = 0; g < CTemp; g++) {
            treesumBuffer[g] = ((Q31_T)convBuffer2[g]) * ((Q31_T)filter3[g * COut + i]);
          }

          q31_v_treesum(treesumBuffer, CTemp, depth3, 0);
          #ifdef SHIFT
            output[n * HOut * WOut * COut + hout * WOut * COut + wout * COut + i] =
              (((((Q31_T)(((treesumBuffer[0] << shlU3) >> shrU3) + ((BN3B[i] << shlB3) >> shrB3))) *
                ((Q31_T) BN3W[i])) << shlW3) >> shrW3);
          #else
            output[n * HOut * WOut * COut + hout * WOut * COut + wout * COut + i] =
              ((((Q31_T)((treesumBuffer[0] * shlU3) / shrU3 + (BN3B[i] * shlB3) / shrB3)) *
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
  Q15_T* const convBuffer1, Q15_T* const convBuffer2, Q31_T* const treesumBuffer,
  ITER_T N, ITER_T H, ITER_T W, ITER_T CIn, ITER_T CTemp, ITER_T HF, ITER_T WF,
  ITER_T COut, ITER_T HOut, ITER_T WOut, S_ITER_T HPadU, S_ITER_T HPadD,
  S_ITER_T WPadL, S_ITER_T WPadR, ITER_T HStride, ITER_T WStride, SCALE_T depth1,
  SCALE_T depth2, SCALE_T depth3, Q31_T limit1, Q31_T limit2, L_SCALE_T shrU1,
  L_SCALE_T shrB1, L_SCALE_T shrX1, L_SCALE_T shrU2, L_SCALE_T shrB2,
  L_SCALE_T shrX2, L_SCALE_T shrU3, L_SCALE_T shrB3, L_SCALE_T shrW3,
  L_SCALE_T shlU1, L_SCALE_T shlB1, L_SCALE_T shlX1, L_SCALE_T shlU2,
  L_SCALE_T shlB2, L_SCALE_T shlX2, L_SCALE_T shlU3, L_SCALE_T shlB3,
  L_SCALE_T shlW3) {

  S_ITER_T HOffsetFL = (HF - 1) >> 1;
  S_ITER_T WOffsetFL = (WF - 1) >> 1;
  S_ITER_T HOffsetFR = HF >> 1;
  S_ITER_T WOffsetFR = WF >> 1;

  S_ITER_T HOffsetL = HOffsetFL - HPadU;
  S_ITER_T WOffsetL = WOffsetFL - WPadL;
  S_ITER_T HOffsetR = HOffsetFR - HPadD;
  S_ITER_T WOffsetR = WOffsetFR - WPadR;

  for (ITER_T n = 0; n < N; n++) {
    ITER_T margin = 0, nstart = 0;
    if ((S_ITER_T)HF - HPadU - (S_ITER_T)HStride > 0) {
      margin = (ITER_T)((S_ITER_T)HF - HPadU - (S_ITER_T)HStride);
    }
    if (HPadU < 0) {
      // nstart will always be zero unless HPadU is negative.
      nstart = (ITER_T)(-HPadU);
    }

    for (ITER_T i = nstart; i < margin; i++) {
      for (ITER_T j = 0; j < W; j++) {
        for (ITER_T k = 0; k < CTemp; k++) {
          for (ITER_T l = 0; l < CIn; l++) {
            treesumBuffer[l] = ((Q31_T)input[n * H * W * CIn + i * W * CIn + j * CIn + l]) *
                               ((Q31_T)filter1[l * CTemp + k]);
          }

          q31_v_treesum(treesumBuffer, CIn, depth1, 0);
          #ifdef SHIFT
            Q31_T x = (((Q31_T)(((treesumBuffer[0] << shlU1) >> shrU1) + ((BN1B[k] << shlB1) >> shrB1))) *
                        ((Q31_T)BN1W[k]));
          #else
            Q31_T x = (((Q31_T)((treesumBuffer[0] * shlU1) / shrU1 + (BN1B[k] * shlB1) / shrB1)) *
                        ((Q31_T)BN1W[k]));
          #endif
          x = q31_relu(x, limit1);
          #ifdef SHIFT
            convBuffer1[i * W * CTemp + j * CTemp + k] = ((x << shlX1) >> shrX1);
          #else
            convBuffer1[i * W * CTemp + j * CTemp + k] = (x * shlX1) / shrX1;
          #endif
        }
      }
    }

    ITER_T hout = 0;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; hout++, h += (S_ITER_T)HStride) {
      for (ITER_T i = 0; i < HStride; i++) {
        for (ITER_T j = 0; j < W; j++) {
          for (ITER_T k = 0; k < CTemp; k++) {
            ITER_T iRed = (i + margin + hout * HStride) % HF;
            ITER_T iFull = i + margin + hout * HStride;
            for (ITER_T l = 0; l < CIn; l++) {
              if (iFull < H) {
                treesumBuffer[l] = ((Q31_T)input[n * H * W * CIn + iFull * W * CIn + j * CIn + l]) *
                                   ((Q31_T)filter1[l * CTemp + k]);
              } else {
                treesumBuffer[l] = 0;
              }
            }

            q31_v_treesum(treesumBuffer, CIn, depth1, 0);
            #ifdef SHIFT
              Q31_T x = (((Q31_T)(((treesumBuffer[0] << shlU1) >> shrU1) + ((BN1B[k] << shlB1) >> shrB1))) *
                          ((Q31_T)BN1W[k]));
            #else
              Q31_T x = (((Q31_T)((treesumBuffer[0] * shlU1) / shrU1 + (BN1B[k] * shlB1) / shrB1)) *
                          ((Q31_T)BN1W[k]));
            #endif
            x = q31_relu(x, limit1);
            #ifdef SHIFT
              convBuffer1[iRed * W * CTemp + j * CTemp + k] = ((x << shlX1) >> shrX1);
            #else
              convBuffer1[iRed * W * CTemp + j * CTemp + k] = (x * shlX1) / shrX1;
            #endif
          }
        }
      }

      ITER_T wout = 0;
      for (S_ITER_T w = WOffsetL; w < ((S_ITER_T)W) - WOffsetR; wout++, w += ((S_ITER_T)WStride)) {
        for (ITER_T g = 0; g < CTemp; g++) {
          ITER_T counter = 0;
          for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
            for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
              if (((h + hf) < 0) || ((h + hf) >= (S_ITER_T)H) || ((w + wf) < 0) || ((w + wf) >= (S_ITER_T)W)) {
                treesumBuffer[counter] = 0;
              } else {
                treesumBuffer[counter] = ((Q31_T)convBuffer1[(((ITER_T)(h + hf)) % HF) * W * CTemp + ((ITER_T)(w + wf)) * CTemp + g]) *
                                         ((Q31_T)filter2[g * HF * WF + ((ITER_T)(hf + HOffsetFL)) * WF + ((ITER_T)(wf + WOffsetFL))]);
              }
              counter++;
            }
          }

          q31_v_treesum(treesumBuffer, HF * WF, depth2, 0);
          #ifdef SHIFT
            Q31_T x = (((Q31_T)(((treesumBuffer[0] << shlU2) >> shrU2) + ((BN2B[g] << shlB2) >> shrB2))) *
                        ((Q31_T)BN2W[g]));
          #else
            Q31_T x = (((Q31_T)((treesumBuffer[0] * shlU2) / shrU2 + (BN2B[g] * shlB2) / shrB2)) *
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
          for (ITER_T g = 0; g < CTemp; g++) {
            treesumBuffer[g] = ((Q31_T)convBuffer2[g]) * ((Q31_T)filter3[g * COut + i]);
          }

          q31_v_treesum(treesumBuffer, CTemp, depth3, 0);
          #ifdef SHIFT
            output[n * HOut * WOut * COut + hout * WOut * COut + wout * COut + i] =
              (((((Q31_T)(((treesumBuffer[0] << shlU3) >> shrU3) + ((BN3B[i] << shlB3) >> shrB3))) *
                ((Q31_T) BN3W[i])) << shlW3) >> shrW3);
          #else
            output[n * HOut * WOut * COut + hout * WOut * COut + wout * COut + i] =
              ((((Q31_T)((treesumBuffer[0] * shlU3) / shrU3 + (BN3B[i] * shlB3) / shrB3)) *
                ((Q31_T) BN3W[i])) * shlW3) / shrW3;
          #endif
        }
      }
    }
  }
}
