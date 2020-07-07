// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "quantized_mbconv.h"

void MBConv(const INT_T* const A, const INT_T* const F1, const INT_T* const BN1W,
            const INT_T* const BN1B, const INT_T* const F2, const INT_T* const BN2W,
            const INT_T* const BN2B, const INT_T* const F3, const INT_T* const BN3W,
            const INT_T* const BN3B, INT_T* const C, INT_T* const X, INT_T* const T,
            INTM_T* const U, ITER_T N, ITER_T H, ITER_T W, ITER_T Cin, ITER_T Ct,
            ITER_T HF, ITER_T WF, ITER_T Cout, ITER_T Hout, ITER_T Wout,
            ITER_T HPADL, ITER_T HPADR, ITER_T WPADL, ITER_T WPADR, ITER_T HSTR,
            ITER_T WSTR, SCALE_T D1, SCALE_T D2, SCALE_T D3, INTM_T limit1,
            INTM_T limit2, INTM_T shr1, INTM_T shr2, INTM_T shr3, INTM_T shr4,
            INTM_T shr5, INTM_T shr6, INTM_T shr7, INTM_T shr8, INTM_T shr9,
            INTM_T shl1, INTM_T shl2, INTM_T shl3, INTM_T shl4, INTM_T shl5,
            INTM_T shl6, INTM_T shl7, INTM_T shl8, INTM_T shl9) {
	ITER_T HOffsetL = (HF / 2) - HPADL;
	ITER_T WOffsetL = (WF / 2) - WPADL;
	ITER_T HOffsetR = (HF / 2) - HPADR;
	ITER_T WOffsetR = (WF / 2) - WPADR;

	for (ITER_T n = 0; n < N; n++) {
    ITER_T margin = 0, nstart = 0;
    if (HOffsetL + (HF / 2 + 1) - HSTR > 0) {
      margin = HOffsetL + (HF / 2 + 1) - HSTR;
    }
    if (HOffsetL - (HF / 2) > 0) {
      nstart = HOffsetL - (HF / 2);
    }

		for (ITER_T i = nstart; i < margin; i++) {
			for (ITER_T j = 0; j < W; j++) {
				for (ITER_T k = 0; k < Ct; k++) {
					for (ITER_T l = 0; l < Cin; l++) {
						U[l] = ((INTM_T) A[n * H * W * Cin + i * W * Cin + j * Cin + l]) *
                   ((INTM_T) F1[l * Ct + k]);
					}

          v_q_treesum(U, Cin, D1, 0);
					INTM_T x = (((INTM_T)((U[0] * shl1) / shr1 + (BN1B[k] * shl2) / shr2)) *
                     ((INTM_T)BN1W[k]));
					X[i * W * Ct + j * Ct + k] =  (truncate_relu(x, limit1) * shl3) / shr3;
				}
			}
		}

		for (ITER_T h = HOffsetL, hout = 0; h < H - HOffsetR; hout++, h += HSTR) {
			for (ITER_T i = 0; i < HSTR; i++) {
				for (ITER_T j = 0; j < W; j++) {
					for (ITER_T k = 0; k < Ct; k++) {
						ITER_T iRed = (i + margin + hout * HSTR) % HF;
            ITER_T iFull = i + margin + hout * HSTR;
						X[iRed * W * Ct + j * Ct + k] = 0;
						for (ITER_T l = 0; l < Cin; l++) {
              if (iFull < H) {
                U[l] = ((INTM_T) A[n * H * W * Cin + iFull * W * Cin + j * Cin + l]) *
                       ((INTM_T) F1[l * Ct + k]);
              } else {
                U[l] = 0;
              }
						}

						v_q_treesum(U, Cin, D1, 0);
						INTM_T x = (((INTM_T)((U[0] * shl1) / shr1 + (BN1B[k] * shl2) / shr2)) *
                        ((INTM_T)BN1W[k]));
						X[iRed * W * Ct + j * Ct + k] = (truncate_relu(x, limit1) * shl3) / shr3;
					}
				}
			}

			for (ITER_T w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
				for (ITER_T g = 0; g < Ct; g++) {
					ITER_T counter = 0;
					for (ITER_T hf = -(HF >> 1); hf <= (HF >> 1); hf++) {
						for (ITER_T wf = -(WF >> 1); wf <= (WF >> 1); wf++) {
              if (((h + hf) < 0) || ((h + hf) >= H) || ((w + wf) < 0) || ((w + wf) >= W)) {
                U[counter] = 0;
              } else {
                U[counter] = ((INTM_T) X[((h + hf) % HF) * W * Ct + (w + wf) * Ct + g]) *
                             ((INTM_T) F2[g * HF * WF + (hf + (HF >> 1)) * WF + (wf + (WF >> 1))]);
              }
							counter++;
						}
					}

          v_q_treesum(U, HF * WF, D2, 0);
					INTM_T x = (((INTM_T)((U[0] * shl4) / shr4 + (BN2B[g] * shl5) / shr5)) *
                     ((INTM_T)BN2W[g]));
					T[g] = (truncate_relu(x, limit2) * shl6) / shr6;
				}

				for (ITER_T i = 0; i < Cout; i++) {
					for (ITER_T g = 0; g < Ct; g++) {
						U[g] = ((INTM_T) T[g]) * ((INTM_T) F3[g * Cout + i]);
          }

          v_q_treesum(U, Ct, D3, 0);
					C[n * Hout * Wout * Cout + hout * Wout * Cout + wout * Cout + i] =
            ((((INTM_T)((U[0] * shl7) / shr7 + (BN3B[i] * shl8) / shr8)) *
              ((INTM_T) BN3W[i])) * shl9) / shr9;
				}
			}
		}
	}
}
